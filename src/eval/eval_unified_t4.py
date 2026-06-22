"""
EVALUATION NOTEBOOK (unified, v5) - 2x T4 COMPUTE BUILD
======================================================
Identical evaluation logic to v5 (same modes, same items, same gold labels,
same greedy decoding, same extractors). The ONLY differences are in the COMPUTE
section, adapted from a single RTX Pro 6000 (96 GB) to 2x NVIDIA T4 (2x16 GB).

What changed for the T4s (all marked with  # >>> T4):
  1. dtype: T4 is Turing (compute 7.5) and has NO hardware bf16. bfloat16 runs
     emulated and slow, so we use float16 instead. (auto-detected; falls back to
     bf16 only if the GPU actually supports it.)
  2. sharding: model + KV cache are spread across BOTH GPUs with
     device_map="auto" + max_memory, instead of device_map="cuda" (one card).
  3. input device: with a sharded model, model.device is unreliable, so inputs
     are moved to the input-embedding layer's device (the first shard).
  4. batch size / input length: 16 GB per card is far smaller than 96 GB, so
     BATCH_SIZE and INPUT_MAX are reduced. Think modes (1024 new tokens) carry a
     big KV cache, so they use an even smaller batch.

MODES (unchanged)
-----------------
  base_fewshot   : raw completion + few-shot exemplars + base extractors.
  think_zeroshot : canonical <|im_start|> chat, zero-shot, <answer> parsing.
  think_fewshot  : canonical chat + few-shot exemplars in <think>/<answer> form.
"""

import re, json, time, os, random, torch
from collections import defaultdict
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================================================================
# CONFIG
# ======================================================================
EVAL_MODE    = "think_zeroshot"   # "base_fewshot" | "think_zeroshot" | "think_fewshot"
MODEL_PATH   = "inputs/models/distilled-qwen-sft-basic/pear_sft_epoch1"
DATASET_ROOT = "inputs/data/qwen-riva-datasets"
OUT          = "outputs"
MMLU_SEED    = 42

# >>> T4: 16 GB/card instead of 96 GB. Smaller batches; think modes hold a much
#         larger KV cache (1024 new tokens), so they get a smaller batch still.
IS_THINK     = EVAL_MODE in ("think_zeroshot", "think_fewshot")
BATCH_SIZE   = 4 if IS_THINK else 8
INPUT_MAX    = 2048               # was 4096; few-shot prompts still fit, saves KV memory

# >>> T4: cap per-card allocation so accelerate leaves headroom for the KV cache
#         and CUDA context. Tune down if you still see OOM (e.g. "13GiB").
PER_GPU_MEM  = "14GiB"
CPU_MEM      = "24GiB"            # spill target if the model is too big for 2 cards

os.makedirs(f"{OUT}/eval_results", exist_ok=True)

# think modes reason before answering, so they need large budgets; base does not
MAX_NEW = ({"gsm8k": 1024, "mmlu": 1024, "strategyqa": 1024} if IS_THINK
           else {"gsm8k": 512, "mmlu": 8, "strategyqa": 128})

GSM_STOPS  = ["\nQuestion:", "\nQ:"]
MMLU_STOPS = ["\nQuestion:", "\nQ:"]
SQA_STOPS  = ["\nQ:", "\nQuestion:"]

# ======================================================================
# FEW-SHOT EXEMPLARS  (shared question pool; reformatted per mode)
# ======================================================================
GSM8K_SHOTS = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6"),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5"),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39"),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8"),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9"),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. #### 29"),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. #### 33"),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. #### 8"),
]
MMLU_SHOTS = [
    ("Which of the following is the body cavity that contains the pituitary gland?\nA) Abdominal\nB) Cranial\nC) Pleural\nD) Spinal", "B"),
    ("What is the embryological origin of the hyoid bone?\nA) The first pharyngeal arch\nB) The first and second pharyngeal arches\nC) The second pharyngeal arch\nD) The second and third pharyngeal arches", "D"),
    ("In a genetic test of a newborn, a rare genetic disorder is found that has X-linked recessive transmission. Which of the following statements is likely true?\nA) The newborn is a carrier\nB) The newborn is unaffected\nC) The mother is a carrier\nD) The father is affected", "C"),
    ("A set of batteries can power a flashlight for up to 500 hours. If the flashlight has already been used for 238 hours, for how many more hours can it be used?\nA) 262\nB) 264\nC) 738\nD) 976", "A"),
    ("Which of the following statements about the lanthanide elements is NOT correct?\nA) They are all metals.\nB) They have atomic numbers between 57 and 71.\nC) They are also called the rare earth elements.\nD) Their chemical properties vary widely across the series.", "D"),
]
SQA_SHOTS = [
    ("Do hamsters provide food for any animals?",
     "Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. Yes"),
    ("Could Brooke Shields succeed at the 1992 Olympics?",
     "Brooke Shields was born on May 31, 1965. The 1992 Olympics were in Barcelona. Brooke Shields would have been 27 at the time. 27 year old can compete in the Olympics. Yes"),
    ("Hydrogen's atomic number squared exceeds number of Spice Girls?",
     "Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. 1 does not exceed 5. No"),
    ("Is it common to see frost during some college commencements?",
     "College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, frost can be common at some college commencements. Yes"),
    ("Could a llama birth twice during War in Vietnam (1945-46)?",
     "The War in Vietnam was 6 months. The gestation period of a llama is 11 months. Thus, a llama could not give birth twice during the War in Vietnam. No"),
    ("Would a pear sink in water?",
     "The density of a pear is about 0.6 g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would not sink. No"),
]

# ======================================================================
# CANONICAL THINK FORMAT  (copied verbatim from FILE 3 trace generation)
# ======================================================================
SYS = {
    "math": ("Solve the problem. Reason step by step inside <think> and </think>, "
             "then give only the final answer inside <answer> and </answer>."),
    "commonsense": ("Decide whether the answer is Yes or No. Inside <think> and </think>, "
                    "break the question into sub-questions, answer each one, then conclude. "
                    "Put only Yes or No inside <answer> and </answer>."),
    "mmlu": ("Choose the correct option. Reason inside <think> and </think>, then put only "
             "the letter (A, B, C, or D) inside <answer> and </answer>."),
}
def cprompt(domain, qblock):
    return (f"<|im_start|>system\n{SYS[domain]}<|im_end|>\n"
            f"<|im_start|>user\n{qblock}<|im_end|>\n<|im_start|>assistant\n")

def mmlu_block(ex):
    return f"Question: {ex['question']}\n" + "\n".join(
        f"{'ABCD'[i]}) {c}" for i, c in enumerate(ex["choices"]))

# think few-shot: rewrite the shared exemplars into <think>/<answer> turns
def _gsm_shot_ta(q, a):
    reasoning, final = (a.split("####", 1) if "####" in a else (a, a))
    return f"Problem: {q}", f"<think>\n{reasoning.strip()}\n</think>\n<answer>{final.strip()}</answer>"
def _sqa_shot_ta(q, a):
    m = re.findall(r"\b(Yes|No)\b", a)
    final = m[-1] if m else "No"
    reasoning = a.rsplit(final, 1)[0].strip()
    return f"Question: {q}", f"<think>\n{reasoning}\n</think>\n<answer>{final}</answer>"
def _mmlu_shot_ta(q, letter):
    return f"Question: {q}", f"<think>\nThe correct option is {letter}.\n</think>\n<answer>{letter}</answer>"

GSM_TA  = [_gsm_shot_ta(q, a)  for q, a in GSM8K_SHOTS]
SQA_TA  = [_sqa_shot_ta(q, a)  for q, a in SQA_SHOTS]
MMLU_TA = [_mmlu_shot_ta(q, l) for q, l in MMLU_SHOTS]

def think_fewshot(domain, qblock, shots_ta):
    s = f"<|im_start|>system\n{SYS[domain]}<|im_end|>\n"
    for sq, sa in shots_ta:
        s += f"<|im_start|>user\n{sq}<|im_end|>\n<|im_start|>assistant\n{sa}<|im_end|>\n"
    s += f"<|im_start|>user\n{qblock}<|im_end|>\n<|im_start|>assistant\n"
    return s

# ======================================================================
# BASE (v3) PROMPT BUILDERS
# ======================================================================
def build_gsm_base(question):
    ctx = "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a in GSM8K_SHOTS)
    return ctx + f"Question: {question}\nAnswer:"
def build_mmlu_base(question, choices, subject):
    ctx = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    ctx += "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a in MMLU_SHOTS)
    fc = "\n".join(f"{'ABCD'[i]}) {c}" for i, c in enumerate(choices))
    return ctx + f"Question: {question}\n{fc}\nAnswer:"
def build_sqa_base(question):
    ctx = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in SQA_SHOTS)
    return ctx + f"Q: {question}\nA:"

# ======================================================================
# EXTRACTORS
# ======================================================================
def normalize_number(s):
    s = str(s).strip().rstrip(".,;:").replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s

# --- base (v3) ---
def truncate_at_stops(text, stops):
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]
def extract_gsm_base(r):
    block = truncate_at_stops(r, GSM_STOPS)
    if "####" in block:
        raw = block.split("####")[1].strip().split("\n")[0]
        parts = raw.split()
        return normalize_number(parts[0]) if parts else None
    nums = re.findall(r"-?\$?[\d,]+\.?\d*", block)
    return normalize_number(nums[-1]) if nums else None
def extract_mmlu_base(r):
    block = truncate_at_stops(r, MMLU_STOPS).strip()
    m = re.match(r"^([A-Da-d])", block)
    if m: return m.group(1).upper()
    m = re.search(r"\b([A-Da-d])\b", block)
    return m.group(1).upper() if m else None
def extract_sqa_base(r):
    block = truncate_at_stops(r, SQA_STOPS)
    m = re.findall(r"\b(Yes|No)\b", block, re.IGNORECASE)
    return m[-1].capitalize() if m else None

# --- think (v4) ---
def extract_tag(text, tag="answer"):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None
def extract_boxed(text):
    idx = text.rfind(r"\boxed{")
    if idx == -1: return None
    start = text.index("{", idx) + 1
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{": depth += 1
        elif text[i] == "}": depth -= 1
        i += 1
    return text[start:i - 1].strip() if depth == 0 else None
def extract_gsm_think(r):
    tagged = extract_tag(r)
    if tagged is not None:
        b = extract_boxed(tagged); val = b if b is not None else tagged
        nums = re.findall(r"-?[\d,]+\.?\d*(?:/[\d,]+)?", val)
        return normalize_number(nums[-1]) if nums else normalize_number(val)
    b = extract_boxed(r)
    if b is not None: return normalize_number(b)
    nums = re.findall(r"-?\$?[\d,]+\.?\d*", r)
    return normalize_number(nums[-1]) if nums else None
def extract_mmlu_think(r):
    tagged = extract_tag(r)
    if tagged is not None:
        m = re.search(r"\b([ABCD])\b", tagged.strip(), re.IGNORECASE)
        if m: return m.group(1).upper()
    m = re.findall(r"\b([ABCD])\b", r, re.IGNORECASE)
    return m[-1].upper() if m else None
def extract_sqa_think(r):
    tagged = extract_tag(r)
    src = tagged if tagged is not None else r
    m = re.findall(r"\b(Yes|No)\b", src, re.IGNORECASE)
    return m[-1].capitalize() if m else None

def has_answer_tag(t):
    return re.search(r"<answer>.*?</answer>", t, re.DOTALL | re.IGNORECASE) is not None

# ======================================================================
# MODE DISPATCH
# ======================================================================
if EVAL_MODE == "base_fewshot":
    gsm_build  = lambda ex: build_gsm_base(ex["question"])
    mmlu_build = lambda ex: build_mmlu_base(ex["question"], ex["choices"], ex.get("subject", "general"))
    sqa_build  = lambda ex: build_sqa_base(ex["question"])
    gsm_ext, mmlu_ext, sqa_ext = extract_gsm_base, extract_mmlu_base, extract_sqa_base
elif EVAL_MODE == "think_zeroshot":
    gsm_build  = lambda ex: cprompt("math", f"Problem: {ex['question']}")
    mmlu_build = lambda ex: cprompt("mmlu", mmlu_block(ex))
    sqa_build  = lambda ex: cprompt("commonsense", f"Question: {ex['question']}")
    gsm_ext, mmlu_ext, sqa_ext = extract_gsm_think, extract_mmlu_think, extract_sqa_think
elif EVAL_MODE == "think_fewshot":
    gsm_build  = lambda ex: think_fewshot("math", f"Problem: {ex['question']}", GSM_TA)
    mmlu_build = lambda ex: think_fewshot("mmlu", mmlu_block(ex), MMLU_TA)
    sqa_build  = lambda ex: think_fewshot("commonsense", f"Question: {ex['question']}", SQA_TA)
    gsm_ext, mmlu_ext, sqa_ext = extract_gsm_think, extract_mmlu_think, extract_sqa_think
else:
    raise ValueError(f"unknown EVAL_MODE: {EVAL_MODE}")

# ======================================================================
# MODEL  (2x T4 COMPUTE)
# ======================================================================
print(f"EVAL_MODE = {EVAL_MODE}")
print(f"Loading model: {MODEL_PATH}")

# >>> T4: pick a dtype the hardware actually accelerates. Turing (T4) has no
#         hardware bf16, so use fp16 there. bf16 only if the GPU supports it.
# >>> T4: DO NOT trust torch.cuda.is_bf16_supported() - it returns True on T4
#         but bf16 is EMULATED (no hardware), ~10x slower. Force fp16 tensor cores.
DTYPE = torch.float16
print("dtype: float16 (forced - T4 has no hardware bf16; emulated bf16 is ~10x slower)")

n_gpu = torch.cuda.device_count()
print(f"visible GPUs: {n_gpu}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# >>> T4: shard across BOTH cards instead of device_map="cuda" (single card).
#         max_memory keeps headroom per card for the KV cache + CUDA context.
max_memory = {i: PER_GPU_MEM for i in range(n_gpu)}
max_memory["cpu"] = CPU_MEM
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map="auto",
    max_memory=max_memory,
).eval()

# >>> T4: with a sharded model, model.device is unreliable. Inputs must land on
#         the device that holds the input-embedding layer (the first shard).
INPUT_DEVICE = model.get_input_embeddings().weight.device
print(f"device map: {getattr(model, 'hf_device_map', 'single')}")
print(f"input device: {INPUT_DEVICE}")

EOS_IDS = [tokenizer.eos_token_id]
if IS_THINK:
    _im = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(_im, int) and _im >= 0 and _im != tokenizer.eos_token_id:
        EOS_IDS.append(_im)
print(f"Model loaded. eos ids: {EOS_IDS}\n")

def generate_batch(prompts, max_new_tokens):
    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    order   = sorted(range(len(prompts)), key=lambda i: lengths[i], reverse=True)
    inputs  = tokenizer([prompts[i] for i in order], return_tensors="pt",
                        truncation=True, max_length=INPUT_MAX, padding=True).to(INPUT_DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id, eos_token_id=EOS_IDS)
    p_len = inputs["input_ids"].shape[1]
    dec = [tokenizer.decode(out[i][p_len:], skip_special_tokens=True).strip() for i in range(len(order))]
    res = [None] * len(prompts)
    for si, oi in enumerate(order):
        res[oi] = dec[si]
    # >>> T4: 16 GB fills fast across many batches; release cached blocks so
    #         fragmentation doesn't accumulate into an OOM mid-run.
    del inputs, out
    torch.cuda.empty_cache()
    return res

# ======================================================================
# TEST SETS
# ======================================================================
gsm_test = load_from_disk(f"{DATASET_ROOT}/test/gsm8k")
sqa_test = load_from_disk(f"{DATASET_ROOT}/test/strategyqa")
mml_test = load_from_disk(f"{DATASET_ROOT}/test/mmlu")

rng = random.Random(MMLU_SEED)
subj_to_idx = defaultdict(list)
for idx, ex in enumerate(mml_test):
    subj_to_idx[ex["subject"]].append(idx)
mmlu_indices = []
for subj in sorted(subj_to_idx):
    mmlu_indices.extend(rng.sample(subj_to_idx[subj], min(10, len(subj_to_idx[subj]))))

print(f"GSM8K {len(gsm_test)} | StrategyQA {len(sqa_test)} | MMLU {len(mmlu_indices)} (stratified)\n")
phase_name = os.path.basename(MODEL_PATH.rstrip("/"))

def run_benchmark(name, items, build, extract, gold_fn, mx):
    print(f"--- {name} ---")
    correct = tags = 0
    n = len(items)
    for bs in range(0, n, BATCH_SIZE):
        batch = [items[i] for i in range(bs, min(bs + BATCH_SIZE, n))]
        resps = generate_batch([build(ex) for ex in batch], mx)
        for ex, resp in zip(batch, resps):
            if IS_THINK and has_answer_tag(resp):
                tags += 1
            pred, gold = extract(resp), gold_fn(ex)
            if pred is not None and str(pred) == str(gold):
                correct += 1
        if bs % (BATCH_SIZE * 10) == 0:
            done = min(bs + BATCH_SIZE, n)
            print(f"  {done}/{n} | acc {round(correct / done * 100, 1)}%")
    score = round(correct / n * 100, 1)
    tag_str = f" | <answer> rate {tags}/{n} = {tags/n:.0%}" if IS_THINK else ""
    print(f"{name}: {score}%{tag_str}\n")
    return score, (round(tags / n, 3) if IS_THINK else None)

gsm_items = list(gsm_test)
sqa_items = list(sqa_test)
mml_items = [mml_test[i] for i in mmlu_indices]

# GSM8K: gsm extractors already return normalize_number(...), and gold is
# normalized here, so run_benchmark's str(pred)==str(gold) is a clean match.
gsm_score, gsm_tag = run_benchmark(
    "GSM8K", gsm_items, gsm_build, gsm_ext,
    lambda ex: normalize_number(ex["answer"].split("####")[-1].strip()), MAX_NEW["gsm8k"])

mmlu_score, mmlu_tag = run_benchmark(
    "MMLU", mml_items, mmlu_build, mmlu_ext,
    lambda ex: "ABCD"[ex["answer"]], MAX_NEW["mmlu"])

sqa_score, sqa_tag = run_benchmark(
    "StrategyQA", sqa_items, sqa_build, sqa_ext,
    lambda ex: "Yes" if ex["answer"] else "No", MAX_NEW["strategyqa"])

# ======================================================================
# RESULTS + SAVE
# ======================================================================
avg = round((gsm_score + mmlu_score + sqa_score) / 3, 1)
print("=" * 55)
print(f"RESULTS: {phase_name}  [{EVAL_MODE}]")
print("=" * 55)
print(f"GSM8K      : {gsm_score}%")
print(f"MMLU       : {mmlu_score}%")
print(f"StrategyQA : {sqa_score}%")
print(f"Average    : {avg}%")
print("=" * 55)

output = {
    "phase": phase_name, "eval_mode": EVAL_MODE,
    "model_type": "base" if EVAL_MODE == "base_fewshot" else "sft_think_answer",
    "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    "scores": {"gsm8k": gsm_score, "mmlu": mmlu_score, "strategyqa": sqa_score, "average": avg},
    "answer_tag_rate": {"gsm8k": gsm_tag, "mmlu": mmlu_tag, "strategyqa": sqa_tag},
    "compute": {
        "gpus": f"{n_gpu}x (device_map=auto, max_memory per card {PER_GPU_MEM})",
        "dtype": str(DTYPE), "batch_size": BATCH_SIZE, "input_max": INPUT_MAX,
    },
    "eval_protocol": {
        "mode": EVAL_MODE, "decoding": "greedy (do_sample=False, no repetition_penalty)",
        "max_new_tokens": MAX_NEW,
        "gsm8k": f"full test ({len(gsm_items)} items)",
        "mmlu": f"stratified {len(mml_items)}-item sample (seed={MMLU_SEED}, 10/subject)",
        "strategyqa": f"full test ({len(sqa_items)} items)",
    },
    "citations": {
        "harness": "Gao et al. 2023, zenodo 10256836",
        "gsm8k": "Cobbe et al. 2021, arXiv:2110.14168",
        "mmlu": "Hendrycks et al. 2021, arXiv:2009.03300",
        "strategyqa": "Geva et al. 2021, arXiv:2101.02235",
        "few_shot": "Brown et al. 2020, arXiv:2005.14165",
        "cot": "Wei et al. 2022, arXiv:2201.11903",
    },
}
out_path = f"{OUT}/eval_results/{phase_name}__{EVAL_MODE}.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")
