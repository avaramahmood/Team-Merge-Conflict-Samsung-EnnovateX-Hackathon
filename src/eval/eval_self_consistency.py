"""
EVALUATION NOTEBOOK - maj@8 SELF-CONSISTENCY (v3-sc)
=====================================================
Same harness as the greedy v3 eval, but each question is answered by
majority vote over K sampled chains (Wang et al. 2022, self-consistency).

  GSM8K      - 8-shot CoT, maj@K
  MMLU       - 5-shot per-subject, stratified 570-item sample, maj@K
  StrategyQA - 6-shot, maj@K

Prompts, extractors, gold parsing, stratified MMLU sample: IDENTICAL to greedy v3.
Only the decoder changes (greedy -> temperature sampling) plus a vote step.

*** FAIRNESS - READ THIS ***
Your hackathon baseline (qwen2.5-7b: G 82.9 / M 72.8 / S 77.9) was measured GREEDY.
maj@K lifts every model, including the base. For a defensible "+5 over baseline"
claim, run THIS notebook on the BASE model too and compare maj@K-to-maj@K, OR
report greedy-to-greedy. Do not compare your maj@K vs the base's greedy - judges
will normalize that. maj@K is a margin booster and an honest second column, not a
substitute for a model-level win.

NOTE ON MMLU + SC: MMLU here emits only a letter (no CoT), so sampling diversity
is low and maj@K barely moves it. That is expected. GSM8K and SQA have real CoT,
so SC helps them. Left as-is to keep the structure identical to your greedy eval.

CHANGES vs greedy v3
--------------------
[SC 1] do_sample=True, temperature=SC_TEMP, top_p=SC_TOP_P, num_return_sequences=K.
[SC 2] generate_batch returns K decodings per prompt; majority_vote() picks the mode
       of the extracted answers (None extractions dropped before voting).
[SC 3] Per-benchmark BATCH_SIZE (GSM8K smaller: K×512 tokens is the heavy case).
All FIX 1-6 from the greedy notebook are preserved verbatim.

CITATIONS
---------
  [1] Gao et al. (2023). lm-evaluation-harness. https://doi.org/10.5281/zenodo.10256836
  [2] Cobbe et al. (2021). GSM8K. arXiv:2110.14168
  [3] Hendrycks et al. (2021). MMLU. arXiv:2009.03300
  [4] Geva et al. (2021). StrategyQA. arXiv:2101.02235
  [5] Brown et al. (2020). Few-shot. arXiv:2005.14165
  [6] Wei et al. (2022). Chain-of-thought. arXiv:2201.11903
  [7] Wang et al. (2022). Self-Consistency. arXiv:2203.11171
"""

import re, json, time, os, random, torch
from collections import defaultdict, Counter
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================================================================
# CONFIG
# ======================================================================
MODEL_PATH   = "inputs/models/qwen2-5-7b/qwen2.5-7b"
DATASET_ROOT = "inputs/data/qwen-riva-datasets"
OUT          = "outputs"
MMLU_SEED    = 42   # fixed -> reproducible 570-item stratified sample

# [SC 1] self-consistency knobs
K_SAMPLES = 8       # maj@K
SC_TEMP   = 0.7     # Wang et al. 2022 use ~0.5-0.7
SC_TOP_P  = 0.95
SC_SEED   = 1234    # sampling reproducibility

# [SC 3] per-benchmark batch (questions per generate call). K samples each are
# produced via num_return_sequences, so GPU load ≈ BATCH_SIZE × K sequences.
BATCH_SIZE = {"gsm8k": 8, "mmlu": 16, "strategyqa": 12}

os.makedirs(f"{OUT}/eval_results", exist_ok=True)
torch.manual_seed(SC_SEED); random.seed(SC_SEED)

# MMLU only needs 1 letter; GSM8K/SQA need room for CoT chain
MAX_NEW = {"gsm8k": 512, "mmlu": 8, "strategyqa": 128}

GSM_STOPS  = ["\nQuestion:", "\nQ:"]
MMLU_STOPS = ["\nQuestion:", "\nQ:"]
SQA_STOPS  = ["\nQ:", "\nQuestion:"]

# ======================================================================
# FEW-SHOT EXAMPLES  (identical to greedy v3)
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
# PROMPT BUILDERS  (identical to greedy v3)
# ======================================================================
def build_gsm_prompt(question):
    ctx = ""
    for q, a in GSM8K_SHOTS:
        ctx += f"Question: {q}\nAnswer: {a}\n\n"
    return ctx + f"Question: {question}\nAnswer:"

def build_mmlu_prompt(question, choices, subject):
    ctx = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    for q, a in MMLU_SHOTS:
        ctx += f"Question: {q}\nAnswer: {a}\n\n"
    letters = "ABCD"
    formatted_choices = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(choices))
    return ctx + f"Question: {question}\n{formatted_choices}\nAnswer:"

def build_sqa_prompt(question):
    ctx = ""
    for q, a in SQA_SHOTS:
        ctx += f"Q: {q}\nA: {a}\n\n"
    return ctx + f"Q: {question}\nA:"

# ======================================================================
# ANSWER EXTRACTION  (identical to greedy v3)
# ======================================================================
def normalize_number(s):
    s = str(s).strip().rstrip(".,;:").replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s

def truncate_at_stops(text, stops):
    cut = len(text)
    for s in stops:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]

def extract_gsm(response):
    block = truncate_at_stops(response, GSM_STOPS)
    if "####" in block:
        raw   = block.split("####")[1].strip().split("\n")[0]
        parts = raw.split()
        return normalize_number(parts[0]) if parts else None
    nums = re.findall(r"-?\$?[\d,]+\.?\d*", block)
    return normalize_number(nums[-1]) if nums else None

def extract_mmlu(response):
    block = truncate_at_stops(response, MMLU_STOPS).strip()
    m = re.match(r"^([A-Da-d])", block)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-Da-d])\b", block)
    return m.group(1).upper() if m else None

def extract_sqa(response):
    block   = truncate_at_stops(response, SQA_STOPS)
    matches = re.findall(r"\b(Yes|No)\b", block, re.IGNORECASE)
    return matches[-1].capitalize() if matches else None

# ======================================================================
# [SC 2] MAJORITY VOTE
# ======================================================================
def majority_vote(answers):
    """Mode of the non-None extracted answers. Ties -> Counter's first-seen max.
    Returns (voted_answer, n_valid_votes)."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 0
    winner, _ = Counter(valid).most_common(1)[0]
    return winner, len(valid)

# ======================================================================
# MODEL LOADING
# ======================================================================
print(f"Loading model: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print("Model loaded.\n")

# ======================================================================
# [SC 1] BATCH GENERATION - K samples/prompt via num_return_sequences
# ======================================================================
def generate_batch_sc(prompts, max_new_tokens, k=K_SAMPLES):
    """Returns a list (len = len(prompts)) of lists (len = k) of decoded strings."""
    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    order   = sorted(range(len(prompts)), key=lambda i: lengths[i], reverse=True)
    inputs  = tokenizer(
        [prompts[i] for i in order],
        return_tensors="pt", truncation=True, max_length=3072, padding=True,
    ).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=SC_TEMP, top_p=SC_TOP_P,
            num_return_sequences=k,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    p_len = inputs["input_ids"].shape[1]
    # outputs is [len(order)*k]; row (si*k + j) is sample j of sorted input si
    result = [None] * len(prompts)
    for si, oi in enumerate(order):
        samples = []
        for j in range(k):
            seq = outputs[si * k + j][p_len:]
            samples.append(tokenizer.decode(seq, skip_special_tokens=True).strip())
        result[oi] = samples
    return result

# ======================================================================
# LOAD TEST SETS
# ======================================================================
gsm_test = load_from_disk(f"{DATASET_ROOT}/test/gsm8k")
sqa_test = load_from_disk(f"{DATASET_ROOT}/test/strategyqa")
mml_test = load_from_disk(f"{DATASET_ROOT}/test/mmlu")

rng         = random.Random(MMLU_SEED)
subj_to_idx = defaultdict(list)
for idx, ex in enumerate(mml_test):
    subj_to_idx[ex["subject"]].append(idx)
mmlu_indices = []
for subj in sorted(subj_to_idx):
    mmlu_indices.extend(rng.sample(subj_to_idx[subj], min(10, len(subj_to_idx[subj]))))

print("Test sets:")
print(f"  GSM8K:      {len(gsm_test)} items (full)")
print(f"  StrategyQA: {len(sqa_test)} items (full)")
print(f"  MMLU:       {len(mmlu_indices)} items (stratified, {len(subj_to_idx)} subjects)")
print(f"  Self-consistency: maj@{K_SAMPLES} | T={SC_TEMP} top_p={SC_TOP_P}\n")

phase_name = os.path.basename(MODEL_PATH) + f"_maj{K_SAMPLES}"

# ======================================================================
# GSM8K
# ======================================================================
print("--- GSM8K ---")
gsm_correct = 0; gsm_n = len(gsm_test); B = BATCH_SIZE["gsm8k"]
for bs in range(0, gsm_n, B):
    batch     = gsm_test.select(range(bs, min(bs + B, gsm_n)))
    prompts   = [build_gsm_prompt(ex["question"]) for ex in batch]
    samples   = generate_batch_sc(prompts, MAX_NEW["gsm8k"])
    for ex, k_resps in zip(batch, samples):
        pred, _ = majority_vote([extract_gsm(r) for r in k_resps])
        gold = ex["answer"].split("####")[-1].strip()
        if pred is not None and normalize_number(pred) == normalize_number(gold):
            gsm_correct += 1
    if bs % (B * 20) == 0:
        done = min(bs + B, gsm_n)
        print(f"  {done}/{gsm_n} | acc {round(gsm_correct / done * 100, 1)}%")
gsm_score = round(gsm_correct / gsm_n * 100, 1)
print(f"GSM8K: {gsm_score}%\n")

# ======================================================================
# MMLU
# ======================================================================
print("--- MMLU ---")
mmlu_correct = 0; mmlu_n = len(mmlu_indices); B = BATCH_SIZE["mmlu"]
for bs in range(0, mmlu_n, B):
    batch     = [mml_test[i] for i in mmlu_indices[bs: bs + B]]
    prompts   = [build_mmlu_prompt(ex["question"], ex["choices"], ex.get("subject", "general")) for ex in batch]
    samples   = generate_batch_sc(prompts, MAX_NEW["mmlu"])
    for ex, k_resps in zip(batch, samples):
        pred, _ = majority_vote([extract_mmlu(r) for r in k_resps])
        gold_letter = "ABCD"[ex["answer"]]
        if pred is not None and pred.upper() == gold_letter:
            mmlu_correct += 1
    if bs % (B * 5) == 0:
        done = min(bs + B, mmlu_n)
        print(f"  {done}/{mmlu_n} | acc {round(mmlu_correct / done * 100, 1)}%")
mmlu_score = round(mmlu_correct / mmlu_n * 100, 1)
print(f"MMLU: {mmlu_score}%\n")

# ======================================================================
# StrategyQA
# ======================================================================
print("--- StrategyQA ---")
sqa_correct = 0; sqa_n = len(sqa_test); B = BATCH_SIZE["strategyqa"]
for bs in range(0, sqa_n, B):
    batch     = sqa_test.select(range(bs, min(bs + B, sqa_n)))
    prompts   = [build_sqa_prompt(ex["question"]) for ex in batch]
    samples   = generate_batch_sc(prompts, MAX_NEW["strategyqa"])
    for ex, k_resps in zip(batch, samples):
        pred, _ = majority_vote([extract_sqa(r) for r in k_resps])
        gold = "Yes" if ex["answer"] else "No"
        if pred is not None and pred.capitalize() == gold:
            sqa_correct += 1
    if bs % (B * 20) == 0:
        done = min(bs + B, sqa_n)
        print(f"  {done}/{sqa_n} | acc {round(sqa_correct / done * 100, 1)}%")
sqa_score = round(sqa_correct / sqa_n * 100, 1)
print(f"StrategyQA: {sqa_score}%\n")

# ======================================================================
# RESULTS
# ======================================================================
avg = round((gsm_score + mmlu_score + sqa_score) / 3, 1)
print("=" * 55)
print(f"RESULTS: {phase_name}")
print("=" * 55)
print(f"GSM8K      (8-shot CoT,  maj@{K_SAMPLES}, {gsm_n} items): {gsm_score}%")
print(f"MMLU       (5-shot,      maj@{K_SAMPLES}, {mmlu_n} items): {mmlu_score}%")
print(f"StrategyQA (6-shot,      maj@{K_SAMPLES}, {sqa_n} items): {sqa_score}%")
print(f"Average:                              {avg}%")
print("=" * 55)

# ======================================================================
# SAVE
# ======================================================================
output = {
    "phase":      phase_name,
    "model_type": "base",
    "timestamp":  time.strftime("%Y-%m-%d %H:%M"),
    "scores": {"gsm8k": gsm_score, "mmlu": mmlu_score, "strategyqa": sqa_score, "average": avg},
    "eval_protocol": {
        "gsm8k":      f"8-shot CoT, maj@{K_SAMPLES}, full test split ({gsm_n} items)",
        "mmlu":       f"5-shot generic, maj@{K_SAMPLES}, stratified {mmlu_n}-item sample (seed={MMLU_SEED}, 10/subject)",
        "strategyqa": f"6-shot, maj@{K_SAMPLES}, full test split ({sqa_n} items)",
        "decoding":   f"self-consistency: do_sample=True, T={SC_TEMP}, top_p={SC_TOP_P}, K={K_SAMPLES}, vote=majority",
        "max_new_tokens": MAX_NEW,
        "format":     "raw completion, no chat template",
        "fairness":   "compare maj@K vs base maj@K, or greedy vs greedy; never maj@K vs base greedy",
    },
    "citations": {
        "self_consistency": "Wang et al. 2022, arXiv:2203.11171",
        "gsm8k": "Cobbe et al. 2021, arXiv:2110.14168",
        "mmlu": "Hendrycks et al. 2021, arXiv:2009.03300",
        "strategyqa": "Geva et al. 2021, arXiv:2101.02235",
    },
}
out_path = f"{OUT}/eval_results/{phase_name}.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")
