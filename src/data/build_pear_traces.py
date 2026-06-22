"""
FILE 3 - PEAR TRACE GENERATION (32B AWQ teacher distillation)
=============================================================
  pi_beta  = R1-Distill-Qwen-32B AWQ   pi_theta = Qwen2.5-7B base
  Shared Qwen2.5 vocab -> token-level PEAR weight: delta_t = log pi_theta - log pi_beta

Key optimisation vs v1: logprob scoring is now BATCHED.
  v1: N traces -> N serial forward passes per model  (GPU idle between each)
  v2: N traces -> 1 batched forward pass per model   (4-6x faster on logprob step)

Domain order: commonsense (must-win) -> math (floor) -> mmlu (retention)
"""

import os, subprocess, sys

WHEEL_DIR = "inputs/data/gptqmodel-wheel"

SKIP_PREFIXES = (
    "torch-", "torch_", "torchvision", "torchaudio",
    "transformers-", "numpy-", "pandas-",
    "nvidia-", "triton-", "cuda-",
)
wheels = []
for root, _, files in os.walk(WHEEL_DIR):
    for f in files:
        if f.endswith(".whl") and not any(f.lower().startswith(p.lower()) for p in SKIP_PREFIXES):
            wheels.append(os.path.join(root, f))
wheels.sort()
print(f"Installing {len(wheels)} wheels (torch/transformers/nvidia excluded):")
for w in wheels: print(f"  {os.path.basename(w)}")
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"] + wheels, check=True)

import json, re, hashlib, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_from_disk, concatenate_datasets

os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"]       = "true"

import glob

# ── Config ──────────────────────────────────────────────────────────────
TARGET_MODEL = "inputs/models/qwen2-5-7b/qwen2.5-7b"
SPLIT_DATA   = "inputs/data/qwen-pear-dataset"
TEACHER_NAME = "inputs/models/deepseek-r1-distill-qwen-32b-awq-gemm-128-v01/transformers/default/1"

def resolve_model_dir(name_hint):
    hits = [p for p in glob.glob("inputs/**/config.json", recursive=True)
            if name_hint in p.lower()]
    if not hits:
        raise FileNotFoundError(f"Could not find '{name_hint}' under the local inputs/ tree.")
    return os.path.dirname(sorted(hits)[0])

TEACHER_MODEL = resolve_model_dir(TEACHER_NAME)
print(f"Teacher resolved to: {TEACHER_MODEL}")

OUT       = "outputs"
OUT_JSONL = f"{OUT}/pear_sft_final.jsonl"
PROGRESS  = f"{OUT}/progress.txt"

BATCH_SIZE   = 8
ROLLOUTS     = {"math": 4, "commonsense": 6, "mmlu": 3}
MAX_NEW      = {"math": 1024, "commonsense": 768, "mmlu": 256}
THINK_MAX_CHARS      = 1800
RATIONALIZE_RESIDUAL = True
# From scratch, BOTH GSM8K and StrategyQA must be taught by distillation and
# both need +5, so math and commonsense get an EQUAL budget. Commonsense always
# includes ALL StrategyQA (the actual target) first, then fills with CSQA2.
COMMONSENSE_CAP = 4000   # all StrategyQA (~2290) + ~1700 CSQA2
MATH_CAP        = 4000   # equal footing with commonsense
MMLU_CAP        = None    # full (~1500, small) - retention
random.seed(42)

SYS = {
    "math": ("Solve the problem. Reason step by step inside <think> and </think>, "
             "then give only the final answer inside <answer> and </answer>."),
    "commonsense": ("Decide whether the answer is Yes or No. Inside <think> and </think>, "
                    "break the question into sub-questions, answer each one, then conclude. "
                    "Put only Yes or No inside <answer> and </answer>."),
    "mmlu": ("Choose the correct option. Reason inside <think> and </think>, then put only "
             "the letter (A, B, C, or D) inside <answer> and </answer>."),
}

# ── Load models ──────────────────────────────────────────────────────────
print("Loading teacher (pi_beta, 32B AWQ)...")
teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True, local_files_only=True)
if teacher_tok.pad_token is None: teacher_tok.pad_token = teacher_tok.eos_token
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL, torch_dtype=torch.float16, device_map="cuda",
    trust_remote_code=True, local_files_only=True).eval()

print("Loading target (pi_theta, Qwen2.5-7B base)...")
base_tok = AutoTokenizer.from_pretrained(TARGET_MODEL, local_files_only=True)
if base_tok.pad_token is None: base_tok.pad_token = base_tok.eos_token
base = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()

assert teacher_tok.vocab_size == base_tok.vocab_size, \
    "Tokenizer vocab mismatch -> PEAR weight undefined. Stop."

GEN = {
    "math":        GenerationConfig(do_sample=True, temperature=0.7, top_p=0.95),
    "commonsense": GenerationConfig(do_sample=True, temperature=0.7, top_p=0.95),
    "mmlu":        GenerationConfig(do_sample=True, temperature=0.3, top_p=0.95),
}

# ── Canonical format ─────────────────────────────────────────────────────
def canonical_prompt(domain, qblock):
    return (f"<|im_start|>system\n{SYS[domain]}<|im_end|>\n"
            f"<|im_start|>user\n{qblock}<|im_end|>\n"
            f"<|im_start|>assistant\n")

def canonical_trace(thinking, final):
    thinking = thinking.strip()
    if len(thinking) > THINK_MAX_CHARS:
        cut = thinking[:THINK_MAX_CHARS]
        thinking = cut[:cut.rfind(".") + 1] if "." in cut else cut
    return f"<think>\n{thinking}\n</think>\n<answer>{final}</answer><|im_end|>"

# ── Answer parsing ───────────────────────────────────────────────────────
def normalize_number(s):
    s = str(s).strip().rstrip(".,;:").replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s.lower()

def extract_boxed(text):
    idx = text.rfind(r"\boxed{")
    if idx == -1: return None
    start = text.index("{", idx) + 1
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{": depth += 1
        elif text[i] == "}": depth -= 1
        i += 1
    return text[start:i-1].strip() if depth == 0 else None

def extract_tag(text, tag="answer"):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None

def split_think(raw):
    raw = raw.strip()
    if raw.startswith("<think>"): raw = raw[len("<think>"):]
    if "</think>" in raw:
        think, tail = raw.split("</think>", 1)
    else:
        think, tail = raw, raw
    return think.strip(), tail.strip()

def parse_answer(raw, tail, domain):
    tagged = extract_tag(raw)
    if domain == "math":
        if tagged:
            b = extract_boxed(tagged)
            return normalize_number(b if b is not None else tagged)
        b = extract_boxed(raw)
        if b is not None: return normalize_number(b)
        nums = re.findall(r"-?[\d,]+\.?\d*(?:/[\d,]+)?", tail)
        return normalize_number(nums[-1]) if nums else None
    if domain == "commonsense":
        src = tagged if tagged else tail
        m = re.findall(r"\b(Yes|No)\b", src, re.IGNORECASE)
        return m[-1].capitalize() if m else None
    if domain == "mmlu":
        src = tagged if tagged else tail
        m = re.search(r"\b([ABCD])\b", src.strip(), re.IGNORECASE)
        return m.group(1).upper() if m else None
    return None

def n_steps(trace): return len([l for l in trace.split("\n") if l.strip()])
def _eq(pred, gold, domain):
    return normalize_number(pred) == normalize_number(gold) if domain == "math" else str(pred) == str(gold)

# ── Selection (no scoring - returns raw think/final/role tuples) ─────────
def select_traces(rollouts, gold, domain):
    parsed = []
    for raw in rollouts:
        think, tail = split_think(raw)
        ans = parse_answer(raw, tail, domain)
        if ans is not None and think:
            parsed.append((think, ans))
    correct = [(t, a) for t, a in parsed if _eq(a, gold, domain)]
    if not correct: return [], parsed
    selected = []
    best_t, best_a = max(correct, key=lambda x: n_steps(x[0]))
    selected.append((best_t, best_a, "positive"))
    if domain in ("math", "commonsense"):
        wrong = [(t, a) for t, a in parsed if not _eq(a, gold, domain)]
        if wrong:
            wt, wa = max(wrong, key=lambda x: n_steps(x[0]))
            if n_steps(wt) >= 2:
                selected.append((wt, wa, "negative"))
    return selected, parsed

# ── Batched logprobs (THE FIX: one forward pass per model per batch) ─────
def batch_token_logprobs(model, tok, prompt_trace_pairs, max_total=4096, sub_batch=6):
    """
    Score a list of (prompt, trace) pairs in sub-batched forward passes.
    Uses right-padding so trace positions are simple slices [p_len:total_real].
    Returns a list of per-token logprob lists (or None if trace truncated).
    """
    results = [None] * len(prompt_trace_pairs)
    prev_side = tok.padding_side
    tok.padding_side = "right"
    for sb in range(0, len(prompt_trace_pairs), sub_batch):
        chunk   = prompt_trace_pairs[sb:sb+sub_batch]
        prompts = [p for p, _ in chunk]
        fulls   = [p + t for p, t in chunk]
        p_lens  = [tok(p, truncation=True, max_length=max_total,
                       return_tensors="pt").input_ids.shape[1]
                   for p in prompts]
        enc = tok(fulls, return_tensors="pt", truncation=True,
                  max_length=max_total, padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits          # [B, seq, vocab] in model dtype
        for i, p_len in enumerate(p_lens):
            total_real = int(enc.attention_mask[i].sum().item())
            if total_real <= p_len:
                results[sb+i] = None
                continue
            sl  = logits[i, p_len-1:total_real-1].float()
            ids = enc.input_ids[i, p_len:total_real]
            lp  = torch.log_softmax(sl, dim=-1)
            results[sb+i] = lp[torch.arange(ids.shape[0], device=lp.device), ids].tolist()
        del logits; torch.cuda.empty_cache()
    tok.padding_side = prev_side
    return results

def trace_id(prompt, trace):
    return hashlib.md5((prompt+trace).encode()).hexdigest()[:20]

# ── Teacher generation ───────────────────────────────────────────────────
def gen_text(domain, qblock, hint_facts=None):
    instr = SYS[domain]
    if hint_facts:
        instr += ("\nBackground you may use: " + " ".join(hint_facts) +
                  "\nWrite the reasoning as if recalling these yourself; do not mention "
                  "being given any facts.")
    return teacher_tok.apply_chat_template(
        [{"role": "user", "content": f"{instr}\n\n{qblock}"}],
        add_generation_prompt=True, tokenize=False)

def generate(qblocks, domain, hint_facts_list=None):
    texts = [gen_text(domain, q, (hint_facts_list[i] if hint_facts_list else None))
             for i, q in enumerate(qblocks)]
    teacher_tok.padding_side = "left"
    inp = teacher_tok(texts, return_tensors="pt", truncation=True,
                      max_length=2048, padding=True).to(teacher.device)
    n = ROLLOUTS[domain]
    with torch.no_grad():
        out = teacher.generate(**inp, generation_config=GEN[domain],
                               max_new_tokens=MAX_NEW[domain],
                               num_return_sequences=n,
                               pad_token_id=teacher_tok.eos_token_id)
    p_len = inp.input_ids.shape[1]
    flat  = [teacher_tok.decode(out[i][p_len:], skip_special_tokens=True).strip()
             for i in range(len(qblocks)*n)]
    return [flat[i*n:(i+1)*n] for i in range(len(qblocks))]

# ── Resume ───────────────────────────────────────────────────────────────
done = set()
if os.path.exists(PROGRESS): done = set(open(PROGRESS).read().split())
prog_f = open(PROGRESS, "a")
out_f  = open(OUT_JSONL, "a")

def qid(source, text): return source + "_" + hashlib.md5(text.encode()).hexdigest()[:12]
def emit(recs, qkey):
    for r in recs: out_f.write(json.dumps(r) + "\n")
    out_f.flush(); prog_f.write(qkey + "\n"); prog_f.flush(); done.add(qkey)

stats = {"positive": 0, "negative": 0, "skipped": 0, "rationalized": 0}

def score_and_build(to_score, traces):
    """Batch-score all (prompt,think,ans,gold,source,role) + pre-built trace strings."""
    if not to_score: return []
    pairs = [(p, tr) for (p, *_), tr in zip(to_score, traces)]
    b_lps = batch_token_logprobs(teacher, teacher_tok, pairs)
    t_lps = batch_token_logprobs(base,    base_tok,    pairs)
    recs = []
    for j, (p, think, ans, gold, source, role) in enumerate(to_score):
        if b_lps[j] is None or t_lps[j] is None:
            print(f"  [WARN] skip truncated trace (source={source})")
            recs.append(None)
            continue
        recs.append({"id": trace_id(p, traces[j]), "prompt": p, "trace": traces[j],
                     "gold": gold, "source": source, "role": role,
                     "behavior_logprobs": b_lps[j], "base_logprobs": t_lps[j]})
    return recs

# ══════════════════════════════════════════════════════════════════════
# COMMONSENSE  (must-win - first)
# ══════════════════════════════════════════════════════════════════════
def apply_cap(ds, cap):
    if cap is None: return ds
    return ds.shuffle(seed=42, keep_in_memory=True).select(
        range(min(cap, len(ds))), keep_in_memory=True)

# Commonsense: guarantee ALL StrategyQA (the benchmark), then fill with CSQA2.
_cs_full = load_from_disk(f"{SPLIT_DATA}/pear_commonsense")
_sqa  = _cs_full.filter(lambda r: r["dataset_source"] == "strategyqa", keep_in_memory=True)
_csqa = _cs_full.filter(lambda r: r["dataset_source"] != "strategyqa", keep_in_memory=True)
if COMMONSENSE_CAP is None:
    cs = _cs_full.shuffle(seed=42, keep_in_memory=True)
else:
    _n_csqa = max(0, COMMONSENSE_CAP - len(_sqa))
    _csqa = _csqa.shuffle(seed=42, keep_in_memory=True).select(
        range(min(_n_csqa, len(_csqa))), keep_in_memory=True)
    cs = concatenate_datasets([_sqa, _csqa]).shuffle(seed=42, keep_in_memory=True)
print(f"\nCOMMONSENSE: {len(cs)} rows (all {len(_sqa)} StrategyQA + {len(cs)-len(_sqa)} CSQA2)")

for bs in range(0, len(cs), BATCH_SIZE):
    rows = [cs[i] for i in range(bs, min(bs+BATCH_SIZE, len(cs)))]
    rows = [r for r in rows if r["question"]
            and qid(r["dataset_source"], r["question"]) not in done]
    if not rows: continue

    rollouts = generate([f"Question: {r['question']}" for r in rows], "commonsense")

    to_score, row_map, retry_rows, retry_q, retry_facts = [], {}, [], [], []

    for r, rolls in zip(rows, rollouts):
        prompt = canonical_prompt("commonsense", f"Question: {r['question']}")
        qkey   = qid(r["dataset_source"], r["question"])
        sel, _ = select_traces(rolls, r["gold_answer"], "commonsense")
        if sel:
            start = len(to_score)
            traces_here = [canonical_trace(t, a) for t, a, _ in sel]
            for (think, ans, role), tr in zip(sel, traces_here):
                to_score.append((prompt, think, ans, r["gold_answer"], r["dataset_source"], role))
            row_map[qkey] = (start, len(sel))
        elif RATIONALIZE_RESIDUAL and r.get("facts"):
            retry_rows.append((r, qkey, prompt))
            retry_q.append(f"Question: {r['question']}")
            retry_facts.append(r["facts"])
        else:
            stats["skipped"] += 1
            emit([], qkey)

    if retry_rows:
        rr = generate(retry_q, "commonsense", hint_facts_list=retry_facts)
        for (r, qkey, prompt), rolls in zip(retry_rows, rr):
            sel, _ = select_traces(rolls, r["gold_answer"], "commonsense")
            if sel:
                start = len(to_score)
                for think, ans, role in sel:
                    to_score.append((prompt, think, ans, r["gold_answer"], r["dataset_source"], role))
                row_map[qkey] = (start, len(sel))
                stats["rationalized"] += 1
            else:
                stats["skipped"] += 1
                emit([], qkey)

    if to_score:
        all_traces = [canonical_trace(t, a) for _, t, a, *_ in to_score]
        scored     = score_and_build(to_score, all_traces)
        for qkey, (start, count) in row_map.items():
            recs = [r for r in scored[start:start+count] if r is not None]
            for rec in recs: stats[rec["role"]] += 1
            if not recs: stats["skipped"] += 1
            emit(recs, qkey)

    if bs % (BATCH_SIZE * 20) == 0:
        print(f"  {bs}/{len(cs)} pos={stats['positive']} neg={stats['negative']} "
              f"rat={stats['rationalized']} skip={stats['skipped']}")

# ══════════════════════════════════════════════════════════════════════
# MATH  (capped - GSM8K banked, floor only)
# ══════════════════════════════════════════════════════════════════════
math_ds = apply_cap(load_from_disk(f"{SPLIT_DATA}/pear_math"), MATH_CAP)
print(f"\nMATH: {len(math_ds)} rows (cap={MATH_CAP})")

for bs in range(0, len(math_ds), BATCH_SIZE):
    rows = [math_ds[i] for i in range(bs, min(bs+BATCH_SIZE, len(math_ds)))]
    rows = [r for r in rows if r["problem"] and r["gold_answer"]
            and qid(r["dataset_source"], r["problem"]) not in done]
    if not rows: continue

    rollouts = generate([f"Problem: {r['problem']}" for r in rows], "math")
    to_score, row_map = [], {}

    for r, rolls in zip(rows, rollouts):
        prompt = canonical_prompt("math", f"Problem: {r['problem']}")
        qkey   = qid(r["dataset_source"], r["problem"])
        sel, _ = select_traces(rolls, r["gold_answer"], "math")
        if sel:
            start = len(to_score)
            for think, ans, role in sel:
                to_score.append((prompt, think, ans, r["gold_answer"], r["dataset_source"], role))
            row_map[qkey] = (start, len(sel))
        else:
            stats["skipped"] += 1
            emit([], qkey)

    if to_score:
        all_traces = [canonical_trace(t, a) for _, t, a, *_ in to_score]
        scored     = score_and_build(to_score, all_traces)
        for qkey, (start, count) in row_map.items():
            recs = [r for r in scored[start:start+count] if r is not None]
            for rec in recs: stats[rec["role"]] += 1
            if not recs: stats["skipped"] += 1
            emit(recs, qkey)

    if bs % (BATCH_SIZE * 20) == 0:
        print(f"  {bs}/{len(math_ds)} pos={stats['positive']} neg={stats['negative']} "
              f"skip={stats['skipped']}")

# ══════════════════════════════════════════════════════════════════════
# MMLU  (positives only - retention)
# ══════════════════════════════════════════════════════════════════════
mmlu_ds = apply_cap(load_from_disk(f"{SPLIT_DATA}/pear_mmlu"), MMLU_CAP)
print(f"\nMMLU: {len(mmlu_ds)} rows (cap={MMLU_CAP})")
letters = "ABCD"

for bs in range(0, len(mmlu_ds), BATCH_SIZE):
    rows = [mmlu_ds[i] for i in range(bs, min(bs+BATCH_SIZE, len(mmlu_ds)))]
    rows = [r for r in rows if qid("mmlu", r["question"]) not in done]
    if not rows: continue

    def block(r):
        return f"Question: {r['question']}\n" + "\n".join(f"{letters[i]}) {r['choices'][i]}" for i in range(4))

    rollouts = generate([block(r) for r in rows], "mmlu")
    to_score, row_map = [], {}

    for r, rolls in zip(rows, rollouts):
        prompt = canonical_prompt("mmlu", block(r))
        qkey   = qid("mmlu", r["question"])
        sel, _ = select_traces(rolls, r["gold_answer"], "mmlu")
        if sel:
            start = len(to_score)
            for think, ans, role in sel:
                to_score.append((prompt, think, ans, r["gold_answer"], "mmlu", role))
            row_map[qkey] = (start, len(sel))
        else:
            stats["skipped"] += 1
            emit([], qkey)

    if to_score:
        all_traces = [canonical_trace(t, a) for _, t, a, *_ in to_score]
        scored     = score_and_build(to_score, all_traces)
        for qkey, (start, count) in row_map.items():
            recs = [r for r in scored[start:start+count] if r is not None]
            for rec in recs: stats[rec["role"]] += 1
            if not recs: stats["skipped"] += 1
            emit(recs, qkey)

    if bs % (BATCH_SIZE * 20) == 0:
        print(f"  {bs}/{len(mmlu_ds)} pos={stats['positive']} skip={stats['skipped']}")

# ── Stats ─────────────────────────────────────────────────────────────────
out_f.close(); prog_f.close()
stats["total"]            = stats["positive"] + stats["negative"]
stats["neg_to_pos_ratio"] = round(stats["negative"] / max(1, stats["positive"]), 3)
with open(f"{OUT}/trace_stats.json", "w") as f: json.dump(stats, f, indent=2)
print("\n=== DONE ===")
print(json.dumps(stats, indent=2))
print(f"\nTraces: {OUT_JSONL}")
