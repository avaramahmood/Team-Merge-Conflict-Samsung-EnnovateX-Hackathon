"""
═══════════════════════════════════════════════════════════════════════════════
POOL BUILDER - pass@8 difficulty pools for math + sqa + mmlu (vLLM, offline)
═══════════════════════════════════════════════════════════════════════════════
FINAL. Matches your actual datasets under train/:
  gsm8k, math_hendrycks, strategyqa, csqa2, mmlu

Schema handling per source:
  gsm8k          question / answer(#### N)          -> numeric
  math_hendrycks problem  / answer (\\dfrac ok)      -> numeric (fraction-safe)
  strategyqa     question / answer (bool)           -> yes/no
  csqa2          question / answer ("yes"/"no")     -> yes/no  (sqa fallback)
  mmlu           question / choices(list) / answer(int idx) -> options folded
                 into prompt, gold index -> letter

Fixes carried in: prompt aligned to think_fewshot, large MAXTOK + truncation
logging, robust + fraction-safe extraction, multiple-choice support.

DIAGNOSTIC_ONLY=True runs 5 greedy gens per domain and exits. Confirm sqa shows
yes/no gold and mmlu shows lettered options + a matching letter pred, then set
False and build.

CRITICAL: normalize_* / extract_* MUST stay byte-identical to GRPO reward + SFT.
Two-session resume via progress_{domain}.json (flush per chunk).
"""

# ── ENV ───────────────────────────────────────────────────────────────────────
import os
os.environ["TOKENIZERS_PARALLELISM"]       = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["WANDB_DISABLED"]               = "true"
os.environ["HF_DATASETS_OFFLINE"]          = "1"
os.environ["TRANSFORMERS_OFFLINE"]         = "1"
os.environ["HF_DATASETS_CACHE"]            = "outputs/cache"
os.makedirs("outputs/cache", exist_ok=True)

# ── INSTALL vLLM from offline wheels ──────────────────────────────────────────
import subprocess, sys
WHEELS = "inputs/data/vllm-wheels"
subprocess.run([sys.executable, "-m", "pip", "install",
                "--no-index", "--find-links", WHEELS, "vllm"], check=True)

# ── IMPORTS ─────────────────────────────────────────────────────────────────
import hashlib, json, math, random, re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import datasets
from datasets import Dataset, load_from_disk
from vllm import LLM, SamplingParams

datasets.disable_caching()
random.seed(42)

# ── PATHS ──────────────────────────────────────────────────────────────────────
POLICY    = "inputs/models/distilled-qwen-sft-basic/pear_sft_epoch1"
DATA_ROOT = "inputs/data/qwen-riva-datasets/train"
OUT       = "outputs"
os.makedirs(OUT, exist_ok=True)

SRC: Dict[str, List[Tuple[str, str]]] = {
    "math": [
        ("gsm8k",          f"{DATA_ROOT}/gsm8k"),
        ("hendrycks_math", f"{DATA_ROOT}/math_hendrycks"),
    ],
    "sqa": [
        ("strategyqa",     f"{DATA_ROOT}/strategyqa"),
        ("csqa2",          f"{DATA_ROOT}/csqa2"),       # CommonsenseQA-2, yes/no
    ],
    "mmlu": [
        ("mmlu",           f"{DATA_ROOT}/mmlu"),
    ],
}

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DIAGNOSTIC_ONLY = False      # FIRST RUN: inspect, then set False for full build

PASS_K     = 8
BAND       = {"math": (2, 5), "sqa": (2, 5), "mmlu": (2, 5)}
TARGET     = {"math": 1000,   "sqa": 1500,   "mmlu": 800}
MAXTOK     = {"math": 2048,   "sqa": 1024,   "mmlu": 768}
CHUNK_SIZE = 500
TEMP, TOPP = 1.0, 1.0

# mmlu gold is an int index. cais/mmlu is 0-indexed (0->A). If the diagnostic
# shows mmlu preds systematically one letter off, flip this to False (1-indexed).
MMLU_GOLD_ZERO_INDEXED = True

BATCH_BY_LENGTH: List[Tuple[int, int]] = [
    (256, 384), (512, 256), (1024, 160), (2048, 96),
]
def get_batch_size(prompt_len: int) -> int:
    for max_len, bs in BATCH_BY_LENGTH:
        if prompt_len <= max_len:
            return bs
    return 48


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESS HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def progress_path(domain: str) -> str:
    return f"{OUT}/progress_{domain}.json"

def load_progress(domain: str) -> Tuple[List[dict], Dict[str, int], set]:
    p = progress_path(domain)
    if not os.path.exists(p):
        print(f"  [resume] No progress file for {domain} - starting fresh.")
        return [], {}, set()
    with open(p) as f:
        state = json.load(f)
    kept       = state.get("kept", [])
    passcounts = state.get("passcounts", {})
    ev_hashes  = set(state.get("evaluated_hashes", []))
    print(f"  [resume] Loaded progress for {domain}: kept={len(kept)}, evaluated={len(ev_hashes)}")
    return kept, passcounts, ev_hashes

def save_progress(domain, kept, passcounts, evaluated_hashes):
    p, tmp = progress_path(domain), progress_path(domain) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"kept": kept, "passcounts": passcounts,
                   "evaluated_hashes": list(evaluated_hashes)}, f)
    os.replace(tmp, p)
    print(f"  [checkpoint] {domain}: kept={len(kept)}, evaluated={len(evaluated_hashes)} → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZERS (byte-identical to GRPO reward + SFT prep)
# ══════════════════════════════════════════════════════════════════════════════
def normalize_math_gold(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\\[,!;]", "", s)
    s = re.sub(r"^\$(.+)\$$", r"\1", s).strip()
    s = re.sub(r"\\[td]frac\b", r"\\frac", s)
    frac = re.match(r"^(-?)\\frac\{([^}]+)\}\{([^}]+)\}$", s)
    if frac:
        sign, num, den = frac.group(1), frac.group(2), frac.group(3)
        try:
            val = float(num) / float(den)
            s = f"{sign}{val}" if sign else str(val)
        except (ValueError, ZeroDivisionError):
            s = f"{sign}{num}/{den}"
    s = s.replace(",", "").replace("$", "").replace("\\!", "")
    try:
        f = float(s)
        if math.isfinite(f) and abs(f) < 1e15 and f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, OverflowError):
        return s.strip()

def normalize_sqa(s: str) -> str:
    a = str(s).strip().lower()
    if re.search(r"\b(yes|true)\b", a):  return "yes"
    if re.search(r"\b(no|false)\b", a):  return "no"
    return a

def normalize_mmlu(s: str) -> str:
    m = re.search(r"\b([A-D])\b", str(s).strip().upper())
    return m.group(1) if m else str(s).strip().upper()

NORM: Dict[str, Callable[[str], str]] = {
    "math": normalize_math_gold, "sqa": normalize_sqa, "mmlu": normalize_mmlu,
}


# ── EXTRACTORS (robust + fraction-safe) ───────────────────────────────────────
def extract_boxed(text: str) -> Optional[str]:
    last = text.rfind("\\boxed{")
    if last == -1:
        return None
    start, depth, i = last + 7, 1, last + 7
    while i < len(text):
        if   text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None

def _last_number(s: str) -> Optional[str]:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s)
    return nums[-1] if nums else None

def _answer_block(trace: str) -> Optional[str]:
    m = re.search(r"<answer>(.*?)</answer>", trace, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None

_NUMERIC = re.compile(r"-?\d+(\.\d+)?$")

def _math_from_candidate(cand: str) -> str:
    """Normalize a math answer candidate. Strips a boxed wrapper, normalizes
    directly (so \\frac/\\dfrac survive as decimals), and only falls back to the
    last number when the candidate is prose like 'the answer is 72 clips'."""
    b = extract_boxed(cand)
    if b is not None:
        cand = b
    norm = normalize_math_gold(cand)
    if _NUMERIC.match(norm) or ("/" in norm and not any(c.isalpha() for c in norm)):
        return norm
    n = _last_number(cand)
    return normalize_math_gold(n) if n else norm

def extract_answer(domain: str, trace: str) -> Optional[str]:
    blk = _answer_block(trace)
    if domain == "math":
        if blk is not None:
            return _math_from_candidate(blk)
        b = extract_boxed(trace)
        if b is not None:
            return _math_from_candidate(b)
        am = re.search(r"ANSWER:\s*\$?([^\n$]+)", trace)
        if am:
            return _math_from_candidate(am.group(1))
        if "####" in trace:
            n = _last_number(trace.split("####", 1)[1].split("\n")[0])
            if n:
                return normalize_math_gold(n)
        n = _last_number(trace)
        return normalize_math_gold(n) if n else None
    if domain == "sqa":
        return normalize_sqa(blk if blk is not None else trace[-200:])
    if domain == "mmlu":
        src = blk if blk is not None else trace[-160:]
        letters = re.findall(r"\b([A-D])\b", src.upper())
        return letters[-1] if letters else normalize_mmlu(src)
    return None

def is_correct(domain: str, trace: str, gold: str) -> bool:
    p = extract_answer(domain, trace)
    return p is not None and p == NORM[domain](gold)


# ── MULTIPLE-CHOICE HELPERS ────────────────────────────────────────────────────
def format_mc_question(question: str, choices: List[str]) -> str:
    lines = [str(question).strip()]
    for i, c in enumerate(choices):
        lines.append(f"{chr(65 + i)}. {str(c).strip()}")
    return "\n".join(lines)

def _letter_from_gold(raw, choices: List[str]) -> str:
    s = str(raw).strip()
    if re.fullmatch(r"\d+", s):
        i = int(s) if MMLU_GOLD_ZERO_INDEXED else int(s) - 1
        if 0 <= i < len(choices):
            return chr(65 + i)
    if re.fullmatch(r"[A-Za-z]", s):
        return s.upper()
    for i, c in enumerate(choices):
        if str(c).strip().lower() == s.lower():
            return chr(65 + i)
    return s


# ── GOLD + SOURCE LOADER (choice-aware) ────────────────────────────────────────
def _extract_gold(row, a_col: str, choices: Optional[List[str]] = None) -> Optional[str]:
    raw = row[a_col]
    if isinstance(raw, dict):
        raw = raw.get("label") or raw.get("text") or raw.get("value") or str(raw)
    if choices:
        return _letter_from_gold(raw, choices)
    raw_str = str(raw).strip()
    if "####" in raw_str:
        raw_str = raw_str.split("####", 1)[1].strip().split("\n")[0].strip()
    return raw_str if raw_str else None

def load_source(source_name: str, path: str, domain: str, skip_hashes: set) -> List[dict]:
    print(f"\n  [{domain}] Loading source: {source_name}  path={path}")
    if not os.path.isdir(path):
        print(f"    SKIP - path does not exist"); return []

    ds = load_from_disk(path)
    print(f"    Rows={len(ds)}  Cols={ds.column_names}")
    if "problem_source" in ds.column_names:
        before = len(ds)
        ds = ds.filter(lambda r: r["problem_source"] == "math")
        print(f"    Filtered to problem_source='math': {len(ds)}/{before}")

    cols  = ds.column_names
    q_col = next((c for c in ["question", "problem", "prompt", "input", "text", "user"]
                  if c in cols), None)
    a_col = next((c for c in ["answer", "answers", "answerKey", "expected_answer",
                              "gold", "target", "label", "output"] if c in cols), None)
    c_col = next((c for c in ["choices", "options", "candidates"] if c in cols), None)
    if not q_col: raise ValueError(f"No question column in {cols}")
    if not a_col: raise ValueError(f"No answer column in {cols}")
    print(f"    question_col='{q_col}'  answer_col='{a_col}'  choices_col={c_col!r}")

    out, internal_seen = [], set()
    skipped_empty = skipped_dupe = 0
    for row in ds:
        q = str(row[q_col]).strip()

        choices = None
        if c_col is not None:
            raw_c = row[c_col]
            if isinstance(raw_c, dict):
                raw_c = raw_c.get("text") or raw_c.get("choices") or list(raw_c.values())[0]
            if isinstance(raw_c, (list, tuple)) and len(raw_c) >= 2:
                choices = [str(x) for x in raw_c]

        g = _extract_gold(row, a_col, choices)
        if not q or not g:
            skipped_empty += 1; continue
        if choices:
            q = format_mc_question(q, choices)

        h = hashlib.sha256(re.sub(r"\s+", " ", q.strip().lower()).encode()).hexdigest()
        if h in skip_hashes or h in internal_seen:
            skipped_dupe += 1; continue
        internal_seen.add(h)
        out.append({"id": h[:16], "domain": domain, "question": q,
                    "gold": g, "source": source_name, "hash": h})

    print(f"    Loaded={len(out)}  skipped(empty={skipped_empty}, dupe={skipped_dupe})"
          + ("  [multiple-choice: options folded into prompt]" if c_col else ""))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# vLLM SETUP
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60); print("Loading policy model into vLLM ..."); print("=" * 60)
llm = LLM(model=POLICY, dtype="bfloat16", gpu_memory_utilization=0.88,
          max_model_len=4096, seed=42, enforce_eager=False)
tok = llm.get_tokenizer()

# ── PROMPT - align to eval_unified_v5 (think_fewshot). Paste eval exemplars. ──
SYSTEM_BY_DOMAIN = {
    "math": "Solve the problem. Reason inside <think></think>, then give only the "
            "final answer inside <answer></answer>.",
    "sqa":  "Answer the yes/no question. Reason inside <think></think>, then put "
            "exactly Yes or No inside <answer></answer>.",
    "mmlu": "Answer the multiple-choice question. Reason inside <think></think>, "
            "then put only the option letter inside <answer></answer>.",
}
FEWSHOT: Dict[str, List[Tuple[str, str]]] = {"math": [], "sqa": [], "mmlu": []}

def build_prompt(domain: str, question: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM_BY_DOMAIN[domain]}]
    for ex_q, ex_a in FEWSHOT.get(domain, []):
        msgs.append({"role": "user", "content": ex_q})
        msgs.append({"role": "assistant", "content": ex_a})
    msgs.append({"role": "user", "content": question})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def _tok_lengths(prompts: List[str]) -> List[int]:
    lengths = []
    for i in range(0, len(prompts), 512):
        enc = tok(prompts[i:i+512], truncation=True, max_length=2048, add_special_tokens=True)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return lengths


# ── OPTIONAL: list real dirs + schemas (uncomment to inspect) ─────────────────
def inspect_sources():
    print("DATA_ROOT contents:", sorted(os.listdir(DATA_ROOT)))
    for name in sorted(os.listdir(DATA_ROOT)):
        path = os.path.join(DATA_ROOT, name)
        if not os.path.isdir(path): continue
        try: ds = load_from_disk(path)
        except Exception as e: print(f"\n[{name}] not a dataset ({e})"); continue
        print(f"\n[{name}] rows={len(ds)} cols={ds.column_names}")
        r = ds[0]
        for k in ds.column_names:
            print(f"    {k!r:14} ({type(r[k]).__name__}) = {str(r[k])[:90]}")
# inspect_sources()


# ── DIAGNOSTIC ─────────────────────────────────────────────────────────────────
def run_diagnostic():
    print("\n" + "#" * 70)
    print("DIAGNOSTIC: 5 greedy generations per domain (gold / pred / finish / tail)")
    print("#" * 70)
    for domain in ("math", "sqa", "mmlu"):
        src = SRC[domain][0]
        probs = load_source(src[0], src[1], domain, set())[:5]
        if not probs:
            print(f"  [{domain}] no problems loaded, skipping"); continue
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=MAXTOK[domain])
        outs = llm.generate([build_prompt(domain, p["question"]) for p in probs], sp)
        for p, o in zip(probs, outs):
            out  = o.outputs[0]
            pred = extract_answer(domain, out.text)
            gold = NORM[domain](p["gold"])
            print("=" * 80)
            print(f"[{domain}] Q   :", p["question"][:140].replace("\n", " | "))
            print(f"[{domain}] GOLD:", gold, "| PRED:", pred, "| correct:", pred == gold,
                  "| finish:", out.finish_reason, "| out_tokens:", len(out.token_ids))
            print(f"[{domain}] TAIL:", repr(out.text[-280:]))
    print("=" * 80)
    print("READ: finish=='length' on most -> raise MAXTOK.")
    print("      no <answer>/\\boxed in TAIL -> prompt wrong (align FEWSHOT/system).")
    print("      mmlu preds one letter off systematically -> set MMLU_GOLD_ZERO_INDEXED=False.")


# ── PASS@K SCORER ───────────────────────────────────────────────────────────────
def score_pass_at_k(problems: List[dict], domain: str,
                    n_rollouts: int = PASS_K, max_tokens: int = 512) -> Tuple[List[int], List[str]]:
    n = len(problems)
    prompts = [build_prompt(domain, p["question"]) for p in problems]
    golds   = [p["gold"] for p in problems]
    counts, last_traces = [0] * n, [""] * n
    n_trunc = n_gen = 0

    print(f"    Tokenising {n} prompts ...", end=" ", flush=True)
    tok_lens = _tok_lengths(prompts)
    print(f"done. range [{min(tok_lens)}, {max(tok_lens)}] tokens")

    order = sorted(range(n), key=lambda i: tok_lens[i])
    s_prompts = [prompts[i] for i in order]
    s_golds   = [golds[i]   for i in order]
    s_lens    = [tok_lens[i] for i in order]

    batches, i = [], 0
    while i < n:
        bs = get_batch_size(s_lens[i]); j = min(i + bs, n)
        while j > i + 1 and get_batch_size(s_lens[j-1]) < bs: j -= 1
        batches.append((i, j)); i = j

    print(f"    {len(batches)} batches | {n_rollouts} rollouts × {len(batches)} = "
          f"{n_rollouts * len(batches)} generate() calls | max_tokens={max_tokens}")

    for rollout in range(n_rollouts):
        for bstart, bend in batches:
            sp = SamplingParams(n=1, temperature=TEMP, top_p=TOPP,
                                max_tokens=max_tokens, seed=42 + rollout)
            outs = llm.generate(s_prompts[bstart:bend], sp)
            for idx, (o, g) in enumerate(zip(outs, s_golds[bstart:bend])):
                orig = order[bstart + idx]; out0 = o.outputs[0]
                n_gen += 1
                if out0.finish_reason == "length": n_trunc += 1
                if is_correct(domain, out0.text, g): counts[orig] += 1
                last_traces[orig] = out0.text
        print(f"    rollout {rollout+1}/{n_rollouts} | ≥1 correct: {sum(c>0 for c in counts)}/{n}")

    pct = 100.0 * n_trunc / max(1, n_gen)
    print(f"    truncated (finish=='length'): {n_trunc}/{n_gen} ({pct:.1f}%)"
          + ("   <-- raise MAXTOK" if pct > 10 else ""))
    return counts, last_traces


# ── FINALISE DOMAIN ─────────────────────────────────────────────────────────────
def finalise_domain(domain: str, kept: List[dict], passcounts: Dict[str, int]) -> None:
    lo, hi = BAND[domain]; target = TARGET[domain]
    seen, dupes = set(), 0
    for r in kept:
        if r["hash"] in seen: dupes += 1
        seen.add(r["hash"])
    if dupes:
        raise ValueError(f"[{domain}] dedup failure: {dupes} duplicate hashes")

    random.shuffle(kept)
    save_cols = ["id", "domain", "question", "gold", "pass_count", "source"]
    final_rows = [{k: r[k] for k in save_cols} for r in kept]
    pool_dir = f"{OUT}/{domain}_pool"
    Dataset.from_list(final_rows).save_to_disk(pool_dir)
    print(f"\nSaved pool → {pool_dir}  ({len(final_rows)} rows)")

    with open(f"{OUT}/{domain}_passcounts.json", "w") as f: json.dump(passcounts, f)
    with open(f"{OUT}/{domain}_kept_hashes.json", "w") as f:
        json.dump([r["hash"] for r in kept], f)

    pass_dist, source_dist = defaultdict(int), defaultdict(int)
    for r in kept:
        pass_dist[r["pass_count"]] += 1; source_dist[r["source"]] += 1
    summary = {
        "domain": domain, "scoring_model": POLICY.split("/")[-1],
        "sources_tried": [n for n, _ in SRC[domain]], "n_rollouts": PASS_K,
        "window": [lo, hi], "target": target, "total_kept": len(final_rows),
        "total_evaluated": len(passcounts),
        "pass_distribution": {str(k): v for k, v in sorted(pass_dist.items())},
        "source_distribution": dict(source_dist),
    }
    with open(f"{OUT}/{domain}_summary.json", "w") as f: json.dump(summary, f, indent=2)
    print(f"Summary:\n{json.dumps(summary, indent=2)}")

    verify = load_from_disk(pool_dir)
    print(f"Reload check: {len(verify)} rows | cols: {verify.column_names}")
    for i in range(min(3, len(verify))):
        r = verify[i]
        print(f"  [{i}] src={r['source']} pass={r['pass_count']} gold={str(r['gold'])[:40]} id={r['id']}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY
# ══════════════════════════════════════════════════════════════════════════════
if DIAGNOSTIC_ONLY:
    run_diagnostic()
    print("\nDIAGNOSTIC_ONLY=True - stopping before the full build. "
          "Inspect output, then set DIAGNOSTIC_ONLY=False.")
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
for domain in ("math", "sqa", "mmlu"):
    lo, hi = BAND[domain]; target = TARGET[domain]; max_tokens = MAXTOK[domain]
    print("\n" + "=" * 60)
    print(f"DOMAIN: {domain.upper()}  band=[{lo},{hi}]  target={target}")
    print(f"Sources: {' → '.join(n for n,_ in SRC[domain])}")
    print("=" * 60)

    kept, passcounts, evaluated_hashes = load_progress(domain)
    if len(kept) >= target:
        print(f"  Already complete ({len(kept)}/{target}). Finalising ...")
        finalise_domain(domain, kept, passcounts); continue
    print(f"  Resuming: kept={len(kept)}, evaluated={len(evaluated_hashes)}, need {target - len(kept)} more.")

    skip_hashes = set(evaluated_hashes)
    for source_name, source_path in SRC[domain]:
        if len(kept) >= target:
            print(f"\n  Target reached, skipping {source_name}."); break
        problems = load_source(source_name, source_path, domain, skip_hashes)
        if not problems:
            continue
        for p in problems:
            skip_hashes.add(p["hash"])
        random.shuffle(problems)
        print(f"\n  [{source_name}] {len(problems)} new problems to evaluate.")

        for cs in range(0, len(problems), CHUNK_SIZE):
            if len(kept) >= target:
                print(f"  Target hit mid-source, stopping {source_name}."); break
            chunk = problems[cs: cs + CHUNK_SIZE]
            print(f"\n  [{source_name}] chunk [{cs}:{cs+len(chunk)}] - {len(chunk)} problems")
            counts, _ = score_pass_at_k(chunk, domain, max_tokens=max_tokens)

            hist = defaultdict(int)
            for prob, nc in zip(chunk, counts):
                hist[nc] += 1
                passcounts[prob["hash"]] = int(nc)
                evaluated_hashes.add(prob["hash"])
                if lo <= nc <= hi and len(kept) < target:
                    kept.append({"id": prob["id"], "domain": domain,
                                 "question": prob["question"], "gold": prob["gold"],
                                 "pass_count": int(nc), "source": source_name,
                                 "hash": prob["hash"]})
            print(f"  pass dist={dict(sorted(hist.items()))}  kept={len(kept)}/{target}")
            save_progress(domain, kept, passcounts, evaluated_hashes)

    if len(kept) < target:
        print(f"\nWARNING: target {target} not met for {domain}. Got {len(kept)}.")
    finalise_domain(domain, kept, passcounts)

print("\n" + "=" * 60); print("All pools ready under", OUT); print("=" * 60)
