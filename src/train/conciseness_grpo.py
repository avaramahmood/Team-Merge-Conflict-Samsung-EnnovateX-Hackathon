"""
═══════════════════════════════════════════════════════════════════════════════
CONCISENESS GRPO - Layer 4: correctness-dependent length shaping (post-GRPO pass)
═══════════════════════════════════════════════════════════════════════════════
Goal: compress reasoning WITHOUT losing the accuracy the main GRPO run earned.
Single ~1.5h session. Resumes from grpo_best (NOT SFT). Same pool, same verifier,
same rollout prompts as grpo.py v7 / grpo_sqa_*.

NO EVAL AT ALL (matches grpo_sqa_* convention): eval is fully removed. Brevity is
monitored live from the iter log - len_sqa / len_math = mean CORRECT-trace token
length, which must trend DOWN while pass_sqa / pass_math hold. Save ONCE at the
end (mid-run saves are avoided to keep the working tree small); the end-of-run save fires
on BOTH the wall-clock break and normal completion. Run eval separately afterward
with eval_unified_v5 against grpo_concise_best.

MECHANISM (GRPO-LEAD arXiv:2504.09696 - correctness-dependent brevity reward)
────────────────────────────────────────────────────────────────────────────
  wrong   : r = -1.0                       (flat; NEVER rewarded for being short)
  correct : r = exp(-ALPHA * L / budget)   (exponential length decay, in (0,1])
Advantage = r - mean(FULL G-group), NO std (Dr.GRPO-style, difficulty-aware).
Even the LONGEST correct trace (r≈0.3) beats any wrong trace (-1) by >1.3, so
correctness always dominates; brevity only re-ranks among the correct traces.

WHAT CHANGES vs grpo.py (and WHY)
─────────────────────────────────
1. NO 8/8 RETIREMENT. Binary GRPO retired all-correct items (zero advantage).
   Here an 8/8 group with varied trace LENGTHS is the richest brevity signal,
   so RETIRE_ALL_CORRECT=False and we DO NOT inherit the prior retired list -
   former 8/8 items are exactly what we want back. Only 0/8x2 still retires.
2. KL ANCHORS TO grpo_best, not SFT. We refine near the GRPO optimum; anchoring
   to SFT would pull the policy back and undo the run.
3. PRM DROPPED. PROF process-selection is orthogonal to brevity and the 7B PRM
   swap costs ~15 GB transient + wall-clock per iter. Math runs plain GRPO on
   all rollouts with the same length shaping. Buys several more iters in 1.5h.
4. GENTLER STEPS. LR 3e-6 (vs 8e-6), KL_COEF 0.01 (vs 0.005): style nudge, not
   relearning. Shorter MAXTOK caps (math 1024 / sqa 512) both speed generation
   and reinforce brevity without truncating typical correct solutions.

MEMORY (96 GB): policy 15.2 + ref 15.2 + AdamW8bit ~15 resident. No PRM. <50 GB.
═══════════════════════════════════════════════════════════════════════════════
"""

# ── 0. ENV ───────────────────────────────────────────────────────────────────
import os
os.environ["PYTORCH_ALLOC_CONF"]    = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"]        = "true"
os.environ["HF_DATASETS_OFFLINE"]   = "1"
os.environ["TRANSFORMERS_OFFLINE"]  = "1"
os.environ["HF_DATASETS_CACHE"]     = "/tmp/hf_cache"   # keep the HF cache out of the working tree
os.makedirs("/tmp/hf_cache", exist_ok=True)

# ── 1. INSTALLS ──────────────────────────────────────────────────────────────
import subprocess, sys
WHEEL_BNB = "inputs/data/bitsandbyteswheel/bnb_wheel"
try:
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "--no-index", "--find-links", WHEEL_BNB, "--no-deps",
                    "bitsandbytes", "safetensors"],
                   check=True, capture_output=True, text=True)
    print("Packages installed.")
except subprocess.CalledProcessError as e:
    print("PIP INSTALL FAILED"); print(e.stderr[-2000:]); sys.exit(1)

# ── 2. PATHS ─────────────────────────────────────────────────────────────────
# PREV_RUN points at the previous GRPO run output directory. It must be the
# root that contains grpo_best/ (the trained policy). Edit PREV_RUN to your path.
PREV_RUN  = "inputs/data/qwen-grpo-v7-output"
INIT      = f"{PREV_RUN}/grpo_best"               # policy AND ref start here
POOL_ROOT = "inputs/data/qwen-f-grpo-pool"   # SAME pool

OUT      = "outputs"
CKPT_DIR = f"{OUT}/grpo_concise_best"             # do NOT overwrite grpo_best
LOG_DIR  = f"{OUT}/logs_concise"
PROGRESS = f"{OUT}/concise_progress.json"
os.makedirs(LOG_DIR, exist_ok=True)

# ── 3. HYPERPARAMETERS ───────────────────────────────────────────────────────
ITERATIONS          = 24       # CAP; the wall-clock guard usually binds first
MAX_SESSION_SECONDS = 5000     # ~83m train, leaves room for load + final save < 90m
PROMPTS_PER_ITER    = 64
G                   = 8
LR               = 3e-6        # gentle: style nudge near the GRPO optimum
WARMUP_ITERS     = 1
KL_COEF          = 0.01        # anchor harder to grpo_best (resist style drift)
EPS_LOW, EPS_HIGH = 0.20, 0.28
GRAD_CLIP        = 0.5
MICRO_BSZ        = 4

# correctness-dependent length shaping (GRPO-LEAD)
ALPHA_LEN      = 0.6           # decay strength; bigger = stronger brevity push
CONCISE_BUDGET = {"math": 512, "sqa": 256}   # soft target token lengths (reward)

MIX = {"sqa": 0.60, "math": 0.40}            # same target weighting as v7

# online retirement - brevity-specific
RETIRE_ALL_CORRECT = False     # 8/8 with varied lengths IS the brevity signal
RETIRE_ZERO_AFTER  = 2         # 0/8 unlearnable -> retire

# shorter caps: speed + brevity, still above typical correct-solution lengths
MAXTOK         = {"math": 1024, "sqa": 512}
TEMP, TOPP     = 1.0, 1.0
MAX_PROMPT_LEN = 1024
GEN_BS         = 48

CKPT_EVERY = 10**9             # SAVE ONCE, at session end (mid saves overflow quota)
SEED       = 24

# ── 4. IMPORTS ───────────────────────────────────────────────────────────────
import gc, json, math, random, re, time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
RNG = random.Random(SEED)
DEV = "cuda"

def gpu_gb():
    return torch.cuda.memory_allocated() / 1e9

# ═════════════════════════════════════════════════════════════════════════════
# 5. STRUCTURED LOGGER
# ═════════════════════════════════════════════════════════════════════════════
class Logger:
    def __init__(self, log_dir: str):
        self._iter_f = open(f"{log_dir}/iter_metrics.jsonl", "a", buffering=1)
        self._run_f  = open(f"{log_dir}/run.log",            "a", buffering=1)

    def _tee(self, msg: str):
        print(msg)
        self._run_f.write(msg + "\n"); self._run_f.flush()

    def log(self, msg: str):
        self._tee(msg)

    def banner(self, msg: str):
        sep = "=" * 70
        self._tee(f"\n{sep}\n{msg}\n{sep}")

    def iter(self, rec: dict):
        self._iter_f.write(json.dumps(rec) + "\n"); self._iter_f.flush()
        self._tee(
            f"iter {rec['iter']:3d}/{ITERATIONS} | "
            f"pass sqa={rec['pass_sqa']:.3f} math={rec['pass_math']:.3f} | "
            f"len sqa={rec['len_sqa']:4.0f} math={rec['len_math']:4.0f} | "
            f"surv={rec['survivors']:2d} buf={rec['buffer']:3d} | "
            f"pg={rec['loss_pg']:+.4f} kl={rec['kl']:.4f} "
            f"gn={rec['grad_norm']:.2f} | ret={rec['retired_total']} | "
            f"gpu={rec['gpu_gb']}GB | {rec['iter_min']}m (tot {rec['total_min']}m)")

    def close(self):
        for f in (self._iter_f, self._run_f):
            try: f.close()
            except Exception: pass

logger = Logger(LOG_DIR)

# ═════════════════════════════════════════════════════════════════════════════
# 6. VERIFIER - byte-identical to the pool builder / grpo.py v7.
# ═════════════════════════════════════════════════════════════════════════════
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

# ── rollout prompt: byte-identical to the pool builder ───────────────────────
SYSTEM_BY_DOMAIN = {
    "math": ("Solve the problem. Reason inside <think></think>, then give only the "
             "final answer inside <answer></answer>."),
    "sqa":  ("Answer the yes/no question. Reason inside <think></think>, then put "
             "exactly Yes or No inside <answer></answer>."),
}

def build_prompt(domain: str, question: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM_BY_DOMAIN[domain]},
            {"role": "user",   "content": question}]
    return policy_tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)

# ═════════════════════════════════════════════════════════════════════════════
# 7. ADVANTAGES - correctness-dependent length shaping (the conciseness core)
#    wrong   -> -1 (flat)            correct -> exp(-ALPHA * L/budget) in (0,1]
#    advantage = r - mean(FULL G group), NO std  (difficulty-aware, Dr.GRPO)
# ═════════════════════════════════════════════════════════════════════════════
def shaped_advantages(corr: List[bool], lengths: List[int],
                      budget: int, alpha: float) -> List[float]:
    r = []
    for c, L in zip(corr, lengths):
        if c:
            r.append(math.exp(-alpha * (L / max(1, budget))))   # shorter -> higher
        else:
            r.append(-1.0)                                       # never length-rewarded
    r = np.array(r, dtype=np.float64)
    return (r - r.mean()).tolist()

# ═════════════════════════════════════════════════════════════════════════════
# 8. BATCHED LOGPROBS + GENERATION (OOM-safe) - identical to grpo.py
# ═════════════════════════════════════════════════════════════════════════════
def batched_resp_logprobs(model, items: List[dict], micro: int = 8,
                          with_grad: bool = False) -> List[torch.Tensor]:
    res   = [None] * len(items)
    order = sorted(range(len(items)),
                   key=lambda i: len(items[i]["prompt"]) + len(items[i]["response"]))
    ctx = torch.enable_grad if with_grad else torch.no_grad
    for bs in range(0, len(order), micro):
        idxs  = order[bs:bs + micro]
        fulls = [items[i]["prompt"] + items[i]["response"] for i in idxs]
        enc   = policy_tok(fulls, return_tensors="pt", padding=True,
                           truncation=True,
                           max_length=MAX_PROMPT_LEN + max(MAXTOK.values()),
                           padding_side="right").to(DEV)
        plens = [len(policy_tok(items[i]["prompt"], truncation=True,
                                max_length=MAX_PROMPT_LEN).input_ids)
                 for i in idxs]
        with ctx():
            logits = model(**enc).logits
            logp   = torch.gather(
                F.log_softmax(logits[:, :-1].float(), dim=-1), 2,
                enc.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        for row, i in enumerate(idxs):
            pl    = plens[row]
            total = int(enc.attention_mask[row].sum().item())
            rl    = total - pl
            res[i] = (logp[row, pl - 1: pl - 1 + rl] if rl > 0
                      else torch.zeros(1, device=DEV, requires_grad=with_grad))
        if not with_grad:
            del enc, logits, logp
    return res

def oom_safe_generate(model, prompts: List[str], max_new: int,
                      gen_bs: int = GEN_BS) -> List[str]:
    n   = len(prompts)
    out: List[Optional[str]] = [None] * n
    order = sorted(range(n), key=lambda i: len(prompts[i]))
    pos, bs = 0, gen_bs
    model.config.use_cache = True
    while pos < n:
        idxs = order[pos: pos + bs]
        try:
            enc  = policy_tok([prompts[i] for i in idxs], return_tensors="pt",
                              truncation=True, max_length=MAX_PROMPT_LEN,
                              padding=True, padding_side="left").to(DEV)
            plen = enc.input_ids.shape[1]
            with torch.inference_mode():
                gen = model.generate(
                    **enc, do_sample=True, temperature=TEMP, top_p=TOPP,
                    max_new_tokens=max_new,
                    pad_token_id=policy_tok.eos_token_id,
                    eos_token_id=policy_tok.eos_token_id)
            for row, i in enumerate(idxs):
                out[i] = policy_tok.decode(gen[row, plen:],
                                           skip_special_tokens=True)
            del enc, gen
            torch.cuda.empty_cache()
            pos += bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                out[order[pos]] = ""; pos += 1
            else:
                bs = max(1, bs // 2)
                logger.log(f"  [GEN OOM] halving batch -> {bs}")
    model.config.use_cache = False
    return [o if o is not None else "" for o in out]

# ═════════════════════════════════════════════════════════════════════════════
# 9. LOAD MODELS (policy + ref both from grpo_best). NO eval, NO PRM.
# ═════════════════════════════════════════════════════════════════════════════
logger.banner(f"CONCISENESS GRPO | {time.strftime('%Y-%m-%d %H:%M')} | init={INIT}")

logger.log(f"\nLoading policy from {INIT} ...")
policy_tok = AutoTokenizer.from_pretrained(INIT)
if policy_tok.pad_token is None:
    policy_tok.pad_token = policy_tok.eos_token

policy = AutoModelForCausalLM.from_pretrained(
    INIT, torch_dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="sdpa")
policy.gradient_checkpointing_enable()
policy.config.use_cache = False
logger.log(f"  policy on GPU: {gpu_gb():.1f} GB")

logger.log("Loading frozen ref (= grpo_best, anchor KL to the GRPO optimum) ...")
ref = AutoModelForCausalLM.from_pretrained(
    INIT, torch_dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="sdpa")
ref.eval()
for p in ref.parameters():
    p.requires_grad_(False)
logger.log(f"  + ref: {gpu_gb():.1f} GB")

optimizer = bnb.optim.AdamW8bit(
    [p for p in policy.parameters() if p.requires_grad],
    lr=LR, weight_decay=0.0)

# ═════════════════════════════════════════════════════════════════════════════
# 10. POOLS (same grpo pool) - FRESH retirement (recover former 8/8 items)
# ═════════════════════════════════════════════════════════════════════════════
state = {
    "iter": 0, "opt_steps": 0,
    "cursors": {"math": 0, "sqa": 0},
    "retired": {"math": [], "sqa": []},   # FRESH: do not inherit prior 8/8 retires
    "zero_visits": {},
}

def load_pool(dom: str) -> List[dict]:
    ds   = load_from_disk(os.path.join(POOL_ROOT, f"{dom}_pool"))
    rows = [dict(r) for r in ds]
    rng  = random.Random(SEED)
    rng.shuffle(rows)
    logger.log(f"[pool] {dom}: {len(rows)} active rows (fresh, no inherited retires)")
    return rows

pools = {d: load_pool(d) for d in ("math", "sqa")}

def next_items() -> List[dict]:
    items = []
    for dom, frac in MIX.items():
        k = max(1, round(frac * PROMPTS_PER_ITER))
        for _ in range(k):
            if not pools[dom]:
                continue
            c = state["cursors"][dom] % len(pools[dom])
            items.append(pools[dom][c])
            state["cursors"][dom] = c + 1
    RNG.shuffle(items)
    return items

def retire(dom: str, item_id: str):
    state["retired"].setdefault(dom, []).append(item_id)
    pools[dom] = [r for r in pools[dom] if r["id"] != item_id]

def save_progress():
    tmp = PROGRESS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, PROGRESS)

def save_ckpt():
    policy.save_pretrained(CKPT_DIR, safe_serialization=True)
    policy_tok.save_pretrained(CKPT_DIR)
    logger.log(f"  ✓ checkpoint -> {CKPT_DIR}")

# ═════════════════════════════════════════════════════════════════════════════
# 11. TRAINING LOOP - length-shaped GRPO, no PRM, NO eval, wall-clock guard
# ═════════════════════════════════════════════════════════════════════════════
logger.banner(
    f"CONCISENESS GRPO | iters<= {ITERATIONS} (cap), budget {MAX_SESSION_SECONDS}s | "
    f"prompts/iter={PROMPTS_PER_ITER} G={G} | plain-GRPO(all-{G}) both domains | "
    f"reward: wrong=-1, correct=exp(-{ALPHA_LEN}*L/budget) {CONCISE_BUDGET} | "
    f"adv=r-groupmean (no std) | eps=[{EPS_LOW},{EPS_HIGH}] KL={KL_COEF} LR={LR} "
    f"clip={GRAD_CLIP} | mix={MIX} | retire: 8/8=OFF, 0/8x{RETIRE_ZERO_AFTER}")

t_start = time.time()

while state["iter"] < ITERATIONS:
    if (time.time() - t_start) >= MAX_SESSION_SECONDS:
        logger.log(f"\n[session] {MAX_SESSION_SECONDS}s budget reached at "
                   f"iter {state['iter']}/{ITERATIONS}; stopping cleanly. "
                   f"Checkpoint saved below.")
        break

    it    = state["iter"] + 1
    t_it  = time.time()
    items = next_items()

    # ── 11a. rollouts ─────────────────────────────────────────────────────────
    policy.eval()
    by_dom: Dict[str, List[dict]] = defaultdict(list)
    for rec in items:
        by_dom[rec["domain"]].append(rec)

    roll_texts: Dict[int, List[str]] = {}
    for dom, group in by_dom.items():
        prompts = []
        for rec in group:
            prompts.extend([build_prompt(dom, rec["question"])] * G)
        texts = oom_safe_generate(policy, prompts, MAXTOK[dom])
        for gi, rec in enumerate(group):
            roll_texts[id(rec)] = texts[gi * G: (gi + 1) * G]

    # ── 11b. outcome + retirement (8/8 KEPT for brevity; 0/8x2 retired) ───────
    survivors   = []
    pass_by_dom = defaultdict(list)
    retired_this_iter = 0
    for rec in items:
        texts = roll_texts.get(id(rec), [])
        if not texts:
            continue
        dom  = rec["domain"]
        corr = [is_correct(dom, t, rec["gold"]) for t in texts]
        nc   = sum(corr)
        pass_by_dom[dom].append(nc / len(texts))
        if nc == len(texts) and RETIRE_ALL_CORRECT:
            retire(dom, rec["id"]); retired_this_iter += 1
            continue
        if nc == 0:
            zv = state["zero_visits"].get(rec["id"], 0) + 1
            state["zero_visits"][rec["id"]] = zv
            if zv >= RETIRE_ZERO_AFTER:
                retire(dom, rec["id"]); retired_this_iter += 1
            continue   # all-wrong: advantages all 0, no gradient anyway
        state["zero_visits"].pop(rec["id"], None)
        survivors.append((rec, texts, corr))

    # ── 11c. buffer with length-shaped, difficulty-aware advantages ───────────
    buf: List[dict] = []
    len_by_dom = defaultdict(list)
    for rec, texts, corr in survivors:
        dom     = rec["domain"]
        lengths = [len(policy_tok(t, add_special_tokens=False).input_ids)
                   for t in texts]
        advs    = shaped_advantages(corr, lengths, CONCISE_BUDGET[dom], ALPHA_LEN)
        prompt  = build_prompt(dom, rec["question"])
        for i in range(len(texts)):            # plain GRPO: keep ALL rollouts
            if texts[i].strip():
                buf.append({"prompt": prompt, "response": texts[i],
                            "advantage": advs[i], "domain": dom})
                if corr[i]:
                    len_by_dom[dom].append(lengths[i])   # track CORRECT lengths

    if len(buf) < MICRO_BSZ:
        logger.log(f"iter {it}: thin buffer ({len(buf)}), skipping")
        state["iter"] = it
        save_progress()
        continue
    RNG.shuffle(buf)

    # ── 11d. old_logp (behaviour policy) + ref_logp (frozen grpo_best) ────────
    policy.eval()
    old_lps = batched_resp_logprobs(policy, buf, micro=8, with_grad=False)
    for b, lp in zip(buf, old_lps):
        b["old_logp"] = lp.detach()
    ref_lps = batched_resp_logprobs(ref, buf, micro=8, with_grad=False)
    for b, lp in zip(buf, ref_lps):
        b["ref_logp"] = lp.detach()
    torch.cuda.empty_cache()

    # ── 11e. PPO update: batched backward in micro-groups ─────────────────────
    policy.train()
    lr_now = LR * min(1.0, it / max(1, WARMUP_ITERS))
    for pg_ in optimizer.param_groups:
        pg_["lr"] = lr_now

    agg_pg = agg_kl = 0.0
    n_terms = 0
    last_gn = 0.0
    for cs in range(0, len(buf), MICRO_BSZ):
        chunk = buf[cs:cs + MICRO_BSZ]
        try:
            cur_list = batched_resp_logprobs(policy, chunk,
                                             micro=MICRO_BSZ, with_grad=True)
            optimizer.zero_grad()
            loss_total = None
            valid = 0
            for b, cur in zip(chunk, cur_list):
                n = min(cur.shape[0], b["old_logp"].shape[0],
                        b["ref_logp"].shape[0])
                if n == 0:
                    continue
                cur_n, old_n, ref_n = cur[:n], b["old_logp"][:n], b["ref_logp"][:n]
                ratio = (cur_n - old_n).exp()
                adv   = torch.tensor(b["advantage"], device=DEV,
                                     dtype=torch.float32)
                pg = -torch.min(ratio * adv,
                                ratio.clamp(1 - EPS_LOW, 1 + EPS_HIGH) * adv).mean()
                kl = (torch.exp(ref_n - cur_n) - (ref_n - cur_n) - 1.0).mean()
                term = pg + KL_COEF * kl
                loss_total = term if loss_total is None else loss_total + term
                agg_pg += pg.item(); agg_kl += kl.item()
                n_terms += 1; valid += 1
            if valid == 0 or loss_total is None:
                continue
            (loss_total / valid).backward()
            last_gn = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), GRAD_CLIP).item()
            optimizer.step()
            state["opt_steps"] += 1
            del cur_list, loss_total
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            logger.log(f"  [BWD OOM] chunk {cs} skipped")
            continue

    # ── 11f. log iter (len_* = mean CORRECT-trace token length, the KPI) ──────
    logger.iter({
        "iter":          it,
        "opt_steps":     state["opt_steps"],
        "pass_sqa":      float(np.mean(pass_by_dom["sqa"]))  if pass_by_dom["sqa"]  else 0.0,
        "pass_math":     float(np.mean(pass_by_dom["math"])) if pass_by_dom["math"] else 0.0,
        "len_sqa":       float(np.mean(len_by_dom["sqa"]))   if len_by_dom["sqa"]   else 0.0,
        "len_math":      float(np.mean(len_by_dom["math"]))  if len_by_dom["math"]  else 0.0,
        "survivors":     len(survivors),
        "buffer":        len(buf),
        "loss_pg":       agg_pg / max(1, n_terms),
        "kl":            agg_kl / max(1, n_terms),
        "grad_norm":     last_gn,
        "lr":            lr_now,
        "retired_iter":  retired_this_iter,
        "retired_total": sum(len(v) for v in state["retired"].values()),
        "pool_sqa":      len(pools["sqa"]),
        "pool_math":     len(pools["math"]),
        "gpu_gb":        round(gpu_gb(), 1),
        "iter_min":      round((time.time() - t_it) / 60, 1),
        "total_min":     round((time.time() - t_start) / 60, 1),
    })

    state["iter"] = it
    save_progress()

    if it % CKPT_EVERY == 0:        # disabled by default; save once at end below
        save_ckpt()

# single end-of-run save (fires on BOTH the wall-clock break and normal completion)
save_ckpt()
save_progress()
logger.close()

print("\n" + "=" * 70)
print("CONCISENESS PASS DONE.")
print(f"  iterations run : {state['iter']}/{ITERATIONS}")
print(f"  model          : {CKPT_DIR}")
print("  NO eval ran in-session. Eval separately with eval_unified_v5 against")
print("  grpo_concise_best, watching accuracy (held) + mean length (down).")
print("  Next: feed grpo_concise_best into quantization (QAT + UD-Q4_K_XL).")
print("=" * 70)
