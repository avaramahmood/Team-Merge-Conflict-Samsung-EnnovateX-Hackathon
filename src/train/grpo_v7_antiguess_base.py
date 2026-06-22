"""
═══════════════════════════════════════════════════════════════════════════════
GRPO v7 - FINAL RUN: sqa plain-GRPO + math PROF, difficulty-aware advantages
═══════════════════════════════════════════════════════════════════════════════
Targets vs fresh baseline: GSM8K +2pp, StrategyQA +4pp, MMLU hold (eval-only).
START FRESH from pear_sft_epoch1. Do NOT resume the v6 checkpoint (KL-drift
baggage, no real gain to preserve).

WHY v6 DID NOT LEARN (and what v7 changes)
──────────────────────────────────────────
1. ADVANTAGE BUG (the killer). PROF balanced every group to 2+/2- then
   standardized within the group => every sequence got advantage ±1 regardless
   of problem difficulty. 220 symmetric ±1 tugs per iter = gradient noise;
   KL drifted while accuracy random-walked. v7: advantage = r - mean(FULL
   8-rollout group), NO std normalization, NO balancing for sqa. A 2/8-pass
   problem now pushes its correct traces with +1.5; a 6/8 problem with +0.5.
   Difficulty is finally encoded in the gradient (Dr.GRPO-style unbiased).

2. PROF ON BINARY TASKS = NOISE SELECTION. min(step scores) from VersaPRM over
   2-4 short sqa steps is near-random, so PROF reinforced randomly chosen
   correct traces. v7: sqa = plain GRPO on raw correctness, ALL 8 rollouts of
   surviving prompts enter the buffer. PROF (Qwen Math PRM, "both", mean-agg)
   is kept for MATH ONLY, with advantages still computed vs the full-group
   mean so math is difficulty-aware too. VersaPRM is dropped entirely.

3. MMLU EVAL WAS BROKEN: 25.2% baseline = random chance, because the v6 eval
   builder never folded the choices into the prompt. Fixed (choices folded
   exactly like eval_unified_v5 mmlu_block). MMLU is removed from TRAINING;
   it remains in EVAL only to verify it holds.

4. POOL STALENESS -> ONLINE RETIREMENT. No expensive re-score pass. Every
   training rollout already measures the item's live pass rate: items that
   roll 8/8 are retired permanently; items that roll 0/8 twice are retired.
   The pool self-refreshes for free as the policy improves.

5. DECISIVE STEPS. LR 8e-6 (1.5e-6 was homeopathic), KL_COEF 0.005 (KL drifted
   to 0.029 unopposed at 0.001), GRAD_CLIP 0.5, batched backward in
   micro-groups of 4 (faster than per-sequence singles).

UNCHANGED AND ALREADY VERIFIED: ratio denominator = old_logp (behaviour policy),
KL vs frozen SFT ref, byte-identical verifier/prompts to the pool builder,
think_fewshot eval aligned to eval_unified_v5, structured logger, resume via
RESUME_FROM (grpo_best/ + grpo_progress.json at the root), eval-gated best
checkpoint under the 20 GB cap.

MEMORY (96 GB): policy 15.2 + ref 15.2 + AdamW8bit ~15 resident; Qwen PRM swaps
in/out (+15 transient); no VersaPRM. Peak well under 70 GB.
═══════════════════════════════════════════════════════════════════════════════
"""

# ── 0. ENV ───────────────────────────────────────────────────────────────────
import os
os.environ["PYTORCH_ALLOC_CONF"]    = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"]        = "true"
os.environ["HF_DATASETS_OFFLINE"]   = "1"
os.environ["TRANSFORMERS_OFFLINE"]  = "1"
os.environ["HF_DATASETS_CACHE"]     = "outputs/cache"
os.makedirs("outputs/cache", exist_ok=True)

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
POLICY_INIT = "inputs/models/distilled-qwen-sft-basic/pear_sft_epoch1"
QWEN_PRM    = "inputs/models/qwen2-5-math-prm-7b/qwen2.5-math-prm-7b"
POOL_ROOT   = "inputs/data/qwen-f-grpo-pool"
EVAL_ROOT   = "inputs/data/qwen-riva-datasets/test"

RESUME_FROM = None   # session 2+: root containing grpo_best/ + grpo_progress.json

OUT      = "outputs"
CKPT_DIR = f"{OUT}/grpo_best"
LOG_DIR  = f"{OUT}/logs"
PROGRESS = f"{OUT}/grpo_progress.json"
os.makedirs(LOG_DIR, exist_ok=True)

# ── 3. HYPERPARAMETERS ───────────────────────────────────────────────────────
ITERATIONS       = 32          # sized for one 12h session incl. evals
PROMPTS_PER_ITER = 64
G                = 8
MATH_M_KEEP      = 4           # PROF keep for math only
LR               = 8e-6        # decisive movement (v6's 1.5e-6 was homeopathic)
WARMUP_ITERS     = 2
KL_COEF          = 0.005       # resist drift (v6 KL hit 0.029 unopposed)
EPS_LOW, EPS_HIGH = 0.20, 0.28
GRAD_CLIP        = 0.5
MICRO_BSZ        = 4           # sequences per batched backward

# training mix: SQA is the target with the most headroom. MMLU is EVAL-ONLY.
MIX = {"sqa": 0.60, "math": 0.40}

# math PROF
PROF_AGG_MATH = "mean"
LAMBDA_REG    = 0.5
H_LAMBDA      = 20

# online pool retirement
RETIRE_ALL_CORRECT = True      # 8/8 once -> retire (no gradient there anyway)
RETIRE_ZERO_AFTER  = 2         # 0/8 this many visits -> retire (unlearnable)

MAXTOK         = {"math": 2048, "sqa": 1024, "mmlu": 1024}
TEMP, TOPP     = 1.0, 1.0
MAX_PROMPT_LEN = 1024
GEN_BS         = 48

EVAL_EVERY   = 8
EVAL_N       = {"math": 200, "sqa": 350, "mmlu": 100}   # precision where it matters
TARGET_DELTA = {"math": 2.0, "sqa": 4.0, "mmlu": -1.0}
COMPOSITE_W  = {"math": 1.0, "sqa": 2.0, "mmlu": 0.25}

SEED      = 24
PRM_BATCH = 4

# ── 4. IMPORTS ───────────────────────────────────────────────────────────────
import gc, hashlib, json, math, random, re, time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from safetensors import safe_open
from datasets import load_from_disk
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

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
        self._eval_f = open(f"{log_dir}/eval_records.jsonl", "a", buffering=1)
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
            f"surv={rec['survivors']:2d} buf={rec['buffer']:3d} | "
            f"pg={rec['loss_pg']:+.4f} kl={rec['kl']:.4f} "
            f"gn={rec['grad_norm']:.2f} |adv|={rec['adv_abs_mean']:.2f} | "
            f"ret={rec['retired_total']} | gpu={rec['gpu_gb']}GB | "
            f"{rec['iter_min']}m (tot {rec['total_min']}m)")

    def eval(self, rec: dict):
        self._eval_f.write(json.dumps(rec) + "\n"); self._eval_f.flush()
        self._tee(
            f"  [eval@{rec['iter']}] {rec['scores']} deltas={rec['deltas']} "
            f"composite={rec['composite']:.2f} (best {rec['best_composite']:.2f}) "
            f"tag_rates={rec['tag_rates']} ({rec['eval_min']:.1f}m)")

    def close(self):
        for f in (self._iter_f, self._eval_f, self._run_f):
            try: f.close()
            except Exception: pass

logger = Logger(LOG_DIR)

# ═════════════════════════════════════════════════════════════════════════════
# 6. VERIFIER - byte-identical to the pool builder.
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

def extract_think(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

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

def split_steps(text: str) -> List[str]:
    body  = extract_think(text) or text
    steps = [s.strip() for s in body.split("\n\n") if s.strip()]
    if len(steps) <= 1 and "\n" in body:
        steps = [s.strip() for s in body.split("\n") if s.strip()]
    return steps or [body.strip() or ""]

# ═════════════════════════════════════════════════════════════════════════════
# 7. EVAL BUILDERS  (think_fewshot, byte-identical to eval_unified_v5)
#    FIX vs v6: MMLU choices are folded into the prompt (v6 evaluated MC blind,
#    hence the 25.2% = random-chance baseline).
# ═════════════════════════════════════════════════════════════════════════════
_EVAL_SYS = {
    "math": ("Solve the problem. Reason step by step inside <think> and </think>, "
             "then give only the final answer inside <answer> and </answer>."),
    "commonsense": ("Decide whether the answer is Yes or No. Inside <think> and </think>, "
                    "break the question into sub-questions, answer each one, then conclude. "
                    "Put only Yes or No inside <answer> and </answer>."),
    "mmlu": ("Choose the correct option. Reason inside <think> and </think>, then put only "
             "the letter (A, B, C, or D) inside <answer> and </answer>."),
}

def _think_fewshot(domain_key: str, qblock: str,
                   shots_ta: List[Tuple[str, str]]) -> str:
    s = f"<|im_start|>system\n{_EVAL_SYS[domain_key]}<|im_end|>\n"
    for sq, sa in shots_ta:
        s += (f"<|im_start|>user\n{sq}<|im_end|>\n"
              f"<|im_start|>assistant\n{sa}<|im_end|>\n")
    s += f"<|im_start|>user\n{qblock}<|im_end|>\n<|im_start|>assistant\n"
    return s

_GSM8K_SHOTS = [
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
_MMLU_SHOTS = [
    ("Which of the following is the body cavity that contains the pituitary gland?\nA) Abdominal\nB) Cranial\nC) Pleural\nD) Spinal", "B"),
    ("What is the embryological origin of the hyoid bone?\nA) The first pharyngeal arch\nB) The first and second pharyngeal arches\nC) The second pharyngeal arch\nD) The second and third pharyngeal arches", "D"),
    ("In a genetic test of a newborn, a rare genetic disorder is found that has X-linked recessive transmission. Which of the following statements is likely true?\nA) The newborn is a carrier\nB) The newborn is unaffected\nC) The mother is a carrier\nD) The father is affected", "C"),
    ("A set of batteries can power a flashlight for up to 500 hours. If the flashlight has already been used for 238 hours, for how many more hours can it be used?\nA) 262\nB) 264\nC) 738\nD) 976", "A"),
    ("Which of the following statements about the lanthanide elements is NOT correct?\nA) They are all metals.\nB) They have atomic numbers between 57 and 71.\nC) They are also called the rare earth elements.\nD) Their chemical properties vary widely across the series.", "D"),
]
_SQA_SHOTS = [
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

def _gsm_shot_ta(q, a):
    reasoning, final = (a.split("####", 1) if "####" in a else (a, a))
    return (f"Problem: {q}",
            f"<think>\n{reasoning.strip()}\n</think>\n<answer>{final.strip()}</answer>")

def _sqa_shot_ta(q, a):
    m = re.findall(r"\b(Yes|No)\b", a)
    final = m[-1] if m else "No"
    reasoning = a.rsplit(final, 1)[0].strip()
    return (f"Question: {q}",
            f"<think>\n{reasoning}\n</think>\n<answer>{final}</answer>")

def _mmlu_shot_ta(q, letter):
    return (f"Question: {q}",
            f"<think>\nThe correct option is {letter}.\n</think>\n<answer>{letter}</answer>")

_GSM_TA  = [_gsm_shot_ta(q, a)  for q, a in _GSM8K_SHOTS]
_SQA_TA  = [_sqa_shot_ta(q, a)  for q, a in _SQA_SHOTS]
_MMLU_TA = [_mmlu_shot_ta(q, l) for q, l in _MMLU_SHOTS]

def _mmlu_block(ex: dict) -> str:
    """Identical to eval_unified_v5 mmlu_block: choices folded into the prompt."""
    return f"Question: {ex['question']}\n" + "\n".join(
        f"{'ABCD'[i]}) {c}" for i, c in enumerate(ex["choices"]))

def eval_build_math(ex):
    q = str(ex.get("question") or ex.get("problem", "")).strip()
    return _think_fewshot("math", f"Problem: {q}", _GSM_TA)

def eval_build_sqa(ex):
    return _think_fewshot("commonsense",
                          f"Question: {str(ex['question']).strip()}", _SQA_TA)

def eval_build_mmlu(ex):
    return _think_fewshot("mmlu", _mmlu_block(ex), _MMLU_TA)

def _normalize_number_eval(s: str) -> str:
    s = str(s).strip().rstrip(".,;:").replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s

def eval_extract_math(resp):
    tagged = _answer_block(resp)
    if tagged is not None:
        b = extract_boxed(tagged)
        val = b if b is not None else tagged
        nums = re.findall(r"-?[\d,]+\.?\d*(?:/[\d,]+)?", val)
        return _normalize_number_eval(nums[-1]) if nums else _normalize_number_eval(val)
    b = extract_boxed(resp)
    if b is not None:
        return _normalize_number_eval(b)
    nums = re.findall(r"-?\$?[\d,]+\.?\d*", resp)
    return _normalize_number_eval(nums[-1]) if nums else None

def eval_extract_mmlu(resp):
    tagged = _answer_block(resp)
    if tagged is not None:
        m = re.search(r"\b([ABCD])\b", tagged.strip(), re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.findall(r"\b([ABCD])\b", resp, re.IGNORECASE)
    return m[-1].upper() if m else None

def eval_extract_sqa(resp):
    tagged = _answer_block(resp)
    src = tagged if tagged is not None else resp
    m = re.findall(r"\b(Yes|No)\b", src, re.IGNORECASE)
    return m[-1].capitalize() if m else None

def _has_answer_tag(t):
    return bool(re.search(r"<answer>.*?</answer>", t, re.DOTALL | re.IGNORECASE))

def eval_gold_math(ex):
    raw = ex.get("answer", "")
    if "####" in str(raw):
        raw = str(raw).split("####")[-1].strip()
    return _normalize_number_eval(str(raw).strip())

def eval_gold_mmlu(ex):
    return "ABCD"[int(ex["answer"])]

def eval_gold_sqa(ex):
    return "Yes" if ex["answer"] else "No"

_EVAL_BUILD   = {"math": eval_build_math,   "sqa": eval_build_sqa,   "mmlu": eval_build_mmlu}
_EVAL_EXTRACT = {"math": eval_extract_math, "sqa": eval_extract_sqa, "mmlu": eval_extract_mmlu}
_EVAL_GOLD    = {"math": eval_gold_math,    "sqa": eval_gold_sqa,    "mmlu": eval_gold_mmlu}

# ═════════════════════════════════════════════════════════════════════════════
# 8. QWEN MATH PRM (validated reconstruction)
# ═════════════════════════════════════════════════════════════════════════════
class QwenMathPRM:
    SYS = "Please reason step by step, and put your final answer within \\boxed{}."

    def __init__(self, path: str):
        logger.log(f"\nLoading Qwen Math PRM from {path} ...")
        self.tok = AutoTokenizer.from_pretrained(path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        sep_ids = self.tok.encode("<extra_0>", add_special_tokens=False)
        assert len(sep_ids) == 1, f"<extra_0> not single token: {sep_ids}"
        self.sep_id = sep_ids[0]

        self.base = AutoModel.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="cpu",
            attn_implementation="sdpa")
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        H = self.base.config.hidden_size

        w = {}
        idx_path = os.path.join(path, "model.safetensors.index.json")
        files = (sorted(set(json.load(open(idx_path))["weight_map"][k]
                            for k in json.load(open(idx_path))["weight_map"]
                            if k.startswith("score.")))
                 if os.path.exists(idx_path)
                 else sorted(f for f in os.listdir(path) if f.endswith(".safetensors")))
        for fn in files:
            with safe_open(os.path.join(path, fn), framework="pt") as f:
                for k in f.keys():
                    if k.startswith("score."):
                        w[k] = f.get_tensor(k)
        inter = w["score.0.weight"].shape[0]
        self.head = nn.Sequential(
            nn.Linear(H, inter), nn.ReLU(),
            nn.Linear(inter, w["score.2.weight"].shape[0]))
        with torch.no_grad():
            self.head[0].weight.copy_(w["score.0.weight"])
            self.head[0].bias.copy_(w["score.0.bias"])
            self.head[2].weight.copy_(w["score.2.weight"])
            self.head[2].bias.copy_(w["score.2.bias"])
        self.head = self.head.to(torch.bfloat16).eval()
        for p in self.head.parameters():
            p.requires_grad_(False)
        self.on = False
        logger.log(f"  Qwen PRM ready (hidden={H}, inter={inter}).")

    def to_gpu(self):
        if not self.on:
            self.base.to(DEV); self.head.to(DEV); self.on = True

    def to_cpu(self):
        if self.on:
            self.base.to("cpu"); self.head.to("cpu"); self.on = False
            torch.cuda.empty_cache()

    def _fmt(self, q, trace):
        steps = split_steps(trace)
        msgs = [{"role": "system", "content": self.SYS},
                {"role": "user",   "content": q},
                {"role": "assistant",
                 "content": "<extra_0>".join(steps) + "<extra_0>"}]
        return self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)

    @torch.no_grad()
    def score(self, questions, traces, batch_size: int = PRM_BATCH):
        out = [[0.5] for _ in traces]
        texts = [self._fmt(q, t) for q, t in zip(questions, traces)]
        for bs in range(0, len(texts), batch_size):
            batch = texts[bs:bs + batch_size]
            try:
                enc = self.tok(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=3072,
                               padding_side="left").to(DEV)
                hid    = self.base(**enc).last_hidden_state
                logits = self.head(hid)
                probs  = F.softmax(logits.float(), dim=-1)[..., 1]
                sep    = (enc["input_ids"] == self.sep_id)
                for i in range(len(batch)):
                    pos = sep[i].nonzero(as_tuple=True)[0]
                    if len(pos):
                        out[bs + i] = probs[i, pos].cpu().tolist()
                del enc, hid, logits, probs, sep
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size > 1:
                    sub = self.score(questions[bs:bs + batch_size],
                                     traces[bs:bs + batch_size], batch_size=1)
                    for j, v in enumerate(sub):
                        out[bs + j] = v
        torch.cuda.empty_cache()
        return out

# ═════════════════════════════════════════════════════════════════════════════
# 9. ADVANTAGES (THE FIX)
#    Group-mean baseline over the FULL G rollouts, NO std normalization.
#    sqa  -> keep all rollouts (no filtering); raw r in {+1,-1}.
#    math -> PROF selects WHICH m=4 to train on (Qwen PRM consistency), but the
#            advantage is still r - full_group_mean, so difficulty survives.
# ═════════════════════════════════════════════════════════════════════════════
def group_advantages(corr: List[bool]) -> List[float]:
    r = np.array([1.0 if c else -1.0 for c in corr])
    return (r - r.mean()).tolist()

def prof_select_math(rolls: List[dict]) -> List[int]:
    """rolls: [{correct, prm, n_steps}]. Returns kept indices (<= MATH_M_KEEP),
    balanced removal per PROF eq 2; correct kept by HIGH rpro, wrong by LOW."""
    for r in rolls:
        v = (sum(r["prm"]) / len(r["prm"])) if r["prm"] else 0.0
        ro  = 1.0 if r["correct"] else -1.0
        reg = LAMBDA_REG if (r["n_steps"] == 1 or r["n_steps"] >= H_LAMBDA) else 0.0
        r["rpro"] = v - reg * ro
    pos = [i for i, r in enumerate(rolls) if r["correct"]]
    neg = [i for i, r in enumerate(rolls) if not r["correct"]]
    rm  = len(rolls) - MATH_M_KEEP
    if rm <= 0:
        return pos + neg
    delta = len(pos) - len(neg)
    k_pos = max(0, min(len(pos), min(rm, math.ceil((delta + rm) / 2))))
    k_neg = max(0, min(len(neg), rm - k_pos))
    if k_pos + k_neg < rm:
        k_pos = min(len(pos), rm - k_neg)
    pos_sorted = sorted(pos, key=lambda i: rolls[i]["rpro"])      # asc
    keep_pos = pos_sorted[k_pos:]                                  # drop lowest
    neg_sorted = sorted(neg, key=lambda i: rolls[i]["rpro"])      # asc
    keep_neg = neg_sorted[:len(neg) - k_neg]                       # keep lowest
    return keep_pos + keep_neg

# ═════════════════════════════════════════════════════════════════════════════
# 10. BATCHED LOGPROBS + GENERATION (OOM-safe)
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
# 11. EVAL
# ═════════════════════════════════════════════════════════════════════════════
def load_eval_sets() -> Dict[str, List[dict]]:
    name_map = {"math": "gsm8k", "sqa": "strategyqa", "mmlu": "mmlu"}
    sets = {}
    for dom, name in name_map.items():
        path = os.path.join(EVAL_ROOT, name)
        if not os.path.isdir(path):
            logger.log(f"[eval] missing {path}; eval disabled for {dom}")
            continue
        ds   = load_from_disk(path)
        rng  = random.Random(123)
        idxs = list(range(len(ds))); rng.shuffle(idxs)
        rows = [dict(ds[j]) for j in idxs[: EVAL_N[dom]]]
        sets[dom] = rows
        logger.log(f"[eval] {dom}: {len(rows)} items")
    return sets

@torch.no_grad()
def run_eval(model, eval_sets):
    scores, tag_rates = {}, {}
    model.eval()
    for dom, rows in eval_sets.items():
        build, extract, gold_fn = _EVAL_BUILD[dom], _EVAL_EXTRACT[dom], _EVAL_GOLD[dom]
        prompts = [build(r) for r in rows]
        n, correct, tags = len(prompts), 0, 0
        pos, bs = 0, GEN_BS
        order = sorted(range(n), key=lambda i: len(prompts[i]))
        model.config.use_cache = True
        while pos < n:
            idxs = order[pos:pos + bs]
            try:
                enc  = policy_tok([prompts[i] for i in idxs],
                                  return_tensors="pt", truncation=True,
                                  max_length=4096,
                                  padding=True, padding_side="left").to(DEV)
                plen = enc.input_ids.shape[1]
                gen  = model.generate(
                    **enc, do_sample=False, max_new_tokens=MAXTOK[dom],
                    pad_token_id=policy_tok.eos_token_id,
                    eos_token_id=policy_tok.eos_token_id)
                for row, i in enumerate(idxs):
                    resp = policy_tok.decode(gen[row, plen:],
                                             skip_special_tokens=True)
                    if _has_answer_tag(resp):
                        tags += 1
                    pred = extract(resp)
                    if pred is not None and str(pred) == str(gold_fn(rows[i])):
                        correct += 1
                del enc, gen
                torch.cuda.empty_cache()
                pos += bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
        model.config.use_cache = False
        scores[dom]    = round(100.0 * correct / max(1, n), 1)
        tag_rates[dom] = round(tags / max(1, n), 3)
    return scores, tag_rates

def composite(scores):
    return sum(COMPOSITE_W.get(d, 0.0) * v for d, v in scores.items())

# ═════════════════════════════════════════════════════════════════════════════
# 12. LOAD MODELS + RESUME
# ═════════════════════════════════════════════════════════════════════════════
init_path = POLICY_INIT
state = {
    "iter": 0, "opt_steps": 0,
    "cursors": {"math": 0, "sqa": 0},
    "retired": {"math": [], "sqa": []},     # retired item ids (online refresh)
    "zero_visits": {},                       # id -> consecutive 0/8 count
    "baseline": None, "best_composite": -1.0,
    "best_scores": None, "history": [],
}

if RESUME_FROM:
    rp = os.path.join(RESUME_FROM, "grpo_progress.json")
    rc = os.path.join(RESUME_FROM, "grpo_best")
    if os.path.exists(rp) and os.path.isdir(rc):
        with open(rp) as f:
            state = json.load(f)
        state.setdefault("retired", {"math": [], "sqa": []})
        state.setdefault("zero_visits", {})
        state["cursors"] = {d: state.get("cursors", {}).get(d, 0)
                            for d in ("math", "sqa")}
        init_path = rc
        logger.log(f"[resume] iter={state['iter']} "
                   f"best_composite={state['best_composite']:.2f} "
                   f"baseline={state['baseline']}")
    else:
        logger.log("[resume] progress/ckpt missing; starting fresh.")

logger.banner(
    f"GRPO v7  |  session {time.strftime('%Y-%m-%d %H:%M')}  |  "
    f"resume={RESUME_FROM or 'None'}  |  start_iter={state['iter']+1}/{ITERATIONS}")

logger.log(f"\nLoading policy from {init_path} ...")
policy_tok = AutoTokenizer.from_pretrained(init_path)
if policy_tok.pad_token is None:
    policy_tok.pad_token = policy_tok.eos_token

policy = AutoModelForCausalLM.from_pretrained(
    init_path, torch_dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="sdpa")
policy.gradient_checkpointing_enable()
policy.config.use_cache = False
logger.log(f"  policy on GPU: {gpu_gb():.1f} GB")

logger.log("Loading frozen ref (pear_sft_epoch1) on GPU ...")
ref = AutoModelForCausalLM.from_pretrained(
    POLICY_INIT, torch_dtype=torch.bfloat16, device_map="cuda",
    attn_implementation="sdpa")
ref.eval()
for p in ref.parameters():
    p.requires_grad_(False)
logger.log(f"  + ref: {gpu_gb():.1f} GB")

qwen_prm = QwenMathPRM(QWEN_PRM)
qwen_prm.to_gpu()
g_sc = float(np.mean(qwen_prm.score(
    ["Compute 25 x 8."],
    ["<think>25 x 8 = 200 since 25 x 4 = 100, doubled.</think><answer>200</answer>"])[0]))
b_sc = float(np.mean(qwen_prm.score(
    ["Compute 25 x 8."],
    ["<think>25 x 8 is roughly 250, maybe 230.</think><answer>250</answer>"])[0]))
qwen_prm.to_cpu()
logger.log(f"Qwen PRM gap: good={g_sc:.3f} bad={b_sc:.3f} -> {g_sc - b_sc:+.3f}")
assert g_sc - b_sc > 0.05, "Qwen PRM reconstruction broken; do not train."

optimizer = bnb.optim.AdamW8bit(
    [p for p in policy.parameters() if p.requires_grad],
    lr=LR, weight_decay=0.0)

# ═════════════════════════════════════════════════════════════════════════════
# 13. POOLS (sqa + math only) + ONLINE RETIREMENT
# ═════════════════════════════════════════════════════════════════════════════
def load_pool(dom: str) -> List[dict]:
    ds   = load_from_disk(os.path.join(POOL_ROOT, f"{dom}_pool"))
    rows = [dict(r) for r in ds]
    rng  = random.Random(SEED)
    rng.shuffle(rows)
    retired = set(state["retired"].get(dom, []))
    if retired:
        before = len(rows)
        rows = [r for r in rows if r["id"] not in retired]
        logger.log(f"[pool] {dom}: dropped {before - len(rows)} retired on resume")
    logger.log(f"[pool] {dom}: {len(rows)} active rows")
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

def save_best_ckpt():
    policy.save_pretrained(CKPT_DIR, safe_serialization=True)
    policy_tok.save_pretrained(CKPT_DIR)
    logger.log(f"  ✓ best checkpoint -> {CKPT_DIR}")

eval_sets = load_eval_sets()

# ═════════════════════════════════════════════════════════════════════════════
# 14. BASELINE EVAL
# ═════════════════════════════════════════════════════════════════════════════
if state["baseline"] is None and eval_sets:
    logger.log("\nBaseline eval (think_fewshot, greedy; MMLU now with choices) ...")
    t0 = time.time()
    base_scores, base_tags = run_eval(policy, eval_sets)
    base_comp = composite(base_scores)
    state["baseline"]       = base_scores
    state["best_composite"] = base_comp
    state["best_scores"]    = dict(base_scores)
    logger.eval({"iter": 0, "scores": base_scores, "deltas": {},
                 "composite": round(base_comp, 2),
                 "best_composite": round(base_comp, 2),
                 "tag_rates": base_tags,
                 "eval_min": round((time.time() - t0) / 60, 1),
                 "note": "baseline"})
    save_progress()

def targets_met(scores):
    if not state["baseline"]:
        return False
    for d, delta in TARGET_DELTA.items():
        if d not in scores or d not in state["baseline"]:
            return False
        if scores[d] - state["baseline"][d] < delta:
            return False
    return True

# ═════════════════════════════════════════════════════════════════════════════
# 15. TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════
logger.banner(
    f"GRPO v7 | start_iter={state['iter']+1}/{ITERATIONS} | "
    f"prompts/iter={PROMPTS_PER_ITER} G={G} | sqa=plain-GRPO(all-{G}) "
    f"math=PROF(m={MATH_M_KEEP}) | adv=r-groupmean (no std) | "
    f"eps=[{EPS_LOW},{EPS_HIGH}] KL={KL_COEF} LR={LR} clip={GRAD_CLIP} | "
    f"mix={MIX} | retire 8/8 + 0/8x{RETIRE_ZERO_AFTER}")

t_start = time.time()

while state["iter"] < ITERATIONS:
    it    = state["iter"] + 1
    t_it  = time.time()
    items = next_items()

    # ── 15a. rollouts ────────────────────────────────────────────────────────
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

    # ── 15b. outcome + dynamic sampling + ONLINE RETIREMENT ─────────────────
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
            continue
        state["zero_visits"].pop(rec["id"], None)
        survivors.append((rec, texts, corr))

    # ── 15c. PRM scoring (math only) ─────────────────────────────────────────
    prm_lists: Dict[int, List[List[float]]] = {}
    math_jobs = [(rec, texts) for rec, texts, _ in survivors
                 if rec["domain"] == "math"]
    if math_jobs:
        qwen_prm.to_gpu()
        for rec, texts in math_jobs:
            prm_lists[id(rec)] = qwen_prm.score(
                [rec["question"]] * len(texts), texts)
        qwen_prm.to_cpu()

    # ── 15d. build buffer with difficulty-aware advantages ──────────────────
    buf: List[dict] = []
    for rec, texts, corr in survivors:
        dom    = rec["domain"]
        advs   = group_advantages(corr)        # r - full-group mean, no std
        prompt = build_prompt(dom, rec["question"])
        if dom == "math":
            pl = prm_lists.get(id(rec), [None] * len(texts))
            rolls = [{"correct": c, "prm": p, "n_steps": len(split_steps(t))}
                     for t, c, p in zip(texts, corr, pl)]
            keep_idx = prof_select_math(rolls)
        else:   # sqa: plain GRPO, keep ALL rollouts
            keep_idx = list(range(len(texts)))
        for i in keep_idx:
            if texts[i].strip():
                buf.append({"prompt": prompt, "response": texts[i],
                            "advantage": advs[i], "domain": dom})

    if len(buf) < MICRO_BSZ:
        logger.log(f"iter {it}: thin buffer ({len(buf)}), skipping")
        state["iter"] = it
        save_progress()
        continue
    RNG.shuffle(buf)

    # ── 15e. old_logp (behaviour policy) + ref_logp (frozen SFT) ────────────
    policy.eval()
    old_lps = batched_resp_logprobs(policy, buf, micro=8, with_grad=False)
    for b, lp in zip(buf, old_lps):
        b["old_logp"] = lp.detach()
    ref_lps = batched_resp_logprobs(ref, buf, micro=8, with_grad=False)
    for b, lp in zip(buf, ref_lps):
        b["ref_logp"] = lp.detach()
    torch.cuda.empty_cache()

    # ── 15f. PPO update: batched backward in micro-groups ───────────────────
    policy.train()
    lr_now = LR * min(1.0, it / max(1, WARMUP_ITERS))
    for pg_ in optimizer.param_groups:
        pg_["lr"] = lr_now

    agg_pg = agg_kl = 0.0
    n_terms = 0
    last_gn = 0.0
    adv_abs = []
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
                adv_abs.append(abs(b["advantage"]))
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

    # ── 15g. log iter ────────────────────────────────────────────────────────
    logger.iter({
        "iter":          it,
        "opt_steps":     state["opt_steps"],
        "pass_sqa":      float(np.mean(pass_by_dom["sqa"]))  if pass_by_dom["sqa"]  else 0.0,
        "pass_math":     float(np.mean(pass_by_dom["math"])) if pass_by_dom["math"] else 0.0,
        "survivors":     len(survivors),
        "buffer":        len(buf),
        "loss_pg":       agg_pg / max(1, n_terms),
        "kl":            agg_kl / max(1, n_terms),
        "grad_norm":     last_gn,
        "adv_abs_mean":  float(np.mean(adv_abs)) if adv_abs else 0.0,
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

    # ── 15h. eval gate + best checkpoint ─────────────────────────────────────
    if eval_sets and (it % EVAL_EVERY == 0 or it == ITERATIONS):
        t_e = time.time()
        scores, tag_rates = run_eval(policy, eval_sets)
        comp = composite(scores)
        deltas = ({d: round(scores[d] - state["baseline"][d], 2) for d in scores}
                  if state["baseline"] else {})
        logger.eval({"iter": it, "scores": scores, "deltas": deltas,
                     "composite": round(comp, 2),
                     "best_composite": round(state["best_composite"], 2),
                     "tag_rates": tag_rates,
                     "eval_min": round((time.time() - t_e) / 60, 1)})
        state["history"].append({"iter": it, "scores": scores, "deltas": deltas})
        if comp > state["best_composite"]:
            state["best_composite"] = comp
            state["best_scores"]    = scores
            save_best_ckpt()
        save_progress()
        if targets_met(scores):
            logger.log("\n*** TARGETS MET (GSM8K +2, SQA +4, MMLU held). Stopping. ***")
            break
        policy.train()

logger.close()

if not os.path.isdir(CKPT_DIR):
    save_best_ckpt()

print("\n" + "=" * 70)
print("SESSION DONE.")
print(f"  best scores : {state.get('best_scores')}")
print(f"  baseline    : {state.get('baseline')}")
print(f"  iterations  : {state['iter']}/{ITERATIONS}")
print("  To continue, point RESUME_FROM at this run directory and set")
print("  RESUME_FROM to its root (contains grpo_best/ + grpo_progress.json).")
print("  Final model for eval_unified_v5: outputs/grpo_best")
print("=" * 70)