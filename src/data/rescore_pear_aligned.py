"""
PEAR RE-SCORE  (fixes the +2 logprob misalignment)
==================================================
The diagnostic showed behavior_logprobs (pi_beta, 32B AWQ teacher) and
base_logprobs (pi_theta, 7B base) are off by a constant +2 tokens on every
trace: FILE 3 tokenized each trace with each model's OWN tokenizer and they
disagree on the structural tokens (<think>/</think>/<|im_end|>). So delta_t
compared different tokens.

Fix: tokenize each (prompt, trace) ONCE with the BASE tokenizer, then run BOTH
models on those SAME input_ids. The reasoning body is identical-id Qwen2.5 BPE
in both models, so the per-token logprobs become aligned and equal-length by
construction. Teacher logprobs on the tag sub-pieces are mildly OOD but aligned,
which is exactly what the delta needs.

Inputs : existing pear_sft_final.jsonl (only the two logprob arrays are bad)
Output : pear_sft_final_aligned.jsonl (same records, recomputed arrays)

Then: re-run pear_diagnostic.py on the aligned file (expect equal-length 8764/8764),
and only then consider raw / centered weighting in v13.
"""

import os, subprocess, sys, glob

WHEEL_DIR = "inputs/data/gptqmodel-wheel"
SKIP_PREFIXES = ("torch-", "torch_", "torchvision", "torchaudio",
                 "transformers-", "numpy-", "pandas-", "nvidia-", "triton-", "cuda-")
wheels = []
for root, _, files in os.walk(WHEEL_DIR):
    for f in files:
        if f.endswith(".whl") and not any(f.lower().startswith(p.lower()) for p in SKIP_PREFIXES):
            wheels.append(os.path.join(root, f))
wheels.sort()
print(f"Installing {len(wheels)} wheels (torch/transformers/nvidia excluded)")
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"] + wheels, check=True)

import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"]       = "true"

# ── Paths ──────────────────────────────────────────────────────────────
TARGET_MODEL = "inputs/models/qwen2-5-7b/qwen2.5-7b"
TEACHER_HINT = "inputs/models/deepseek-r1-distill-qwen-32b-awq-gemm-128-v01/transformers/default/1"
IN_JSONL     = "inputs/data/32b-qwen-pear-traces/pear_sft_final.jsonl"
OUT_JSONL    = "outputs/pear_sft_final_aligned.jsonl"

BATCH     = 8          # 2 models resident; raise if VRAM allows
MAX_TOTAL = 4096

def resolve_model_dir(hint):
    hits = [p for p in glob.glob("inputs/**/config.json", recursive=True) if hint in p.lower()]
    if not hits:
        raise FileNotFoundError(f"Could not find '{hint}' under the local inputs/ tree.")
    return os.path.dirname(sorted(hits)[0])

TEACHER_MODEL = resolve_model_dir(TEACHER_HINT)
print(f"Teacher: {TEACHER_MODEL}")

# ── Load ───────────────────────────────────────────────────────────────
# ONE tokenizer drives tokenization for BOTH models (this is the whole fix).
tok = AutoTokenizer.from_pretrained(TARGET_MODEL, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Loading teacher (pi_beta, 32B AWQ)...")
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL, torch_dtype=torch.float16, device_map="cuda",
    trust_remote_code=True, local_files_only=True).eval()

print("Loading student (pi_theta, 7B base)...")
student = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
    local_files_only=True).eval()

assert teacher.config.vocab_size == student.config.vocab_size, \
    "Vocab size mismatch: shared-id scoring is invalid. Stop."

# ── Shared-tokenization logprobs ───────────────────────────────────────
@torch.no_grad()
def trace_logprobs(model, input_ids, attn, p_lens):
    """Per-token logprobs over the trace span [p_len:real_len], fp32, aligned."""
    logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B, seq, vocab]
    out = []
    for i, p_len in enumerate(p_lens):
        real = int(attn[i].sum().item())
        if real <= p_len:
            out.append(None); continue
        sl  = logits[i, p_len - 1:real - 1].float()
        ids = input_ids[i, p_len:real]
        lp  = torch.log_softmax(sl, dim=-1)
        out.append(lp[torch.arange(ids.shape[0], device=lp.device), ids].tolist())
    del logits; torch.cuda.empty_cache()
    return out

def score_batch(prompts, traces):
    tok.padding_side = "right"
    fulls  = [p + t for p, t in zip(prompts, traces)]
    p_lens = [tok(p, truncation=True, max_length=MAX_TOTAL,
                  return_tensors="pt").input_ids.shape[1] for p in prompts]
    enc = tok(fulls, return_tensors="pt", truncation=True,
              max_length=MAX_TOTAL, padding=True)
    ids_t = enc.input_ids.to(teacher.device); att_t = enc.attention_mask.to(teacher.device)
    beta  = trace_logprobs(teacher, ids_t, att_t, p_lens)
    ids_s = enc.input_ids.to(student.device); att_s = enc.attention_mask.to(student.device)
    theta = trace_logprobs(student, ids_s, att_s, p_lens)
    return beta, theta

# ── Run ────────────────────────────────────────────────────────────────
records = [json.loads(l) for l in open(IN_JSONL)]
print(f"Re-scoring {len(records)} records...")

out_f = open(OUT_JSONL, "w")
written = dropped = 0
buf = []

def flush(buf):
    global written, dropped
    if not buf: return
    prompts = [r["prompt"] for r in buf]
    traces  = [r["trace"]  for r in buf]
    beta, theta = score_batch(prompts, traces)
    for r, b, t in zip(buf, beta, theta):
        if b is None or t is None or len(b) != len(t):
            dropped += 1; continue
        r["behavior_logprobs"] = b
        r["base_logprobs"]     = t
        out_f.write(json.dumps(r) + "\n")
        written += 1
    out_f.flush()

for i, r in enumerate(records):
    buf.append(r)
    if len(buf) >= BATCH:
        flush(buf); buf = []
    if (i + 1) % (BATCH * 50) == 0:
        print(f"  {i+1}/{len(records)}  written={written} dropped={dropped}")
flush(buf)
out_f.close()

print(f"\nDONE. written={written} dropped={dropped}")
print(f"Aligned traces: {OUT_JSONL}")
print("Next: re-run pear_diagnostic.py pointed at this file. Expect equal-length to be ~100%.")
