"""
═══════════════════════════════════════════════════════════════════════════════
QUANTIZE v1 - domain-calibrated 4-bit GGUF (Q4_K_XL) for the conciseness model
═══════════════════════════════════════════════════════════════════════════════
Turns the trained bf16 checkpoints into phone-deployable 4-bit GGUFs that keep
the accuracy they earned. Offline-safe: the llama.cpp source and the gguf
wheels are staged locally, so nothing downloads at runtime.

RECIPE (evidence)
─────────────────
1. W4 weight-only is near-lossless for reasoning models; low-bit activations / KV
   are where accuracy dies          -> COLM 2025, arXiv:2504.04823.
2. A DOMAIN-MATCHED importance matrix is the #1 lever for reasoning models - the
   calibration domain matters far more than for base/instruct models. We build it
   from each model's OWN greedy CoT, in its EXACT rollout chat template (byte-
   identical SYSTEM_BY_DOMAIN / build_prompt to grpo.py).
3. Embeddings + lm_head kept at Q8_0 is what makes "XL" beat plain Q4_K_M (those
   tensors dominate quant error on small models, ~no size cost). Hand-built
   equivalent of Unsloth Dynamic 2.0 UD-Q4_K_XL.
4. Validate with KL-divergence vs a Q8_0 reference (catches silent drift that
   accuracy spot-checks misses).

WHAT NEEDS THE GPU (rest is CPU / indifferent)
──────────────────────────────────────────────
  calibration generation (torch)  ·  llama-imatrix  ·  KL perplexity.
  convert + Q8 + the actual Q4 quantize are CPU and ignore the GPU.

BLACKWELL NOTE: torch GPU support is already proven by training (Layer 2 ok). The
ONLY untested piece is whether nvcc can COMPILE llama.cpp for sm_120 - needs CUDA
toolkit >= 12.8. Section 6 checks nvcc and falls back to a CPU-only llama.cpp
build automatically; calibration still runs on GPU via torch either way.

OFFLINE PREP (stage these once, then run fully offline):
  git clone https://github.com/ggml-org/llama.cpp && zip -r llamacpp_src.zip llama.cpp
  pip download gguf sentencepiece protobuf -d gguf_wheels
═══════════════════════════════════════════════════════════════════════════════
"""

# ── 0. ENV ───────────────────────────────────────────────────────────────────
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"]    = "1"
os.environ["TRANSFORMERS_OFFLINE"]   = "1"
os.environ["HF_DATASETS_CACHE"]      = "/tmp/hf_cache"   # keep the HF cache out of the working tree
os.makedirs("/tmp/hf_cache", exist_ok=True)

# ── 1. INSTALLS (offline, from staged wheels) ────────────────────────────────
import subprocess, sys
WHEEL_GGUF = "inputs/data/gguf-wheels/gguf_wheels"          # staged: gguf + sentencepiece + protobuf
try:
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "--no-index", "--find-links", WHEEL_GGUF,
                    "gguf", "sentencepiece", "protobuf"],
                   check=True, capture_output=True, text=True)
    print("Packages installed (gguf, sentencepiece, protobuf).")
except subprocess.CalledProcessError as e:
    print("PIP INSTALL FAILED"); print(e.stderr[-2000:]); sys.exit(1)

# ── 2. PATHS ─────────────────────────────────────────────────────────────────
# Trained checkpoints: point these at the previous run output directories.
# "input_checkpoint_dir : output_basename"
MODELS = {
    "inputs/models/grpo-reward-shaping-ckpt/grpo_concise_best": "concise"
}

LLAMA_SRC = "inputs/data/llamacpp/llamacpp_src/llama.cpp"        # staged source (read-only)
EVAL_ROOT = "inputs/data/qwen-riva-datasets/train"       # gsm8k / strategyqa

OUT       = "outputs"            # finals (.gguf) land here, kept on commit
TMP       = "/tmp/quant"                 # f16 / q8 / imatrix / calib (off quota)
BUILD_DIR = "/tmp/llama_build"           # out-of-source llama.cpp build
BIN       = f"{BUILD_DIR}/bin"
CONVERT   = f"{LLAMA_SRC}/convert_hf_to_gguf.py"
os.makedirs(TMP, exist_ok=True)

# ── 3. CONFIG ────────────────────────────────────────────────────────────────
QUANT_BASE   = "Q4_K_M"        # W4 base; XL = this + Q8_0 embeddings/output
CHUNKS       = 300             # imatrix calibration chunks
NGL          = 99             # GPU layers for imatrix / perplexity (full offload)
CALIB_N      = {"sqa": 180, "math": 120}        # ~60/40, matches training mix
CALIB_MAXTOK = {"sqa": 512, "math": 1024}
CALIB_BS     = 16
SEED         = 24

# ── 4. IMPORTS ───────────────────────────────────────────────────────────────
import gc, random, re, time
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda"

def banner(msg: str):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70)

def run(cmd: list, **kw):
    print("  $ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)

# ═════════════════════════════════════════════════════════════════════════════
# 5. PREFLIGHT - toolchain report + fail-loud path checks (before the long build)
# ═════════════════════════════════════════════════════════════════════════════
def preflight():
    banner("PREFLIGHT: toolchain")
    for cmd in (["nvcc", "--version"],
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"]):
        try:
            print(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())
        except FileNotFoundError:
            print(f"  ({cmd[0]} not found)")
    try:
        print(f"  torch {torch.__version__} cuda={torch.version.cuda} "
              f"capability={torch.cuda.get_device_capability()}")
    except Exception as e:
        print(f"  torch CUDA unavailable: {e}")

    banner("PREFLIGHT: paths")
    required = [WHEEL_GGUF, LLAMA_SRC, CONVERT, EVAL_ROOT] + list(MODELS.keys())
    required += [os.path.join(EVAL_ROOT, d) for d in ("gsm8k", "strategyqa")]
    missing = [p for p in required if not os.path.exists(p)]
    for p in required:
        print(f"  [{'ok ' if p not in missing else 'MISS'}] {p}")
    if missing:
        raise FileNotFoundError(
            "Missing inputs (fix the dataset mount names in §2):\n  "
            + "\n  ".join(missing))

# ── 6. ROLLOUT PROMPT - byte-identical to grpo.py (calibration distribution) ──
SYSTEM_BY_DOMAIN = {
    "math": ("Solve the problem. Reason inside <think></think>, then give only the "
             "final answer inside <answer></answer>."),
    "sqa":  ("Answer the yes/no question. Reason inside <think></think>, then put "
             "exactly Yes or No inside <answer></answer>."),
}
EVAL_NAME = {"math": "gsm8k", "sqa": "strategyqa"}

def build_prompt(tok, domain: str, question: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM_BY_DOMAIN[domain]},
            {"role": "user",   "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def question_of(domain: str, ex: dict) -> str:
    if domain == "math":
        return str(ex.get("question") or ex.get("problem", "")).strip()
    return str(ex["question"]).strip()

# ═════════════════════════════════════════════════════════════════════════════
# 7. BUILD llama.cpp (offline; CUDA if nvcc supports the GPU, else CPU-only)
# ═════════════════════════════════════════════════════════════════════════════
def build_llama():
    if os.path.exists(f"{BIN}/llama-quantize"):
        banner("llama.cpp already built - skipping")
        return
    import shutil
    if os.path.isdir(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)        # clear any stale/failed configure cache
    use_cuda = False
    try:
        v = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
        m = re.search(r"release (\d+)\.(\d+)", v)
        if m:
            major, minor = int(m.group(1)), int(m.group(2))
            cc   = torch.cuda.get_device_capability()      # (12,0) = Blackwell sm_120
            need = 128 if cc[0] >= 12 else 120             # need 12.8 for sm_120
            use_cuda = (major * 10 + minor) >= need
            print(f"  nvcc {major}.{minor}, GPU sm_{cc[0]}{cc[1]} -> CUDA build={use_cuda}")
    except FileNotFoundError:
        print("  nvcc not found -> CPU-only llama.cpp build")
    # silence git "dubious ownership" on the read-only dataset mount (version probe)
    subprocess.run(["git", "config", "--global", "--add", "safe.directory", "*"],
                   check=False)
    cuda_flags = []
    if use_cuda:
        # FindCUDAToolkit needs an UNVERSIONED libcuda.so to create the
        # CUDA::cuda_driver target; without it ggml-cuda fails to CONFIGURE. This
        # image ships only the runtime libcuda.so.1 and no toolkit stub, so locate
        # any libcuda and symlink it as libcuda.so into a dir we hand to CMake.
        import glob
        found = []
        for pat in ("/usr/local/cuda/lib64/stubs/libcuda.so",
                    "/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so",
                    "/usr/lib/x86_64-linux-gnu/libcuda.so",
                    "/usr/lib/x86_64-linux-gnu/libcuda.so.1"):
            found += glob.glob(pat)
        if not found:
            found = glob.glob("/usr/**/libcuda.so*", recursive=True)
        if found:
            stubdir = "/tmp/cuda_stub"; os.makedirs(stubdir, exist_ok=True)
            link = f"{stubdir}/libcuda.so"
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(found[0], link)
            print(f"  libcuda: {found[0]}  ->  {link}")
            cuda_flags = ["-DCMAKE_CUDA_ARCHITECTURES=native",
                          f"-DCMAKE_LIBRARY_PATH={stubdir}"]
        else:
            print("  !! no libcuda.so* found anywhere - falling back to CPU build")
            use_cuda = False
    banner(f"Building llama.cpp ({'CUDA' if use_cuda else 'CPU-only'})")
    run(["cmake", "-S", LLAMA_SRC, "-B", BUILD_DIR,
         f"-DGGML_CUDA={'ON' if use_cuda else 'OFF'}",
         "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"] + cuda_flags)
    run(["cmake", "--build", BUILD_DIR, "-j", "--config", "Release",
         "--target", "llama-quantize", "llama-imatrix", "llama-perplexity"])

# ═════════════════════════════════════════════════════════════════════════════
# 8. CALIBRATION - each model's OWN greedy CoT in its rollout chat template (GPU)
# ═════════════════════════════════════════════════════════════════════════════
def make_calibration(src: str, calib_path: str):
    banner(f"Calibration corpus -> {calib_path}")
    tok = AutoTokenizer.from_pretrained(src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=torch.bfloat16, device_map=DEV,
        attn_implementation="sdpa").eval()

    blocks = []
    for dom, n in CALIB_N.items():
        ds   = load_from_disk(os.path.join(EVAL_ROOT, EVAL_NAME[dom]))
        idxs = list(range(len(ds))); random.Random(SEED).shuffle(idxs)
        rows = [ds[i] for i in idxs[:n]]
        for bs in range(0, len(rows), CALIB_BS):
            chunk   = rows[bs:bs + CALIB_BS]
            prompts = [build_prompt(tok, dom, question_of(dom, r)) for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left",
                      truncation=True, max_length=1024).to(DEV)
            plen = enc.input_ids.shape[1]
            with torch.inference_mode():
                gen = model.generate(**enc, do_sample=False,
                                     max_new_tokens=CALIB_MAXTOK[dom],
                                     pad_token_id=tok.eos_token_id,
                                     eos_token_id=tok.eos_token_id)
            for row in range(len(chunk)):
                comp = tok.decode(gen[row, plen:], skip_special_tokens=False)
                comp = comp.replace(tok.pad_token, "").replace(tok.eos_token, "")
                blocks.append(prompts[row] + comp)
            print(f"  {dom}: {bs + len(chunk)}/{n}")

    random.Random(SEED).shuffle(blocks)
    with open(calib_path, "w") as f:
        f.write("\n\n".join(blocks))
    print(f"  wrote {len(blocks)} calibration blocks")

    # free VRAM before the llama.cpp GPU steps (imatrix / perplexity)
    del model, tok
    gc.collect(); torch.cuda.empty_cache()

# ═════════════════════════════════════════════════════════════════════════════
# 9. QUANTIZE ONE MODEL  (convert -> Q8 ref -> calib -> imatrix -> Q4_K_XL -> KL)
# ═════════════════════════════════════════════════════════════════════════════
def quantize_one(src: str, name: str):
    f16   = f"{TMP}/{name}-f16.gguf"
    q8    = f"{TMP}/{name}-q8.gguf"
    calib = f"{TMP}/{name}-calib.txt"
    imat  = f"{TMP}/{name}-imatrix.dat"
    out   = f"{OUT}/{name}-Q4_K_XL.gguf"
    klb   = f"{TMP}/{name}-klbase.dat"

    banner(f"MODEL: {name}   ({src})")

    print("[1/6] convert -> f16")                                 # CPU
    run([sys.executable, CONVERT, src, "--outfile", f16, "--outtype", "f16"])

    print("[2/6] Q8_0 reference")                                 # CPU
    run([f"{BIN}/llama-quantize", f16, q8, "Q8_0"])

    print("[3/6] calibration")                                    # GPU (torch)
    make_calibration(src, calib)

    print("[4/6] imatrix")                                        # GPU
    run([f"{BIN}/llama-imatrix", "-m", f16, "-f", calib, "-o", imat,
         "--chunks", CHUNKS, "-c", 2048, "-ngl", NGL])

    print("[5/6] quantize -> Q4_K_XL")                            # CPU
    run([f"{BIN}/llama-quantize", "--imatrix", imat,
         "--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0",
         f16, out, QUANT_BASE])

    print("[6/6] KL-divergence validation")                       # GPU
    run([f"{BIN}/llama-perplexity", "-m", q8,  "-f", calib,
         "--kl-divergence-base", klb, "-ngl", NGL])
    run([f"{BIN}/llama-perplexity", "-m", out, "-f", calib,
         "--kl-divergence", "--kl-divergence-base", klb, "-ngl", NGL])

    sz = os.path.getsize(out) / 1e9
    print(f"  ✓ {out}  ({sz:.2f} GB)")

# ═════════════════════════════════════════════════════════════════════════════
# 10. RUN
# ═════════════════════════════════════════════════════════════════════════════
banner(f"QUANTIZE v1 | {time.strftime('%Y-%m-%d %H:%M')} | models={list(MODELS.values())}")
preflight()
build_llama()
for src, name in MODELS.items():
    quantize_one(src, name)

banner("ALL DONE")
print(f"  quantized models in {OUT}:")
for name in MODELS.values():
    print(f"    {OUT}/{name}-Q4_K_XL.gguf")
print("\n  ACCEPT a model only if mean KL-divergence is low (~<0.01) AND your")
print("  eval_unified_v5 GSM8K/StrategyQA deltas vs bf16 are within ~1pp.")
print("  Watch arithmetic errors specifically (quant's known math weak spot).")
print("  The .gguf files are written to the outputs/ directory.")
