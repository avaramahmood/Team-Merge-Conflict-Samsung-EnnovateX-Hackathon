# Installation & Reproducibility

## 1. Environment

```bash
git clone <this-repo> && cd <this-repo>
python -m venv .venv && source .venv/bin/activate     # Python 3.12
pip install -r requirements.txt
```

- **Training (Layers 1-4):** 1 GPU with ≥ 48 GB (we used a 96 GB RTX PRO 6000 Blackwell;
  ≥ 24 GB works with smaller `PROMPTS_PER_ITER` / `GEN_BS`). CUDA 12.x, `bitsandbytes`.
- **Eval:** runs on 1 GPU, or 2× T4 (16 GB) via the `_t4` / `lm_eval` scripts.
- **Quantization (Layer 5):** needs `llama.cpp` built from source (CUDA optional).
- **Figures / paper only:** CPU-only, `pip install matplotlib numpy python-docx`.

## 2. Data & model layout

Scripts read from `./inputs` and write to `./outputs` (configurable in
[`src/config.py`](../src/config.py) or via `INPUTS=`, `DATA=`, `MODELS=`, `WORK=`). Place or
symlink open weights/datasets like so:

```
inputs/
  models/
    qwen2-5-7b/                                 # Qwen/Qwen2.5-7B
    deepseek-r1-distill-qwen-32b-awq/           # teacher (distillation)
    qwen2-5-math-prm-7b/qwen2.5-math-prm-7b/    # math PRM
    versa-prm/{Llama-PRM800K,VersaPRM}          # commonsense PRM (base + LoRA)
    distilled-qwen-sft-basic/pear_sft_epoch1/   # produced by Layer 1
  data/
    qwen-pear-dataset/                          # PEAR source prompts
    qwen-riva-datasets/{train,test}/            # gsm8k, math_hendrycks, strategyqa, csqa2, mmlu
    qwen-f-grpo-pool/                           # produced by Layer 2a
```

Download helpers (examples):

```bash
huggingface-cli download Qwen/Qwen2.5-7B            --local-dir inputs/models/qwen2-5-7b
huggingface-cli download Qwen/Qwen2.5-Math-PRM-7B   --local-dir inputs/models/qwen2-5-math-prm-7b/qwen2.5-math-prm-7b
# datasets: openai/gsm8k, cais/mmlu, ChilleD/StrategyQA, tasksource/commonsense_qa_2.0, hendrycks/competition_math
```

## 3. Full run order

```bash
# ── Layer 1: cold-start SFT ────────────────────────────────────────────
python src/data/build_pear_traces.py          # outputs/pear_sft_final.jsonl
python src/data/rescore_pear_aligned.py       # aligned logprobs
python src/train/pear_sft.py                  # outputs/pear_sft_epoch1

# ── Layers 2-3: GRPO (selected variant) ───────────────────────────────
python src/data/build_grpo_pools.py           # outputs/{math,sqa,mmlu}_pool
python src/train/grpo_sqa_antiguess.py        # outputs/grpo_best
#   ablations: grpo_sqa_prm_shaping.py | grpo_sqa_prm_select.py | grpo_v7_antiguess_base.py

# ── Layer 4: conciseness ──────────────────────────────────────────────
python src/train/conciseness_grpo.py          # outputs/grpo_concise_best

# ── Layer 5: quantize ─────────────────────────────────────────────────
python src/quantize/make_calib.py outputs/grpo_concise_best outputs/calib_concise.txt
bash   src/quantize/quantize.sh               # outputs/concise-Q4_K_XL.gguf

# ── Evaluate (edit MODEL_PATH / EVAL_MODE at top of file) ─────────────
python src/eval/eval_unified.py
```

## 4. Build llama.cpp (Layer 5 + on-device serving)

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON && cmake --build build -j        # drop -DGGML_CUDA=ON for CPU
```

`quantize.sh` calls `convert_hf_to_gguf.py`, `llama-imatrix` (domain calibration),
`llama-quantize` (UD-Q4_K_XL), and `llama-perplexity --kl-divergence` (validation).

## 5. Notes & gotchas

- All training scripts are **offline-safe** (`HF_*_OFFLINE=1`) and **resumable** (checkpoint +
  progress JSON); GRPO saves once at session end to respect output-size quotas.
- The reward verifier functions are intentionally duplicated verbatim across pool builder /
  GRPO / eval, **do not "DRY" them apart**; byte-identical correctness is a correctness
  invariant, not redundancy.
- Reproducibility video (linked in the README) walks through install → download → train →
  evaluate end to end.
