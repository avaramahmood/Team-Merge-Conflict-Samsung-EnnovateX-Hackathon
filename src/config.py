"""
Central path configuration for the Stacked-RL SLM pipeline.

The training/eval/quant scripts reference open models and datasets through a
single local layout under ./inputs and write to ./outputs. Point these at your
own copies (HF cache, a mounted dataset disk, etc.) either by editing the
constants below or by exporting the matching environment variables.

Layout
------
inputs/
  models/
    qwen2-5-7b/                      # Qwen/Qwen2.5-7B
    distilled-qwen-sft-basic/pear_sft_epoch1   # our Layer-1 SFT checkpoint
    deepseek-r1-distill-qwen-32b-awq/          # distillation teacher (pi_beta)
    qwen2-5-math-prm-7b/            # math process reward model
    versa-prm/{Llama-PRM800K,VersaPRM}         # commonsense PRM (base + LoRA)
  data/
    qwen-pear-dataset/             # PEAR source prompts (pear_{math,commonsense,mmlu})
    qwen-riva-datasets/{train,test}            # benchmark splits (gsm8k, mmlu, ...)
    qwen-f-grpo-pool/              # pass@8 difficulty pools for GRPO
outputs/                           # checkpoints, logs, gguf

Every script keeps these paths as plain top-of-file constants so a single grep
shows exactly what it reads. Override at runtime, e.g.:

    INPUTS=/mnt/disk/inputs WORK=/mnt/disk/outputs python src/train/pear_sft.py
"""
import os

INPUTS = os.environ.get("INPUTS", "inputs")
MODELS = os.environ.get("MODELS", os.path.join(INPUTS, "models"))
DATA = os.environ.get("DATA", os.path.join(INPUTS, "data"))
WORK = os.environ.get("WORK", "outputs")

# Convenience handles (mirror the constants used inside the scripts)
BASE_MODEL = os.path.join(MODELS, "qwen2-5-7b")
SFT_CKPT = os.path.join(MODELS, "distilled-qwen-sft-basic", "pear_sft_epoch1")
TEACHER = os.path.join(MODELS, "deepseek-r1-distill-qwen-32b-awq")
MATH_PRM = os.path.join(MODELS, "qwen2-5-math-prm-7b", "qwen2.5-math-prm-7b")
BENCH_ROOT = os.path.join(DATA, "qwen-riva-datasets")
GRPO_POOL = os.path.join(DATA, "qwen-f-grpo-pool")

for d in (INPUTS, WORK):
    os.makedirs(d, exist_ok=True)
