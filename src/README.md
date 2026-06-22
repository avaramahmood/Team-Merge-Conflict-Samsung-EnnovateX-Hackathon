# Source layout & run order

All scripts read open models/datasets from `./inputs` and write to `./outputs`
(see [`config.py`](config.py); override with `INPUTS=…`, `DATA=…`, `MODELS=…`,
`WORK=…`). Paths are kept as plain top-of-file constants so one `grep` shows
exactly what each stage touches.

```
src/
├── config.py                 # central path config (env-overridable)
├── data/
│   ├── build_pear_traces.py      # L1a: 32B-AWQ teacher rollouts -> reweighted SFT traces (+ logprobs)
│   ├── rescore_pear_aligned.py   # L1b: re-score traces with shared tokenizer (fixes logprob alignment)
│   └── build_grpo_pools.py       # L2a: pass@8 difficulty-stratified prompt pools (vLLM)
├── train/
│   ├── pear_sft.py               # L1: PEAR-weighted SFT (cold start)
│   ├── grpo_v7_antiguess_base.py # L2-3: reference GRPO (math PROF + SQA plain), difficulty-aware
│   ├── grpo_sqa_antiguess.py     # L2-3: SQA = verifier-only anti-guess reward   [SELECTED]
│   ├── grpo_sqa_prm_shaping.py   # L2-3: SQA = VersaPRM dense reward shaping       (ablation)
│   ├── grpo_sqa_prm_select.py    # L2-3: SQA = VersaPRM hard selection             (collapsed)
│   └── conciseness_grpo.py       # L4: correctness-conditioned brevity (GRPO-LEAD)
├── eval/
│   ├── eval_unified.py           # 3-mode fair eval (base_fewshot / think_fewshot / think_zeroshot)
│   ├── eval_self_consistency.py  # maj@8 self-consistency variant
│   ├── eval_unified_t4.py        # same logic, sharded for 2× T4 (16 GB) boxes
│   └── lm_eval_run.py            # EleutherAI lm-eval-harness runner (official numbers)
├── quantize/
│   ├── make_calib.py             # L5a: domain-matched calibration corpus (model's own CoT)
│   ├── quantize.py               # L5b: imatrix + UD-Q4_K_XL GGUF + KL validation
│   └── quantize.sh               # shell driver for llama.cpp build + quantize
└── figures/
    └── make_figures.py           # regenerate every figure in docs/ and the paper
```

## Pipeline order

```bash
# Layer 1, cold-start SFT
python src/data/build_pear_traces.py        # -> outputs/pear_sft_final.jsonl
python src/data/rescore_pear_aligned.py     # -> aligned logprobs
python src/train/pear_sft.py                # -> outputs/pear_sft_epoch1

# Layer 2-3, GRPO (run the SELECTED variant; others are ablations)
python src/data/build_grpo_pools.py         # -> outputs/{math,sqa,mmlu}_pool
python src/train/grpo_sqa_antiguess.py      # -> outputs/grpo_best

# Layer 4, conciseness
python src/train/conciseness_grpo.py        # -> outputs/grpo_concise_best

# Layer 5, quantize to on-device GGUF
python src/quantize/make_calib.py outputs/grpo_concise_best outputs/calib.txt
bash   src/quantize/quantize.sh             # -> outputs/concise-Q4_K_XL.gguf

# Evaluate any checkpoint (set MODEL_PATH / EVAL_MODE at the top of the file)
python src/eval/eval_unified.py
```

> **Compute.** Layers 1-4 were run on a single 96 GB GPU (RTX PRO 6000 Blackwell);
> `eval_unified_t4.py` and `lm_eval_run.py` shard onto 2× T4 (16 GB) as in the
> hackathon's recommended infra. Quantization (Layer 5) needs `llama.cpp` built
> from source; everything else is pip-installable (`requirements.txt`).
