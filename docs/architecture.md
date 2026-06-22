# Technical Architecture

## System overview

The solution is a **5-layer stacked training pipeline** that converts a stock
**Qwen2.5-7B** into a quantized, on-device reasoner. Each layer is an independent,
checkpoint-to-checkpoint stage with its own reward/objective; the output of one is the input
of the next.

![pipeline](figures/fig_pipeline.png)

```
                 ┌────────────────────────────────────────────────────────────────┐
 open weights ──►│  L1 PEAR-SFT      32B-AWQ teacher → 7B student, token-reweighted │──► pear_sft_epoch1
                 ├────────────────────────────────────────────────────────────────┤
                 │  L2 PROF-GRPO     math: PRM step-filter | sqa: anti-guess reward │
                 │  L3 Curriculum    reward composition: format → accuracy          │──► grpo_best
                 ├────────────────────────────────────────────────────────────────┤
                 │  L4 Conciseness   correctness-conditioned length decay (GRPO-LEAD)│──► grpo_concise_best
                 ├────────────────────────────────────────────────────────────────┤
                 │  L5 Quantize      domain-imatrix → UD-Q4_K_XL GGUF + KL validate │──► concise-Q4_K_XL.gguf (5.09 GB)
                 └────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼  runs fully offline via llama.cpp
                              On-device app (Knowledge / Thinking modes, Python tool)
```

## Component map

| Layer | Module(s) | Reads | Writes | Key models |
|---|---|---|---|---|
| L1a traces | `src/data/build_pear_traces.py` | PEAR prompts, teacher, base | `pear_sft_final.jsonl` (+ per-token logprobs) | R1-Distill-Qwen-32B-AWQ (π_β), Qwen2.5-7B (π_θ) |
| L1b rescore | `src/data/rescore_pear_aligned.py` | traces | aligned logprobs | both, shared tokenizer |
| L1 SFT | `src/train/pear_sft.py` | aligned traces | `pear_sft_epoch1` | Qwen2.5-7B |
| L2a pools | `src/data/build_grpo_pools.py` | benchmark train splits | `{math,sqa,mmlu}_pool` | policy (vLLM, pass@8) |
| L2-3 GRPO | `src/train/grpo_sqa_antiguess.py` (+3 variants) | pools, SFT ckpt | `grpo_best` | policy, Qwen-Math-PRM, VersaPRM |
| L4 concise | `src/train/conciseness_grpo.py` | pools, `grpo_best` | `grpo_concise_best` | policy |
| L5 quant | `src/quantize/{make_calib,quantize}.py` | `grpo_concise_best` | `concise-Q4_K_XL.gguf` | llama.cpp |
| Eval | `src/eval/eval_unified.py` (+ SC, T4, lm-eval) | test splits, any ckpt | `eval_results/*.json` | n/a |

## Data-flow & reward design

- **Verifier-centred design.** A single set of `normalize_*` / `extract_*` functions defines
  correctness for math (numeric, fraction-safe), SQA (yes/no), and MMLU (letter). These are
  **byte-identical** across the pool builder, every GRPO reward, and eval, so RL optimises
  the exact distribution that is measured, with no train/eval skew.
- **Difficulty-aware advantage.** `adv = r − mean(group)` over the full 8-rollout group, no
  std, no balancing (Dr.GRPO). Difficulty is encoded directly in the gradient magnitude.
- **Process signal placement.** Math: PRM as a *filter before* GRPO (PROF). SQA: either a
  *zero-mean within-group shaping* term or a *verifier-only anti-guess* reward, never a hard
  selector (which the policy games; see `methods-and-observations.md`).
- **Reward-weight curriculum (Layer 3).** The *composition* of reward signals shifts across
  stages: format → accuracy (GRPO) → efficiency (conciseness). This staged re-weighting of
  *competing* objectives is the main methodological contribution.

## Memory & compute architecture

- **Training box:** 1× RTX PRO 6000 Blackwell (96 GB). GRPO holds policy (15 GB) + frozen
  reference (15 GB) + AdamW8bit (~15 GB) resident; the 7B PRMs swap in/out around scoring
  (transient +15 GB). Peak < 70 GB.
- **Eval box:** 2× NVIDIA T4 (16 GB), `eval_unified_t4.py` shards with `device_map="auto"`
  and fp16 (Turing has no hardware bf16); `lm_eval_run.py` uses vLLM `tensor_parallel_size=2`.
- **Deployment:** CPU/edge GPU via llama.cpp on the 5.09 GB GGUF.

## Design principles

1. **Stack, don't fuse.** Each layer is checkpoint-clean and independently evaluable, so a
   regression is attributable to one stage (this is how we caught the v8-C collapse).
2. **Outcome dominates, process re-ranks.** Every reward keeps correctness as the top term;
   process/length signals only break ties, the structural defence against reward hacking.
3. **Optimise the measured distribution.** Identical verifiers/prompts end to end.
4. **Don't over-train.** One SFT epoch; capped GRPO/conciseness passes; held-out (not pool)
   numbers reported.
