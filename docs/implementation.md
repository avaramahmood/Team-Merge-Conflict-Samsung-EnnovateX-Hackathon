# Implementation Details, Tech Stack & OSS Inventory

## Technical stack

| Concern | Choice | Why |
|---|---|---|
| Base / policy | Qwen2.5-7B | strong GSM8K/StrategyQA headroom, shared vocab with the 32B teacher |
| Teacher (distill) | DeepSeek-R1-Distill-Qwen-32B (AWQ 4-bit) | long-CoT reasoner, same tokenizer ⇒ token-level PEAR weight is defined |
| Training | PyTorch + HF Transformers, custom GRPO loop | full control of advantage/KL/reward (TRL hides exactly the knobs that mattered) |
| Optimiser | `bitsandbytes` AdamW8bit | fits policy+ref+optimizer in one box |
| Rollouts | vLLM (pass@8 pool build) | paged batched decoding, ~10× HF `.generate` |
| Process rewards | Qwen2.5-Math-PRM-7B, VersaPRM (LoRA via PEFT) | math step scoring + commonsense step scoring |
| Quantization | llama.cpp (imatrix, UD-Q4_K_XL), gguf | on-device GGUF, KL-validated |
| Eval | custom 3-mode harness + EleutherAI lm-eval | native-format fairness + official numbers |
| Figures/paper | matplotlib, python-docx | reproducible visuals + arXiv-style Word doc |

## OSS libraries & projects used (with links)

- PyTorch, https://github.com/pytorch/pytorch
- Hugging Face Transformers, https://github.com/huggingface/transformers
- Hugging Face Datasets, https://github.com/huggingface/datasets
- Accelerate, https://github.com/huggingface/accelerate
- bitsandbytes, https://github.com/bitsandbytes-foundation/bitsandbytes
- PEFT, https://github.com/huggingface/peft
- vLLM, https://github.com/vllm-project/vllm
- llama.cpp, https://github.com/ggml-org/llama.cpp
- gguf (Python), https://pypi.org/project/gguf/
- EleutherAI lm-evaluation-harness, https://github.com/EleutherAI/lm-evaluation-harness
- GPTQModel (AWQ teacher loading), https://github.com/ModelCloud/GPTQModel
- matplotlib, https://github.com/matplotlib/matplotlib · python-docx, https://github.com/python-openxml/python-docx

Open models: Qwen2.5-7B, Qwen2.5-Math-PRM-7B (Apache-2.0); DeepSeek-R1-Distill-Qwen-32B
(MIT); VersaPRM (on Llama-3.1-8B). Open datasets: GSM8K, MATH, StrategyQA, CSQA2, MMLU,
AQuA-RAT (see README for links/licenses).

## Implementation highlights

### PEAR token weighting (`src/train/pear_sft.py`)
Per-trace weights are computed from `δ_t = log π_θ − log π_β`, suffix-accumulated with a
discount, **mean-centered** (removes the 32B-vs-7B confidence offset that otherwise collapses
the weight to `exp(−10)` on early tokens), symmetric-clipped, and **renormalised to mean 1**
so every trace contributes equally regardless of length. Loss is **fp32 per-item
cross-entropy** on the trace span only (prompt masked), PEAR-weighted. `uniform` mode (all
weights = 1) is the shipped, robust control.

### Shared-tokenizer logprob alignment (`src/data/rescore_pear_aligned.py`)
Both teacher and student are run on the **same** base-tokenized `input_ids`, making the two
logprob arrays equal-length and position-aligned by construction (fixed a silent +2-token
misalignment that made `δ_t` compare different tokens).

### Difficulty-aware GRPO (`src/train/grpo_*.py`)
`adv = r − mean(full 8-rollout group)`, no std, no balancing (Dr.GRPO). Importance ratio uses
the behaviour-policy logprob as denominator; KL is to a **frozen SFT reference**; asymmetric
clip ε=[0.20,0.28]; grad-clip 0.5; micro-batched backward (groups of 4). Online pool
retirement (8/8 → retire; 0/8×2 → retire) keeps the pool fresh at no extra cost.

### Reward functions (identical across pool/GRPO/eval)
`normalize_math_gold` (fraction-safe, `\dfrac`→decimal), `normalize_sqa` (yes/no),
`normalize_mmlu` (letter); `extract_answer` prefers `<answer>…</answer>`, then `\boxed{}`,
then last-number/letter fallback. **Byte-identical** everywhere, the single most important
correctness invariant.

### Conciseness reward (`src/train/conciseness_grpo.py`)
`r = −1` if wrong, else `exp(−0.6·L/budget)`. Correctness dominates; brevity only re-ranks
correct traces. 8/8 retirement disabled (varied-length 8/8 groups are the brevity signal).

### Domain-matched quantization (`src/quantize/`)
`make_calib.py` generates the importance-matrix corpus from the model's **own** greedy CoT in
its exact chat template; `quantize.py` builds llama.cpp, runs `llama-imatrix`, quantizes to
UD-Q4_K_XL (W4 + Q8_0 embeddings/lm_head), and validates with `--kl-divergence` vs a Q8_0
reference.

## Salient features

1. **End-to-end SLM-RL pipeline** from open weights to a 5 GB on-device GGUF.
2. **Novel reward-weight curriculum** (format → accuracy → efficiency) over *competing*
   objectives across discrete stages.
3. **Verifier-only anti-guess reward** that beats two PRM-based designs on commonsense without
   any extra model.
4. **Documented reward-hacking analysis**: one overt collapse (v8-C) and one subtle case
   (v8-B vs held-out), with the curves to prove it.
5. **KL-validated 4-bit quantization** (0.008) that keeps reasoning accuracy on-device.
6. **Fair, multi-mode, reproducible evaluation** + official lm-eval runner.
