# Prior Iterations: How the Design Evolved

The final pipeline did not arrive fully formed. It is the third iteration, and the two earlier
ones produced the findings that shaped it. This page documents what we tried first, what each run
showed, and why we changed course. The raw logs are in
[`../results/previous_runs/`](../results/previous_runs/).

| Iteration | Notebook | What it did | What it taught us |
|---|---|---|---|
| v1 | `workinprogress.ipynb` | math-first PROF-GRPO (math only), PEAR epoch selection | math RL forgets commonsense; pick the SFT epoch for RL-readiness, not SFT accuracy |
| v2 | `progression.ipynb` | fixed eval harness, added CSQA2, proper PROF Stage 1, difficulty Pool A | a correct eval changes every number; commonsense needs its own data |
| final | `final.ipynb` | joint multi-domain GRPO, anti-guess reward, conciseness, quantization | the staged pipeline in this repo |

---

## v1 (workinprogress.ipynb): we trained mathematics first, then reasoning

The first plan was the obvious one: get mathematics working with PROF-GRPO, then move on to
commonsense. We ran PROF-GRPO on a math-only filtered pool (G = 8 rollouts, keep m = 4) starting
from the PEAR-SFT checkpoint.

**Finding 1: math-only RL lifts GSM8K and forgets commonsense.** Measured under the v1 eval
harness, the math-only checkpoint at step 100 moved GSM8K from 75.6 to 86.8 (+11.2) and MMLU
from 73.3 to 73.7, but StrategyQA fell from 72.5 to 66.4 (-6.1). Optimising one domain in
isolation degraded the others. This negative transfer is the single reason the final pipeline
trains commonsense and mathematics together, gives commonsense a budget equal to mathematics,
and uses a commonsense-specific reward.

![math-first transfer](figures/fig_math_first_transfer.png)

(These v1 numbers use an earlier evaluation harness with three bugs that v2 fixed, so they are
not directly comparable to the numbers elsewhere in this repository. The valid comparison is the
within-v1 delta: GSM8K up, StrategyQA down.)

**Finding 2: choose the SFT checkpoint for RL-readiness, not for SFT accuracy.** Before RL we
evaluated three PEAR-SFT epochs on four proxy metrics: a 50-step mini-GRPO proxy loss, the
fraction of correct rollouts (the "Goldilocks" target is 0.30 to 0.70, where there is signal to
learn from), the PEAR token-weight entropy, and the KL to the base model. Epoch 1 scored highest
on RL-readiness (0.386) ahead of epoch 2 (0.351), even though later epochs fit the SFT data
better. This is PEAR's thesis made concrete, and it is why the final pipeline ships the
single-epoch checkpoint. The same probe also showed correct fractions of 0.76 to 0.80, above the
Goldilocks band, which told us the math pool was too easy and motivated the difficulty-stratified
2-to-5-of-8 pool used later.

**Finding 3: engineering limits are real.** The v1 run died at step 200 with a disk-space error
during checkpointing. The final pipeline saves once at session end and resumes from a small
progress file, a direct response to that failure.

Logs: [`v1_math_first_prof_grpo.txt`](../results/previous_runs/v1_math_first_prof_grpo.txt),
[`v1_pear_epoch_selection.txt`](../results/previous_runs/v1_pear_epoch_selection.txt), and the
per-step series in
[`v1_math_first_prof_grpo_steps.csv`](../results/previous_runs/v1_math_first_prof_grpo_steps.csv).

---

## v2 (progression.ipynb): fix the measurement, then fix the data

**The eval harness was wrong, and fixing it moved every number.** v2 corrected three bugs: GSM8K
was reading the last `####` block, which on base models is often a hallucinated extra question,
so it now reads the first; StrategyQA was scanning hallucinated continuations for Yes or No, so
it now truncates at the first follow-on delimiter; and MMLU was scored without the answer choices
in the prompt. With the corrected harness the base model measured 82.9 / 72.8 / 77.9 rather than
the v1 numbers, and these corrected values are the baseline used throughout the final work. The
lesson is plain: a benchmark number is only as trustworthy as its extractor.

**Commonsense needs its own data.** v2 rebuilt the dataset and added CommonsenseQA 2.0, which is
structurally identical to StrategyQA (pure Yes/No, about 9.3k rows) and therefore directly
strengthens the commonsense target. The PEAR-SFT mix was rebalanced to roughly 40% mathematics,
30% commonsense, and the remainder split between GSM8K and MMLU auxiliary data.

**PROF, implemented properly.** v2 implemented PROF Stage 1 (Ye et al. 2025, arXiv:2509.03403)
faithfully: generate G rollouts, score each with the PRM and the outcome verifier, balance the
kept positive and negative groups, drop the low-process-score correct traces, and standardise
the advantage within the kept set. It also introduced difficulty Pool A, the pass@8 filter in the
2-to-5 band that the final pipeline generalised to all three domains.

Logs: [`v2_progression_evals.txt`](../results/previous_runs/v2_progression_evals.txt).

---

## How the prior runs map onto the final design

Every change in the shipped pipeline traces back to one of these findings.

- Math-only forgetting (v1) leads to joint multi-domain GRPO with a commonsense budget and a
  commonsense-specific anti-guess reward.
- RL-readiness over SFT accuracy (v1) leads to shipping the single-epoch PEAR-SFT checkpoint.
- Correct fractions above the Goldilocks band (v1) lead to the difficulty-stratified 2-to-5-of-8
  pool.
- The disk failure (v1) leads to save-once-and-resume training.
- The eval bugs (v2) lead to the byte-identical verifier shared across the pool builder, every
  reward, and evaluation.
- The CSQA2 addition (v2) lead to the commonsense corpus the final run trains on.

The full final-pipeline analysis is in [methods-and-observations.md](methods-and-observations.md);
the cell-by-cell map of the final notebook is in [notebook-walkthrough.md](notebook-walkthrough.md).
