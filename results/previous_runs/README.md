# Previous runs (raw logs from earlier iterations)

These are the original logs from the two iterations that preceded the final pipeline. They are
the evidence behind [`../../docs/prior-iterations.md`](../../docs/prior-iterations.md), which
explains what each run taught us and how it shaped the final design.

| File | Iteration | Source | Contents |
|---|---|---|---|
| `v1_math_first_prof_grpo.txt` | v1 | `workinprogress.ipynb` cells 20, 22-24 | math-only PROF-GRPO execution log, the disk-space failure at step 200, and before/after eval (GSM8K +11.2, StrategyQA -6.1) |
| `v1_math_first_prof_grpo_steps.csv` | v1 | parsed from the log above | per-step loss, kept/step, and pass-rate (of 8) over 200 steps |
| `v1_pear_epoch_selection.txt` | v1 | `workinprogress.ipynb` cells 3, 18 | RL-readiness scoring of three PEAR-SFT epochs (proxy loss, correct fraction, weight entropy, KL) |
| `v2_progression_evals.txt` | v2 | `progression.ipynb` cells 11, 13, 14 | corrected-harness eval of the base model and the PEAR-SFT checkpoint (few-shot and zero-shot) |

Headline finding (v1): training mathematics first with PROF-GRPO lifted GSM8K by 11.2 points but
cost 6.1 points on StrategyQA, which is why the final pipeline trains the domains jointly. See
the figure `docs/figures/fig_math_first_transfer.png`.

Note on comparability: the v1 numbers use an earlier eval harness with three extractor bugs that
v2 fixed, so only within-v1 deltas are meaningful. The v2 corrected numbers (base 82.9 / 72.8 /
77.9) are the baseline used throughout the rest of the repository.
