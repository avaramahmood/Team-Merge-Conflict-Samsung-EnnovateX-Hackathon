# Training logs (parsed series)

These CSVs are parsed directly from the per-iteration / per-step lines printed in
`final.ipynb` (the same lines preserved verbatim in `../notebook_outputs/`). They are the
single source of truth for every plot in `docs/figures/`, `src/figures/make_figures.py` reads
them, nothing is hand-entered into the charts.

| File | Source cells | Rows | Columns |
|---|---|---|---|
| `pear_sft.csv` | 5 | 4 logged steps | step, loss, pos_loss, lr, grad_norm |
| `pear_sft_paper.csv` | PEAR ablation run | 189 steps | step, loss, lr, grad_norm |
| `pear_sft_centered.csv` | PEAR ablation run | 189 steps | step, loss, lr, grad_norm |
| `pear_sft_modes.csv` | derived | 3 modes | mode, val_loss, steady_grad_norm, step1_grad_norm, shipped |
| `grpo_prm_shaping.csv` | 14, 15 | 86 iters | iter, sqa, math, surv, buf, pg, kl, gn, ret |
| `grpo_antiguess.csv` | 17, 18 | 85 iters | (shipped run) same columns |
| `grpo_prm_select.csv` | 20, 21 | 90 iters | (collapsed run) same columns |
| `conciseness.csv` | 23 | 12 iters | + len_sqa, len_math |
| `kpi_summary.csv` | 8,10,12,16,19,22 | 6 | held-out GSM8K/MMLU/StrategyQA/avg per checkpoint |

Column key: `sqa`/`math` = pool pass-rate; `surv` = surviving prompts; `buf` = buffer size;
`pg` = policy-gradient term; `kl` = KL to frozen reference; `gn` = grad norm; `ret` = cumulative
retired prompts; `len_*` = mean correct-trace token length.

Aggregated pass@8 difficulty distributions per domain are in
`../pool_pass8_distribution.json` (from cell 13). Raw, human-readable cell logs are in
`../notebook_outputs/`. See `../../docs/notebook-walkthrough.md` for the full cell→artifact map.

## PEAR-SFT weighting ablation (why we ship `uniform`)
Three single-epoch (189-step) cold-start runs differing only in the PEAR token-weight mode.
The reweighting buys no reliable validation gain while roughly doubling gradient noise:

| Mode | Val loss | Steady-state grad norm (steps 50-150) | Step-1 grad norm | Shipped |
|---|---|---|---|---|
| **uniform** | 0.552 | **2.34** | 16.5 | ✅ |
| paper (raw PEAR weight) | 0.536 | 4.64 (~2x) | 27.25 | |
| centered (mean-subtracted) | 0.569 | 3.50 (~1.5x) | 28.62 | |

`paper` is only 0.016 below `uniform` (within single-seed, single-epoch noise) and `centered`
is actually 0.017 **worse**, so the reweighting's effect on generalisation is inconsistent and
not worth ~2x the gradient variance on a cold start whose only job is a clean, low-variance RL
launch point. Plotted in `docs/figures/fig_sft_variants.png`.

## Headline runs captured
- PEAR-SFT: 1 epoch / 188 steps, val loss 0.552 (shipped `uniform`; `paper`/`centered` ablations above)
- GRPO v8-A PRM-shaping: 86 iters, eval 85.7 / 71.2 / 77.3
- GRPO v8-B anti-guess: 85 iters, eval 88.3 / 69.8 / 79.8  **[selected]**
- GRPO v8-C PRM-select: 90 iters, capability collapse, eval 85.8 / 70.9 / 75.1
- Conciseness GRPO: 12 iters, lengths −8 to −10%, accuracy held
- Quantization: KL 0.008, PPL ratio 1.013, same-top-1 97.8%
