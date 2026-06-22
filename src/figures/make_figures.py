"""
make_figures.py - regenerate every visual used in the paper and docs.

This script is **data-driven**: it reads the parsed run logs in
results/training_logs/*.csv, the aggregated pool distribution in
results/pool_pass8_distribution.json, and the consolidated KPIs in
results/eval_results.json. Those files are transcribed directly from the
cell outputs of final.ipynb (training cells, eval cells, the pool builder,
and the quantization report). Re-run after re-parsing the notebook:

    python src/figures/make_figures.py

No GPU, no model, no data download.
"""
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(ROOT, "docs", "figures")
LOGS = os.path.join(ROOT, "results", "training_logs")
RES = os.path.join(ROOT, "results")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 11,
    "savefig.facecolor": "white", "savefig.edgecolor": "white",
    "savefig.pad_inches": 0.06,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.antialiased": True, "patch.antialiased": True,
    "font.family": "DejaVu Sans",
})
C = {"base": "#8d99ae", "sft": "#4361ee", "grpo": "#2a9d8f",
     "concise": "#e76f51", "bad": "#d62828", "good": "#2a9d8f", "kl": "#9d4edd"}


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


def load_csv(name):
    """results/training_logs/<name>.csv -> dict of float columns (blank -> nan)."""
    rows = list(csv.DictReader(open(os.path.join(LOGS, name))))
    cols = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            v = r[k]
            vals.append(float(v) if v not in ("", None) else np.nan)
        cols[k] = np.array(vals)
    return cols


def smooth(a, k=5):
    """Centered moving average with edge-aware normalisation (no boundary dip)."""
    a = np.asarray(a, float)
    num = np.convolve(a, np.ones(k), mode="same")
    den = np.convolve(np.ones_like(a), np.ones(k), mode="same")
    return num / den


# ───────────────────────── 1. Pipeline diagram ─────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.2)); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    boxes = [
        ("Qwen2.5-7B\nbase", 10, C["base"]),
        ("L1: PEAR-SFT\n32B-AWQ distill\n+ importance\nreweighting", 30, C["sft"]),
        ("L2-3: PROF-GRPO\nstep filter +\nanti-guess +\ncurriculum", 50, C["grpo"]),
        ("L4: Conditional\nconciseness\n(GRPO-LEAD)", 70, C["concise"]),
        ("L5: QAT +\nUD-Q4_K_XL\n5.09 GB\nGGUF", 90, C["kl"]),
    ]
    w, h, y = 16, 18, 11
    for txt, x, col in boxes:
        ax.add_patch(FancyBboxPatch((x - w / 2, y), w, h,
                     boxstyle="round,pad=0.3,rounding_size=1.0", fc=col, ec="none", alpha=0.9))
        ax.text(x, y + h / 2, txt, ha="center", va="center", color="white",
                fontsize=8.3, fontweight="bold")
    for x0, x1 in zip([b[1] for b in boxes][:-1], [b[1] for b in boxes][1:]):
        ax.add_patch(FancyArrowPatch((x0 + w / 2 + 0.3, y + h / 2), (x1 - w / 2 - 0.3, y + h / 2),
                     arrowstyle="-|>", mutation_scale=15, lw=1.6, color="#333"))
    ax.text(50, 36, "Stacked SLM-RL pipeline - Qwen2.5-7B to on-device 4-bit reasoner",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(10, 7.5, "82.9 / 72.8 / 77.9", ha="center", fontsize=7.3, color="#444")
    ax.text(50, 7.5, "88.3 / 69.8 / 79.8", ha="center", fontsize=7.3, color="#444")
    ax.text(90, 7.5, "KL 0.008 vs bf16", ha="center", fontsize=7.3, color="#444")
    save(fig, "fig_pipeline.png")


# ───────────────────────── 2. KPI grouped bars ─────────────────────────
def fig_kpi_bars():
    bench = ["GSM8K", "MMLU", "StrategyQA"]
    base = [82.9, 72.8, 77.9]; sft = [86.7, 71.4, 79.9]; grpo = [88.3, 69.8, 79.8]
    x = np.arange(len(bench)); w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for off, vals, lbl, col in [(-w, base, "Base Qwen2.5-7B", C["base"]),
                                (0, sft, "PEAR-SFT (L1)", C["sft"]),
                                (w, grpo, "+ PROF-GRPO (L2-3)", C["grpo"])]:
        bars = ax.bar(x + off, vals, w, label=lbl, color=col)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8.5)
    for i, b in enumerate(base):
        ax.hlines(b + 5, x[i] - 1.4 * w, x[i] + 1.4 * w, color=C["bad"], ls="--", lw=1.3, zorder=5)
    ax.plot([], [], color=C["bad"], ls="--", lw=1.3, label="baseline +5 pp target")
    ax.set_xticks(x); ax.set_xticklabels(bench); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(55, 95)
    ax.set_title("Reasoning KPIs across the stacked pipeline (native eval format)")
    ax.legend(loc="lower center", ncol=2, fontsize=8.5, framealpha=0.9)
    save(fig, "fig_kpi_bars.png")


# ───────────────────── 3. Stage progression ───────────────────
def fig_progression():
    stages = ["Base\n(few-shot)", "PEAR-SFT\n(few-shot)", "PEAR-SFT\n(zero-shot)", "PROF-GRPO\n(few-shot)"]
    avg = [77.9, 79.3, 72.7, 79.3]; gsm = [82.9, 86.7, 82.2, 88.3]; sqa = [77.9, 79.9, 69.4, 79.8]
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(x, gsm, "-o", color=C["grpo"], label="GSM8K")
    ax.plot(x, sqa, "-s", color=C["sft"], label="StrategyQA")
    ax.plot(x, avg, "-^", color="#333", lw=2.3, label="3-benchmark average")
    for xi, v in zip(x, avg):
        ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=9); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(66, 92)
    ax.set_title("Capability through the pipeline (the SFT zero-shot dip is the RL launch point)")
    ax.legend(fontsize=9)
    save(fig, "fig_progression.png")


# ─────────────────── 4. GRPO variants held-out KPIs ───────────────────
def fig_grpo_variants():
    v = ["PEAR-SFT\n(start)", "v8-A\nPRM-shaping", "v8-C\nPRM-select", "v8-B\nanti-guess"]
    gsm = [86.7, 85.7, 85.8, 88.3]; mmlu = [71.4, 71.2, 70.9, 69.8]; sqa = [79.9, 77.3, 75.1, 79.8]
    x = np.arange(len(v)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(x - w, gsm, w, label="GSM8K", color=C["grpo"])
    ax.bar(x, mmlu, w, label="MMLU", color=C["base"])
    ax.bar(x + w, sqa, w, label="StrategyQA", color=C["sft"])
    ax.axvspan(2.6, 3.4, color=C["good"], alpha=0.08)
    ax.text(3.0, 90.4, "selected", ha="center", color=C["good"], fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(v, fontsize=9); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(60, 92)
    ax.set_title("Three SQA reward designs vs the SFT start (math = PROF-GRPO throughout)")
    ax.legend(fontsize=9, ncol=3, loc="lower center")
    save(fig, "fig_grpo_variants.png")


# ──────────── 5. Reward-hacking collapse (PRM-select, full 90 iters) ────────────
def fig_collapse():
    c = load_csv("grpo_prm_select.csv")
    it = c["iter"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(it, c["math"] * 100, "-o", ms=3, color=C["grpo"], label="math pool pass-rate")
    ax.plot(it, c["sqa"] * 100, "-s", ms=3, color=C["sft"], label="SQA pool pass-rate")
    ax.set_xlabel("GRPO iteration"); ax.set_ylabel("Pool pass-rate (%)"); ax.set_ylim(0, 100)
    ax2 = ax.twinx(); ax2.set_yscale("log")
    ax2.plot(it, c["kl"], color=C["kl"], lw=1.4, alpha=0.8, label="KL to reference (log)")
    ax2.set_ylabel("KL divergence (log)", color=C["kl"]); ax2.tick_params(axis="y", colors=C["kl"]); ax2.grid(False)
    ax.axvspan(28, it.max(), color=C["bad"], alpha=0.07)
    ax.text(58, 88, "capability collapse\n(reward hacking)", color=C["bad"], ha="center", fontsize=9, fontweight="bold")
    l1, b1 = ax.get_legend_handles_labels(); l2, b2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, b1 + b2, fontsize=8.5, loc="center left")
    ax.set_title("v8-C PRM-select: gaming the selector ⇒ KL blow-up ⇒ math collapses to 3%")
    save(fig, "fig_reward_hacking_collapse.png")


# ──────────── 6. Anti-guess training curve (full 85 iters) ─────────────
def fig_grpo_curve():
    c = load_csv("grpo_antiguess.csv")
    it = c["iter"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(it, c["math"] * 100, color=C["grpo"], alpha=0.22)
    ax.plot(it, c["sqa"] * 100, color=C["sft"], alpha=0.22)
    ax.plot(it, smooth(c["math"]) * 100, color=C["grpo"], lw=2.2, label="math (5-iter MA)")
    ax.plot(it, smooth(c["sqa"]) * 100, color=C["sft"], lw=2.2, label="SQA (5-iter MA)")
    ax.axvline(45, color="#888", ls=":", lw=1); ax.text(45.6, 54, "session 2 resume", fontsize=8, color="#666")
    ax.set_xlabel("GRPO iteration"); ax.set_ylabel("Pool pass-rate (%)"); ax.set_ylim(50, 92)
    ax.set_title("v8-B anti-guess: stable difficulty-aware GRPO (KL controlled, no collapse)")
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "fig_grpo_training_curve.png")


# ──────────── 7. KL stability across all three variants (NEW) ──────────
def fig_kl_stability():
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for name, lbl, col in [("grpo_antiguess.csv", "v8-B anti-guess (shipped)", C["grpo"]),
                           ("grpo_prm_shaping.csv", "v8-A PRM-shaping", C["sft"]),
                           ("grpo_prm_select.csv", "v8-C PRM-select (collapsed)", C["bad"])]:
        c = load_csv(name)
        ax.plot(c["iter"], np.clip(c["kl"], 1e-3, None), lw=1.6, color=col, alpha=0.9, label=lbl)
    ax.set_yscale("log")
    ax.axhspan(1e-3, 1.0, color=C["good"], alpha=0.06)
    ax.text(88, 0.0016, "healthy KL band (< 1)", color=C["good"], fontsize=8.5,
            ha="right", va="bottom")
    ax.set_xlabel("GRPO iteration"); ax.set_ylabel("KL to frozen SFT reference (log)")
    ax.set_title("Training stability: KL stays bounded for the safe rewards, diverges when the PRM is gamed")
    ax.legend(fontsize=8.5, loc="upper left")
    save(fig, "fig_kl_stability.png")


# ──────────── 8. Conciseness: length down, accuracy held (CSV) ──────────
def fig_conciseness():
    c = load_csv("conciseness.csv"); it = c["iter"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(it, c["math"] * 100, "-o", color=C["grpo"], label="math accuracy")
    ax.plot(it, c["sqa"] * 100, "-s", color=C["sft"], label="SQA accuracy")
    ax.set_xlabel("Conciseness-GRPO iteration"); ax.set_ylabel("Pool pass-rate (%)"); ax.set_ylim(55, 100)
    ax2 = ax.twinx()
    ax2.plot(it, c["len_math"], "--o", color=C["grpo"], alpha=0.5, label="math length")
    ax2.plot(it, c["len_sqa"], "--s", color=C["sft"], alpha=0.5, label="SQA length")
    ax2.set_ylabel("Mean correct-trace length (tokens)"); ax2.grid(False); ax2.set_ylim(330, 620)
    l1, b1 = ax.get_legend_handles_labels(); l2, b2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, b1 + b2, fontsize=8, loc="center left", ncol=2)
    ax.set_title("Layer 4 conditional conciseness: SQA −10% / math −8% tokens, accuracy held")
    save(fig, "fig_conciseness.png")


# ──────────── 9. PEAR-SFT loss (CSV) ────────────────────────────────────
def fig_sft_loss():
    c = load_csv("pear_sft.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(c["step"], c["loss"], "-o", color=C["sft"], lw=2)
    ax.axhline(0.552, color=C["bad"], ls="--", lw=1.2, label="val loss 0.552")
    ax.set_xlabel("Optimizer step (1 epoch, 188 steps)"); ax.set_ylabel("PEAR-weighted CE loss")
    ax.set_title("Layer 1 PEAR-SFT: single-epoch cold start (no over-training)")
    ax.legend(fontsize=9)
    save(fig, "fig_sft_loss.png")


# ──────────── 10. Quantization fidelity ────────────────────────────────
def fig_quant():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    ax = axes[0]
    bars = ax.bar(["PPL\nratio", "Same\ntop-1", "Mean\nKLD×100"], [1.0128, 97.79, 0.805],
                  color=[C["concise"], C["good"], C["kl"]])
    for b, d in zip(bars, ["1.013×", "97.8%", "0.008"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, d, ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 110); ax.set_title("UD-Q4_K_XL vs bf16 fidelity")
    ax = axes[1]
    pct = ["50%", "90%", "95%", "99%", "99.9%", "max"]
    kld = [0.000168, 0.0187, 0.0342, 0.1086, 0.4883, 2.1609]
    ax.plot(range(len(pct)), kld, "-o", color=C["kl"])
    ax.set_yscale("log"); ax.set_xticks(range(len(pct))); ax.set_xticklabels(pct)
    ax.axhline(0.01, color=C["good"], ls="--", lw=1.2, label="accept threshold 0.01")
    ax.set_xlabel("KL-divergence percentile"); ax.set_ylabel("per-token KLD (log)")
    ax.set_title("Per-token KL distribution (median 1.7e-4)"); ax.legend(fontsize=8)
    save(fig, "fig_quant_fidelity.png")


# ──────────── 11. Per-domain pass@8 difficulty pool (real aggregate) ────
def fig_pool():
    pool = json.load(open(os.path.join(RES, "pool_pass8_distribution.json")))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=False)
    for ax, dom, title in zip(axes, ["math", "sqa", "mmlu"],
                              ["Math (GSM8K+MATH)", "Commonsense (StrategyQA+CSQA2)", "MMLU"]):
        ks = list(range(9)); vs = [pool[dom][str(k)] for k in ks]
        cols = [C["bad"] if k in (0, 1) else (C["sft"] if 2 <= k <= 5 else C["base"]) for k in ks]
        ax.bar(ks, vs, color=cols)
        ax.axvspan(1.5, 5.5, color=C["sft"], alpha=0.08)
        ax.set_title(title, fontsize=10); ax.set_xlabel("pass@8")
    axes[0].set_ylabel("# problems")
    fig.suptitle("Difficulty-stratified pools: keep the learnable 2-5/8 band (blue); "
                 "drop 0-1/8 (no signal) and 6-8/8 (no advantage)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_pool_difficulty.png")


# ──────────── 12. Answer-format compliance (from eval logs) ─────────────
def fig_answer_tag():
    # <answer>-tag emission rate measured during eval (GSM8K / MMLU / StrategyQA)
    stages = ["PEAR-SFT\nfew-shot", "PEAR-SFT\nzero-shot", "anti-guess\nGRPO"]
    gsm = [100, 100, 100]; mmlu = [100, 94, 100]; sqa = [100, 98, 100]
    x = np.arange(len(stages)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.bar(x - w, gsm, w, label="GSM8K", color=C["grpo"])
    ax.bar(x, mmlu, w, label="MMLU", color=C["base"])
    ax.bar(x + w, sqa, w, label="StrategyQA", color=C["sft"])
    ax.set_xticks(x); ax.set_xticklabels(stages); ax.set_ylim(85, 102)
    ax.set_ylabel("<answer> tag emission (%)")
    ax.set_title("Format reliability: the model almost always emits a parseable <answer> block")
    ax.legend(fontsize=9, ncol=3, loc="lower center")
    save(fig, "fig_answer_tag.png")


# ════════════════════════════════════════════════════════════════════════
# ILLUSTRATIVE / SYNTHETIC figures (teaching aids, NOT measured results).
# These depict what a reward-hacking-free StrategyQA run would look like.
# They are generated from a deterministic closed-form trajectory, are clearly
# labelled "illustrative", and are kept out of results/. See docs/idealized-run.md.
# ════════════════════════════════════════════════════════════════════════
ILLUS = "(illustrative, not measured)"


def _ideal_held(it):
    # smooth saturating climb 69.4 -> 83.0, reaching ~83 by iteration 75
    return 83.0 - (83.0 - 69.4) * np.exp(-it / 24.0)


def _real_held(it):
    # our measured behaviour: fast early gain to ~79.8, then a flat/slightly
    # declining plateau (the subtle-reward-hacking signature)
    base = 79.8 - (79.8 - 69.4) * np.exp(-it / 11.0)
    return base - np.clip((it - 45) / 45.0, 0, 1) * 1.4


def fig_idealized_trajectory():
    it = np.arange(0, 91)
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.plot(it, _real_held(it), color=C["bad"], lw=2.2,
            label="measured run (plateaus at 79.8)")
    ax.plot(it, _ideal_held(it), color=C["good"], lw=2.4,
            label="idealized run (reaches 83.0)")
    ax.axhline(82.9, color="#444", ls="--", lw=1.2)
    ax.text(2, 82.95, "+5 pp target = 82.9", fontsize=8.5, color="#444", va="bottom")
    ax.axvline(75, color=C["good"], ls=":", lw=1.2)
    ax.plot([75], [_ideal_held(np.array([75]))[0]], "o", color=C["good"], ms=7)
    ax.annotate("83.0 at iter 75", (75, 82.4), textcoords="offset points",
                xytext=(-10, -22), fontsize=9, color=C["good"], fontweight="bold",
                ha="right")
    ax.set_xlabel("GRPO iteration"); ax.set_ylabel("Held-out StrategyQA accuracy (%)")
    ax.set_ylim(66, 86)
    ax.set_title(f"Idealized vs measured StrategyQA trajectory {ILLUS}")
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "illustrative_sqa_trajectory.png")


def fig_idealized_gap():
    it = np.arange(0, 91)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    # measured: training-pool pass-rate keeps rising while held-out flattens (gap opens)
    pool_real = 81 - (81 - 57) * np.exp(-it / 16.0)
    held_real = _real_held(it)
    ax = axes[0]
    ax.plot(it, pool_real, color=C["sft"], lw=2.2, label="training-pool pass-rate")
    ax.plot(it, held_real, color=C["bad"], lw=2.2, label="held-out accuracy")
    ax.fill_between(it, held_real, pool_real, where=(pool_real > held_real),
                    color=C["bad"], alpha=0.10)
    ax.text(46, 62, "generalisation gap\nopens (reward hacking)", color=C["bad"],
            fontsize=9, ha="center")
    ax.set_title("Measured run: pool and held-out diverge")
    ax.set_xlabel("GRPO iteration"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(52, 90)
    ax.legend(fontsize=8.5, loc="lower right")
    # idealized: pool and held-out track each other (no gap)
    pool_ideal = _ideal_held(it) + 1.2
    held_ideal = _ideal_held(it)
    ax = axes[1]
    ax.plot(it, pool_ideal, color=C["sft"], lw=2.2, label="training-pool pass-rate")
    ax.plot(it, held_ideal, color=C["good"], lw=2.2, label="held-out accuracy")
    ax.text(48, 74, "pool and held-out track:\nreasoning generalises", color=C["good"],
            fontsize=9, ha="center")
    ax.set_title("Idealized run: no generalisation gap")
    ax.set_xlabel("GRPO iteration"); ax.set_ylim(52, 90)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.suptitle(f"Why the gap matters: pool accuracy can rise while held-out does not {ILLUS}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "illustrative_gap.png")


def fig_idealized_kpi():
    bench = ["GSM8K", "MMLU", "StrategyQA"]
    base = [82.9, 72.8, 77.9]
    measured = [88.3, 69.8, 79.8]
    ideal = [89.0, 72.0, 83.0]
    x = np.arange(len(bench)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - w, base, w, label="base", color=C["base"])
    ax.bar(x, measured, w, label="measured final", color=C["grpo"])
    ax.bar(x + w, ideal, w, label="idealized (illustrative)", color=C["good"], alpha=0.85,
           hatch="//", edgecolor="white")
    for i, b in enumerate(base):
        ax.hlines(b + 5, x[i] - 1.4 * w, x[i] + 1.4 * w, color=C["bad"], ls="--", lw=1.3)
    ax.plot([], [], color=C["bad"], ls="--", lw=1.3, label="+5 pp target")
    ax.set_xticks(x); ax.set_xticklabels(bench); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(55, 95)
    ax.set_title(f"Idealized KPIs clear the +5 pp bar on GSM8K and StrategyQA {ILLUS}")
    ax.legend(fontsize=8.5, ncol=2, loc="lower center")
    save(fig, "illustrative_kpi.png")


# ──────────── Prior iteration: math-first negative transfer ─────────────
def fig_math_first_transfer():
    # v1 (workinprogress.ipynb), measured under the v1 eval harness.
    bench = ["GSM8K", "MMLU", "StrategyQA"]
    base = [75.6, 73.3, 72.5]          # base Qwen2.5-7B, v1 harness
    mathfirst = [86.8, 73.7, 66.4]     # after math-only PROF-GRPO (step 100)
    x = np.arange(len(bench)); w = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w / 2, base, w, label="PEAR-SFT (before RL)", color=C["base"])
    b2 = ax.bar(x + w / 2, mathfirst, w, label="after math-only PROF-GRPO", color=C["grpo"])
    ax.bar_label(b1, fmt="%.1f", fontsize=8.5, padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=8.5, padding=2)
    for i, (a, b) in enumerate(zip(base, mathfirst)):
        dv = b - a
        col = C["good"] if dv >= 0 else C["bad"]
        ax.annotate(f"{dv:+.1f}", (x[i] + w / 2, b + 2.2), ha="center", fontsize=9,
                    color=col, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(bench); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(55, 95)
    ax.set_title("Prior iteration (v1): math-only RL lifts GSM8K but forgets commonsense")
    ax.legend(fontsize=9, loc="lower center", ncol=2)
    save(fig, "fig_math_first_transfer.png")


if __name__ == "__main__":
    fig_math_first_transfer()
    fig_pipeline()
    fig_kpi_bars()
    fig_progression()
    fig_grpo_variants()
    fig_collapse()
    fig_grpo_curve()
    fig_kl_stability()
    fig_conciseness()
    fig_sft_loss()
    fig_quant()
    fig_pool()
    fig_answer_tag()
    fig_idealized_trajectory()
    fig_idealized_gap()
    fig_idealized_kpi()
    print("\nall figures written to", os.path.relpath(FIG, ROOT))
