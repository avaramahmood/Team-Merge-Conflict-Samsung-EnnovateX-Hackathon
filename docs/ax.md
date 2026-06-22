# AX: Agentic AI and Open-Weight Tooling Used to Build This Solution

*(Required artefact.)* How we used **open-weight models** and **agentic development tools** to
implement the solution, the setup, the workflows, what worked, and what did not.

## 1. Open-weight models as components (not just outputs)

Every model in the pipeline is open-weight, and several are used *agentically*, i.e. one model
supervises or grades another:

| Open model | Role | Used as |
|---|---|---|
| DeepSeek-R1-Distill-Qwen-32B (AWQ) | distillation teacher `π_β` | generates + scores reasoning traces that train the 7B |
| Qwen2.5-Math-PRM-7B | math process critic | filters/grades reasoning **steps** inside GRPO (PROF) |
| VersaPRM (LoRA/Llama-3.1-8B) | commonsense process critic | step grading for the SQA shaping/selection ablations |
| Qwen2.5-7B | policy `π_θ` | the model we actually train and ship |

This is a **multi-model orchestration** at training time: teacher → student distillation, then
two PRM "judges" supervising the policy's reasoning during RL. The verifier (exact-match) is a
deterministic non-model agent that gates every reward.

## 2. Agentic coding harness

The training/eval/quant code in this repo was authored with an **agentic coding assistant**
(Claude Code / CLI-style harness) driving the loop:

- **Reasoning & planning pipeline.** Each layer was specified as a goal ("fix the SFT→RL
  mismatch", "stop SQA lucky-guessing", "shrink to 4-bit without losing math"), the agent
  proposed a design, we reviewed, it implemented, ran, and iterated on the logs. The dense
  docstrings at the top of every `src/train/*.py` are the agent's own design rationale captured
  in-code (e.g. the "WHY v6 DID NOT LEARN" block in `grpo_*`, the advantage-standardisation
  bug was found and fixed inside this loop).
- **Tool use / tool chaining.** The agent chained: file edits → GPU run → parse `iter …` logs
  → diagnose (KL blow-up, +2 logprob misalignment) → patch → re-run. The KL/PPL quantization
  validation was added by the agent specifically because accuracy spot-checks missed silent
  drift.
- **Memory / context handling.** Long runs (90-iter GRPO across 12 h sessions) used
  checkpoint + `*_progress.json` resume so context survived session boundaries; the agent
  tracked which variant produced which checkpoint (`grpo_best`, `failed-grpo`).

## 3. MCP servers & agents.md (operationalising the workflow)

Per our Phase-1 plan we authored a custom agentic workflow over an MCP-style toolset:

- **Filesystem MCP**: direct read/write of training scripts, logs, and checkpoints.
- **Custom Hugging Face MCP server (authored from scratch)**: lets the agent pull datasets
  (GSM8K, AQuA-RAT, …) and push fine-tuned adapters/checkpoints to the Hub directly from the
  training scripts, eliminating manual hub interaction across 2-3 base models and 10+ dataset
  configs.
- **GitHub MCP**: version control, issues, and progress tracking.
- **`agents.md` / `skills.md`**: declared the agent's capabilities and the tool-use workflow
  (model pulls, adapter pushes, dataset streaming) so a single controller could execute the
  pipeline definition end to end.

The **on-device app** (see `reasoning-agent-app.md`) is itself an agentic system at *inference*
time: a ReAct planner that chains a Python-execution tool, grounded in PAL/PoT/ReAct.

## 4. What worked

- **Model-as-judge supervision.** A 32B teacher + 7B PRMs let a 7B student exceed its own
  baseline (GSM8K +5.4) without any human labels.
- **Agent-in-the-loop debugging.** The two highest-impact fixes, the **+2-token logprob
  misalignment** and the **advantage-standardisation bug**, were diagnosed from logs by the
  coding agent, not by hand.
- **Capturing rationale in-code.** Forcing every script to carry a "why this differs from the
  last version" docstring made regressions attributable and made this documentation almost
  write itself.
- **Resumable, offline-safe runs.** Checkpoint + progress JSON made multi-session GRPO robust.

## 5. What did **not** work

- **PRM as a hard selector (agentic over-trust).** Letting VersaPRM make keep/drop decisions
  handed the policy a signal to *game* → full capability collapse (v8-C). Process models are
  safe as **soft, zero-mean** advice, dangerous as **hard gates**.
- **Naïve "let it train longer".** The agent's instinct (and ours, chasing the +5) to keep
  iterating traded held-out generalisation for reward-surface optimisation, the subtle
  reward-hacking gap on StrategyQA. The fix is a **held-out-gated early stop**, which we now
  recommend over pool-metric-driven continuation.
- **TRL/black-box RL.** Off-the-shelf GRPO hid exactly the knobs (advantage normalisation, KL
  target, clip asymmetry) that determined success/failure; a transparent custom loop was
  necessary for an SLM.
- **Generic quantization calibration.** wikitext imatrix quietly cost reasoning accuracy;
  domain-matched calibration (model's own CoT) was required to hit KL 0.008.

## 6. Reproducibility of the agentic process

The agent's decisions are legible: read the top-of-file docstrings in `src/train/*` for the
design log, `docs/methods-and-observations.md` for the experiment narrative, and
`src/figures/make_figures.py` for the exact per-iteration data behind every claim.
