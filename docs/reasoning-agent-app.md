# Companion On-Device App: Reasoning Agent

**Repository:** https://github.com/avaramahmood/reasoning-agentic-harness

> **Scope.** This document describes the deliverable's *deployment vehicle*: a desktop app that
> runs our trained 4-bit GGUF **fully on-device** via llama.cpp, with a research-grade
> tool-integrated reasoning (TIR) agent on top. The app is **shipped separately** (it embeds
> ~hundreds of MB of native llama.cpp/Electron binaries and a 5 GB model) and is intentionally
> **not** included in this training repo, keeping this repo a clean, reviewable training/eval
> codebase. This page documents what it is and how it uses paper-grade reasoning agents.

## Why it exists

PS06 is about **on-device** reasoning. A model checkpoint alone does not demonstrate that. The
app closes the loop: pick the GGUF, and a private, offline, low-latency reasoner answers, with
a **hard correctness guarantee** on the things small models reliably get wrong (counting,
arithmetic, enumeration), because those are routed to a real Python interpreter rather than the
model's head.

## Architecture

```
React + Vite UI (One-UI-style theme; model picker, live role cards, token streaming)
   │  /api (discover + load models)            │  /v1 (streaming inference)
   ▼                                           ▼
control server (Node, :8081) ── spawns ──► llama-server (llama.cpp, :8080)
   • lists .gguf / .tar.gz, extracts archives          ▼
   • (re)launches llama-server on model select   concise-Q4_K_XL.gguf  (our model)
```

Pick any `.gguf` in the UI; the control server (re)launches `llama-server` pointed at it, no
rebuild to switch models. Packaged as a Linux `.deb` / AppImage (Electron).

## Two modes

| Mode | Pipeline | For |
|---|---|---|
| **Knowledge** | 1 model pass | facts / recall, fast, no tools |
| **Thinking** | ReAct loop: reason ⇄ write Python ⇄ execute ⇄ continue ⇄ answer | math, counting, puzzles, tool-grounded |

A **deterministic router** forces mechanical sub-steps down the code path (with a regex
fallback that writes the code itself), so those answers are correct *for sure*; when the tool
yields ground truth, redundant later model passes are skipped (faster).

## Research-grade reasoning agents (what it implements)

| Idea | Paper |
|---|---|
| LLM writes a program, interpreter runs it | PAL, Gao et al. 2022, arXiv:2211.10435 |
| Disentangle reasoning from computation (+12%) | PoT, Chen et al. 2022, arXiv:2211.12588 |
| Thought → Act(tool) → Observe loop | ReAct, Yao et al. 2022, arXiv:2210.03629 |
| Sample + majority vote (+17.9% GSM8K) | Self-Consistency, Wang et al. 2022, arXiv:2203.11171 |
| Small models are weak self-critics ⇒ ground the verifier on executed output | Han et al. 2024, arXiv:2404.17140 |
| "Strawberry"/counting is a **tokenization** failure, not reasoning | arXiv:2412.18626, arXiv:2410.19730 |

The agent prompts the model in its **trained `<think>`/`<answer>` format**, so the on-device
model stays exactly in-distribution with how it was SFT/GRPO-trained in this repo.

> **Key insight.** A small model can neither count reliably **nor** reliably *decide* to use a
> tool. So the guarantee comes from the **router + code execution**, not the model. Verified on
> our exact model: asked "how many r in strawberry" it answers **2** in its head (wrong); the
> tool returns **3**.

## What the agentic loop actually fixes

The trained model is a strong reasoner, but like every small model it fails a specific class of
questions that look trivial to a human and are really limitations of tokenization and
single-pass decoding. The agentic loop turns each of these from "usually wrong" into "always
right", because the answer comes from executed code, not from the model's head. These are our
concrete, demonstrable contributions on top of the checkpoint.

| Failure class | Example prompt | Model alone (single pass) | With the agentic loop |
|---|---|---|---|
| Letter counting (tokenization) | "How many r in strawberry?" | 2 (wrong) | writes `"strawberry".count("r")` -> **3** |
| Filtered enumeration | "How many months start with J?" | often 2 or 4 | enumerates the 12 months, filters by first letter -> **3** (January, June, July) |
| Letter-in-set counting | "How many J's appear across all month names?" | guesses | counts J across the joined month names in code -> exact |
| Multi-step arithmetic | "What is 17 percent of 2,384, rounded?" | drifts on long arithmetic | evaluates in Python -> exact |
| Date and calendar logic | "What day of the week was 2014-03-09?" | unreliable | `datetime` in code -> exact |
| Concurrency / rate word problems | "Two pipes fill a tank in 6 and 9 minutes; together?" | mishandles the combined-rate algebra | sets up `1/6 + 1/9` and solves -> **3.6 minutes** |

The pattern is the same in every row: the model is reliable at *deciding the method* and
*setting up* the computation, and unreliable at *executing* it in one forward pass. The loop
splits those two jobs, the model plans and writes code, the interpreter executes, and the model
reads the real result back before answering. A deterministic router additionally forces
counting, enumeration, and arithmetic onto the code path even when the model would have tried to
answer from memory, and a regex fallback writes the code itself if the model forgets to, so the
mechanical cases are guaranteed rather than merely encouraged.

### Concurrency at the serving layer

"Concurrency" also matters in the engineering of the app itself. The control server (Node, port
8081) and `llama-server` (llama.cpp, port 8080) run as separate processes, so model loading,
token streaming, and tool execution do not block one another. The Python tool runs in an
isolated subprocess per call, so a long or runaway computation cannot stall the UI or the model
stream, and switching models at runtime simply relaunches the inference server without a
rebuild. The result is that a single on-device session can stream a reasoning chain, fire a tool
call, and keep the interface responsive at the same time.

## Relationship to the training repo

- Consumes `concise-Q4_K_XL.gguf` produced by `src/quantize/`.
- Uses the same domain system prompts and `<think>/<answer>` contract as `src/eval/`.
- Attribution: built on **llama.cpp** (MIT) and the PAL/PoT/ReAct literature above.
