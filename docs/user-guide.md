# User Guide

Two ways to use the deliverable: (A) run inference with the trained model, (B) reproduce/extend
the training.

## A. Run the model (inference)

### A1. Quick CLI via llama.cpp (the on-device path)

```bash
# after building llama.cpp (see installation.md) and obtaining concise-Q4_K_XL.gguf
./llama.cpp/build/bin/llama-cli -m concise-Q4_K_XL.gguf -ngl 99 -c 4096 \
  -p "<|im_start|>system
Solve the problem. Reason inside <think></think>, then give only the final answer inside <answer></answer>.<|im_end|>
<|im_start|>user
Problem: Natalia sold clips to 48 friends in April, then half as many in May. How many clips total?<|im_end|>
<|im_start|>assistant
"
```

The model emits a `<think>…</think>` chain then `<answer>72</answer>`. Use the **system
prompt that matches the domain** (math / commonsense / multiple-choice), they are listed at
the top of `src/eval/eval_unified.py` (`SYS`).

### A2. Server (OpenAI-compatible)

```bash
./llama.cpp/build/bin/llama-server -m concise-Q4_K_XL.gguf -ngl 99 -c 4096 --port 8080
# then POST to http://localhost:8080/v1/chat/completions
```

### A3. HF Transformers (the bf16 checkpoints)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tok = AutoTokenizer.from_pretrained("outputs/grpo_concise_best")
m = AutoModelForCausalLM.from_pretrained("outputs/grpo_concise_best",
        torch_dtype=torch.bfloat16, device_map="cuda").eval()
SYS = ("Decide whether the answer is Yes or No. Inside <think> and </think>, break the "
       "question into sub-questions, answer each, then conclude. Put only Yes or No inside "
       "<answer> and </answer>.")
q = "Would a pear sink in water?"
prompt = (f"<|im_start|>system\n{SYS}<|im_end|>\n"
          f"<|im_start|>user\nQuestion: {q}<|im_end|>\n<|im_start|>assistant\n")
ids = tok(prompt, return_tensors="pt").to("cuda")
out = m.generate(**ids, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
```

## B. Evaluate

Edit the top of `src/eval/eval_unified.py`:

```python
EVAL_MODE  = "think_zeroshot"   # base_fewshot | think_fewshot | think_zeroshot
MODEL_PATH = "outputs/grpo_concise_best"
```

then `python src/eval/eval_unified.py`. Results print per-benchmark and save to
`outputs/eval_results/<name>__<mode>.json`. For the official EleutherAI numbers:
`python src/eval/lm_eval_run.py --model_path outputs/grpo_concise_best`.

## C. On-device app (Knowledge / Thinking modes)

The companion desktop app wraps the GGUF with a **deterministic router + Python tool** so
mechanical sub-steps (counting, arithmetic) are *guaranteed* correct. See
[`reasoning-agent-app.md`](reasoning-agent-app.md). The app is shipped separately (it embeds
large native llama.cpp binaries) and is not part of this training repo.

## D. Choosing a checkpoint

| You want… | Use |
|---|---|
| Best held-out reasoning (bf16) | `grpo_best` (anti-guess GRPO) |
| Same accuracy, shorter answers | `grpo_concise_best` |
| On-device / phone / laptop | `concise-Q4_K_XL.gguf` |
| Just the cold start | `pear_sft_epoch1` |

## E. Tips

- Always use the **domain-matched system prompt**; the model was trained on those exact
  strings and drifts without them.
- Greedy decoding for single-shot accuracy; `eval_self_consistency.py` (maj@8) for a margin
  boost (compare maj@K ↔ base maj@K only).
- For arithmetic-heavy use, prefer the app's Thinking mode (tool-grounded), 4-bit math is
  quant's known weak spot.
