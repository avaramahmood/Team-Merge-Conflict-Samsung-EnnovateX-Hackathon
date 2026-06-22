"""
Build a DOMAIN-MATCHED calibration corpus for imatrix quantization of the
post-conciseness checkpoint. This is the single biggest lever for keeping the
trained accuracy after 4-bit quant (reasoning models are highly sensitive to
calibration-domain mismatch; generic wikitext quietly costs you GSM8K/SQA).

Output: calib_concise.txt - prompt+completion blocks in the EXACT chat template
the model runs at inference, with the model's OWN concise <think>/<answer> traces.
Feed this to llama-imatrix.
"""
import os, random, sys
os.environ["TRANSFORMERS_OFFLINE"] = "1"; os.environ["HF_DATASETS_OFFLINE"] = "1"
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# usage: python make_calib.py <model_dir> <out.txt>   (or set CKPT / OUT env vars)
CKPT      = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CKPT", "outputs/grpo_concise_best")
OUT       = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("OUT", "calib_concise.txt")
EVAL_ROOT = os.environ.get("EVAL_ROOT", "inputs/data/qwen-riva-datasets/test")
N         = {"sqa": 180, "math": 120}                    # ~300 total, 60/40 like training
MAXTOK    = {"sqa": 512, "math": 1024}
SEED      = 24

SYSTEM_BY_DOMAIN = {
    "math": ("Solve the problem. Reason inside <think></think>, then give only the "
             "final answer inside <answer></answer>."),
    "sqa":  ("Answer the yes/no question. Reason inside <think></think>, then put "
             "exactly Yes or No inside <answer></answer>."),
}
NAME = {"math": "gsm8k", "sqa": "strategyqa"}

tok = AutoTokenizer.from_pretrained(CKPT)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    CKPT, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
model.eval()

def question_of(dom, ex):
    if dom == "math":
        return str(ex.get("question") or ex.get("problem", "")).strip()
    return str(ex["question"]).strip()

def build_prompt(dom, q):
    msgs = [{"role": "system", "content": SYSTEM_BY_DOMAIN[dom]},
            {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

random.seed(SEED)
blocks = []
for dom, n in N.items():
    ds = load_from_disk(os.path.join(EVAL_ROOT, NAME[dom]))
    idxs = list(range(len(ds))); random.shuffle(idxs)
    rows = [ds[i] for i in idxs[:n]]
    for bs in range(0, len(rows), 16):
        chunk = rows[bs:bs + 16]
        prompts = [build_prompt(dom, question_of(dom, r)) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left", truncation=True, max_length=1024).to("cuda")
        plen = enc.input_ids.shape[1]
        with torch.inference_mode():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=MAXTOK[dom],
                                 pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        for p, row in zip(prompts, range(len(chunk))):
            comp = tok.decode(gen[row, plen:], skip_special_tokens=False)
            comp = comp.replace(tok.pad_token, "").replace(tok.eos_token, "")
            blocks.append(p + comp)          # full template kept (chat tokens literal)
        print(f"{dom}: {bs + len(chunk)}/{n}")

random.shuffle(blocks)
with open(OUT, "w") as f:
    f.write("\n\n".join(blocks))
print(f"\nWrote {len(blocks)} calibration blocks -> {OUT}")
