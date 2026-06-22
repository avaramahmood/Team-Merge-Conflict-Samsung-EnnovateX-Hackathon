"""
NOTEBOOK 2 - PEAR SFT TRAINING  (v13)
======================================
Reads:  pear_sft_final.jsonl  (32B AWQ teacher traces, Qwen2.5-7B base student)
Saves:  outputs/pear_sft_epoch1                    (model)
        outputs/logs/run_{ts}/train.jsonl          (per-step metrics)
        outputs/logs/run_{ts}/metadata.json        (frozen config)
        outputs/logs/run_{ts}/summary.json         (final rollup)

Why v13 differs from v11/v12
============================
pi_beta = R1-Distill-Qwen-32B (AWQ 4-bit), pi_theta = Qwen2.5-7B BASE.
These are far apart, so raw token delta = log pi_theta - log pi_beta is
net-negative on almost every reasoning token (a 7B base model is less
confident than a 32B specialized reasoner) and saturates the clip on
format tokens. Accumulated backwards, the raw PEAR weight collapses to
~exp(-10) on early reasoning tokens and ~1 on the final tokens, so SFT
effectively trains only the answer region. That reproduces the StrategyQA
format-drift / shortcut failure. Run pear_diagnostic.py to see it.

Changes vs v12
--------------
1. PEAR_MODE switch:
     "centered" (DEFAULT, robust): subtract the per-trace mean delta so the
       weight tracks WITHIN-trace relative agreement, not the global model-gap
       offset; symmetric clip; then normalize weights to mean 1 per trace so
       every trace (any length) contributes equally and nothing collapses.
     "raw"     : original v12 PEAR (kept for ablation only).
     "uniform" : plain weighted SFT control (all token weights = 1).
2. Per-trace weight normalization to mean 1 (kills the collapse + length bias).
3. fp32 cross_entropy per item (v12 ran log_softmax in bf16 over full vocab,
   numerically weak). Also lighter on memory: no full-batch log_softmax tensor.
4. INCLUDE_NEG flag (default False): repulsion on "long wrong" traces is risky
   in a from-scratch distillation SFT and its weight goes inert anyway; off for
   the first robust run, flip True to ablate.
5. use_cache=False with gradient checkpointing (removes the conflict warning).

Single epoch, lr 1e-5 (PEAR thesis: do not over-train SFT or it hurts RL).
"""

# -- 1. INSTALL ---------------------------------------------------------
import subprocess, sys

WHEEL_BNB = "inputs/data/bitsandbyteswheel/bnb_wheel"
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "--no-index", "--find-links", WHEEL_BNB, "--no-deps", "bitsandbytes"],
        check=True, capture_output=True, text=True,
    )
    print("bitsandbytes installed from wheel.")
except subprocess.CalledProcessError as e:
    print("BNB INSTALL FAILED, falling back to plain AdamW.")
    print(e.stderr[-1000:])

import os, json, random, math, time, shutil, torch
import torch.nn.functional as F
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import bitsandbytes as bnb
    _USE_BNB = True
except ImportError:
    _USE_BNB = False
    print("WARNING: bitsandbytes not available, using plain AdamW (slow).")

os.environ["WANDB_DISABLED"]       = "true"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_CACHE"]    = "outputs/cache"
os.makedirs("outputs/cache", exist_ok=True)


# -- 2. LOGGER ----------------------------------------------------------
class RunLogger:
    def __init__(self, base_dir="outputs/logs"):
        self.run_id   = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir  = os.path.join(base_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.train_path    = os.path.join(self.run_dir, "train.jsonl")
        self.metadata_path = os.path.join(self.run_dir, "metadata.json")
        self.summary_path  = os.path.join(self.run_dir, "summary.json")
        self.start_time    = time.time()
        open(self.train_path, "w").close()
        self.epoch_records = []
        print(f"Logging to: {self.run_dir}")

    def log_metadata(self, payload):
        payload["run_id"]    = self.run_id
        payload["timestamp"] = datetime.now().isoformat()
        with open(self.metadata_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def log_step(self, payload):
        payload["wall_time"] = round(time.time() - self.start_time, 2)
        with open(self.train_path, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n"); f.flush()

    def log_epoch(self, payload):
        payload["wall_time"] = round(time.time() - self.start_time, 2)
        self.epoch_records.append(payload)
        self.log_step({"event": "epoch_end", **payload})

    def finalize(self, extras=None):
        summary = {"run_id": self.run_id,
                   "total_time_sec": round(time.time() - self.start_time, 2),
                   "epochs": self.epoch_records}
        if extras:
            summary.update(extras)
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary saved: {self.summary_path}")


logger = RunLogger()


# -- 3. CONFIG ----------------------------------------------------------
BASE_MODEL  = "inputs/models/qwen2-5-7b/qwen2.5-7b"
TRACES_PATH = "inputs/data/32b-qwen-pear-traces"
OUT         = "outputs"

PEAR_MODE   = "uniform"     # "centered" (robust default) | "raw" (paper) | "uniform" (control)
INCLUDE_NEG = False          # repulsion off for first robust run; True to ablate

EPOCHS        = 1            # PEAR thesis: 1 epoch
BATCH_SIZE    = 16           # plenty of headroom now (no full-vocab log_softmax); 24 is safe too
GRAD_ACCUM    = 2            # effective batch = 32
LR            = 1e-5
NEG_LAMBDA    = 0.1          # only used if INCLUDE_NEG

GAMMA         = 0.999
LOG_G_MIN     = -10.0
LOG_G_MAX     =  5.0
DELTA_MIN_RAW = -0.08        # raw asymmetric clip (paper)
DELTA_MAX_RAW =  0.30
DELTA_ABS     =  0.15        # symmetric clip applied AFTER mean-centering
MAX_SEQ_LEN   = 1024

random.seed(42)
torch.manual_seed(42)


# -- 4. PEAR WEIGHTS ----------------------------------------------------
def compute_pear_weights(b_lps, t_lps, T, mode):
    """
    Returns (token_weights [T], neg_seq_weight float).
    raw delta_t = log pi_theta - log pi_beta.
    centered mode removes the per-trace mean (the global model-gap offset that
    causes collapse) and keeps within-trace relative agreement, then normalizes
    the final weights to mean 1 so no trace's signal vanishes.
    """
    raw = [t_lps[j] - b_lps[j] for j in range(T)]
    cl  = [max(DELTA_MIN_RAW, min(DELTA_MAX_RAW, d)) for d in raw]

    if mode == "uniform":
        return torch.ones(T), 1.0

    if mode == "centered":
        m      = sum(cl) / max(1, T)
        deltas = [max(-DELTA_ABS, min(DELTA_ABS, x - m)) for x in cl]
    else:  # raw
        deltas = cl

    log_gamma = math.log(GAMMA)
    w         = torch.zeros(T)
    suffix    = 0.0
    for t in reversed(range(T)):
        log_g = (T - 1 - t) * log_gamma + suffix
        w[t]  = math.exp(max(LOG_G_MIN, min(LOG_G_MAX, log_g)))
        suffix += deltas[t]
    neg_seq_w = math.exp(max(LOG_G_MIN, min(LOG_G_MAX, suffix)))

    s = float(w.sum())          # normalize to mean 1 -> (w*nll).mean() is a true weighted mean
    if s > 0:
        w = w * (T / s)
    return w, neg_seq_w


# -- 5. DATASET ---------------------------------------------------------
class PEARDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len, mode, include_neg):
        self.tok, self.max_len, self.mode = tokenizer, max_len, mode
        self.items = []
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                if len(r.get("behavior_logprobs", [])) == 0 or len(r.get("base_logprobs", [])) == 0:
                    continue
                if (not include_neg) and r.get("role") == "negative":
                    continue
                self.items.append(r)
        pos = sum(1 for x in self.items if x["role"] == "positive")
        neg = sum(1 for x in self.items if x["role"] == "negative")
        print(f"PEARDataset[{mode}]: {len(self.items)} records - pos: {pos}, neg: {neg}, "
              f"ratio: {neg/max(1,pos):.2f}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        role = item.get("role", "positive")

        full_enc   = self.tok(item["prompt"] + item["trace"],
                              max_length=self.max_len, truncation=True, return_tensors="pt")
        prompt_enc = self.tok(item["prompt"],
                              max_length=self.max_len, truncation=True, return_tensors="pt")
        input_ids  = full_enc["input_ids"][0]
        prompt_len = prompt_enc["input_ids"].shape[1]
        trace_len  = len(input_ids) - prompt_len

        if trace_len <= 0:
            return {"input_ids": input_ids, "prompt_len": prompt_len,
                    "pear_weights": torch.ones(1), "neg_seq_w": 1.0, "role": role}

        T     = min(trace_len, len(item["behavior_logprobs"]), len(item["base_logprobs"]))
        b_lps = item["behavior_logprobs"][:T]
        t_lps = item["base_logprobs"][:T]
        w, neg_seq_w = compute_pear_weights(b_lps, t_lps, T, self.mode)

        return {"input_ids": input_ids, "prompt_len": prompt_len,
                "pear_weights": w, "neg_seq_w": neg_seq_w, "role": role}


def pear_collate(batch):
    max_len       = max(x["input_ids"].shape[0] for x in batch)
    input_ids_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
    attn_mask     = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        L = x["input_ids"].shape[0]
        input_ids_pad[i, :L] = x["input_ids"]
        attn_mask[i, :L]     = 1
    return {"input_ids": input_ids_pad, "attention_mask": attn_mask, "items": batch}


# -- 6. MODEL -----------------------------------------------------------
print("Loading base model for SFT...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="cuda")
model.config.use_cache = False          # required with gradient checkpointing
model.gradient_checkpointing_enable()
model.train()


# -- 7. SPLIT + LOADERS -------------------------------------------------
full_dataset = PEARDataset(f"{TRACES_PATH}/pear_sft_final.jsonl",
                           tokenizer, MAX_SEQ_LEN, PEAR_MODE, INCLUDE_NEG)
val_size   = max(1, int(0.05 * len(full_dataset)))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | mode={PEAR_MODE} | neg={INCLUDE_NEG}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=pear_collate, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=pear_collate, num_workers=0)


# -- 8. OPTIMIZER + SCHEDULER -------------------------------------------
batches_per_epoch   = len(train_loader)
total_steps         = batches_per_epoch * EPOCHS
num_optimizer_steps = max(2, total_steps // GRAD_ACCUM)
num_warmup_steps    = max(1, int(0.05 * num_optimizer_steps))

if _USE_BNB:
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=0.01)
    print("Using AdamW8bit.")
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    print("Using plain AdamW (slow, fix bnb wheel).")

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_optimizer_steps)
print(f"Rows: {len(train_dataset)} | Batches/epoch: {batches_per_epoch} | "
      f"Optimizer steps: {num_optimizer_steps} | Warmup: {num_warmup_steps}")

logger.log_metadata({
    "model": BASE_MODEL, "traces_path": TRACES_PATH,
    "pear_mode": PEAR_MODE, "include_neg": INCLUDE_NEG,
    "hyperparams": {"EPOCHS": EPOCHS, "BATCH_SIZE": BATCH_SIZE, "GRAD_ACCUM": GRAD_ACCUM,
                    "LR": LR, "GAMMA": GAMMA, "DELTA_MIN_RAW": DELTA_MIN_RAW,
                    "DELTA_MAX_RAW": DELTA_MAX_RAW, "DELTA_ABS": DELTA_ABS,
                    "LOG_G_MIN": LOG_G_MIN, "LOG_G_MAX": LOG_G_MAX,
                    "NEG_LAMBDA": NEG_LAMBDA, "MAX_SEQ_LEN": MAX_SEQ_LEN},
    "dataset_stats": {"total_records": len(full_dataset),
                      "train_records": len(train_dataset), "val_records": len(val_dataset),
                      "positive_count": sum(1 for x in full_dataset.items if x["role"] == "positive"),
                      "negative_count": sum(1 for x in full_dataset.items if x["role"] == "negative")},
    "training_schedule": {"batches_per_epoch": batches_per_epoch,
                          "optimizer_steps_total": num_optimizer_steps,
                          "warmup_steps": num_warmup_steps},
    "hardware": {"gpu_name": torch.cuda.get_device_name(0),
                 "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                 "cuda_version": torch.version.cuda, "torch_version": torch.__version__,
                 "bnb_enabled": _USE_BNB},
})


# -- 9. LOSS ------------------------------------------------------------
def compute_item_losses(logits, input_ids, batch):
    """Per-item fp32 cross_entropy on the trace span, PEAR-weighted. Returns SUMS."""
    pos_loss_sum = torch.tensor(0.0, device=logits.device)
    neg_loss_sum = torch.tensor(0.0, device=logits.device)
    pos_count = neg_count = 0

    for i, item in enumerate(batch["items"]):
        t_start, role = item["prompt_len"], item["role"]
        if t_start < 1:
            continue
        T = min(item["input_ids"].shape[0] - t_start, len(item["pear_weights"]))
        if T <= 0:
            continue

        logits_i  = logits[i, t_start - 1 : t_start - 1 + T, :].float()    # [T, vocab] fp32
        targets   = input_ids[i, t_start : t_start + T]                    # [T]
        token_nll = F.cross_entropy(logits_i, targets, reduction="none")   # [T]

        if role == "positive":
            w = item["pear_weights"][:T].to(logits.device, dtype=token_nll.dtype).detach()
            pos_loss_sum += (w * token_nll).mean()
            pos_count    += 1
        else:
            seq_w = float(torch.clamp(torch.tensor(item["neg_seq_w"]), min=0.01, max=2.0))
            neg_loss_sum += seq_w * token_nll.mean()
            neg_count    += 1

    return pos_loss_sum, pos_count, neg_loss_sum, neg_count


def combine_losses(pos_loss_sum, pos_count, neg_loss_sum, neg_count):
    has_pos, has_neg = pos_count > 0, neg_count > 0
    if not has_pos and not has_neg:
        return None
    if has_pos and has_neg:
        return (pos_loss_sum / pos_count) - NEG_LAMBDA * (neg_loss_sum / neg_count)
    return pos_loss_sum / pos_count if has_pos else -NEG_LAMBDA * (neg_loss_sum / neg_count)


# -- 10. VALIDATION -----------------------------------------------------
@torch.no_grad()
def run_validation(loader):
    model.eval()
    sum_loss, sum_count = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to("cuda")
        attn_mask = batch["attention_mask"].to("cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attn_mask)
            pls, pc, _, _ = compute_item_losses(out.logits, input_ids, batch)
        sum_loss += pls.item(); sum_count += pc
    model.train()
    return sum_loss / max(1, sum_count)


# -- 11. TRAIN LOOP -----------------------------------------------------
def optimizer_step():
    g = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step(); scheduler.step(); optimizer.zero_grad()
    return float(g)

run_loss_sum = run_pos_loss_sum = run_neg_loss_sum = 0.0
run_pos_count = run_neg_count = run_tokens = run_micro = 0
update_step = total_pos = total_neg = 0

optimizer.zero_grad()
step_start = time.time()
print("Starting PEAR SFT (1 epoch)...\n")

for epoch in range(EPOCHS):
    print(f"\n{'='*60}\nEPOCH {epoch+1}/{EPOCHS}\n{'='*60}")
    epoch_start = time.time()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to("cuda")
        attn_mask = batch["attention_mask"].to("cuda")

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attn_mask)
            pls, pc, nls, nc = compute_item_losses(out.logits, input_ids, batch)
            loss = combine_losses(pls, pc, nls, nc)
        if loss is None:
            continue

        (loss / GRAD_ACCUM).backward()

        run_loss_sum     += loss.item()
        run_pos_loss_sum += pls.item() if pc > 0 else 0.0
        run_pos_count    += pc
        run_neg_loss_sum += nls.item() if nc > 0 else 0.0
        run_neg_count    += nc
        run_tokens       += int(attn_mask.sum().item())
        run_micro        += 1
        total_pos        += pc
        total_neg        += nc

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == batches_per_epoch:
            grad_norm = optimizer_step()
            update_step += 1
            step_time   = time.time() - step_start

            mean_loss     = run_loss_sum     / run_micro
            mean_pos_loss = run_pos_loss_sum / max(1, run_pos_count)
            mean_neg_loss = run_neg_loss_sum / max(1, run_neg_count)
            tok_s         = run_tokens / max(1e-6, step_time)

            logger.log_step({
                "step": update_step, "epoch": epoch + 1, "loss": mean_loss,
                "mean_pos_loss": mean_pos_loss, "mean_neg_loss": mean_neg_loss,
                "pos_items_in_step": run_pos_count, "neg_items_in_step": run_neg_count,
                "lr": scheduler.get_last_lr()[0], "grad_norm": grad_norm,
                "gpu_mem_alloc_gb": torch.cuda.memory_allocated() / 1e9,
                "gpu_mem_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "step_time_sec": step_time, "tokens_per_sec": tok_s,
            })

            run_loss_sum = run_pos_loss_sum = run_neg_loss_sum = 0.0
            run_pos_count = run_neg_count = run_tokens = run_micro = 0
            step_start = time.time()

            if update_step % 200 == 0:
                torch.cuda.empty_cache()
            if update_step % 50 == 0 or update_step == 1:
                print(f"  [ep {epoch+1}] step {update_step}/{num_optimizer_steps} | "
                      f"loss: {mean_loss:.4f} | pos: {mean_pos_loss:.4f} | "
                      f"neg: {mean_neg_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.2e} | "
                      f"grad: {grad_norm:.2f} | tok/s: {tok_s:.0f} | "
                      f"gpu: {torch.cuda.memory_allocated()/1e9:.1f}GB | "
                      f"cum_pos: {total_pos} | cum_neg: {total_neg}")

    val_loss   = run_validation(val_loader)
    epoch_time = time.time() - epoch_start
    print(f"\nEpoch {epoch+1} val loss: {val_loss:.4f} | time: {epoch_time/60:.1f} min")
    logger.log_epoch({"epoch": epoch + 1, "step": update_step, "val_loss": val_loss,
                      "epoch_time_sec": epoch_time,
                      "cum_pos_samples": total_pos, "cum_neg_samples": total_neg})


# -- 12. SAVE + FINALIZE ------------------------------------------------
ckpt = f"{OUT}/pear_sft_epoch1"
if os.path.exists(ckpt):
    shutil.rmtree(ckpt)
model.save_pretrained(ckpt)
tokenizer.save_pretrained(ckpt)
print(f"\nCheckpoint saved: {ckpt}")

logger.finalize({"checkpoint_path": ckpt, "final_val_loss": val_loss,
                 "pear_mode": PEAR_MODE, "include_neg": INCLUDE_NEG})
print("\nPEAR SFT complete.")
print(f"Checkpoint: {ckpt}")
print(f"Logs:       {logger.run_dir}")
