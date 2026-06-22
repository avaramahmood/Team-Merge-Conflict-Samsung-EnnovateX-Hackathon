"""
OFFICIAL lm-evaluation-harness RUNNER (2x 16GB GPU)
========================================================
Runs the EleutherAI lm-eval harness (the "official" numbers) for:

    gsm8k       5-shot, generative CoT, exact_match
    mmlu        5-shot, loglikelihood multiple-choice (57 subjects)
    strategyqa  6-shot, loglikelihood Yes/No  (registered as a custom task)

Backend: vLLM with tensor_parallel_size=2 so the model is sharded across BOTH
T4s and decoding is batched/paged - far faster than HF .generate(). T4 is
Turing (no hardware bf16) so we run float16.

USAGE (terminal):

    !python lm_eval_run.py --model_path inputs/models/.../pear_sft_epoch1

Only --model_path is required. Everything else has sane defaults.

Typical wall-clock on 2x T4 (vLLM): one model load (~2-3 min) then
gsm8k ~8-12 min, strategyqa ~3 min, mmlu(full) ~12-18 min  => ~30-40 min.
If you want a hard guarantee of <50 min, pass --limit_mmlu 10 (10 docs per
subject = 570 items, mirrors a stratified sample) -> total ~20 min.
"""

import argparse
import json
import os
import subprocess
import sys
import time


# ----------------------------------------------------------------------
# 0. dependencies (install fresh, quietly)
# ----------------------------------------------------------------------
def ensure_deps():
    try:
        import lm_eval  # noqa: F401
        import vllm      # noqa: F401
        return
    except ImportError:
        print(">> installing lm_eval + vllm (one-time, ~5-12 min)...", flush=True)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "lm_eval==0.4.9", "vllm==0.6.6",
            # transformers >=4.50 removed AutoModelForVision2Seq, which lm_eval
            # 0.4.9 imports at load time -> AttributeError. Pin below 4.50.
            "transformers==4.48.3",
            "numpy<2.0",   # keep pandas/datasets/numpy on the same ABI
        ])


# ----------------------------------------------------------------------
# 1. custom StrategyQA task  (not built into the harness)
#    Registered via a YAML on disk + TaskManager(include_path=...).
#    Few-shot exemplars are drawn from the train split automatically.
# ----------------------------------------------------------------------
STRATEGYQA_YAML = """\
task: strategyqa
dataset_path: ChilleD/StrategyQA
output_type: multiple_choice
training_split: train
fewshot_split: train
test_split: test
doc_to_text: "Q: {{question}}\\nA:"
doc_to_choice: [" No", " Yes"]
doc_to_target: "{{ 1 if answer else 0 }}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""


def write_custom_tasks(task_dir):
    os.makedirs(task_dir, exist_ok=True)
    path = os.path.join(task_dir, "strategyqa.yaml")
    with open(path, "w") as f:
        f.write(STRATEGYQA_YAML)
    return task_dir


# ----------------------------------------------------------------------
# 2. main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="local path or HF id of the model to evaluate")
    ap.add_argument("--out_dir", default="outputs/lm_eval_results")
    ap.add_argument("--task_dir", default="outputs/lm_eval_tasks")
    ap.add_argument("--tp", type=int, default=2, help="tensor_parallel_size (# GPUs)")
    ap.add_argument("--dtype", default="float16", help="T4/Turing has no bf16")
    ap.add_argument("--max_model_len", type=int, default=4096,
                    help="must cover the 5-shot MMLU prompt")
    ap.add_argument("--gpu_mem", type=float, default=0.90,
                    help="vLLM gpu_memory_utilization per card")
    ap.add_argument("--gsm8k_max_tokens", type=int, default=512,
                    help="max NEW tokens for gsm8k generation (harness default is "
                         "only 256; raise for long CoT/think traces). mmlu/strategyqa "
                         "are loglikelihood -> no generation, so this doesn't apply.")
    ap.add_argument("--gsm8k_shots", type=int, default=5)
    ap.add_argument("--mmlu_shots", type=int, default=5)
    ap.add_argument("--strategyqa_shots", type=int, default=6)
    ap.add_argument("--limit_mmlu", type=int, default=None,
                    help="docs PER mmlu subject (e.g. 10 -> 570 items). None = full.")
    ap.add_argument("--limit", type=int, default=None,
                    help="global debug cap on docs per task (smoke test)")
    args = ap.parse_args()

    ensure_deps()

    # imports must come AFTER ensure_deps so the fresh versions are used
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager
    from lm_eval.models.vllm_causallms import VLLM

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    os.makedirs(args.out_dir, exist_ok=True)
    write_custom_tasks(args.task_dir)
    task_manager = TaskManager(include_path=args.task_dir)

    model_name = os.path.basename(args.model_path.rstrip("/"))
    print("=" * 60)
    print(f"lm-eval (official) | model: {model_name}")
    print(f"backend: vllm  tp={args.tp}  dtype={args.dtype}  "
          f"max_len={args.max_model_len}")
    print("=" * 60, flush=True)

    # ---- load the model ONCE; reuse across all task groups ----
    t0 = time.time()
    lm = VLLM(
        pretrained=args.model_path,
        tensor_parallel_size=args.tp,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        batch_size="auto",
        trust_remote_code=True,
    )
    print(f">> model loaded in {time.time() - t0:.0f}s\n", flush=True)

    # Each task gets its own num_fewshot, so run as separate passes that
    # share the one loaded model. (gsm8k/strategyqa/mmlu want different shots.)
    passes = [
        dict(tasks=["gsm8k"],      num_fewshot=args.gsm8k_shots,      limit=args.limit,
             gen_kwargs=f"max_gen_toks={args.gsm8k_max_tokens}"),
        dict(tasks=["strategyqa"], num_fewshot=args.strategyqa_shots, limit=args.limit,
             gen_kwargs=None),
        dict(tasks=["mmlu"],       num_fewshot=args.mmlu_shots,
             limit=args.limit if args.limit is not None else args.limit_mmlu,
             gen_kwargs=None),
    ]

    merged = {}
    for p in passes:
        name = p["tasks"][0]
        print(f"\n----- running {name} ({p['num_fewshot']}-shot, "
              f"limit={p['limit']}) -----", flush=True)
        t = time.time()
        res = simple_evaluate(
            model=lm,
            tasks=p["tasks"],
            num_fewshot=p["num_fewshot"],
            limit=p["limit"],
            gen_kwargs=p["gen_kwargs"],   # only set for the generative task (gsm8k)
            task_manager=task_manager,
            batch_size="auto",
            bootstrap_iters=0,   # skip stderr bootstrapping -> faster
        )
        if res is not None:  # rank 0 only
            merged.update(res["results"])
            print(f"   {name} done in {time.time() - t:.0f}s", flush=True)

    if not merged:  # non-zero rank in a TP worker
        return

    # ---- pull the headline metric for each task ----
    def acc(task):
        r = merged.get(task, {})
        for k in ("exact_match,strict-match", "exact_match,flexible-extract",
                  "exact_match", "acc,none", "acc"):
            if k in r:
                return round(r[k] * 100, 1)
        # mmlu group score lives under acc,none of the group row
        return None

    gsm = acc("gsm8k")
    sqa = acc("strategyqa")
    mml = acc("mmlu")
    scores = {"gsm8k": gsm, "mmlu": mml, "strategyqa": sqa}
    valid = [v for v in scores.values() if v is not None]
    avg = round(sum(valid) / len(valid), 1) if valid else None

    print("\n" + "=" * 60)
    print(f"OFFICIAL lm-eval RESULTS: {model_name}")
    print("=" * 60)
    print(f"GSM8K      ({args.gsm8k_shots}-shot) : {gsm}%")
    print(f"MMLU       ({args.mmlu_shots}-shot) : {mml}%")
    print(f"StrategyQA ({args.strategyqa_shots}-shot) : {sqa}%")
    print(f"Average             : {avg}%")
    print("=" * 60)
    print(f"total wall-clock: {time.time() - t0:.0f}s", flush=True)

    out_path = os.path.join(args.out_dir, f"{model_name}__lm_eval_fewshot.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": model_name,
            "harness": "EleutherAI/lm-evaluation-harness 0.4.9 (vllm backend)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M"),
            "few_shot": {"gsm8k": args.gsm8k_shots, "mmlu": args.mmlu_shots,
                         "strategyqa": args.strategyqa_shots},
            "scores": {**scores, "average": avg},
            "raw_results": merged,
        }, f, indent=2, default=str)
    print(f"saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
