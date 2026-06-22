#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# quantize.sh - research-backed 4-bit GGUF quantization that preserves the
# accuracy you trained. One run does BOTH models. Designed for the 96GB Blackwell
# (full GPU offload everywhere). Expected wall-clock: ~15-30 min per model.
#
# WHY THIS RECIPE (evidence):
#   * W4-weight-only is near-lossless for reasoning models; lower-bit activations
#     / KV are where accuracy dies  -> COLM 2025, arXiv:2504.04823.
#   * A DOMAIN-MATCHED importance matrix (imatrix) is the #1 lever for reasoning
#     models - calibration domain matters far more than for base/instruct models.
#   * Keeping embeddings + lm_head at Q8_0 is what makes "XL" beat plain Q4_K_M:
#     those tensors dominate quant error on small models, cost ~nothing in size.
#     This is the hand-built equivalent of Unsloth Dynamic 2.0 UD-Q4_K_XL.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── CONFIG ─────────────────────────────────────────────────────────────────────
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"     # llama.cpp checkout
WORK="${WORK:-$PWD/quant_out}"                 # all artifacts land here
CHUNKS=300                                      # imatrix calibration chunks
NGL=99                                          # full GPU offload (96GB Blackwell)

# "HF_checkpoint_dir|output_basename" - add/edit your two models here
MODELS=(
  "outputs/grpo_best|grpo"
  "outputs/grpo_concise_best|concise"
)

mkdir -p "$WORK"
BIN="$LLAMA_DIR/build/bin"
CONVERT="$LLAMA_DIR/convert_hf_to_gguf.py"

# ── 0. BUILD llama.cpp (one-time; skipped if already built) ────────────────────
if [[ ! -x "$BIN/llama-quantize" ]]; then
  echo "### Building llama.cpp with CUDA ..."
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA_DIR/build" -j --config Release
  pip install -q gguf sentencepiece protobuf
fi

quantize_one () {
  local SRC="$1" NAME="$2"
  local F16="$WORK/${NAME}-f16.gguf"
  local Q8="$WORK/${NAME}-q8.gguf"           # near-lossless reference for KL
  local IMAT="$WORK/${NAME}-imatrix.dat"
  local CALIB="$WORK/${NAME}-calib.txt"
  local OUT="$WORK/${NAME}-Q4_K_XL.gguf"
  local KLBASE="$WORK/${NAME}-klbase.dat"

  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "### MODEL: $NAME   ($SRC)"
  echo "═══════════════════════════════════════════════════════════════"

  # 1. HF -> high-precision GGUF (no precision lost before imatrix)
  echo "### [1/6] convert -> f16 ..."
  python "$CONVERT" "$SRC" --outfile "$F16" --outtype f16

  # 2. Q8_0 reference (near-lossless; used as the KL-divergence ground truth)
  echo "### [2/6] Q8_0 reference ..."
  "$BIN/llama-quantize" "$F16" "$Q8" Q8_0

  # 3. DOMAIN calibration corpus - model's own concise CoT in its chat template
  echo "### [3/6] generate calibration corpus ..."
  python "$PWD/make_calib.py" "$SRC" "$CALIB"

  # 4. importance matrix on YOUR data (the accuracy-preserving step)
  echo "### [4/6] build imatrix ..."
  "$BIN/llama-imatrix" -m "$F16" -f "$CALIB" -o "$IMAT" \
      --chunks "$CHUNKS" -c 2048 -ngl "$NGL"

  # 5. 4-bit quant: imatrix + embeddings/output kept at Q8_0  (= UD-Q4_K_XL idea)
  echo "### [5/6] quantize -> Q4_K_XL ..."
  "$BIN/llama-quantize" --imatrix "$IMAT" \
      --token-embedding-type q8_0 \
      --output-tensor-type q8_0 \
      "$F16" "$OUT" Q4_K_M

  # 6. VALIDATION: KL-divergence vs the Q8 reference (catches silent drift)
  echo "### [6/6] KL-divergence check ..."
  "$BIN/llama-perplexity" -m "$Q8"  -f "$CALIB" --kl-divergence-base "$KLBASE" -ngl "$NGL"
  "$BIN/llama-perplexity" -m "$OUT" -f "$CALIB" --kl-divergence --kl-divergence-base "$KLBASE" -ngl "$NGL"

  echo "### DONE: $OUT"
  ls -lh "$OUT"
}

for entry in "${MODELS[@]}"; do
  IFS='|' read -r src name <<< "$entry"
  quantize_one "$src" "$name"
done

echo ""
echo "ALL DONE. Quantized models in: $WORK"
echo "Accept a model only if mean KL-divergence is low (~<0.01) AND your"
echo "eval_unified_v5 GSM8K/StrategyQA deltas vs bf16 are within ~1pp."
echo "Watch arithmetic errors specifically - quant's known weak spot for math."
