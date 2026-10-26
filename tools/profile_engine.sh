#!/usr/bin/env bash
# Profile one prefill and a short greedy decode, then write nsys summaries.
# Nsight Compute is installed but cannot collect hardware counters on this Jetson
# without root. Nsight Systems traces every kernel and CUDA API call without that.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NSYS="${NSYS:-/tmp/nsys-extract/opt/nvidia/nsight-systems/2024.5.4/target-linux-tegra-armv8/nsys}"
BIN="${BIN:-$ROOT/build/cuda_cublas_json}"
MODEL="${MODEL:-/mnt/nvme/models/Llama-3.2-1B-Instruct/model.safetensors}"
PROMPT="${PROMPT:-$ROOT/bench/prompt_128.txt}"
OUT="${OUT:-$ROOT/docs/benchmark}"
STEPS="${STEPS:-8}"

if [[ ! -x "$NSYS" ]]; then
  if command -v nsys >/dev/null 2>&1; then
    NSYS="$(command -v nsys)"
  else
    echo "nsys not found. Set NSYS to the Nsight Systems binary." >&2
    exit 1
  fi
fi

mkdir -p "$OUT"

profile() {
  local name="$1"
  shift
  "$NSYS" profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    -o "$OUT/$name" \
    "$BIN" "$MODEL" "$PROMPT" "$@"
  "$NSYS" stats \
    --report cuda_gpu_kern_sum \
    --report cuda_gpu_mem_time_sum \
    --report cuda_api_sum \
    --report cuda_kern_exec_sum \
    --report nvtx_pushpop_sum \
    "$OUT/$name.nsys-rep" > "$OUT/${name}_stats.txt"
}

profile eager --bench-eager --no-warmup --steps "$STEPS"
profile graph --bench-graph --no-warmup --steps "$STEPS"
echo "Wrote $OUT/eager_stats.txt and $OUT/graph_stats.txt"
