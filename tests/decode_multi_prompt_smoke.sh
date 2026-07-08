#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build"
EXEC="${BUILD}/cuda_cublas_json"
MODEL="${LLAMA_MODEL_PATH:-/mnt/nvme/models/Llama-3.2-1B-Instruct/model.safetensors}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
TOKENIZER="${ROOT}/tools/tokenizer.py"

if [[ ! -x "${EXEC}" ]]; then
  echo "Missing executable: ${EXEC}" >&2
  exit 1
fi

if [[ ! -f "${MODEL}" ]]; then
  echo "Skipping decode multi-prompt smoke: model not found at ${MODEL}" >&2
  exit 77
fi

declare -a PROMPTS=(
  "The capital of France is"
  "Hello"
  "1 + 1 ="
  "Explain quantum computing in simple terms."
  "Hi there!"
)

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

printf "%-45s | %8s | %12s | %s\n" "prompt" "tokens" "prefill_out" "decode"
printf "%-45s | %8s | %12s | %s\n" "---------------------------------------------" "--------" "------------" "--------"

failures=0
for prompt in "${PROMPTS[@]}"; do
  token_file="${tmpdir}/tokens.txt"
  log_file="${tmpdir}/run.log"

  if ! python3 "${TOKENIZER}" "${prompt}" --model "${TOKENIZER_MODEL}" -o "${token_file}" >/dev/null 2>&1; then
    echo "FAILED tokenizer for: ${prompt}" >&2
    failures=$((failures + 1))
    continue
  fi

  token_count="$(wc -w < "${token_file}" | tr -d ' ')"
  if ! "${EXEC}" "${MODEL}" "${token_file}" >"${log_file}" 2>&1; then
    printf "%-45s | %8s | %12s | %s\n" "${prompt}" "${token_count}" "CRASH" "FAILED"
    echo "--- stdout/stderr for '${prompt}' ---" >&2
    cat "${log_file}" >&2
    failures=$((failures + 1))
    continue
  fi

  prefill_out="$(grep -o 'token index: [0-9]*' "${log_file}" | tail -1 | awk '{print $3}')"
  if grep -q "Decode forward pass completed" "${log_file}"; then
    decode_status="OK"
  else
    decode_status="MISSING"
    failures=$((failures + 1))
  fi

  printf "%-45s | %8s | %12s | %s\n" "${prompt}" "${token_count}" "${prefill_out}" "${decode_status}"
done

if [[ "${failures}" -ne 0 ]]; then
  echo "${failures} prompt(s) failed" >&2
  exit 1
fi

echo "All prompts completed prefill and decode without crashing."
