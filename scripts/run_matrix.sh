#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

shopt -s nullglob
configs=(configs/*.yaml)

if ((${#configs[@]} == 0)); then
    echo "No config files found under $(pwd)/configs" >&2
    exit 1
fi

failed=()

for cfg in "${configs[@]}"; do
    echo "=== Running: $cfg ==="
    if ! CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} python -m nlp_quantization --config "$cfg"; then
        echo "FAILED: $cfg" >&2
        failed+=("$cfg")
    fi
done

if ((${#failed[@]} > 0)); then
    echo "Matrix completed with ${#failed[@]} failed config(s):" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
