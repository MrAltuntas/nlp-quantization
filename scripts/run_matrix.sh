#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for cfg in configs/*.yaml; do
    echo "=== Running: $cfg ==="
    python -m nlp_quantization --config "$cfg" || echo "FAILED: $cfg"
done
