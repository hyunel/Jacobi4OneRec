#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <input_dir>" >&2
  exit 1
fi

INPUT_DIR="$1"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory does not exist: $INPUT_DIR" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python}"

# Constants for 02_window.py required arguments.
WINDOW_SIZE=16
MIN_NOISY_RATIO=0.2
MAX_NOISY_RATIO=1.0
N_TOKEN_SEQ_LENGTH=32
STRATEGY="progressive"

OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_DIR%/}_n${N_TOKEN_SEQ_LENGTH}w${WINDOW_SIZE}}"
mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
PARQUET_FILES=("$INPUT_DIR"/*.parquet)
shopt -u nullglob

if [[ ${#PARQUET_FILES[@]} -eq 0 ]]; then
  echo "No parquet files found in: $INPUT_DIR" >&2
  exit 1
fi

for input_path in "${PARQUET_FILES[@]}"; do
  base_name="$(basename "$input_path" .parquet)"
  output_path="$OUTPUT_DIR/${base_name}.parquet"

  echo "Processing: $input_path -> $output_path"

  "$PYTHON_BIN" "$SCRIPT_DIR/02_window.py" \
    --input_path "$input_path" \
    --output_path "$output_path" \
    --window_size "$WINDOW_SIZE" \
    --min_noisy_ratio "$MIN_NOISY_RATIO" \
    --max_noisy_ratio "$MAX_NOISY_RATIO" \
    --n_token_seq_len "$N_TOKEN_SEQ_LENGTH" \
    --strategy "$STRATEGY"
done
