#!/bin/bash
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$OUTPUT_DIR" || -z "$MODEL_PATH" || -z "$METHOD" ]]; then
    echo "Usage: $0 --output_dir <dir> --model_path <path> --method <method>"
    exit 1
fi

cd "$(dirname "$0")/../.."
python scripts/infinitebench/compute_scores.py --output_dir ${OUTPUT_DIR} --model_path ${MODEL_PATH} --method ${METHOD}
