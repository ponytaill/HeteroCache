#!/bin/bash
TASKS="code_debug kv_retrieval longbook_choice_eng longbook_qa_chn longbook_qa_eng longbook_sum_eng longdialogue_qa_eng math_find number_string passkey"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --compression_ratio) COMPRESSION_RATIO="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL_PATH" || -z "$METHOD" ]]; then
    echo "Usage: $0 --model_path <path> --method <method> [--compression_ratio <ratio>]"
    exit 1
fi

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for task in ${TASKS}; do
    python scripts/infinitebench/run_infinitebench.py \
        --task ${task} --model_path ${MODEL_PATH} --method ${METHOD} \
        ${COMPRESSION_RATIO:+--compression_ratio $COMPRESSION_RATIO}
done
