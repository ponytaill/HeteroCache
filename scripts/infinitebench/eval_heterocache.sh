#!/bin/bash
TASKS="code_debug kv_retrieval longbook_choice_eng longbook_qa_chn longbook_qa_eng longbook_sum_eng longdialogue_qa_eng math_find number_string passkey"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --compression_ratio) COMPRESSION_RATIO="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --decode_step) DECODE_STEP="$2"; shift 2 ;;
        --topk) TOPK="$2"; shift 2 ;;
        --stable_threshold) STABLE_THRESHOLD="$2"; shift 2 ;;
        --sim_threshold) SIM_THRESHOLD="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL_PATH" ]]; then
    echo "Usage: $0 --model_path <path> [--compression_ratio <ratio>] [--steps <n>] [--decode_step <n>] [--topk <n>] [--stable_threshold <f>] [--sim_threshold <f>]"
    exit 1
fi

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH}"
for task in ${TASKS}; do
    python scripts/infinitebench/run_infinitebench.py \
        --task ${task} --model_path ${MODEL_PATH} --method HeteroCache \
        --real_offload \
        ${COMPRESSION_RATIO:+--compression_ratio $COMPRESSION_RATIO} \
        ${STEPS:+--steps $STEPS} \
        ${DECODE_STEP:+--decode_step $DECODE_STEP} \
        ${TOPK:+--topk $TOPK} \
        ${STABLE_THRESHOLD:+--stable_threshold $STABLE_THRESHOLD} \
        ${SIM_THRESHOLD:+--sim_threshold $SIM_THRESHOLD}
done
