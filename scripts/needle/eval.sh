# This script is adapted from
# https://github.com/FranxYao/Long-Context-Data-Engineering.git
cd "$(dirname "$0")/../.."

mkdir -p ./results_needle/logs/
mkdir -p ./results_needle/img/


CUDA_DEVICE=3
START_LEN=16000
END_LEN=128001
STEP=16000

MODEL_PROVIDER="LLaMA3"
MODEL_PATH="../models/Llama-3.1-8B-Instruct"
MODEL_NAME="../models/Llama-3.1-8B-Instruct"

METHOD='fullkv'       # ['full', 'heterocache', 'pyramidkv', 'snapkv', 'streamingllm', 'h2o']
COMPRESSION_RATIO=0.5
# MAX_CAPACITY_PROMPT=1024  # [64, 96, 128, 256, 512, 1024, ...]
ATTN_IMPLEMENTATION="flash_attention_2" # Support "flash_attention_2", "sdpa", "None".
TAG="v1"
MODEL_VERSION="LlaMA3_${METHOD}_${COMPRESSION_RATIO}_${TAG}"


(
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python -u scripts/needle/run_needle_in_haystack.py \
    --s_len ${START_LEN} \
    --e_len ${END_LEN} \
    --model_provider ${MODEL_PROVIDER} \
    --model_path ${MODEL_PATH} \
    --model_name ${MODEL_NAME} \
    --attn_implementation ${ATTN_IMPLEMENTATION} \
    --step ${STEP} \
    --method ${METHOD} \
    --compression_ratio ${COMPRESSION_RATIO} \
    --model_version ${MODEL_VERSION}
) 2>&1 | tee results_needle/logs/${MODEL_VERSION}.log
