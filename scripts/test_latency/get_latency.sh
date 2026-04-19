# #!/bin/bash
cd "$(dirname "$0")/../.."

MODEL_PATH="../models/Llama-3.1-8B-Instruct"
COMPRESSION_RATIO=0.5

HETEROCACHE_METHOD="HeteroCache"
HETEROCACHE_REAL_OFFLOAD="--real_offload"
HETEROCACHE_DECODE_STEP=5000
HETEROCACHE_STEPS=5
HETEROCACHE_TOPK=1024
HETEROCACHE_STABLE_THRESHOLD=0.5
HETEROCACHE_SIM_THRESHOLD=0.5

CAKE_METHOD="CAKE"
FULLKV_METHOD="FullKV"
SNAPKV_METHOD="SnapKV"
STREAMINGLLM_METHOD="StreamingLLM"
H2O_METHOD="H2O"

echo "Starting latency test script..."


python -m tools.test_latency \
    --method ${HETEROCACHE_METHOD} \
    ${HETEROCACHE_REAL_OFFLOAD} \
    --compression_ratio ${COMPRESSION_RATIO} \
    --model_path ${MODEL_PATH} \
    --decode_step ${HETEROCACHE_DECODE_STEP} \
    --steps ${HETEROCACHE_STEPS} \
    --topk ${HETEROCACHE_TOPK} \
    --stable_threshold ${HETEROCACHE_STABLE_THRESHOLD} \
    --sim_threshold ${HETEROCACHE_SIM_THRESHOLD}

python -m tools.test_latency \
    --method ${FULLKV_METHOD} \
    --model_path ${MODEL_PATH}

python -m tools.test_latency \
    --method ${SNAPKV_METHOD} \
    --compression_ratio ${COMPRESSION_RATIO} \
    --model_path ${MODEL_PATH}

python -m tools.test_latency \
    --method ${CAKE_METHOD} \
    --compression_ratio ${COMPRESSION_RATIO} \
    --model_path ${MODEL_PATH}


python -m tools.test_latency \
    --method ${STREAMINGLLM_METHOD} \
    --compression_ratio ${COMPRESSION_RATIO} \
    --model_path ${MODEL_PATH}

python -m tools.test_latency \
    --method ${H2O_METHOD} \
    --compression_ratio ${COMPRESSION_RATIO} \
    --model_path ${MODEL_PATH}


echo "All tests completed!"

   
