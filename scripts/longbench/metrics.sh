
cd "$(dirname "$(realpath "$0")")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH}"
results_dir=$1

python3 scripts/longbench/eval.py \
    --results_dir ${results_dir}
