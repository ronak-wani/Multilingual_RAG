#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=500G
#SBATCH -J Qwen
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --constraint=H200|H100|RTX6000B
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=TERM@60
#SBATCH --open-mode=append

source ./venv/bin/activate
VENV_PYTHON=$(which python)
#pip install -r requirements.txt

# rm -rf .qdrant-initialized qdrant_sandbox/ storage/ snapshots/ dense_rag.log

module load apptainer

if [ ! -d "qdrant_sandbox" ]; then
  echo "Building Qdrant sandbox"
  apptainer build --sandbox qdrant_sandbox docker://qdrant/qdrant:latest
else
  echo "Qdrant sandbox already exists"
fi

export QDRANT_HOST=localhost
export QDRANT_PORT=6333

QDRANT_SANDBOX=qdrant_sandbox
QDRANT_STORAGE=/home/$USER/CIMCL_RAG_RESEARCH/qdrant_storage
mkdir -p $QDRANT_STORAGE

echo "Starting Qdrant"

apptainer exec \
--bind ${QDRANT_STORAGE}:/qdrant/storage \
--env QDRANT__SERVICE__HOST=0.0.0.0 \
--env QDRANT__STORAGE__WAL__SYNC=false \
$QDRANT_SANDBOX \
$QDRANT_SANDBOX/qdrant/qdrant &

QDRANT_PID=$!
echo "Qdrant PID: $QDRANT_PID"

echo "Waiting for Qdrant"
until curl -s http://localhost:6333/readyz > /dev/null; do
  sleep 2
done
echo "Qdrant is ready."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

trap '
    echo "SIGTERM received — forwarding to Python and requeueing"
    kill -TERM $PID
    wait $PID
    scontrol requeue $SLURM_JOB_ID
    exit 0
' TERM

#
# $VENV_PYTHON -m baselines.dense_rag --skip-loading --retrieval-type monolingual &
# PID=$!
# wait $PID
#
$VENV_PYTHON -m baselines.dense_rag --skip-loading --only-retrieval --retrieval-type multilingual &
PID=$!
wait $PID

# $VENV_PYTHON -m baselines.dense_rag --skip-loading --only-retrieval --retrieval-type crosslingual &
# PID=$!
# wait $PID

# $VENV_PYTHON baselines/translate.py \
#     xor_dev_retrieve_eng_span_v1_1.jsonl \
#     xor_train_retrieve_eng_span.jsonl \
#     --direction en-to-target \
#     --output-dir translated_benchmark_files \
#     --batch-size 64 &
# TRANSLATE_EN_TO_TARGET_PID=$!
# wait $TRANSLATE_EN_TO_TARGET_PID

# $VENV_PYTHON baselines/translate.py \
#     xor_dev_full_v1_1.jsonl \
#     xor_train_full.jsonl \
#     --direction auto \
#     --output-dir translated_benchmark_files \
#     --batch-size 64 &
# TRANSLATE_TARGET_TO_EN_PID=$!
# wait $TRANSLATE_TARGET_TO_EN_PID

# MODEL="CohereLabs/aya-101"
# MODEL="google/gemma-3-27b-it"
# MODEL="Qwen/Qwen3-30B-A3B"

# echo "Running monolingual + xor_full"
# $VENV_PYTHON -m baselines.dense_rag \
#     --skip-retrieval \
#     --retrieval-type monolingual \
#     --span-type xor_full \
#     --model-name "$MODEL" &

# PID=$!                      
# wait $PID                    
# echo "Completed monolingual + xor_full"

# echo "Running crosslingual + xor_english_span"
# $VENV_PYTHON -m baselines.dense_rag \
#     --skip-retrieval \
#     --retrieval-type crosslingual \
#     --span-type xor_english_span \
#     --model-name "$MODEL" &
# PID=$!                      
# wait $PID     
# echo "Completed crosslingual + xor_english_span"

# echo "Running multilingual + xor_english_span"
# $VENV_PYTHON -m baselines.dense_rag \
#     --skip-retrieval \
#     --retrieval-type multilingual \
#     --span-type xor_english_span \
#     --model-name "$MODEL" &
# PID=$!                      
# wait $PID     
# echo "Completed multilingual + xor_english_span"

# echo "Running multilingual + xor_full"
# $VENV_PYTHON -m baselines.dense_rag \
#     --skip-retrieval \
#     --retrieval-type multilingual \
#     --span-type xor_full \
#     --model-name "$MODEL" &
# PID=$!                      
# wait $PID     
# echo "Completed multilingual + xor_full"

echo "Stopping Qdrant"
kill "$QDRANT_PID" 2>/dev/null || true
wait "$QDRANT_PID" 2>/dev/null || true