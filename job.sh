#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=500G
#SBATCH -J NEW_CIMCL_RAG
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --constraint=H200
#SBATCH --gres=gpu:H200:1
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

trap 'echo "SIGTERM received, requeueing."; scontrol requeue $SLURM_JOB_ID; exit 0' TERM

$VENV_PYTHON baselines/dense_rag.py --skip-loading &
PID=$!
wait $PID

echo "Stopping Qdrant"
kill "$QDRANT_PID" 2>/dev/null || true
wait "$QDRANT_PID" 2>/dev/null || true

$VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_train_retrieve_eng_span.jsonl --pred_file output/xor_train_retrieve_eng_span_results.json
$VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_dev_retrieve_eng_span_v1_1.jsonl --pred_file output/xor_dev_retrieve_eng_span_v1_1_results.json
$VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_train_full.jsonl --pred_file output/xor_train_full_results.json
$VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_dev_full_v1_1.jsonl --pred_file output/xor_dev_full_v1_1_results.json

$VENV_PYTHON evals/eval_xor_engspan.py --data_file xor_train_retrieve_eng_span.jsonl --pred_file xor-engspan/xor_train_retrieve_eng_span_results.json
$VENV_PYTHON evals/eval_xor_engspan.py --data_file xor_dev_retrieve_eng_span_v1_1.jsonl --pred_file xor-engspan/xor_dev_retrieve_eng_span_v1_1_results.json
$VENV_PYTHON evals/eval_xor_engspan.py --data_file xor_train_full.jsonl --pred_file xor-engspan/xor_train_full_results.json
$VENV_PYTHON evals/eval_xor_engspan.py --data_file xor_dev_full_v1_1.jsonl --pred_file xor-engspan/xor_dev_full_v1_1_results.json

$VENV_PYTHON evals/eval_xor_full.py --data_file xor_train_retrieve_eng_span.jsonl --pred_file xor-full/xor_train_retrieve_eng_span_results.json
$VENV_PYTHON evals/eval_xor_full.py --data_file xor_dev_retrieve_eng_span_v1_1.jsonl --pred_file xor-full/xor_dev_retrieve_eng_span_v1_1_results.json
$VENV_PYTHON evals/eval_xor_full.py --data_file xor_train_full.jsonl --pred_file xor-full/xor_train_full_results.json
$VENV_PYTHON evals/eval_xor_full.py --data_file xor_dev_full_v1_1.jsonl --pred_file xor-full/xor_dev_full_v1_1_results.json