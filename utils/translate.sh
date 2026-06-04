#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=500G
#SBATCH -J LLM_Translate_Multilingual
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --constraint=H200|H100|RTX6000B
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=TERM@60
#SBATCH --open-mode=append
#SBATCH --array=0


BASE_IN="$HOME/CIMCL_RAG_RESEARCH/multilingual/xor_english_span_llm_predictions"
BASE_OUT="$HOME/CIMCL_RAG_RESEARCH/multilingual/xor_english_span_llm_predictions_translated"
SCRIPT="$HOME/CIMCL_RAG_RESEARCH/utils/llm_pred_translate.py"

INPUT_FILES=(
    # "$BASE_IN/Qwen_Qwen3-30B-A3B/use_translated/xor_dev_full_v1_1_results_predictions.jsonl"
    # "$BASE_IN/Qwen_Qwen3-30B-A3B/use_translated/xor_train_full_results_predictions.jsonl"
    # "$BASE_IN/Qwen_Qwen3-30B-A3B/original_answer/xor_dev_retrieve_eng_span_v1_1_results_predictions.jsonl"
    # "$BASE_IN/Qwen_Qwen3-30B-A3B/original_answer/xor_train_retrieve_eng_span_results_predictions.jsonl"
    # "$BASE_IN/google_gemma-3-27b-it/use_translated/xor_dev_full_v1_1_results_predictions.jsonl"
    "$BASE_IN/google_gemma-3-27b-it/use_translated/xor_train_full_results_predictions.jsonl"
    # "$BASE_IN/google_gemma-3-27b-it/original_answer/xor_dev_retrieve_eng_span_v1_1_results_predictions.jsonl"
    # "$BASE_IN/google_gemma-3-27b-it/original_answer/xor_train_retrieve_eng_span_results_predictions.jsonl"
)

OUTPUT_FILES=(
    # "$BASE_OUT/Qwen_Qwen3-30B-A3B/use_translated/xor_dev_full_v1_1_results_predictions_translated.jsonl"
    # "$BASE_OUT/Qwen_Qwen3-30B-A3B/use_translated/xor_train_full_results_predictions_translated.jsonl"
    # "$BASE_OUT/Qwen_Qwen3-30B-A3B/original_answer/xor_dev_retrieve_eng_span_v1_1_results_predictions_translated.jsonl"
    # "$BASE_OUT/Qwen_Qwen3-30B-A3B/original_answer/xor_train_retrieve_eng_span_results_predictions_translated.jsonl"
    # "$BASE_OUT/google_gemma-3-27b-it/use_translated/xor_dev_full_v1_1_results_predictions_translated.jsonl"
    "$BASE_OUT/google_gemma-3-27b-it/use_translated/xor_train_full_results_predictions_translated.jsonl"
    # "$BASE_OUT/google_gemma-3-27b-it/original_answer/xor_dev_retrieve_eng_span_v1_1_results_predictions_translated.jsonl"
    # "$BASE_OUT/google_gemma-3-27b-it/original_answer/xor_train_retrieve_eng_span_results_predictions_translated.jsonl"
)

source "$HOME/CIMCL_RAG_RESEARCH/venv/bin/activate"
VENV_PYTHON=$(which python)

TASK_ID=${SLURM_ARRAY_TASK_ID}
INPUT="${INPUT_FILES[$TASK_ID]}"
OUTPUT="${OUTPUT_FILES[$TASK_ID]}"

mkdir -p "$(dirname "$OUTPUT")"

$VENV_PYTHON "$SCRIPT" \
    --input-file  "$INPUT" \
    --output-file "$OUTPUT" \
    --batch-size  64

EXIT_CODE=$?

echo "Finished: $(date) | Exit code: $EXIT_CODE"

exit $EXIT_CODE
