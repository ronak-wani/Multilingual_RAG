#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=100G
#SBATCH -J Evaluation
#SBATCH -p short
#SBATCH -t 12:00:00

export MECABRC=$HOME/.local/etc/mecabrc
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

source ./venv/bin/activate
VENV_PYTHON=$(which python)

# $VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_train_retrieve_eng_span.jsonl --pred_file monolingual_output/xor_train_retrieve_eng_span_results.json
# $VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_dev_retrieve_eng_span_v1_1.jsonl --pred_file monolingual_output/xor_dev_retrieve_eng_span_v1_1_results.json
# $VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_train_full.jsonl --pred_file monolingual_output/xor_train_full_results.json
# $VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_dev_full_v1_1.jsonl --pred_file monolingual_output/xor_dev_full_v1_1_results.json

# $VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_train_retrieve_eng_span.jsonl --pred_file multilingual_output/xor_train_retrieve_eng_span_results.json
# $VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_dev_retrieve_eng_span_v1_1.jsonl --pred_file multilingual_output/xor_dev_retrieve_eng_span_v1_1_results.json
# $VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_train_full.jsonl --pred_file multilingual_output/xor_train_full_results.json
# $VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_dev_full_v1_1.jsonl --pred_file multilingual_output/xor_dev_full_v1_1_results.json

# $VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_train_retrieve_eng_span.jsonl --pred_file crosslingual_output/xor_train_retrieve_eng_span_results.json
# $VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_dev_retrieve_eng_span_v1_1.jsonl --pred_file crosslingual_output/xor_dev_retrieve_eng_span_v1_1_results.json
# $VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_train_full.jsonl --pred_file crosslingual_output/xor_train_full_results.json
# $VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_dev_full_v1_1.jsonl --pred_file crosslingual_output/xor_dev_full_v1_1_results.json

$VENV_PYTHON evals/eval_xor_full.py \
    --data_file translated_benchmark_files/xor_dev_full_v1_1.jsonl \
    --pred_file monolingual/xor_full_llm_predictions/google_gemma-3-27b-it/xor_dev_full_v1_1_results_eval.json \
    --use-translated false

$VENV_PYTHON evals/eval_xor_full.py \
    --data_file translated_benchmark_files/xor_dev_retrieve_eng_span_v1_1.jsonl \
    --pred_file monolingual/xor_full_llm_predictions/google_gemma-3-27b-it/xor_dev_retrieve_eng_span_v1_1_results_eval.json \
    --use-translated true

$VENV_PYTHON evals/eval_xor_full.py \
    --data_file translated_benchmark_files/xor_train_full.jsonl \
    --pred_file monolingual/xor_full_llm_predictions/google_gemma-3-27b-it/xor_train_full_results_eval.json \
    --use-translated false

$VENV_PYTHON evals/eval_xor_full.py \
    --data_file translated_benchmark_files/xor_train_retrieve_eng_span.jsonl \
    --pred_file monolingual/xor_full_llm_predictions/google_gemma-3-27b-it/xor_train_retrieve_eng_span_results_eval.json \
    --use-translated true
