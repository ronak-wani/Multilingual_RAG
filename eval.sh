#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=100G
#SBATCH -J Evaluation
#SBATCH -p short
#SBATCH -t 12:00:00

source ./venv/bin/activate
VENV_PYTHON=$(which python)

$VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_train_retrieve_eng_span.jsonl --pred_file multilingual_output/xor_train_retrieve_eng_span_results.json
$VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_dev_retrieve_eng_span_v1_1.jsonl --pred_file multilingual_output/xor_dev_retrieve_eng_span_v1_1_results.json
$VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_train_full.jsonl --pred_file multilingual_output/xor_train_full_results.json
$VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_dev_full_v1_1.jsonl --pred_file multilingual_output/xor_dev_full_v1_1_results.json

$VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_train_retrieve_eng_span.jsonl --pred_file crosslingual_output/xor_train_retrieve_eng_span_results.json
$VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_dev_retrieve_eng_span_v1_1.jsonl --pred_file crosslingual_output/xor_dev_retrieve_eng_span_v1_1_results.json
$VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_train_full.jsonl --pred_file crosslingual_output/xor_train_full_results.json
$VENV_PYTHON evals/eval_xor_retrieve.py --data_file xor_dev_full_v1_1.jsonl --pred_file crosslingual_output/xor_dev_full_v1_1_results.json