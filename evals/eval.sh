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

echo "Monolingual | XOR Train (eng span) | answers_translated"
$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_train_retrieve_eng_span.jsonl --pred_file monolingual/dense_retrieval/xor_train_retrieve_eng_span_results.json --use-translated true
echo "Monolingual | XOR Dev (eng span) | answers_translated"
$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_dev_retrieve_eng_span_v1_1.jsonl --pred_file monolingual/dense_retrieval/xor_dev_retrieve_eng_span_v1_1_results.json --use-translated true
echo "Monolingual | XOR Train (full) | answers"
$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_train_full.jsonl --pred_file monolingual/dense_retrieval/xor_train_full_results.json --use-translated false
echo "Monolingual | XOR Dev (full) | answers"
$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_dev_full_v1_1.jsonl --pred_file monolingual/dense_retrieval/xor_dev_full_v1_1_results.json --use-translated false

echo "Crosslingual | XOR Train (eng span) | answers"
$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_train_retrieve_eng_span.jsonl --pred_file crosslingual/dense_retrieval/xor_train_retrieve_eng_span_results.json --use-translated false
echo "Crosslingual | XOR Dev (eng span) | answers"
$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_dev_retrieve_eng_span_v1_1.jsonl --pred_file crosslingual/dense_retrieval/xor_dev_retrieve_eng_span_v1_1_results.json --use-translated false
# echo "Crosslingual | XOR Train (full) | answers_translated"
# $VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_train_full.jsonl --pred_file crosslingual/dense_retrieval/xor_train_full_results.json --use-translated true
# echo "Crosslingual | XOR Dev (full) | answers_translated"
# $VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_dev_full_v1_1.jsonl --pred_file crosslingual/dense_retrieval/xor_dev_full_v1_1_results.json --use-translated true

# echo "Multilingual | XOR Train (eng span) | answers + answers_translated"
# $VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_train_retrieve_eng_span.jsonl --pred_file multilingual/dense_retrieval/xor_train_retrieve_eng_span_results.json --use-translated both
echo "Multilingual | XOR Dev (eng span) | answers + answers_translated"
$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_dev_retrieve_eng_span_v1_1.jsonl --pred_file multilingual/dense_retrieval/xor_dev_retrieve_eng_span_v1_1_results.json --use-translated both
# echo "Multilingual | XOR Train (full) | answers + answers_translated"
# $VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_train_full.jsonl --pred_file multilingual/dense_retrieval/xor_train_full_results.json --use-translated both
# echo "Multilingual | XOR Dev (full) | answers + answers_translated"
# $VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_dev_full_v1_1.jsonl --pred_file multilingual/dense_retrieval/xor_dev_full_v1_1_results.json --use-translated both

# $VENV_PYTHON evals/eval_xor_full.py \
#     --data_file translated_benchmark_files/xor_dev_full_v1_1.jsonl \
#     --pred_file monolingual/dense_retrieval/xor_full_llm_predictions/google_gemma-3-27b-it/xor_dev_full_v1_1_results_eval.json \
#     --use-translated false

# $VENV_PYTHON evals/eval_xor_full.py \
#     --data_file translated_benchmark_files/xor_dev_retrieve_eng_span_v1_1.jsonl \
#     --pred_file monolingual/dense_retrieval/xor_full_llm_predictions/google_gemma-3-27b-it/xor_dev_retrieve_eng_span_v1_1_results_eval.json \
#     --use-translated true

# $VENV_PYTHON evals/eval_xor_full.py \
#     --data_file translated_benchmark_files/xor_train_full.jsonl \
#     --pred_file monolingual/dense_retrieval/xor_full_llm_predictions/google_gemma-3-27b-it/xor_train_full_results_eval.json \
#     --use-translated false

# $VENV_PYTHON evals/eval_xor_full.py \
#     --data_file translated_benchmark_files/xor_train_retrieve_eng_span.jsonl \
#     --pred_file monolingual/dense_retrieval/xor_full_llm_predictions/google_gemma-3-27b-it/xor_train_retrieve_eng_span_results_eval.json \
#     --use-translated true
