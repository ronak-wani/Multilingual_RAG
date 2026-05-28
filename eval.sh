#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=100G
#SBATCH -J Crosslingual_Evaluation
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --array=0

export MECABRC=$HOME/.local/etc/mecabrc
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

source ./venv/bin/activate
VENV_PYTHON=$(which python)

LABELS=(
    # "Monolingual | XOR Train (eng span) | answers_translated"
    # "Monolingual | XOR Dev (eng span) | answers_translated"
    # "Monolingual | XOR Train (full) | answers"
    # "Monolingual | XOR Dev (full) | answers"
    # "Crosslingual | XOR Train (eng span) | answers"
    # "Crosslingual | XOR Dev (eng span) | answers"
    # "Crosslingual | XOR Train (full) | answers_translated"
    # "Crosslingual | XOR Dev (full) | answers_translated"
    # "Multilingual | XOR Train (eng span) | answers + answers_translated"
    # "Multilingual | XOR Dev (eng span) | answers + answers_translated"
    # "Multilingual | XOR Train (full) | answers + answers_translated"
    "Multilingual | XOR Dev (full) | answers + answers_translated"
)

COMMANDS=(
# "$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_train_retrieve_eng_span.jsonl --pred_file monolingual/dense_retrieval/xor_train_retrieve_eng_span_results.json --use-translated true --retrieval-type monolingual"
# "$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_dev_retrieve_eng_span_v1_1.jsonl --pred_file monolingual/dense_retrieval/xor_dev_retrieve_eng_span_v1_1_results.json --use-translated true --retrieval-type monolingual"
# "$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_train_full.jsonl --pred_file monolingual/dense_retrieval/xor_train_full_results.json --use-translated false --retrieval-type monolingual"
# "$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_dev_full_v1_1.jsonl --pred_file monolingual/dense_retrieval/xor_dev_full_v1_1_results.json --use-translated false --retrieval-type monolingual"

# "$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_train_retrieve_eng_span.jsonl --pred_file crosslingual/dense_retrieval/xor_train_retrieve_eng_span_results.json --use-translated false --retrieval-type crosslingual"
# "$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_dev_retrieve_eng_span_v1_1.jsonl --pred_file crosslingual/dense_retrieval/xor_dev_retrieve_eng_span_v1_1_results.json --use-translated false --retrieval-type crosslingual"
# "$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_train_full.jsonl --pred_file crosslingual/dense_retrieval/xor_train_full_results.json --use-translated true --retrieval-type crosslingual"
# "$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_dev_full_v1_1.jsonl --pred_file crosslingual/dense_retrieval/xor_dev_full_v1_1_results.json --use-translated true --retrieval-type crosslingual"

# "$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_train_retrieve_eng_span.jsonl --pred_file multilingual/dense_retrieval/xor_train_retrieve_eng_span_results.json --use-translated both --retrieval-type multilingual"
# "$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_dev_retrieve_eng_span_v1_1.jsonl --pred_file multilingual/dense_retrieval/xor_dev_retrieve_eng_span_v1_1_results.json --use-translated both --retrieval-type multilingual"
# "$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_train_full.jsonl --pred_file multilingual/dense_retrieval/xor_train_full_results.json --use-translated both --retrieval-type multilingual"
"$VENV_PYTHON evals/eval_xor_retrieve.py --data_file translated_benchmark_files/xor_dev_full_v1_1.jsonl --pred_file multilingual/dense_retrieval/xor_dev_full_v1_1_results.json --use-translated both --retrieval-type multilingual"
)

echo "Running: ${LABELS[$SLURM_ARRAY_TASK_ID]}"
eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"
