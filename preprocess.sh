#!/bin/bash
#SBATCH -N 1                 
#SBATCH -n 8               
#SBATCH --mem=100G
#SBATCH -J Wiki_Process         
#SBATCH -p long
#SBATCH -t 72:00:00
#SBATCH --requeue
#SBATCH --signal=TERM@60
#SBATCH --open-mode=append

conda activate py310
# pip install wikiextractor
VENV_PYTHON=$(which python)

# Download Wikipedia passages (Order: Arabic, Bengali, English, Finnish, Japanese, Korean, Russian, Telugu)
#  list=(enwiki arwiki bnwiki fiwiki jawiki kowiki ruwiki tewiki)

#  for i in "${list[@]}"; do
#      wget "https://archive.org/download/${i}-20190201/${i}-20190201-pages-articles-multistream.xml.bz2"
#      bunzip2 "${i}-20190201-pages-articles-multistream.xml.bz2"
#  done

#  for i in "${list[@]}"; do
#     python -m wikiextractor.WikiExtractor \
#         ${i}-20190201-pages-articles-multistream.xml \
#         --json \
#         --processes 8 \
#         --bytes 1G \
#         --output ${i}_extracted
#  done

# Download XOR-Retrieve and XOR-EnglishSpan train and dev datasets

# wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_train_retrieve_eng_span.jsonl
# wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_retrieve_eng_span_v1_1.jsonl

# Download XOR-TyDiQA XOR-FULL train and dev datasets

# wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_train_full.jsonl
# wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_full_v1_1.jsonl
