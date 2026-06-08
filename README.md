# Multilingual, Crosslingual, and Monolingual RAG Research

This repository contains the codebase for evaluating Retrieval-Augmented Generation (RAG) across multiple strategies (Multilingual, Crosslingual, and Monolingual) on the **XOR-TyDI benchmark** (an open-domain multilingual QA benchmark).

The codebase supports embedding generation, vector indexing using **Qdrant**, multi-lingual passage retrieval, batch LLM inference, and extensive evaluation of retrieval and question-answering accuracy.

---

## Authors & Contributors
- **Ronak Wani**
- **Sai Teja Sunku**

---

## Project Structure

```
multilingual_rag/
├── baselines/
│   ├── __init__.py
│   └── dense_rag.py              # Main pipeline for indexing, retrieval, and LLM inference
├── evals/
│   ├── eval_xor_retrieve.py      # Evaluates retrieval metrics (Recall@2kt, Recall@5kt)
├── utils/
│   ├── __init__.py
│   ├── model_config.py           # Configuration registry and batch generation for LLMs
│   ├── prompts.py                # Few-shot prompts for QA in multiple languages
│   ├── xor_benchmark_translate.py # Translates XOR benchmark ground-truth answers (TranslateGemma)
│   ├── llm_pred_translate.py     # Translates LLM target-language predictions to English
│   ├── preprocess.sh             # Wikipedia data download, extraction, and dataset download
│   └── translate.sh              # Batch script for translating prediction outputs
├── translated_benchmark_files/   # Storage for translated benchmark files
├── requirements.txt              # Project dependencies
├── token_counter.py              # Utility to count tokens in retrieved passages
├── job.sh                        # SLURM script to run Qdrant container and RAG pipeline
└── eval.sh                       # SLURM script to run evaluation scripts
```

---

## RAG Strategies

The project implements and evaluates three distinct RAG strategies:

| Setting | Query Language | Source Index for Retrieval | Target Answer Language |
| :--- | :--- | :--- | :--- |
| **Monolingual** | Target Language ($L$) | Target Language Wikipedia (`{L}wiki`) | Target Language ($L$) |
| **Crosslingual** | Target Language ($L$) | English Wikipedia (`enwiki`) | English (`en`) |
| **Multilingual** | Target Language ($L$) | Composite Wiki Index (all 8 languages) | English or Target Language |

### Supported Languages
The codebase supports 8 languages:
- Arabic (`ar`)
- Bengali (`bn`)
- English (`en`)
- Finnish (`fi`)
- Japanese (`ja`)
- Korean (`ko`)
- Russian (`ru`)
- Telugu (`te`)

---

## Installation & Setup

### 1. Cloning the Repository
Since this repository contains a Git submodule (`multilingual-smile-metric-qna-eval`), clone it recursively to pull the submodule files:

```bash
git clone --recursive https://github.com/ronak-wani/CIMCL_RAG_RESEARCH.git
```

If you have already cloned the repository without the submodule, initialize and pull it using:

```bash
git submodule update --init --recursive
```

If you only want to clone the current RAG repository, simple use:

```bash
git clone https://github.com/ronak-wani/CIMCL_RAG_RESEARCH.git
```

### 2. Requirements
Install the dependencies from the `requirements.txt` file. Note that language-specific tokenizers are used for evaluations (such as MeCab/fugashi for Japanese, Kiwi for Korean, and Indic NLP for Telugu/Bengali).

```bash
pip install -r requirements.txt
# Additional packages for Japanese and Korean tokenization:
pip install fugashi unidic-lite kiwipiepy indic-nlp-library camel-tools
```

### 3. Preprocessing Data
Run the preprocessing script to download the XML dumps of the Wikipedia datasets, extract the articles to JSON using `WikiExtractor`, and download the XOR-QA train and dev benchmark files:

```bash
sbatch utils/preprocess.sh
```

---

## Running the RAG Pipeline

The primary orchestrator script is `job.sh`. It automatically:
1. Spawns an **Apptainer** sandbox container running **Qdrant Server**.
2. Creates the `wiki_composite` collection on the Qdrant instance.
3. Loads the embedding model (`nvidia/llama-embed-nemotron-8b`) on the GPU.
4. Generates embeddings and upserts Wikipedia text blocks into the Qdrant DB.
5. Runs retrieval and inference.

Submit the pipeline job to SLURM via:
```bash
sbatch job.sh
```

### How `dense_rag.py` Works

The python file `baselines/dense_rag.py` supports modular execution. Below are the key command-line arguments:

```bash
python -m baselines.dense_rag \
    --retrieval-type [monolingual|crosslingual|multilingual] \
    --span-type [xor_english_span|xor_full] \
    --model-name [google/gemma-3-27b-it|Qwen/Qwen3-30B-A3B] \
    [--skip-loading] [--skip-retrieval] [--only-retrieval]
```

- `--skip-loading`: Skips loading Wikipedia data into the Qdrant DB (useful if already indexed).
- `--skip-retrieval`: Skips database retrieval and runs LLM inference directly using existing retrieval files.
- `--only-retrieval`: Runs the retrieval pipeline and exits without performing LLM generation.
- `--retrieval-type`: Configures index filtering (`monolingual` filters to the query language, `crosslingual` filters to English, `multilingual` searches all databases).
- `--span-type`: Customizes the few-shot prompt templates to target English answers (`xor_english_span`) or target-language answers (`xor_full`).

#### Output Directories
- **Retrieval Results**: Saved under `<retrieval_type>/dense_retrieval/<dataset_name>_results.json`.
- **LLM Predictions**: Read from `<retrieval_type>/dense_retrieval/` and written to `<retrieval_type>/[span_type]_llm_predictions/[model_name]/[subfolder]/[dataset_name]_predictions.jsonl`.
---

## Translation Pipelines

Because open-domain QA evaluations require cross-lingual translation of queries, ground truths, or model responses, two translation utilities are provided:

### 1. Benchmark Translation
Translates ground-truth answers in benchmark files to target languages (using `google/translategemma-27b-it`):
```bash
python utils/xor_benchmark_translate.py \
    translated_benchmark_files/xor_dev_full_v1_1.jsonl \
    --direction auto \
    --output-dir translated_benchmark_files \
    --batch-size 64
```

### 2. Prediction Translation
Translates the target-language predictions generated by the LLM back into English to evaluate English span QA performance:
```bash
sbatch utils/translate.sh
```
This invokes `utils/llm_pred_translate.py` which detects non-English text using `lingua` and translates it to English using `google/translategemma-27b-it`.

---

## Evaluation

The evaluation scripts are executed via the SLURM array script `eval.sh`.

```bash
sbatch eval.sh
```

### 1. Retrieval Accuracy
Evaluates Recall@2kt and Recall@5kt (hit rate matching at 2,000 and 5,000 character/token budgets):
```bash
python evals/eval_xor_retrieve.py \
    --data_file translated_benchmark_files/xor_dev_full_v1_1.jsonl \
    --pred_file multilingual/dense_retrieval/xor_dev_full_v1_1_results.json \
    --use-translated both \
    --retrieval-type multilingual
```

### 2. LLM Generation Evaluation
To evaluate LLM-generated answers, you can use the `multilingual-smile-metric-qna-eval` submodule. It computes multilingual lexical-semantic evaluations across the 8 supported languages.

Run the evaluation script from the submodule using your generated predictions and ground-truth reference files:

```bash
python multilingual-smile-metric-qna-eval/main.py \
    --input <path_to_predictions.jsonl> \
    --ground-truth <path_to_ground_truth.jsonl> \
    --output <path_to_output_results.csv> \
    --metrics smile rouge exact_match \
    --verbose
```

Supported metrics: SMILE, ROUGE-L, BERTScore, METEOR, Exact Match, sBERT, BLEURT, MoverScore. You can either choose to use all metrics or select specific ones by modifying the `--metrics` argument. By default, the script computes all metrics.

---

## Supported Models

Models are defined and instantiated via `utils/model_config.py`:
1. **`google/gemma-3-27b-it`**
2. **`Qwen/Qwen3-30B-A3B`**

---

## Database Details

- **Database**: Qdrant (hosted locally in container sandbox).
- **Collection Name**: `wiki_composite`
- **Embedding Model**: `nvidia/llama-embed-nemotron-8b` (4096-dimensional vectors).
- **Metric**: Cosine similarity.
- **Indexing Optimizations**: Payload indexing on the keyword field `wiki` for efficient filtering on monolingual and crosslingual runs.
