import argparse, signal, glob
import os, json, logging, sys
import time
from pathlib import Path
from typing import Literal
from utils.model_config import (
    get_model_config,
    load_model,
    build_prompt,
    run_batch_sync,
)

import ijson
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
import uuid, aiofiles
import gc
import torch
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
from qdrant_client import QdrantClient, models


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dense_rag.log')
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

executor = ThreadPoolExecutor(max_workers=os.cpu_count())


class DenseRAG:
    def __init__(self):
        logger.info("Initializing DenseRAG")
        self.shutdown_requested = False
        signal.signal(signal.SIGTERM, self.handle_sigterm)
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            logger.info("Login Successful")
        else:
            logger.warning("No HF_TOKEN found")

        logger.info("Loading the embedding model")

        self.embed_model = SentenceTransformer(
            "nvidia/llama-embed-nemotron-8b",
            trust_remote_code=True,
            model_kwargs={
                "attn_implementation": "flash_attention_2",
                "torch_dtype": "bfloat16"
            },
            tokenizer_kwargs={"padding_side": "left"},
            device="cuda"
        )

        self.embed_model.eval()
        for param in self.embed_model.parameters():
            param.requires_grad = False
        torch.set_grad_enabled(False)

        logger.info("Completed loading the embedding model")

        self.collection_name = "wiki_composite"
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = os.getenv("QDRANT_PORT", "6333")
        qdrant_url = f"http://{qdrant_host}:{qdrant_port}"

        self.qdrant_client = QdrantClient(url=qdrant_url, timeout=1200)
        logger.info(f"Connected to Qdrant server at {qdrant_url}")
        self.collection_exist = self.qdrant_client.collection_exists(self.collection_name)
        if not self.collection_exist:
            logger.info(f"Creating new collection")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                on_disk_payload=True,
                vectors_config=models.VectorParams(
                    size=4096,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                ),
            )
            logger.info("Collection created successfully")
        else:
            logger.info(f"Collection already exists")

        self.qdrant_client.create_payload_index(
            collection_name=self.collection_name,
            field_name="wiki",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        logger.info("Created payload index on 'wiki' field")

    def handle_sigterm(self, signum, frame):
        logger.info("SIGTERM received, will stop after current batch")
        self.shutdown_requested = True

    async def chunk_document(self, doc, text_split):
        return await asyncio.get_event_loop().run_in_executor(
            executor,
            text_split.split_documents,
            [doc]
        )

    async def create_embeddings(self, wiki, text_split, batch_size=256):
        docs, texts = [], []
        num_batch = 0

        checkpoint_count = self.qdrant_client.count(
            collection_name=self.collection_name,
            count_filter=models.Filter(
                must=[models.FieldCondition(
                    key="wiki",
                    match=models.MatchValue(value=wiki)
                )]
            )
        ).count

        logger.info(
            f"[{wiki}] Found {checkpoint_count:,} points already uploaded, skipping first {checkpoint_count} chunks")

        chunks_skipped = 0
        articles_processed = 0

        wiki_files = sorted(glob.glob(f"{wiki}_extracted/**/wiki_*", recursive=True))

        if not wiki_files:
            logger.error(f"[{wiki}] No extracted files found in {wiki}_extracted/")
            raise FileNotFoundError(f"No WikiExtractor files found for {wiki}")

        logger.info(f"[{wiki}] Found {len(wiki_files)} WikiExtractor files to process")

        for wiki_file in wiki_files:
            logger.info(f"[{wiki}] Processing {wiki_file}")

            with open(wiki_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        article = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"[{wiki}] Invalid JSON line in {wiki_file}")
                        continue

                    if not article.get('text') or len(article['text']) < 100:
                        continue

                    articles_processed += 1

                    doc = Document(
                        page_content=article['text'],
                        metadata={
                            'source': wiki,
                            'title': article.get('title', ''),
                            'url': article.get('url', ''),
                            'id': article.get('id', '')
                        }
                    )

                    chunks = await self.chunk_document(doc, text_split)
                    # logger.info(f"[{wiki}] Created {len(chunks)} chunks")
                    for chunk in chunks:
                        if chunks_skipped < checkpoint_count:
                            chunks_skipped += 1
                            if chunks_skipped % 100 == 0:
                                logger.info(f"[{wiki}] Skipped {chunks_skipped:,}/{checkpoint_count:,} chunks")
                        else:
                            docs.append(chunk)
                            texts.append(chunk.page_content)
                            if len(texts) >= batch_size and len(docs) >= batch_size:
                                num_batch += 1
                                torch.cuda.empty_cache()
                                logger.info(
                                    f"[{wiki}] Starting embedding generation for Batch {num_batch} ({articles_processed:,} articles)")
                                embeddings_array = self.embed_model.encode_document(
                                    texts,
                                    batch_size=128,
                                    show_progress_bar=False,
                                    convert_to_tensor=False,
                                    normalize_embeddings=False,
                                    device='cuda',
                                )
                                embeddings = embeddings_array.tolist()
                                yield embeddings, docs
                                logger.info(f"[{wiki}] Generated {len(embeddings)} embeddings")
                                del embeddings_array, embeddings
                                gc.collect()
                                torch.cuda.empty_cache()
                                docs, texts = [], []
                                if self.shutdown_requested:
                                    logger.info(
                                        f"[{wiki}] Shutdown requested, stopping generator after batch {num_batch}")
                                    return
            logger.info(f"[{wiki}] Completed {wiki_file} ({articles_processed:,} total articles)")
        if texts:
            num_batch += 1
            torch.cuda.empty_cache()
            logger.info(
                f"[{wiki}] Starting embedding generation for final Batch {num_batch} ({articles_processed:,} articles)")
            embeddings_array = self.embed_model.encode_document(
                texts,
                batch_size=128,
                show_progress_bar=False,
                convert_to_tensor=False,
                normalize_embeddings=False,
                device='cuda',
            )
            embeddings = embeddings_array.tolist()
            yield embeddings, docs
            logger.info(f"[{wiki}] Generated {len(embeddings)} embeddings")
            del embeddings_array
            gc.collect()
            torch.cuda.empty_cache()
        logger.info(f"[{wiki}] Completed: {articles_processed:,} articles, {num_batch} batches")

    async def upload(self, embeddings, documents, wiki):

        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "wiki": wiki,
                }
            )
            for embedding, doc in zip(embeddings, documents)
        ]

        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"[{wiki}] Uploaded batch of {len(points)} points")
        del points
        torch.cuda.empty_cache()

    async def data_loading(self):
        text_split = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", ";", ""],
            keep_separator=False,
        )

        wikis = ['arwiki', 'bnwiki', 'enwiki', 'fiwiki', 'jawiki', 'kowiki', 'ruwiki', 'tewiki']

        logger.info(f"Procesing wikis: {wikis}")

        for wiki in wikis:
            logger.info(f"[{wiki}] Starting embedding process")
            async for embeddings, docs in self.create_embeddings(wiki, text_split):
                logger.info(f"[{wiki}] Uploading {len(embeddings)} embeddings to Qdrant")
                await self.upload(embeddings, docs, wiki)
                logger.info(f"[{wiki}] Uploading {len(embeddings)} completed")
                if self.shutdown_requested:
                    logger.info(f"[{wiki}] Shutdown requested, stopping after this batch")
                    return

        docs_count = self.qdrant_client.count(collection_name=self.collection_name).count
        logger.info(f"Data loading completed. Total documents in collection: {docs_count}")

    def count_existing_prompts(self, output_file):
        if not os.path.exists(output_file):
            return 0
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content or content == '[':
                    return 0
                count = content.count('"id":')
                logger.info(f"Found {count} existing prompts in {output_file}")
                return count
        except Exception as e:
            logger.warning(f"Error counting existing prompts: {e}, starting from 0")
            return 0

    def read_file(self, file_path, skip_count=0):
        logger.info(f"Reading file: {file_path} (skipping first {skip_count} items)")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    if i < skip_count:
                        continue
                    yield json.loads(line.strip())
            else:
                data = json.load(f)
                for i, item in enumerate(data):
                    if i < skip_count:
                        continue
                    yield item

        logger.info(f"Finished reading {file_path}")

    async def retrieval_pipeline(self, file_path, retrieval_type):
        output_dir = Path(retrieval_type) / "dense_retrieval"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

        output_file = f"{output_dir}/{os.path.splitext(file_path)[0]}_results.json"

        skip_count = self.count_existing_prompts(output_file)
        total_prompts = skip_count

        file_exists = os.path.exists(output_file) and skip_count > 0
        mode = "a" if file_exists else "w"

        logger.info(f"[{file_path}] Starting from prompt #{skip_count + 1}")

        async with aiofiles.open(output_file, mode, encoding="utf-8") as f:

            if not file_exists:
                await f.write("[\n")
                first_item = True
            else:
                first_item = False

            for item in self.read_file(file_path, skip_count):
                if self.shutdown_requested:
                    logger.info(f"[{file_path}] Shutdown requested at prompt {total_prompts + 1}, stopping gracefully")
                    break

                total_prompts += 1

                if total_prompts % 100 == 1:
                    logger.info(f"[{file_path}] Processing prompt {total_prompts}")

                question = item['question']
                query_embedding_raw = self.embed_model.encode_query([question])
                query_embedding = query_embedding_raw[0].tolist()
                search_result = self.retrieve(query_embedding, retrieval_type, item['lang'])

                retrieved_context = [
                    point.payload["text"]
                    for point in search_result.points
                ]
                prompt = {
                    "id": item['id'],
                    "lang": item['lang'],
                    "question": item['question'],
                    "ctxs": retrieved_context,
                }
                if not first_item:
                    await f.write(",\n")
                else:
                    first_item = False

                json_str = json.dumps(prompt, ensure_ascii=False, indent=4)
                await f.write(json_str)

                if total_prompts == 1 or (total_prompts == skip_count + 1 and skip_count > 0):
                    logger.info(f"Sample Prompt: {prompt}")

                await f.flush()

            await f.write("\n]")
            await f.flush()

        if self.shutdown_requested:
            logger.info(
                f"[{file_path}] Graceful shutdown: Saved {total_prompts - skip_count} new prompts (total: {total_prompts})")
        else:
            logger.info(f"[{file_path}] Completed: {total_prompts} total prompts in {output_file}")

    def retrieve(self, embedding, retrieval_type, lang):
        query_filter = None
        if retrieval_type == 'monolingual':
            query_filter = models.Filter(
                must=[models.FieldCondition(
                    key="wiki",
                    match=models.MatchValue(value=f"{lang}wiki")
                )]
            )

        elif retrieval_type == 'crosslingual':
            query_filter = models.Filter(
                must=[models.FieldCondition(
                    key="wiki",
                    match=models.MatchValue(value="enwiki")
                )]
            )

        elif retrieval_type == 'multilingual':
            query_filter = None

        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=100,
            timeout=1200,
            query_filter=query_filter,
        )
        return search_result

    async def _stream_json_array(self, path: str, exec_: ThreadPoolExecutor):
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)

        def _producer():
            try:
                with open(path, "rb") as fh:
                    for item in ijson.items(fh, "item"):
                        loop.call_soon_threadsafe(queue.put_nowait, item)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, StopAsyncIteration())

        fut = loop.run_in_executor(exec_, _producer)
        while True:
            item = await queue.get()
            if isinstance(item, StopAsyncIteration):
                break
            if isinstance(item, Exception):
                raise item
            yield item
        await fut

    async def chat_llm(self, model_name, input_dir, retrieval_type, span_type, inference_batch_size=64):
        cfg = get_model_config(model_name)
        output_dir = Path(retrieval_type) / f"{span_type}_llm_predictions" / model_name.replace("/", "_")
        output_dir.mkdir(parents=True, exist_ok=True)

        input_files = sorted(Path(input_dir).glob("*.json"))
        if not input_files:
            logger.error(f"No JSON files found in {input_dir}")
            return

        logger.info(
            f"LLM model={model_name}  type={cfg.model_type}  "
            f"span_type={span_type}  batch={inference_batch_size}  "
            f"files={len(input_files)}"
        )

        loop = asyncio.get_event_loop()
        tok, model = await loop.run_in_executor(executor, load_model, cfg)

        for input_file in input_files:
            if self.shutdown_requested:
                logger.warning("Shutdown - skipping remaining files")
                break

            output_file = str(Path(output_dir) / f"{input_file.stem}_predictions.json")

            skip_count = self.count_existing_prompts(output_file)
            total_written = skip_count
            file_exists = os.path.exists(output_file) and skip_count > 0
            mode: Literal["a", "w"] = "a" if file_exists else "w"

            logger.info(
                f"[{input_file.name}] → {output_file}  "
                f"(mode={mode}, already_done={skip_count})"
            )

            pending_items: list[dict] = []
            pending_prompts: list = []
            items_seen = 0
            first_item = not file_exists

            async with aiofiles.open(output_file, mode, encoding="utf-8") as fout:
                if not file_exists:
                    await fout.write("{\n")

                async def flush(items: list[dict], prompts: list) -> int:
                    nonlocal first_item
                    if not items:
                        return 0
                    predictions: list[str] = await loop.run_in_executor(
                        executor, run_batch_sync, tok, model, cfg, prompts, inference_batch_size,
                    )
                    n = 0
                    for item, pred in zip(items, predictions):
                        if not first_item:
                            await fout.write(",\n")
                        else:
                            first_item = False
                        await fout.write(
                            f'    {json.dumps(str(item["id"]))}: {json.dumps(pred, ensure_ascii=False)}'
                        )
                        n += 1

                    await fout.flush()
                    logger.info(
                        f"[{input_file.name}] +{n} predictions  "
                        f"(total: {total_written + n})"
                    )
                    return n

                async for item in self._stream_json_array(str(input_file), executor):
                    if items_seen < skip_count:
                        items_seen += 1
                        continue
                    items_seen += 1

                    if self.shutdown_requested:
                        total_written += await flush(pending_items, pending_prompts)
                        logger.warning(
                            f"[{input_file.name}] Shutdown — saved {total_written} predictions"
                        )
                        await fout.write("\n}")
                        await fout.flush()
                        break

                    pending_items.append(item)
                    pending_prompts.append(build_prompt(item, span_type, model_name))

                    if len(pending_items) >= inference_batch_size:
                        total_written += await flush(pending_items, pending_prompts)
                        pending_items, pending_prompts = [], []

                else:
                    if pending_items:
                        total_written += await flush(pending_items, pending_prompts)
                    await fout.write("\n}")
                    await fout.flush()

                if not self.shutdown_requested:
                    logger.info(f"[{input_file.name}] Complete — {total_written} predictions written")

        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("All files processed")

    def _write_eval_json(self, predictions_json: str, eval_json: str) -> None:

        predictions: dict[str, str] = {}
        with open(predictions_json, "r", encoding="utf-8") as f:
            for rec in json.load(f):
                predictions[str(rec["id"])] = rec["prediction"]

        with open(eval_json, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        logger.info(f"Eval JSON written → {eval_json}  ({len(predictions)} predictions)")

    async def main(self, model_name: str = "", skip_retrieval=False, skip_loading=False, retrieval_type="multilingual",
                   span_type="english_span"):
        try:
            if skip_retrieval:
                logger.info("Skipping retrieval — running inference only")
            else:
                if not skip_loading:
                    logger.info("Starting data loading")
                    await self.data_loading()

                logger.info("Starting retrieval pipelines")

                for file_path in [
                    "xor_dev_retrieve_eng_span_v1_1.jsonl",
                    "xor_train_retrieve_eng_span.jsonl",
                    "xor_train_full.jsonl",
                    "xor_dev_full_v1_1.jsonl",
                ]:
                    await self.retrieval_pipeline(file_path, retrieval_type)
                logger.info("Completed all the retrieval pipelines")

            logger.info(f"Starting inference: model={model_name}  span={span_type}")
            await self.chat_llm(
                model_name           = model_name,
                input_dir            = Path(retrieval_type) / "dense_retrieval",
                retrieval_type       = retrieval_type,
                span_type            = span_type,
                inference_batch_size = 64,
            )
            logger.info("All pipelines complete")

        except Exception as e:
            logger.error("Pipeline failed with exception", exc_info=True)
            raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dense RAG Wikipedia Processing')

    parser.add_argument(
        '--skip-retrieval',
        action='store_true',
        help='Skip retrieval and only run inference pipelines'
    )

    parser.add_argument(
        '--skip-loading',
        action='store_true',
        help='Skip data loading and only run retrieval/inference pipelines'
    )

    parser.add_argument(
        '--retrieval-type',
        type=str,
        required=True,
        choices=['monolingual', 'crosslingual', 'multilingual'],
        help='Specify the retrieval type'
    )

    parser.add_argument(
        "--span-type", required=True,
        choices=["xor_english_span", "xor_full"],
        help=(
            "xor_english_span for generating answers in English"
            "xor_full for generating answers in target language"
        ),
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=["CohereLabs/aya-101", "google/gemma-3-27b-it", "Qwen/Qwen3-30B-A3B"],
        help="Inference model to use",
    )

    args = parser.parse_args()

    try:
        xor = DenseRAG()
        asyncio.run(xor.main(
            model_name=args.model_name,
            skip_retrieval=args.skip_retrieval,
            skip_loading=args.skip_loading,
            retrieval_type=args.retrieval_type,
            span_type=args.span_type,
        ))
        logger.info("All XOR tasks completed")
    except Exception as e:
        logger.error("Program crashed abruptly", exc_info=True)
        sys.exit(1)
