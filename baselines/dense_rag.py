import argparse, signal, glob
import os, json, logging, sys
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
from transformers import AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

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
                                logger.info(f"[{wiki}] Starting embedding generation for Batch {num_batch} ({articles_processed:,} articles)")
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
                                    logger.info(f"[{wiki}] Shutdown requested, stopping generator after batch {num_batch}")
                                    return
            logger.info(f"[{wiki}] Completed {wiki_file} ({articles_processed:,} total articles)")
        if texts:
            num_batch += 1
            torch.cuda.empty_cache()
            logger.info(f"[{wiki}] Starting embedding generation for final Batch {num_batch} ({articles_processed:,} articles)")
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
        output_dir = f"{retrieval_type}_output"
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
                search_result = self.retrieve(query_embedding)

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

    def retrieve(self, embedding):
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=15,
            timeout=1200,
        )
        return search_result

    def chat_llm(self, name):
        batch_count = 0
        logger.info(f"[{name}] Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(name)
        logger.info(f"[{name}] Tokenizer loaded successfully")

        logger.info(f"[{name}] Loading pipeline")
        pipe = pipeline(
            "text-generation",
            model=name,
            tokenizer=tokenizer,
            max_new_tokens=512,
            return_full_text=False,
            device_map="auto",
        )
        logger.info(f"[{name}] Pipeline created successfully")

        chat_model = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe), tokenizer=tokenizer)
        logger.info(f"[{name}] LLM loaded successfully")

        input_file = ""

        for item in self.read_file(input_file, skip_count=0):
            if self.shutdown_requested:
                break

    async def main(self, skip_loading=False, retrieval_type="multilingual"):
        try:
            if skip_loading:
                logger.info("Skipping data loading. Starting retrieval pipelines")
            else:
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

            logger.info("Starting LLM inference")

            # self.chat_llm("CohereLabs/aya-101"),
            # self.chat_llm("google/gemma-3-27b-it"),
            # self.chat_llm("Qwen/Qwen3-30B-A3B"),

        except Exception as e:
            logger.error("Pipeline failed with exception", exc_info=True)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dense RAG Wikipedia Processing')

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


    args = parser.parse_args()

    try:
        xor = DenseRAG()
        asyncio.run(xor.main(skip_loading=args.skip_loading))
        logger.info("All XOR tasks completed")
    except Exception as e:
        logger.error("Program crashed abruptly", exc_info=True)
        sys.exit(1)
