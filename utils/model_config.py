from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Any
from prompts import FEW_SHOT_EN, _FEW_SHOT_LANG, _CTX_HEADER
import torch, logging, sys
from transformers import AutoTokenizer, AutoProcessor, AutoModelForSeq2SeqLM, Gemma3ForConditionalGeneration, AutoModelForCausalLM

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

ModelType    = Literal["seq2seq", "causal"]
PipelineType = Literal["text2text-generation", "text-generation"]

@dataclass
class ModelConfig:
    model_id: str
    model_type: ModelType
    uses_chat_template: bool
    system_prompt: str
    pipeline_type: PipelineType
    max_new_tokens: int
    num_beams: int
    uses_processor: bool = False
    enable_thinking: bool = False  # Qwen3 only
    strip_thinking: bool = False  # Qwen3 only
    do_sample: bool = False

MODEL_REGISTRY: dict[str, ModelConfig] = {

    "CohereLabs/aya-101": ModelConfig(
        model_id="CohereLabs/aya-101",
        model_type="seq2seq",
        uses_chat_template=False,
        system_prompt="",
        pipeline_type="text2text-generation",
        max_new_tokens=64,
        num_beams=4,
    ),

    "google/gemma-3-27b-it": ModelConfig(
        model_id="google/gemma-3-27b-it",
        model_type="causal",
        uses_processor=True,
        uses_chat_template=True,
        system_prompt=(
            "You are a concise multilingual question-answering assistant. "
            "Answer questions with the shortest possible factual response — "
            "a single phrase or a few words. No sentences, no explanations."
        ),
        pipeline_type="text-generation",
        max_new_tokens=64,
        num_beams=1,
    ),

    "Qwen/Qwen3-30B-A3B": ModelConfig(
        model_id="Qwen/Qwen3-30B-A3B",
        model_type="causal",
        uses_chat_template=True,
        system_prompt=(
            "You are a concise multilingual question-answering assistant. "
            "Answer with the shortest possible factual response — "
            "a phrase or a few words only. No sentences, no explanations. /no_think"
        ),
        enable_thinking=False,
        strip_thinking=True,
        pipeline_type="text-generation",
        max_new_tokens=64,
        num_beams=1,
    ),
}
QWEN_END_THINK_TOKEN_ID: int = 151668

def get_model_config(model_id: str) -> ModelConfig:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError:
        supported = "\n  ".join(MODEL_REGISTRY.keys())
        raise KeyError(
            f"Model '{model_id}' is not in MODEL_REGISTRY.\n"
            f"Supported models:\n  {supported}"
        ) from None


def load_model(cfg: ModelConfig):

    logger.info(f"[{cfg.model_id}] Loading tokenizer/processor")

    if cfg.uses_processor:
        tok = AutoProcessor.from_pretrained(cfg.model_id)
    else:
        tok = AutoTokenizer.from_pretrained(cfg.model_id)

    logger.info(f"[{cfg.model_id}] Loading model (type={cfg.model_type})")

    if cfg.model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif cfg.model_id.startswith("google/gemma"):
        model = Gemma3ForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype="auto",
            device_map="auto",
        )

    model.eval()
    logger.info(f"[{cfg.model_id}] Model ready")
    return tok, model


def _select_few_shot(lang: str, span_type: str) -> str | None | Any:
    if span_type == "xor_english_span":
        return FEW_SHOT_EN
    return _FEW_SHOT_LANG.get(lang, FEW_SHOT_EN)


def build_raw_prompt(
    item: dict,
    span_type: str,
) -> str:

    lang = item.get("lang", "en")
    question = item.get("question", "")
    ctxs = item.get("ctxs", [])

    ctx_block = ""
    if ctxs:
        body      = "\n\n".join(f"[{i+1}] {c.strip()}" for i, c in enumerate(ctxs) if c.strip())
        ctx_block = _CTX_HEADER.format(context=body)

    return ctx_block + _select_few_shot(lang, span_type).format(question=question)


def build_chat_messages(
    item: dict,
    span_type: str,
    cfg: ModelConfig,
) -> list[dict]:

    user_text = build_raw_prompt(item, span_type)

    # Gemma's processor expects content as a list of typed dicts.
    # Qwen (and other causal models) accept plain strings.
    if cfg.uses_processor:
        def _wrap(text: str) -> list[dict]:
            return [{"type": "text", "text": text}]

        messages: list[dict] = []
        if cfg.system_prompt:
            messages.append({"role": "system",  "content": _wrap(cfg.system_prompt)})
        messages.append(    {"role": "user",    "content": _wrap(user_text)})
    else:
        messages = []
        if cfg.system_prompt:
            messages.append({"role": "system",  "content": cfg.system_prompt})
        messages.append(    {"role": "user",    "content": user_text})

    return messages


def build_prompt(item, span_type, model_id):
    cfg = get_model_config(model_id)
    if not cfg.uses_chat_template:
        return build_raw_prompt(item, span_type)
    return build_chat_messages(item, span_type, cfg)

def _infer_seq2seq(
        tok,
        model,
        cfg: ModelConfig,
        prompts: list[str],
        batch_size: int,
) -> list[str]:
    results: list[str] = []
    pad_id = tok.pad_token_id

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start: start + batch_size]

        enc = tok(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.inference_mode():
            out_ids = model.generate(
                **enc,
                max_new_tokens=cfg.max_new_tokens,
                num_beams=cfg.num_beams,
                do_sample=cfg.do_sample,
                pad_token_id=pad_id,
            )

        for ids in out_ids:
            results.append(tok.decode(ids, skip_special_tokens=True).strip())

    return results

def _infer_causal(
        tok,
        model,
        cfg: ModelConfig,
        messages_list: list[list[dict]],  # one messages list per item
        batch_size: int,
) -> list[str]:

    results: list[str] = []
    pad_id = tok.tokenizer.pad_token_id if cfg.uses_processor else tok.pad_token_id

    tmpl_kwargs: dict = dict(
        tokenize=False,
        add_generation_prompt=True,
    )
    if cfg.model_id.startswith("Qwen"):
        tmpl_kwargs["enable_thinking"] = cfg.enable_thinking

    for start in range(0, len(messages_list), batch_size):
        chunk = messages_list[start: start + batch_size]

        rendered = [
            tok.apply_chat_template(msgs, **tmpl_kwargs)
            for msgs in chunk
        ]

        enc = tok(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            padding_side="left",
        ).to(model.device, dtype=torch.bfloat16)

        prompt_len = enc["input_ids"].shape[1]

        with torch.inference_mode():
            out_ids = model.generate(
                **enc,
                max_new_tokens=cfg.max_new_tokens,
                num_beams=cfg.num_beams,
                do_sample=cfg.do_sample,
                pad_token_id=pad_id,
            )

        for gen_row in out_ids:
            new_tokens = gen_row[prompt_len:].tolist()

            if cfg.strip_thinking:
                try:
                    idx = len(new_tokens) - new_tokens[::-1].index(QWEN_END_THINK_TOKEN_ID)
                    new_tokens = new_tokens[idx:]
                except ValueError:
                    pass
            decoded = tok.decode(new_tokens, skip_special_tokens=True).strip()
            results.append(decoded)

    return results

def run_batch_sync(
        tok,
        model,
        cfg: ModelConfig,
        prompts: list,
        batch_size: int,
) -> list[str]:

    if cfg.model_type == "seq2seq":
        return _infer_seq2seq(tok, model, cfg, prompts, batch_size)
    return _infer_causal(tok, model, cfg, prompts, batch_size)