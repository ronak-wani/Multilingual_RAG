from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
from prompts import FEW_SHOT_EN, _FEW_SHOT_LANG, _CTX_HEADER

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

def _select_few_shot(lang: str, retrieval_type: str) -> str:
    if retrieval_type == "crosslingual":
        return FEW_SHOT_EN
    return _FEW_SHOT_LANG.get(lang, FEW_SHOT_EN)


def build_raw_prompt(
    item: dict,
    retrieval_type: str,
) -> str:

    lang = item.get("lang", "en")
    question = item.get("question", "")
    ctxs = item.get("ctxs", [])

    ctx_block = ""
    if ctxs:
        body      = "\n\n".join(f"[{i+1}] {c.strip()}" for i, c in enumerate(ctxs) if c.strip())
        ctx_block = _CTX_HEADER.format(context=body)

    return ctx_block + _select_few_shot(lang, retrieval_type).format(question=question)


def build_chat_messages(
    item: dict,
    retrieval_type: str,
    cfg: ModelConfig,
) -> list[dict]:

    user_text     = build_raw_prompt(item, retrieval_type)

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


def build_prompt(
    item: dict,
    retrieval_type: str,
    model_id: str,
) -> str | list[dict]:
    cfg = get_model_config(model_id)
    if not cfg.uses_chat_template:
        return build_raw_prompt(item, retrieval_type)
    return build_chat_messages(item, retrieval_type, cfg)