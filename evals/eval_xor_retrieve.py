import jsonlines
import json, unicodedata, re
from statistics import mean
import argparse
from tqdm import tqdm
import nltk

_tokenizer_cache: dict = {}


def _tokenize_whitespace(text: str) -> list[str]:
    return text.split()

def _get_tokenizer(lang: str):
    if lang == "ja":
        try:
            import fugashi
            _tagger = fugashi.Tagger("-Owakati")

            def _ja(text):
                parsed = _tagger.parse(text)
                if isinstance(parsed, str):
                    return parsed.split()
                return [str(node) for node in parsed]

            return _ja, False
        except Exception:
            raise RuntimeError(
                "fugashi is required for Japanese tokenisation. "
                "Install it with: pip install fugashi unidic-lite"
            )

    if lang == "ko":
        try:
            from kiwipiepy import Kiwi
            _kiwi = Kiwi()
            def _ko(text):
                return [token.form for token in _kiwi.tokenize(text)]
            return _ko, False
        except Exception:
            raise RuntimeError(
                "kiwipiepy is required for Korean tokenisation. "
                "Install it with: pip install kiwipiepy"
            )

    if lang == "ar":
        try:
            from camel_tools.tokenizers.word import simple_word_tokenize
            return simple_word_tokenize, False
        except ImportError:
            try:
                return lambda t: nltk.word_tokenize(t, language="arabic"), False
            except Exception:
                return _tokenize_whitespace, False

    if lang in ("bn", "te"):
        try:
            from indicnlp.tokenize import indic_tokenize
            return lambda t: indic_tokenize.trivial_tokenize(t, lang), False
        except ImportError:
            return _tokenize_whitespace, False

    _lang_map = {"en": "english", "fi": "finnish", "ru": "russian"}
    nltk_lang = _lang_map.get(lang, "english")
    try:
        return lambda t: nltk.word_tokenize(t, language=nltk_lang), False
    except Exception:
        return _tokenize_whitespace, False


def get_tokenizer(lang: str):
    if lang not in _tokenizer_cache:
        _tokenizer_cache[lang] = _get_tokenizer(lang)
    return _tokenizer_cache[lang]


def build_search_string(ctx_texts: list[str], lang: str, max_token_num: int) -> str:
    tokenize_fn, char_level = get_tokenizer(lang)

    selected_texts = []
    budget_size = 0

    for ctx in ctx_texts:
        if budget_size >= max_token_num:
            break

        tokens = tokenize_fn(ctx)
        n_tokens = len(tokens)
        remaining = max_token_num - budget_size

        if n_tokens <= remaining:
            selected_texts.append(ctx)
            budget_size += n_tokens
        else:
            if char_level:
                selected_texts.append(ctx[:remaining])
            else:
                allowed_tokens = tokens[:remaining]
                char_pos = 0
                for tok in allowed_tokens:
                    idx = ctx.find(tok, char_pos)
                    if idx != -1:
                        char_pos = idx + len(tok)
                selected_texts.append(ctx[:char_pos] if char_pos > 0 else ctx[:remaining])
            budget_size = max_token_num
            break

    return " ".join(selected_texts)


def normalize(text: str) -> str:
    """NFKC normalization + lowercase. Handles full-width digits, composed chars, etc."""
    return unicodedata.normalize("NFKC", text).lower()


def answer_in_text(answer: str, search_str: str) -> bool:
    norm_answer = normalize(answer)
    norm_search = normalize(search_str)

    if norm_answer.isdigit():
        return bool(re.search(r'(?<!\d)' + re.escape(norm_answer) + r'(?!\d)', norm_search))

    return norm_answer in norm_search

def read_jsonlines(path: str) -> list[dict]:
    print(f"Loading examples from {path}")
    with jsonlines.open(path) as reader:
        return list(reader)


SUPPORTED_LANGS = {"ar", "bn", "fi", "ja", "ko", "ru", "te"}


def evaluate_top_k_hit(
    results: list[dict],
    gt_answers: dict,
    max_token_num: int = 5000,
) -> dict:
    per_lang: dict[str, dict] = {}

    for item in tqdm(results):
        q_id  = item["id"]
        lang  = item["lang"]

        per_lang.setdefault(lang, {"count": 0, "hit": 0})

        if q_id not in gt_answers:
            continue

        answers = gt_answers[q_id]

        span_answers = [a for a in answers if a not in ("yes", "no")]
        if not span_answers:
            continue

        per_lang[lang]["count"] += 1

        ctx_texts   = item["ctxs"]
        search_str  = build_search_string(ctx_texts, lang, max_token_num)

        if any(answer_in_text(ans, search_str) for ans in span_answers):
            per_lang[lang]["hit"] += 1

    return {lang: v for lang, v in per_lang.items() if v["count"] > 0}

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--max_token_num", type=int, default=5000)
    parser.add_argument(
        "--use-translated",
        required=True,
        choices=["true", "false", "both"],
        type=str.lower,
        help=(
            "true  = use answers_translated only\n"
            "false = use answers only\n"
            "both  = hit if answer found in either field"
        ),
    )
    args = parser.parse_args()

    with open(args.pred_file) as f:
        predictions = json.load(f)
    input_data  = read_jsonlines(args.data_file)
    if args.use_translated == "true":
        qid2answers = {item["id"]: item["answers_translated"] for item in input_data}
    elif args.use_translated == "false":
        qid2answers = {item["id"]: item["answers"] for item in input_data}
    elif args.use_translated == "both":
        qid2answers = {
            item["id"]: list(dict.fromkeys(
                item["answers"] + item.get("answers_translated", [])
            ))
            for item in input_data
        }

    lang_order = ["ar", "bn", "fi", "ja", "ko", "ru", "te"]

    for topk in [2, 5]:
        print(f"Evaluating R@{topk}kt  (max {topk*1000} tokens/chars)")

        results = evaluate_top_k_hit(predictions, qid2answers, topk * 1000)
        avg_scores = []

        for lang in lang_order:
            if lang not in results:
                continue
            count = results[lang]["count"]
            score = results[lang]["hit"] / count * 100
            avg_scores.append(score)
            print(f"{lang:>4}  ({count:>4} examples)  R@{topk}kt = {score:.1f}")

        if avg_scores:
            print(f"\nMacro avg over {len(avg_scores)} languages: {mean(avg_scores):.1f}")


if __name__ == "__main__":
    main()