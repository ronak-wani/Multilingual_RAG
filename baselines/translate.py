import argparse, json, logging, os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import signal, time
from collections import defaultdict
from pathlib import Path
from typing import Optional
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from lingua import Language, LanguageDetector, LanguageDetectorBuilder, IsoCode639_1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

class Translate:
    """
    Reads XOR benchmark JSONL files and enriches every record with
    "answers_translated" — the English answers rendered in the language
    of the original question — using google/translategemma-27b-it.

    Checkpoint file:
    A sidecar "<output>.ckpt.json" is written atomically after every
    buffer flush.  It stores the set of record IDs already written so that
    a requeued job can skip them and append to the existing output file.
    The checkpoint is deleted on successful completion of a file.

    Signal handling:

    SIGTERM (and SIGINT) set "self._shutdown_requested".  The processing
    loop checks this flag after every buffer flush, then flushes any
    partially filled buffer, writes the checkpoint, and returns — leaving
    the job in a state that SLURM --requeue can cleanly restart.
    """

    MODEL_ID    = "google/translategemma-27b-it"
    SOURCE_LANG = "en"
    OUTPUT_DIR  = Path("translated_benchmark_files")
    CKPT_SUFFIX = ".ckpt.json"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device_map: str = "auto",
        batch_size: int = 64,
        direction: str = "auto",
    ) -> None:

        self.batch_size = batch_size
        self.direction = direction
        self._shutdown_requested = False
        self._detector_cache: dict[str, LanguageDetector] = {}

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT,  self._handle_signal)
        logger.info("Signal handlers registered (SIGTERM, SIGINT).")

        logger.info("Loading processor from %s", model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"

        logger.info("Loading model from %s  (device_map=%s)", model_id, device_map)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self.model = torch.compile(self.model, mode="reduce-overhead")
        logger.info("Model ready  (batch_size=%d).", batch_size)


    def _handle_signal(self, signum: int, frame) -> None:
        name = signal.Signals(signum).name
        logger.warning(
            "%s received — will checkpoint after the current buffer and exit.",
            name,
        )
        self._shutdown_requested = True

    def _get_detector(self, lang_code: str):
        """Return a cached 2-language detector for (ENGLISH, lang_code)."""
        if lang_code in self._detector_cache:
            return self._detector_cache[lang_code]
        try:
            iso_code = getattr(IsoCode639_1, lang_code.upper())
            target = Language.from_iso_code_639_1(iso_code)
        except KeyError:
            # lang_code not recognised by lingua — fall back to all-languages
            detector = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()
            self._detector_cache[lang_code] = detector
            return detector

        detector = (
            LanguageDetectorBuilder
            .from_languages(Language.ENGLISH, target)
            .build()
        )
        self._detector_cache[lang_code] = detector
        return detector

    def _detect_lang(self, answer: str, record_lang: str) -> tuple[str, str]:
        """
        Decide the translation direction for a single answer string.
        """
        if self.direction == "en-to-target":
            return self.SOURCE_LANG, record_lang
        if self.direction == "target-to-en":
            return record_lang, self.SOURCE_LANG

        detector = self._get_detector(record_lang)
        detected = detector.detect_language_of(answer)

        if detected is None or detected == Language.ENGLISH:
            return self.SOURCE_LANG, record_lang
        return record_lang, self.SOURCE_LANG

    def count_lines(self, path: Path) -> int:
        """Fast byte-level line count for tqdm totals."""
        n = 0
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                n += chunk.count(b"\n")
        return n

    def _ckpt_path(self, output_path: Path) -> Path:
        return output_path.with_name(output_path.name + self.CKPT_SUFFIX)

    def _load_checkpoint(self, output_path: Path) -> set[str]:
        ckpt = self._ckpt_path(output_path)
        if not ckpt.exists():
            return set()
        try:
            data = json.loads(ckpt.read_text(encoding="utf-8"))
            ids: set[str] = set(data.get("processed_ids", []))
            logger.info(
                "Checkpoint found — %d records already done.  Resuming.", len(ids)
            )
            return ids
        except Exception as exc:
            logger.warning(
                "Could not read checkpoint %s: %s — starting fresh.", ckpt, exc
            )
            return set()

    def _save_checkpoint(self, output_path: Path, processed_ids: set[str]) -> None:
        ckpt = self._ckpt_path(output_path)
        payload = {
            "output_path":     str(output_path),
            "processed_ids":   list(processed_ids),
            "processed_count": len(processed_ids),
            "saved_at":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        tmp = ckpt.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(ckpt)

    def _delete_checkpoint(self, output_path: Path) -> None:
        ckpt = self._ckpt_path(output_path)
        if ckpt.exists():
            ckpt.unlink()
            logger.info("Checkpoint deleted (file complete).")

    def translate_batch(
            self,
            texts: list[str],
            source_lang_code: str,
            target_lang_code: str,
    ) -> list[str]:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        sorted_texts = [texts[i] for i in order]

        pad_id = self.processor.tokenizer.pad_token_id
        input_id_list: list[torch.Tensor] = []

        for text in sorted_texts:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": source_lang_code,
                            "target_lang_code": target_lang_code,
                            "text": text,
                        }
                    ],
                }
            ]
            enc = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_id_list.append(enc["input_ids"][0])

        max_len        = max(t.shape[0] for t in input_id_list)
        padded_ids     = torch.full((len(sorted_texts), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sorted_texts), max_len), dtype=torch.long)

        for i, ids in enumerate(input_id_list):
            offset = max_len - ids.shape[0]
            padded_ids[i, offset:] = ids
            attention_mask[i, offset:] = 1

        padded_ids = padded_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        with torch.inference_mode():
            generation = self.model.generate(
                input_ids=padded_ids,
                attention_mask=attention_mask,
                do_sample=False,
                pad_token_id=pad_id,
            )

        sorted_results: list[str] = []
        for gen_row in generation:
            new_tokens = gen_row[max_len:]
            decoded = self.processor.decode(
                new_tokens, skip_special_tokens=True
            ).strip()
            sorted_results.append(decoded)

        results: list[str] = [""] * len(texts)
        for sorted_idx, orig_idx in enumerate(order):
            results[orig_idx] = sorted_results[sorted_idx]

        return results

    def _flush_buffer(
        self,
        buf: list[dict],
        fout,
        processed_ids: set[str],
        output_path: Path,
        pbar: tqdm,
    ) -> int:

        if not buf:
            return 0

        direction_to_jobs: dict[
            tuple[str, str], list[tuple[int, int, str]]
        ] = defaultdict(list)

        for ri, rec in enumerate(buf):
            record_lang = rec.get("lang", self.SOURCE_LANG)
            for ai, ans in enumerate(rec.get("answers", [])):
                if not ans.strip():
                    continue
                src, tgt = self._detect_lang(ans, record_lang)
                direction_to_jobs[(src, tgt)].append((ri, ai, ans))

        translated: dict[int, list[str]] = {
            ri: list(rec.get("answers", []))
            for ri, rec in enumerate(buf)
        }

        for (src, tgt), jobs in direction_to_jobs.items():
            texts = [text for _, _, text in jobs]
            try:
                results = self.translate_batch(texts, src, tgt)
                for (ri, ai, orig), result in zip(jobs, results):
                    translated[ri][ai] = result
                    logger.debug(
                        "[%s→%s] '%s' → '%s'", src, tgt, orig, result
                    )
            except Exception as exc:
                logger.warning(
                    "translate_batch failed (%s→%s, %d texts): %s — keeping originals.",
                    src, tgt, len(texts), exc,
                )

        for ri, rec in enumerate(buf):
            rec["answers_translated"] = translated[ri]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rec_id = str(rec.get("id", ""))
            if rec_id:
                processed_ids.add(rec_id)

        fout.flush()
        os.fsync(fout.fileno())
        self._save_checkpoint(output_path, processed_ids)
        pbar.update(len(buf))
        return len(buf)

    def process_file(
        self,
        input_path: str | Path,
        output_path: Optional[str | Path] = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Translate all answers in *input_path* and write an enriched JSONL
        to *output_path*.

        On the first run the output file is created fresh.  If a checkpoint
        exists from a previous interrupted run, the output file is opened in
        append mode and already-processed records are skipped.

        Args:
            input_path:  Source ``.jsonl`` file.
            output_path: Destination path.  Defaults to
                         ``translated_benchmark_files/<input filename>``.
            overwrite:   Delete any existing output *and* checkpoint and
                         start from scratch.

        Returns:
            Resolved output path.
        """
        input_path = Path(input_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            output_path = self.OUTPUT_DIR / input_path.name
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if overwrite:
            if output_path.exists():
                output_path.unlink()
            self._delete_checkpoint(output_path)

        if (
            output_path.exists()
            and not overwrite
            and not self._ckpt_path(output_path).exists()
        ):
            raise FileExistsError(
                f"Output already exists: {output_path}. "
                "Pass --overwrite to replace it, or delete it manually."
            )

        processed_ids: set[str] = self._load_checkpoint(output_path)
        resuming    = len(processed_ids) > 0
        write_mode  = "a" if resuming else "w"
        total_lines = self.count_lines(input_path)

        logger.info("Input: %s", input_path)
        logger.info("Output: %s  (mode=%s)", output_path, write_mode)
        logger.info("Batch size: %d", self.batch_size)
        if resuming:
            logger.info("Resuming: %d / ~%d records already done.", len(processed_ids), total_lines)

        processed = skipped = parse_errors = 0
        record_buffer: list[dict] = []

        with (
            input_path.open("r", encoding="utf-8") as fin,
            output_path.open(write_mode, encoding="utf-8") as fout,
            tqdm(
                total=total_lines,
                initial=len(processed_ids),
                desc=input_path.name,
                unit="rec",
                dynamic_ncols=True,
            ) as pbar,
        ):
            for line_no, raw_line in enumerate(fin, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    logger.warning("Line %d — JSON error: %s", line_no, exc)
                    parse_errors += 1
                    pbar.update(1)
                    continue

                rec_id = str(record.get("id", ""))
                if rec_id and rec_id in processed_ids:
                    skipped += 1
                    continue

                record_buffer.append(record)

                if len(record_buffer) >= self.batch_size:
                    processed += self._flush_buffer(
                        record_buffer, fout, processed_ids, output_path, pbar
                    )
                    record_buffer = []

                if self._shutdown_requested:
                    logger.warning(
                        "Shutdown — flushing remaining %d records.", len(record_buffer)
                    )
                    if record_buffer:
                        processed += self._flush_buffer(
                            record_buffer, fout, processed_ids, output_path, pbar
                        )
                    logger.info(
                        "Checkpoint saved.  Rerun the same command to resume "
                        "(%d of ~%d records done).",
                        len(processed_ids), total_lines,
                    )
                    return output_path

            if record_buffer:
                processed += self._flush_buffer(
                    record_buffer, fout, processed_ids, output_path, pbar
                )

        self._delete_checkpoint(output_path)
        logger.info(
            "Done.  Processed=%d  Skipped(resume)=%d  ParseErrors=%d",
            processed, skipped, parse_errors,
        )
        return output_path

    def process_files(
        self,
        input_paths: list[str | Path],
        output_dir: Optional[str | Path] = None,
        overwrite: bool = False,
    ) -> list[Path]:

        out_dir = Path(output_dir) if output_dir else self.OUTPUT_DIR
        results: list[Path] = []

        for src in input_paths:
            if self._shutdown_requested:
                logger.warning("Shutdown requested — skipping remaining input files.")
                break
            out = out_dir / Path(src).name
            results.append(self.process_file(src, out, overwrite=overwrite))

        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Translate XOR benchmark answers to the question language using TranslateGemma"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="XOR benchmark .jsonl files to process.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=str(Translate.OUTPUT_DIR),
        help="Directory for output files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard existing output and checkpoint; start from scratch.",
    )

    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Max answer strings per model.generate() call.",
    )


    parser.add_argument(
        "--direction", "-d",
        choices=["auto", "en-to-target", "target-to-en"],
        default="auto",
        help=(
            "Translation direction. "
            "'en-to-target' always translates from English to the record language, "
            "'target-to-en' always translates to English, "
            "'auto' detects per-answer using lingua language detection (default)."
        ),
    )

    args = parser.parse_args()

    translator = Translate(batch_size=args.batch_size, direction=args.direction)

    output_paths = translator.process_files(
        input_paths=args.inputs,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )