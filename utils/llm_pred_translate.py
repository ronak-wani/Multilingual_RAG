"""
translate.py
------------
Translates the `prediction` field in _predictions.jsonl files to English
using google/translategemma-27b-it.

For each input *_predictions.jsonl the script produces two output files:

  *_predictions_translated.jsonl   — full records with prediction_translated
  *_predictions_translated_eval.json — {id: prediction_translated} mapping

Directory structure is mirrored under --output-dir:

  Input:  <input_dir>/Qwen_Qwen3-30B-A3B/use_translated/xor_dev_full_v1_1_results_predictions.jsonl
  Output: <output_dir>/Qwen_Qwen3-30B-A3B/use_translated/xor_dev_full_v1_1_results_predictions_translated.jsonl
          <output_dir>/Qwen_Qwen3-30B-A3B/use_translated/xor_dev_full_v1_1_results_predictions_translated_eval.json

Rules
-----
• If lingua detects the prediction is already English → copy as-is into
  prediction_translated, set translation_skipped=True.  TranslateGemma is
  NOT called.
• Otherwise → translate with TranslateGemma; set translation_skipped=False.
• The original `prediction` field is always preserved unchanged.
• Checkpoint/resume via a sidecar .ckpt.json next to each output file.
• Graceful SIGTERM / SIGINT: flushes the current in-flight batch then exits.
"""

from __future__ import annotations
 
import argparse, asyncio, json, re, logging,os
import signal, sys, time
from pathlib import Path
from typing import AsyncIterator
import aiofiles, torch
from lingua import Language, LanguageDetectorBuilder
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
 
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("translate.log"),
    ],
)
logger = logging.getLogger(__name__)
 
MODEL_ID      = "google/translategemma-27b-it"
CKPT_SUFFIX   = ".ckpt.json"
CKPT_INTERVAL = 200
DEFAULT_BATCH = 64
 
class PredictionTranslator:
 
    def __init__(self, batch_size: int = DEFAULT_BATCH):
        self.batch_size = batch_size
        self._shutdown_requested = False
 
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT,  self._handle_signal)
 
        logger.info("Building lingua language detector …")
        self._detector = (
            LanguageDetectorBuilder
            .from_all_languages()
            .with_preloaded_language_models()
            .build()
        )
 
        logger.info("Loading TranslateGemma processor …")
        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
 
        logger.info("Loading TranslateGemma model (bfloat16, device_map=auto) …")
        self._model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self._model.eval()
        torch.set_grad_enabled(False)
        logger.info("Model ready.")
  
    def _handle_signal(self, signum, frame) -> None:
        logger.warning(
            "Received %s — will stop after current batch.",
            signal.Signals(signum).name,
        )
        self._shutdown_requested = True
  
    def _is_english(self, text: str) -> bool:
        if not text or not text.strip():
            return True
        return self._detector.detect_language_of(text) is Language.ENGLISH
  
    def _translate_batch(
        self,
        texts: list[str],
        src_langs: list[str],
    ) -> list[str]:
        if not texts:
            return []
 
        prompt_strings: list[str] = []
        for text, src_lang in zip(texts, src_langs):
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": src_lang,
                            "target_lang_code": "en",
                            "text": text,
                        }
                    ],
                }
            ]
            prompt_strings.append(
                self._processor.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
 
        inputs = self._processor(
            text=prompt_strings,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(self._model.device, dtype=torch.bfloat16)
 
        prompt_len = inputs["input_ids"].shape[1]
 
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=512,
            )
 
        results: list[str] = []
        for i in range(outputs.shape[0]):
            generated = outputs[i, prompt_len:]
            decoded   = self._processor.decode(generated, skip_special_tokens=True)
            results.append(decoded.strip())
 
        return results
  
    @staticmethod
    def _ckpt_path(output_path: Path) -> Path:
        return output_path.with_name(output_path.name + CKPT_SUFFIX)
 
    def _load_checkpoint(self, output_path: Path) -> set[str]:
        ckpt = self._ckpt_path(output_path)
        if not ckpt.exists():
            return set()
        try:
            data = json.loads(ckpt.read_text(encoding="utf-8"))
            ids: set[str] = set(data.get("processed_ids", []))
            logger.info("Checkpoint — %d records already done.  Resuming.", len(ids))
            return ids
        except Exception as exc:
            logger.warning("Could not read checkpoint %s: %s — fresh start.", ckpt, exc)
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
  
    async def _write_eval_json(self, translated_jsonl: Path) -> None:
        """
        Read the completed *_translated.jsonl and write a companion
        *_translated_eval.json with the format:
 
            { "<id>": "<prediction_translated>", ... }
        """
        eval_path = translated_jsonl.with_name(
            translated_jsonl.name.replace("_translated.jsonl", "_translated_eval.json")
        )
 
        logger.info("Writing eval JSON → %s", eval_path)
 
        mapping: dict[str, str] = {}
        async with aiofiles.open(translated_jsonl, "r", encoding="utf-8") as fh:
            async for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                    mapping[str(rec.get("id", ""))] = rec.get("prediction_translated", "")
                except json.JSONDecodeError:
                    continue
 
        tmp = eval_path.with_suffix(".tmp")
        async with aiofiles.open(tmp, "w", encoding="utf-8") as fout:
            await fout.write(json.dumps(mapping, ensure_ascii=False, indent=2))
        tmp.replace(eval_path)
 
        logger.info("Eval JSON written — %d entries → %s", len(mapping), eval_path)
  
    @staticmethod
    def count_lines(path: Path) -> int:
        n = 0
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                n += chunk.count(b"\n")
        return n
 
    @staticmethod
    async def _iter_jsonl(path: Path) -> AsyncIterator[dict]:
        async with aiofiles.open(path, "r", encoding="utf-8") as fh:
            async for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line in %s: %s", path, exc)
  
    async def _flush_batch(self, records: list[dict], fout) -> int:
        """
        Partition records into already-English vs needs-translation.
        Returns number of records written.
        """
        if not records:
            return 0
 
        need_idx:  list[int] = []
        need_text: list[str] = []
        need_lang: list[str] = []
 
        for idx, rec in enumerate(records):
            if not self._is_english(rec.get("prediction") or ""):
                need_idx.append(idx)
                need_text.append(rec.get("prediction") or "")
                need_lang.append(rec.get("lang", "auto"))
 
        translations: dict[int, str] = {}
        if need_text:
            translated = await asyncio.get_event_loop().run_in_executor(
                None,
                self._translate_batch,
                need_text,
                need_lang,
            )
            translations = dict(zip(need_idx, translated))
 
        n = 0
        for idx, rec in enumerate(records):
            out_rec = dict(rec)
            if idx in translations:
                out_rec["prediction_translated"] = translations[idx]
                out_rec["translation_skipped"]   = False
            else:
                out_rec["prediction_translated"] = rec.get("prediction") or ""
                out_rec["translation_skipped"]   = True
 
            await fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n += 1
 
        await fout.flush()
        return n
  
    async def translate_file(self, input_path: Path, output_path: Path) -> None:
        """
        Translate one predictions file, then write its companion eval JSON.
        Resumes from checkpoint if one exists.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
 
        processed_ids = self._load_checkpoint(output_path)
        total_lines   = self.count_lines(input_path)
        file_mode     = "a" if processed_ids else "w"
        written       = len(processed_ids)
 
        logger.info(
            "=== %s → %s  [%d lines, %d already done] ===",
            input_path, output_path, total_lines, written,
        )
 
        pending: list[dict] = []
 
        async with aiofiles.open(output_path, file_mode, encoding="utf-8") as fout:
            pbar = tqdm(
                total=total_lines, initial=written, unit="rec",
                desc=input_path.name, dynamic_ncols=True,
            )
 
            async for record in self._iter_jsonl(input_path):
                record_id = str(record.get("id", ""))
 
                if record_id in processed_ids:
                    pbar.update(1)
                    continue
 
                if self._shutdown_requested:
                    logger.warning(
                        "Shutdown — flushing %d in-flight records then exiting.",
                        len(pending),
                    )
                    n = await self._flush_batch(pending, fout)
                    written += n
                    pbar.update(n)
                    for r in pending:
                        processed_ids.add(str(r.get("id", "")))
                    self._save_checkpoint(output_path, processed_ids)
                    pbar.close()
                    logger.info("Graceful shutdown — saved %d records total.", written)
                    return
 
                pending.append(record)
 
                if len(pending) >= self.batch_size:
                    n = await self._flush_batch(pending, fout)
                    written += n
                    pbar.update(n)
                    for r in pending:
                        processed_ids.add(str(r.get("id", "")))
                    pending = []
                    if written % CKPT_INTERVAL == 0:
                        self._save_checkpoint(output_path, processed_ids)
 
            if pending:
                n = await self._flush_batch(pending, fout)
                written += n
                pbar.update(n)
                for r in pending:
                    processed_ids.add(str(r.get("id", "")))
 
            pbar.close()
 
        self._save_checkpoint(output_path, processed_ids)
        logger.info("Done — %d records written to %s", written, output_path)
 
        if not self._shutdown_requested:
            self._delete_checkpoint(output_path)
            await self._write_eval_json(output_path)
  
    async def run(self, input_dir: Path, output_dir: Path) -> None:
        """
        Recursively finds all *_predictions.jsonl files under `input_dir`,
        mirrors their relative path under `output_dir` with a _translated suffix.
        """
        prediction_files = sorted(input_dir.rglob("*_predictions.jsonl"))
        if not prediction_files:
            logger.error("No *_predictions.jsonl files found under %s", input_dir)
            return
 
        logger.info("Found %d prediction file(s) to translate.", len(prediction_files))
        for p in prediction_files:
            logger.info("  %s", p.relative_to(input_dir))
 
        for input_path in prediction_files:
            if self._shutdown_requested:
                logger.warning("Shutdown requested — skipping remaining files.")
                break
 
            rel      = input_path.relative_to(input_dir)
            out_name = input_path.stem + "_translated.jsonl"
            out_path = output_dir / rel.parent / out_name
 
            await self.translate_file(input_path, out_path)
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate `prediction` fields to English.\n\n"
            "SINGLE-FILE MODE (use with SLURM arrays):\n"
            "  llm_pred_translate.py \\\n"
            "      --input-file  path/to/foo_predictions.jsonl \\\n"
            "      --output-file path/to/foo_predictions_translated.jsonl\n\n"
            "DIRECTORY MODE (sequential, original behaviour):\n"
            "  llm_pred_translate.py <input_dir> <output_dir>\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
 
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Single *_predictions.jsonl to translate (single-file mode).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Destination *_predictions_translated.jsonl (single-file mode). "
            "A companion *_translated_eval.json is written alongside it."
        ),
    )
 
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Input directory to recursively search for *_predictions.jsonl files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Output directory to write translated files (mirrors input structure).",
    )
 
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Records per TranslateGemma forward pass.  (default: {DEFAULT_BATCH})",
    )
 
    args = parser.parse_args()
 
    single_file_mode = args.input_file is not None or args.output_file is not None
    dir_mode         = args.input_dir  is not None or args.output_dir  is not None
 
    if single_file_mode and dir_mode:
        parser.error(
            "Specify either --input-file/--output-file OR positional input_dir/output_dir, not both."
        )
    if single_file_mode:
        if args.input_file is None or args.output_file is None:
            parser.error("--input-file and --output-file must both be provided together.")
    elif dir_mode:
        if args.input_dir is None or args.output_dir is None:
            parser.error("Both positional arguments input_dir and output_dir are required.")
    else:
        parser.error(
            "No inputs specified.  Use --input-file/--output-file (single-file mode) "
            "or provide input_dir output_dir (directory mode)."
        )
 
    return args
 
 
async def main() -> None:
    args = parse_args()
    translator = PredictionTranslator(batch_size=args.batch_size)
 
    if args.input_file is not None:
        if not args.input_file.exists():
            logger.error("input_file does not exist: %s", args.input_file)
            sys.exit(1)
        await translator.translate_file(args.input_file, args.output_file)
    else:
        if not args.input_dir.exists():
            logger.error("input_dir does not exist: %s", args.input_dir)
            sys.exit(1)
        await translator.run(args.input_dir, args.output_dir)
 
    logger.info("All files processed.")
 
 
if __name__ == "__main__":
    asyncio.run(main())