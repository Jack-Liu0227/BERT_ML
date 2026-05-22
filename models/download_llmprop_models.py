#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer, T5EncoderModel


REPO_RAW = "https://raw.githubusercontent.com/vertaix/LLM-Prop/main"
BASE_DIR = Path(__file__).resolve().parent
LLMPROP_DIR = BASE_DIR / "llmprop"
BASE_MODEL_ID = "google/t5-v1_1-small"
BASE_MODEL_DIR = LLMPROP_DIR / "google_t5_v1_1_small"
TOKENIZER_DIR = LLMPROP_DIR / "tokenizers" / "t5_tokenizer_trained_on_modified_part_of_C4_and_textedge"


def download_base_model(force: bool = False) -> None:
    required_model_exists = (
        (BASE_MODEL_DIR / "config.json").exists()
        and ((BASE_MODEL_DIR / "model.safetensors").exists() or (BASE_MODEL_DIR / "pytorch_model.bin").exists())
        and ((BASE_MODEL_DIR / "spiece.model").exists() or (BASE_MODEL_DIR / "tokenizer.json").exists())
    )
    if BASE_MODEL_DIR.exists() and required_model_exists and not force:
        print(f"[SKIP] Base model already exists: {BASE_MODEL_DIR}")
        return
    BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] {BASE_MODEL_ID} -> {BASE_MODEL_DIR}")
    # Force the slow SentencePiece tokenizer. Newer transformers versions try a
    # fast-tokenizer/tiktoken conversion that can require optional blobfile.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False)
    try:
        model = T5EncoderModel.from_pretrained(BASE_MODEL_ID)
    except Exception:
        from transformers.models.t5.modeling_t5 import T5EncoderModel as ConcreteT5EncoderModel

        model = ConcreteT5EncoderModel.from_pretrained(BASE_MODEL_ID)
    tokenizer.save_pretrained(BASE_MODEL_DIR)
    model.save_pretrained(BASE_MODEL_DIR, safe_serialization=True)


def try_download_modified_tokenizer(force: bool = False) -> None:
    if TOKENIZER_DIR.exists() and any(TOKENIZER_DIR.iterdir()) and not force:
        print(f"[SKIP] Modified tokenizer already exists: {TOKENIZER_DIR}")
        return
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    # GitHub directory downloads are intentionally best-effort.  If the
    # upstream tokenizer layout changes or a rate limit occurs, fall back to
    # t5-small tokenizer files so the training code remains runnable.
    candidate_files = [
        "special_tokens_map.json",
        "tokenizer_config.json",
        "spiece.model",
        "tokenizer.json",
        "added_tokens.json",
    ]
    downloaded = 0
    for filename in candidate_files:
        url = f"{REPO_RAW}/tokenizers/t5_tokenizer_trained_on_modified_part_of_C4_and_textedge/{filename}"
        try:
            print(f"[TRY] {url}")
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            (TOKENIZER_DIR / filename).write_bytes(data)
            downloaded += 1
        except Exception as exc:
            print(f"  [WARN] Could not download {filename}: {exc}")

    if downloaded == 0:
        print("[FALLBACK] Saving t5-small tokenizer as LLM-Prop tokenizer fallback")
        if TOKENIZER_DIR.exists():
            shutil.rmtree(TOKENIZER_DIR)
        TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
        AutoTokenizer.from_pretrained("t5-small", use_fast=False).save_pretrained(TOKENIZER_DIR)
    elif not (TOKENIZER_DIR / "spiece.model").exists() and (BASE_MODEL_DIR / "spiece.model").exists():
        # Upstream modified tokenizer may publish tokenizer.json only.  The
        # training code intentionally uses the slow SentencePiece tokenizer, so
        # provide a local spiece.model fallback to keep the path loadable.
        print("[FALLBACK] Copying base T5 spiece.model into modified tokenizer directory")
        shutil.copy2(BASE_MODEL_DIR / "spiece.model", TOKENIZER_DIR / "spiece.model")


def write_manifest(sample_checkpoint: bool = False) -> None:
    manifest = {
        "base_model_id": BASE_MODEL_ID,
        "base_model_dir": str(BASE_MODEL_DIR),
        "tokenizer_dir": str(TOKENIZER_DIR),
        "sample_checkpoint_downloaded": bool(sample_checkpoint),
        "sample_only": bool(sample_checkpoint),
        "note": (
            "LLM-Prop upstream sample checkpoints are for pipeline testing only; "
            "formal alloy OOD runs fine-tune from the base T5 encoder."
        ),
    }
    LLMPROP_DIR.mkdir(parents=True, exist_ok=True)
    (LLMPROP_DIR / "llmprop_assets_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download assets for the local LLM-Prop OOD workflow")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--download_sample_checkpoint",
        action="store_true",
        help="Reserved for explicit sample-only checkpoint downloads; not used for formal OOD runs.",
    )
    args = parser.parse_args()

    download_base_model(force=args.force)
    try_download_modified_tokenizer(force=args.force)
    write_manifest(sample_checkpoint=args.download_sample_checkpoint)
    print("[DONE] LLM-Prop assets are ready")
    print(f"  base model: {BASE_MODEL_DIR}")
    print(f"  tokenizer:  {TOKENIZER_DIR}")


if __name__ == "__main__":
    main()
