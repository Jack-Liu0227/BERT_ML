from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.LLMProp.ood_adapter import convert_split_to_llmprop_csv
from src.LLMProp.trainer import LLMPropTrainingConfig, run_llmprop_ood_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LLM-Prop on prepared OOD split CSV files")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--target_column", required=True)
    parser.add_argument("--processing_text_column", default="Processing_Description")
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--ood_method", default=None)
    parser.add_argument("--fold_index", type=int, default=None)
    parser.add_argument("--base_model_path", default="models/llmprop/google_t5_v1_1_small")
    parser.add_argument(
        "--tokenizer_path",
        default="models/llmprop/tokenizers/t5_tokenizer_trained_on_modified_part_of_C4_and_textedge",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_len", type=int, default=888)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser


def _load_split_manifest(path: str | None) -> Dict[str, Any] | None:
    if not path:
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    args = build_parser().parse_args()
    result_dir = Path(args.result_dir)
    llmprop_data_dir = result_dir / "llmprop_data"
    train_csv = llmprop_data_dir / "train_llmprop.csv"
    test_csv = llmprop_data_dir / "test_llmprop.csv"

    train_conversion = convert_split_to_llmprop_csv(
        args.train_file,
        train_csv,
        args.target_column,
        processing_text_column=args.processing_text_column,
    )
    test_conversion = convert_split_to_llmprop_csv(
        args.test_file,
        test_csv,
        args.target_column,
        processing_text_column=args.processing_text_column,
    )

    split_manifest = _load_split_manifest(args.split_manifest)
    if split_manifest is not None:
        split_manifest = {
            **split_manifest,
            "llmprop_conversion": {
                "train": train_conversion,
                "test": test_conversion,
            },
        }

    run_llmprop_ood_training(
        LLMPropTrainingConfig(
            result_dir=str(result_dir),
            train_csv=str(train_csv),
            test_csv=str(test_csv),
            target_column=args.target_column,
            base_model_path=args.base_model_path,
            tokenizer_path=args.tokenizer_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_len=args.max_len,
            dropout=args.dropout,
            pooling=args.pooling,
            valid_ratio=args.valid_ratio,
            random_state=args.random_state,
            device=args.device,
            split_manifest=split_manifest,
            fold_index=args.fold_index,
            ood_method=args.ood_method,
        )
    )


if __name__ == "__main__":
    main()
