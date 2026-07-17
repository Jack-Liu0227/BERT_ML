from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd

from src.data_processing.strength_ood_common import PreparedSplit


SUPPORTED_PROTOCOLS = {
    "target_extrapolation",
    "loco",
    "random_cv_baseline",
    "sparse_x_single",
    "sparse_x_cluster",
    "sparse_y_single",
    "sparse_y_cluster",
}


def _safe_component(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def canonical_task_id(alloy_type: object, dataset_name: object, target_column: object) -> str:
    alloy = str(alloy_type).strip()
    if alloy.lower() in {"matbenchsteels", "matbench_steel", "matbench steel"}:
        alloy = "Matbench Steel"
    return "__".join(
        [_safe_component(alloy), _safe_component(dataset_name), _safe_component(target_column)]
    )


def sample_identity(frame: pd.DataFrame, target_column: str) -> list[tuple[int, float]]:
    if "ID" not in frame.columns:
        raise ValueError("Canonical OOD split requires an ID column.")
    if target_column not in frame.columns:
        raise ValueError(f"Canonical OOD split is missing target column: {target_column}")
    ids = pd.to_numeric(frame["ID"], errors="raise").astype("int64").tolist()
    values = pd.to_numeric(frame[target_column], errors="raise").round(10).tolist()
    return list(zip(ids, values))


def sample_identity_hash(frame: pd.DataFrame, target_column: str) -> str:
    payload = json.dumps(sample_identity(frame, target_column), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_artifact_path(raw_path: object, manifest_path: Path, root: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path.resolve()
    candidates = [Path.cwd() / path, root.parent.parent / path, manifest_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


@dataclass(frozen=True)
class CanonicalSplitEntry:
    task_id: str
    alloy_type: str
    dataset_name: str
    target_column: str
    protocol: str
    split_id: str
    train_file: Path
    test_file: Path
    summary_file: Path
    manifest_file: Path
    method_params: dict[str, Any]
    random_state: int
    train_label: str
    test_label: str
    split_summary: dict[str, Any]

    @property
    def key(self) -> tuple[str, str, str]:
        return self.task_id, self.protocol, self.split_id

    def load_train(self) -> pd.DataFrame:
        return pd.read_csv(self.train_file)

    def load_test(self) -> pd.DataFrame:
        return pd.read_csv(self.test_file)

    def load_prepared_split(self) -> PreparedSplit:
        return PreparedSplit(
            split_strategy=self.protocol,
            target_col=self.target_column,
            train_df=self.load_train(),
            test_df=self.load_test(),
            summary=dict(self.split_summary),
            train_label=self.train_label,
            test_label=self.test_label,
        )

    def identity_record(self) -> dict[str, Any]:
        train = self.load_train()
        test = self.load_test()
        return {
            "task_id": self.task_id,
            "protocol": self.protocol,
            "split_id": self.split_id,
            "target_column": self.target_column,
            "train_file": str(self.train_file),
            "test_file": str(self.test_file),
            "train_count": len(train),
            "test_count": len(test),
            "train_hash": sample_identity_hash(train, self.target_column),
            "test_hash": sample_identity_hash(test, self.target_column),
            "manifest_file": str(self.manifest_file),
        }


class CanonicalSplitRegistry:
    def __init__(self, root: Path, entries: Iterable[CanonicalSplitEntry]):
        self.root = root.resolve()
        self._entries: dict[tuple[str, str, str], CanonicalSplitEntry] = {}
        for entry in entries:
            if entry.key in self._entries:
                raise ValueError(f"Duplicate canonical split key: {entry.key}")
            self._entries[entry.key] = entry

    @classmethod
    def from_root(cls, root: str | Path) -> "CanonicalSplitRegistry":
        root_path = Path(root).resolve()
        entries: list[CanonicalSplitEntry] = []
        for manifest_path in root_path.rglob("split_manifest.json"):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            protocol = str(payload.get("split_strategy", ""))
            if protocol not in SUPPORTED_PROTOCOLS:
                continue
            alloy_type = str(payload.get("alloy_type", "unknown"))
            data_file = Path(str(payload.get("data_file", "dataset.csv")))
            dataset_name = data_file.stem
            target_column = str(payload["target_column"])
            task_id = canonical_task_id(alloy_type, dataset_name, target_column)
            raw_entries = payload.get("folds", []) if payload.get("is_multi_fold") else [payload["split"]]
            for raw_entry in raw_entries:
                fold_index = raw_entry.get("fold_index")
                split_id = f"fold_{int(fold_index)}" if fold_index is not None else "main"
                summary_file = _resolve_artifact_path(
                    raw_entry.get("summary_file", ""), manifest_path, root_path
                )
                split_summary = raw_entry.get("split_summary") or {}
                if not split_summary and summary_file.exists():
                    split_summary = json.loads(summary_file.read_text(encoding="utf-8"))
                entries.append(
                    CanonicalSplitEntry(
                        task_id=task_id,
                        alloy_type=alloy_type,
                        dataset_name=dataset_name,
                        target_column=target_column,
                        protocol=protocol,
                        split_id=split_id,
                        train_file=_resolve_artifact_path(
                            raw_entry["train_file"], manifest_path, root_path
                        ),
                        test_file=_resolve_artifact_path(
                            raw_entry["test_file"], manifest_path, root_path
                        ),
                        summary_file=summary_file,
                        manifest_file=manifest_path.resolve(),
                        method_params=dict(payload.get("method_params") or {}),
                        random_state=int(payload.get("random_state", 42)),
                        train_label=str(raw_entry.get("train_label", "train")),
                        test_label=str(raw_entry.get("test_label", "test")),
                        split_summary=dict(split_summary),
                    )
                )
        return cls(root_path, entries)

    def require(self, task_id: str, protocol: str, split_id: str) -> CanonicalSplitEntry:
        key = task_id, protocol, split_id
        if key not in self._entries:
            raise KeyError(f"No canonical split for key: {key}")
        return self._entries[key]

    def entries(self) -> list[CanonicalSplitEntry]:
        return [self._entries[key] for key in sorted(self._entries)]

    def __len__(self) -> int:
        return len(self._entries)
