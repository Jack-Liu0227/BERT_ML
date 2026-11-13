from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def build_parameter_hash(parameters: dict[str, Any]) -> str:
    canonical = json.dumps(
        _canonicalize(parameters),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class FrozenParameterRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as handle:
            self.payload = yaml.safe_load(handle) or {}
        self.snapshot_id = str(self.payload.get("snapshot_id", ""))
        self.key_fields = tuple(
            str(value)
            for value in self.payload.get(
                "key_fields",
                ("task_id", "protocol", "split_id", "model_family", "model"),
            )
        )
        self._entries = {
            self._registry_key(entry): entry
            for entry in self.payload.get("entries", [])
            if isinstance(entry, dict)
        }

    @property
    def is_task_model_registry(self) -> bool:
        return self.key_fields == ("task_id", "model_family", "model")

    def _registry_key(self, entry: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(entry[name]) for name in self.key_fields)

    @staticmethod
    def entry_key(entry: dict[str, Any]) -> str:
        fields = (
            ("task_id", "protocol", "split_id", "model_family", "model")
            if "protocol" in entry and "split_id" in entry
            else ("task_id", "model_family", "model")
        )
        return "||".join(str(entry[name]) for name in fields)

    def iter_entries(self) -> list[dict[str, Any]]:
        return [deepcopy(entry) for entry in self._entries.values()]

    def get(self, task_id: str, protocol: str, split_id: str, model_family: str, model: str) -> dict[str, Any]:
        if self.key_fields == ("task_id", "model_family", "model"):
            key = (task_id, model_family, model)
        else:
            key = (task_id, protocol, split_id, model_family, model)
        if key not in self._entries:
            raise KeyError(f"No frozen parameters for key: {key}")
        return self._entries[key]

    def validate_split_hashes(
        self,
        entry: dict[str, Any],
        train_split_hash: str,
        test_split_hash: str,
    ) -> None:
        expected_train = str(entry.get("train_split_hash", "") or "")
        expected_test = str(entry.get("test_split_hash", "") or "")
        if expected_train and expected_train != train_split_hash:
            raise ValueError("Training split hash does not match the frozen parameter snapshot.")
        if expected_test and expected_test != test_split_hash:
            raise ValueError("Test split hash does not match the frozen parameter snapshot.")
