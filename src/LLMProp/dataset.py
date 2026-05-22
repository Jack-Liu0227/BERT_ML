from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def tokenize_dataframe(tokenizer: Any, dataframe: pd.DataFrame, max_length: int, pooling: str = "cls") -> tuple[list, list]:
    if "description" not in dataframe.columns:
        raise ValueError("LLM-Prop dataframe requires a 'description' column")
    descriptions = dataframe["description"].fillna("").astype(str).tolist()
    if pooling == "cls":
        descriptions = ["[CLS] " + descr for descr in descriptions]
    encoded = tokenizer(
        text=descriptions,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    return encoded["input_ids"], encoded["attention_mask"]


def create_dataloader(
    tokenizer: Any,
    dataframe: pd.DataFrame,
    max_length: int,
    batch_size: int,
    property_value: str,
    pooling: str = "cls",
    labels_mean: float | None = None,
    labels_std: float | None = None,
    shuffle: bool = False,
) -> DataLoader:
    if property_value not in dataframe.columns:
        raise ValueError(f"LLM-Prop dataframe missing target column: {property_value}")
    input_ids, attention_masks = tokenize_dataframe(tokenizer, dataframe, max_length, pooling=pooling)
    labels = pd.to_numeric(dataframe[property_value], errors="coerce").to_numpy(dtype=np.float32)
    if np.isnan(labels).any():
        raise ValueError(f"Target column contains non-numeric or NaN values: {property_value}")

    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    mask_tensor = torch.tensor(attention_masks, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)

    if labels_mean is not None and labels_std is not None:
        std = float(labels_std) if abs(float(labels_std)) > 1e-12 else 1.0
        normalized_labels = (labels_tensor - float(labels_mean)) / std
        dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor, normalized_labels)
    else:
        dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
