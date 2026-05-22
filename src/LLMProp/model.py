from __future__ import annotations

import torch
import torch.nn as nn


class T5Predictor(nn.Module):
    """LLM-Prop style T5 encoder regressor."""

    def __init__(
        self,
        base_model: nn.Module,
        base_model_output_size: int = 512,
        n_classes: int = 1,
        drop_rate: float = 0.2,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.model = base_model
        self.pooling = pooling
        self.linear_regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(base_model_output_size, n_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_masks)
        last_hidden_state = hidden_states.last_hidden_state
        if self.pooling == "cls":
            input_embedding = last_hidden_state[:, 0, :]
        else:
            mask = attention_masks.unsqueeze(-1).to(last_hidden_state.dtype)
            input_embedding = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        outputs = self.linear_regressor(input_embedding)
        return input_embedding, outputs
