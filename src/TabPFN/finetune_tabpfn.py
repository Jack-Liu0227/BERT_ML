"""Disabled legacy local TabPFN fine-tuning entry points.

The approved runtime is the hosted TabPFN API 2.5 Plus model. The API client
does not expose the local checkpoint fine-tuning workflow that previously
lived in this module, so invoking it would silently violate the runtime lock.
"""

from __future__ import annotations

from typing import Any


_DISABLED_MESSAGE = (
    "Local TabPFN fine-tuning is disabled. Only TabPFN API 2.5 Plus "
    "inference via create_default_for_version(ModelVersion.V2_5) is allowed."
)


class TabPFNFinetuneTrainer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(_DISABLED_MESSAGE)


def run_single_finetune_experiment(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(_DISABLED_MESSAGE)


def run_all_finetune_experiments(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(_DISABLED_MESSAGE)


def main() -> None:
    raise SystemExit(_DISABLED_MESSAGE)


if __name__ == "__main__":
    main()
