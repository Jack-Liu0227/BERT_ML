from __future__ import annotations

from typing import Dict, List, Type

from src.data_processing.strength_extrapolation_data_processor import StrengthExtrapolationDataProcessor
from src.data_processing.strength_loco_data_processor import StrengthLocoDataProcessor
from src.data_processing.strength_ood_common import StrengthOODProcessorBase
from src.data_processing.strength_sparse_x_cluster_data_processor import StrengthSparseXClusterDataProcessor
from src.data_processing.strength_sparse_x_single_data_processor import StrengthSparseXSingleDataProcessor
from src.data_processing.strength_sparse_y_cluster_data_processor import StrengthSparseYClusterDataProcessor
from src.data_processing.strength_sparse_y_single_data_processor import StrengthSparseYSingleDataProcessor
from src.data_processing.strength_random_cv_baseline_data_processor import StrengthRandomCVBaselineDataProcessor


PROCESSOR_REGISTRY: Dict[str, Type[StrengthOODProcessorBase]] = {
    "target_extrapolation": StrengthExtrapolationDataProcessor,
    "sparse_x_single": StrengthSparseXSingleDataProcessor,
    "sparse_y_single": StrengthSparseYSingleDataProcessor,
    "sparse_x_cluster": StrengthSparseXClusterDataProcessor,
    "sparse_y_cluster": StrengthSparseYClusterDataProcessor,
    "loco": StrengthLocoDataProcessor,
    "random_cv_baseline": StrengthRandomCVBaselineDataProcessor,
}


def create_ood_processor(
    split_strategy: str,
    input_file: str,
    random_state: int = 42,
    processing_cols: List[str] | None = None,
) -> StrengthOODProcessorBase:
    if split_strategy not in PROCESSOR_REGISTRY:
        raise ValueError(
            f"Unsupported split_strategy '{split_strategy}'. "
            f"Supported strategies: {', '.join(PROCESSOR_REGISTRY.keys())}"
        )
    processor_cls = PROCESSOR_REGISTRY[split_strategy]
    return processor_cls(
        input_file=input_file,
        random_state=random_state,
        processing_cols=processing_cols,
    )


def get_supported_split_strategies() -> List[str]:
    return list(PROCESSOR_REGISTRY.keys())


def get_processor_registry() -> Dict[str, Type[StrengthOODProcessorBase]]:
    return PROCESSOR_REGISTRY.copy()
