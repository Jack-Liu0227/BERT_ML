"""
TabPFN Module for Alloy Property Prediction
TabPFN 合金性能预测模块

This module provides TabPFN-based models for predicting alloy properties.
"""

from .tabpfn_configs import (
    TABPFN_CONFIGS,
    TABPFN_MODEL_CONFIG,
    get_tabpfn_config,
    get_all_alloy_types
)

from .data_processor import (
    TabPFNDataProcessor,
    create_data_processor
)

from .train_tabpfn import (
    TabPFNTrainer,
    run_single_experiment,
    run_all_experiments
)

__all__ = [
    'TABPFN_CONFIGS',
    'TABPFN_MODEL_CONFIG',
    'get_tabpfn_config',
    'get_all_alloy_types',
    'TabPFNDataProcessor',
    'create_data_processor',
    'TabPFNTrainer',
    'run_single_experiment',
    'run_all_experiments',
]

try:
    from .finetune_tabpfn import (
        TabPFNFinetuneTrainer,
        run_single_finetune_experiment,
        run_all_finetune_experiments,
    )

    __all__.extend([
        'TabPFNFinetuneTrainer',
        'run_single_finetune_experiment',
        'run_all_finetune_experiments',
    ])
except Exception:
    # Finetuning support is optional and may be unavailable when tabpfn extras
    # are not installed. Keep package imports usable for non-finetuning flows.
    pass
