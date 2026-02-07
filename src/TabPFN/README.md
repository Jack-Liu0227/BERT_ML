# TabPFN Model for Alloy Property Prediction

This module provides [TabPFN](https://github.com/PriorLabs/TabPFN) (Tabular Prior-data Fitted Networks) based models for predicting mechanical properties of various alloy types.

## Overview

TabPFN is a transformer-based model pre-trained on synthetic data that can perform in-context learning for tabular prediction tasks. This implementation adapts TabPFN for alloy property prediction tasks.

## Supported Datasets

- **Ti (Titanium Alloys)**: Predicts UTS(MPa), El(%)
- **Al (Aluminum Alloys)**: Predicts UTS(MPa)
- **HEA (High Entropy Alloys)**: Predicts YS(MPa), UTS(MPa), El(%)
- **Steel**: Predicts YS(MPa), UTS(MPa), El(%)

## File Structure

```
src/TabPFN/
├── __init__.py              # Module initialization
├── tabpfn_configs.py        # Configuration for all datasets
├── data_processor.py        # Data loading and preprocessing
├── train_tabpfn.py          # Model training and evaluation
└── README.md                # This file
```

## Installation

First, install the required dependencies:

```bash
pip install tabpfn
pip install pandas numpy scikit-learn matplotlib
```

## Usage

### Quick Start

Run all experiments on all datasets:

```bash
cd src/TabPFN
python train_tabpfn.py
```

When prompted, choose:
- `1` to run all experiments
- `2` to run a single experiment

### Run Specific Experiment

```python
from pathlib import Path
from train_tabpfn import run_single_experiment

# Set project root path
base_path = Path(__file__).parent.parent.parent

# Run experiment
results = run_single_experiment(
    alloy_type="Ti",          # Choose: "Ti", "Al", "HEA", "Steel"
    target_col="UTS(MPa)",    # Target property
    base_path=base_path
)
```

### Custom Training

```python
from tabpfn_configs import get_tabpfn_config
from train_tabpfn import TabPFNTrainer

# Initialize trainer
trainer = TabPFNTrainer(
    alloy_type="Ti",
    target_col="UTS(MPa)",
    base_path="../../.."
)

# Prepare data
trainer.prepare_data(scale=True, drop_na=True)

# Train model
results = trainer.train_classification_for_regression(n_bins=10)

# Plot results
trainer.plot_predictions(save_path="output/predictions.png")
```

### Using Data Processor

```python
from tabpfn_configs import TABPFN_CONFIGS
from data_processor import TabPFNDataProcessor

# Initialize data processor
config = TABPFN_CONFIGS["Ti"]
processor = TabPFNDataProcessor(config)

# Load and process data
X_train, X_test, y_train, y_test = processor.get_full_pipeline(
    target_col="UTS(MPa)",
    base_path="../../..",
    scale=True,
    drop_na=True
)
```

## Configuration

### Dataset Configuration

Edit `tabpfn_configs.py` to modify dataset configurations:

```python
TABPFN_CONFIGS = {
    "Ti": {
        "raw_data": "datasets/Ti_alloys/titanium.csv",
        "targets": ["UTS(MPa)", "El(%)"],
        "feature_cols": [
            "Al(wt%)", "Cr(wt%)", ...,  # Element features
            "Solution Temperature(℃)", ...  # Processing features
        ],
        "test_size": 0.3,
        "random_state": 42
    },
    ...
}
```

### Model Configuration

```python
TABPFN_MODEL_CONFIG = {
    "model_version": "2.5",      # TabPFN version
    "task_type": "regression",   # Task type
    "n_bins": 10,                # Number of bins for regression
}
```

## Output

Results are saved to `output/TabPFN_results/`:

```
output/TabPFN_results/
├── summary_results.csv              # Summary of all experiments
├── Ti/
│   ├── UTSMPa_predictions.png      # Prediction plots
│   └── Elpercent_predictions.png
├── Al/
│   └── UTSMPa_predictions.png
├── HEA/
│   ├── YSMPa_predictions.png
│   ├── UTSMPa_predictions.png
│   └── Elpercent_predictions.png
└── Steel/
    ├── YSMPa_predictions.png
    ├── UTSMPa_predictions.png
    └── Elpercent_predictions.png
```

### Summary Results CSV

The `summary_results.csv` contains:
- Alloy type and target property
- Training metrics: MAE, RMSE, R², MAPE
- Test metrics: MAE, RMSE, R², MAPE

## Important Notes

### Regression with TabPFN

TabPFN supports both **classification and regression** tasks. This implementation uses **TabPFNRegressor (v2)** for direct regression prediction, which provides:

1. **Native regression support** - No need for classification workarounds
2. **Accurate continuous predictions** - Direct prediction of continuous values
3. **Fast in-context learning** - Pre-trained transformer for quick predictions
4. **Open access** - v2 model is freely available without authentication

**Note:** We use TabPFN v2 instead of v2.5 because v2.5 requires HuggingFace authentication. v2 is fully open and works out of the box.

The model leverages TabPFN's powerful in-context learning capabilities for accurate property prediction.

### Data Requirements

- **Sample size**: TabPFN works best with small to medium datasets (hundreds to thousands of samples)
- **Features**: Handles both compositional (element percentages) and processing parameters
- **Missing values**: Automatically handled by dropping or filling with zeros
- **Scaling**: Features are standardized using StandardScaler

### Performance Considerations

- TabPFN is fast for small datasets (no traditional training required)
- May not outperform traditional ML models on all datasets
- Best suited for:
  - Quick baseline predictions
  - Small dataset scenarios
  - Exploratory analysis

## Example Output

```
==============================================================
Processing 钛合金力学性能数据集 / Titanium alloy mechanical properties dataset
Target: UTS(MPa)
==============================================================
Loading data from: datasets/Ti_alloys/titanium.csv
Data loaded: 235 rows, 20 columns
Using 15 features
Final dataset: 235 samples, 15 features
Target range: [519.00, 1725.00]

Train set: 164 samples
Test set: 71 samples
Train/Test split: 70% / 30%
Features scaled using StandardScaler

==============================================================
Converting Regression to 10-class Classification
==============================================================
Binned into 10 classes
Fitting classifier...
Making predictions...

==============================================================
Evaluation Results
==============================================================

Training Set:
  MAE:  45.2341
  RMSE: 62.8973
  R²:   0.8234
  MAPE: 5.67%

Test Set:
  MAE:  52.1234
  RMSE: 71.4521
  R²:   0.7891
  MAPE: 6.23%
```

## Troubleshooting

### Import Error

If you get `ImportError: No module named 'tabpfn'`:
```bash
pip install tabpfn
```

### CUDA/GPU Issues

TabPFN can use GPU acceleration. If you encounter GPU-related errors:
```python
# Force CPU usage
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Memory Issues

For large datasets, reduce bin count or sample size:
```python
trainer.train_classification_for_regression(n_bins=5)
```

## References

1. [TabPFN Paper](https://arxiv.org/abs/2207.01848): Hollmann et al., "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"
2. [TabPFN GitHub](https://github.com/PriorLabs/TabPFN)
3. [TabPFN Documentation](https://priorlabs.ai/docs/)

## License

This implementation follows the same license as the main project.
