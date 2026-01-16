# Feature Engineering Module

This module provides a comprehensive set of tools for feature engineering in materials science, particularly focused on alloy composition and process parameters.

## Module Structure

### Core Components

- `embedding_manager.py`: Manages element and process text embeddings
- `feature_extractor.py`: Extracts features from alloy compositions and process parameters
- `feature_processor.py`: Processes and transforms features
- `data_processor.py`: Handles data loading and preprocessing

### Utility Scripts

- `generate_element_embeddings.py`: Generates element embeddings for the embedding manager
- `example.py`: Example usage of the feature engineering pipeline

## Key Features

### 1. Element Embeddings
- Pre-trained element embeddings using BERT model
- Caching mechanism for efficient embedding retrieval
- Support for custom element embeddings

### 2. Process Parameter Embeddings
- Text-based process parameter encoding
- Support for multiple process parameters
- Efficient batch processing

### 3. Feature Extraction
- Composition-based feature extraction
- Process parameter feature extraction
- Support for custom feature extraction pipelines

## Usage Example

```python
from feature_engineering.embedding_manager import EmbeddingManager
from feature_engineering.feature_extractor import FeatureExtractor
from feature_engineering.feature_processor import FeatureProcessor

# Initialize components
embedding_manager = EmbeddingManager()
feature_extractor = FeatureExtractor(embedding_manager)
feature_processor = FeatureProcessor()

# Process data
compositions = [{'Al': 90, 'Cu': 10}, {'Fe': 80, 'Ni': 20}]
process_params = ['heat treatment at 500C', 'quenching in water']

# Extract features
features = feature_extractor.extract_features(compositions, process_params)

# Process features
processed_features = feature_processor.process_features(features)
```

## Dependencies

- PyTorch
- Transformers
- NumPy
- Pandas
- scikit-learn

## Setup

1. Install required dependencies:
```bash
pip install torch transformers numpy pandas scikit-learn
```

2. Generate element embeddings:
```bash
python generate_element_embeddings.py
```

## Notes

- The module requires a pre-trained BERT model for element embeddings
- Element embeddings are cached for better performance
- Process parameter embeddings are generated on-the-fly
- Feature extraction supports both composition and process parameters

## Contributing

Feel free to submit issues and enhancement requests! 