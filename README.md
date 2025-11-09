# Context-Aware Multi-Agent System

Multi-Agent collaborative system for cost-efficient text classification using K-Means clustering and cosine similarity routing.

## Overview

This project demonstrates how clustering and classification algorithms can solve LLM cost optimization challenges. The system uses K-Means clustering to partition 120K news articles into 4 semantic groups, then routes queries using cosine similarity classification to specialized agents containing only relevant context—reducing API costs by 90%+.

## Project Structure

```
context-aware-multi-agent-system/
├── data/                  # Data directory (not committed to git)
│   ├── raw/              # Raw data from AG News dataset
│   ├── embeddings/       # Cached embeddings
│   ├── interim/          # Intermediate processing results
│   └── processed/        # Final processed data
├── src/                  # Source code
│   └── context_aware_multi_agent_system/
│       ├── data/         # Data loading and preprocessing
│       ├── features/     # Feature engineering (embeddings)
│       ├── models/       # Clustering and classification models
│       ├── evaluation/   # Metrics and evaluation
│       ├── visualization/ # Plotting and visualization
│       └── utils/        # Utility functions
├── notebooks/            # Jupyter notebooks for analysis
├── reports/              # Generated reports and figures
├── results/              # Experiment results (JSON)
├── models/               # Trained models
├── tests/                # Unit tests
├── scripts/              # Utility scripts
└── docs/                 # Project documentation
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd context-aware-multi-agent-system
```

2. Create and activate virtual environment:
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Upgrade pip:
```bash
pip install --upgrade pip
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Configuration

The project uses a centralized configuration system to manage all experimental parameters. Configuration is split between two files:

### config.yaml (Committed to Git)

Contains all experimental parameters for reproducible research:

```yaml
# Dataset Configuration
dataset:
  name: "ag_news"
  categories: 4
  sample_size: null  # null = use full dataset

# Clustering Configuration
clustering:
  algorithm: "kmeans"
  n_clusters: 4
  random_state: 42
  max_iter: 300
  init: "k-means++"

# Embedding Configuration
embedding:
  model: "gemini-embedding-001"
  batch_size: 100
  cache_dir: "data/embeddings"
  output_dimensionality: 768

# Classification Configuration
classification:
  method: "cosine_similarity"
  threshold: 0.7

# Metrics Configuration
metrics:
  cost_per_1M_tokens_under_200k: 3.0
  cost_per_1M_tokens_over_200k: 6.0
  target_cost_reduction: 0.90
```

### .env (Not Committed - Contains Secrets)

Contains API keys and sensitive credentials:

```bash
GEMINI_API_KEY=your_api_key_here
```

### Using Configuration in Code

```python
from context_aware_multi_agent_system.config import Config, Paths

# Load configuration
config = Config()

# Access configuration parameters with dot notation
n_clusters = config.get("clustering.n_clusters")  # Returns 4
model_name = config.get("embedding.model")  # Returns "gemini-embedding-001"
api_key = config.gemini_api_key  # Loads from .env

# Validate configuration
config.validate()  # Raises errors for missing/invalid fields

# Access all project paths
paths = Paths()
data_file = paths.data_raw / "ag_news.csv"
model_file = paths.models / "kmeans_model.pkl"
```

### Reproducibility

All experiments use fixed random seeds for reproducible results:

```python
from context_aware_multi_agent_system.utils.reproducibility import set_seed

# Set random seeds at the beginning of your script
set_seed(42)

# Now all random operations are deterministic
# K-Means clustering will produce identical results across runs
```

**Note**: scikit-learn algorithms use separate `random_state` parameters, which are loaded from `config.yaml`.

## Usage

(To be added as features are implemented)

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

This project uses Ruff for linting and formatting:

```bash
ruff check .
ruff format .
```

## Technology Stack

- **Python**: 3.10
- **Embedding Service**: Google Gemini API (gemini-embedding-001)
- **ML Libraries**: scikit-learn, numpy, pandas
- **Dataset**: AG News (via Hugging Face datasets)
- **Visualization**: matplotlib, seaborn
- **Configuration**: PyYAML, python-dotenv
- **Testing**: pytest
- **Code Quality**: Ruff

## License

MIT License

## Author

Jack YUAN

## Acknowledgments

This project uses the Cookiecutter Data Science project structure.
