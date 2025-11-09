# Development Guide

## Prerequisites

### System Requirements
- **Operating System**: macOS, Linux, or Windows with WSL2
- **Python**: Version 3.10 or higher
- **Package Manager**: pip (bundled with Python)
- **Memory**: Minimum 4GB RAM (8GB recommended for full dataset)
- **Disk Space**: ~1GB free (for embeddings cache and data)

### Required Accounts
- **Google Gemini API**: Obtain API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
  - Free tier: 60 queries per minute
  - Batch API pricing: $0.075/1M tokens

### Recommended Tools
- **IDE**: VS Code, PyCharm, or similar
- **Terminal**: bash, zsh, or PowerShell
- **Version Control**: git

## Environment Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd context-aware-multi-agent-system
```

### 2. Create Virtual Environment

**macOS/Linux**:
```bash
python3.10 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**:
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Verify activation**:
```bash
which python  # Should show path inside venv/
python --version  # Should show Python 3.10+
```

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected output**:
```
Successfully installed google-genai-0.3.0 scikit-learn-1.7.2 numpy-1.24.0 ...
```

**Verify installation**:
```bash
pip list | grep -E '(scikit-learn|google-genai|numpy|pandas)'
```

### 5. Configure Environment Variables

```bash
# Create .env from template
cp .env.example .env

# Edit .env and add your API key
echo "GEMINI_API_KEY=your_actual_api_key_here" >> .env
```

**.env file structure**:
```bash
# Gemini API Configuration
GEMINI_API_KEY=AIzaSy...your_key_here
```

**Security notes**:
- ⚠️ Never commit `.env` to git (already in `.gitignore`)
- ✅ Use `.env.example` as template (safe to commit)
- ✅ Store production keys in secure secret managers

### 6. Verify Setup

Run the test suite to confirm everything is working:

```bash
pytest tests/ -v
```

**Expected output**:
```
tests/epic1/test_config.py::test_config_loads PASSED
tests/epic1/test_dataset.py::test_ag_news_loads PASSED
tests/epic1/test_gemini_auth.py::test_api_connection PASSED
...
==================== 15 passed in 3.2s ====================
```

## Configuration

### config.yaml Structure

The project uses a centralized `config.yaml` file for all experimental parameters:

```yaml
# Dataset Configuration
dataset:
  name: "ag_news"           # AG News dataset from Hugging Face
  categories: 4             # Number of news categories
  sample_size: null         # null = full dataset, or integer for subset

# Clustering Configuration
clustering:
  algorithm: "kmeans"       # Clustering algorithm
  n_clusters: 4             # Number of clusters
  random_state: 42          # Reproducibility seed
  max_iter: 300             # Max iterations for convergence
  init: "k-means++"         # Centroid initialization method

# Embedding Configuration
embedding:
  model: "gemini-embedding-001"  # Gemini embedding model
  batch_size: 100                # Batch size for API calls
  cache_dir: "data/embeddings"   # Cache directory
  output_dimensionality: 768     # Embedding dimensions
  cache_enabled: true            # Enable caching
  use_batch_api: true            # Use batch API for cost savings
  checkpoint_enabled: true       # Enable checkpoint recovery

# Classification Configuration
classification:
  method: "cosine_similarity"  # Similarity metric
  threshold: 0.7               # Minimum similarity threshold

# Metrics Configuration
metrics:
  cost_per_1M_tokens_under_200k: 3.0   # API cost (USD)
  cost_per_1M_tokens_over_200k: 6.0    # API cost (USD)
  target_cost_reduction: 0.90          # 90% cost reduction target
```

### Accessing Configuration in Code

```python
from context_aware_multi_agent_system.config import Config, Paths

# Load configuration
config = Config()

# Access parameters with dot notation
n_clusters = config.get("clustering.n_clusters")  # Returns 4
batch_size = config.get("embedding.batch_size")   # Returns 100
api_key = config.gemini_api_key                   # From .env

# Validate configuration
config.validate()  # Raises errors if missing/invalid

# Access project paths
paths = Paths()
data_dir = paths.data_raw          # project_root/data/raw/
embeddings_dir = paths.data_embeddings  # project_root/data/embeddings/
results_dir = paths.results        # project_root/results/
```

## Build and Run Commands

### Using Makefile (Recommended)

The project includes a Makefile for common development tasks:

```bash
# Display all available commands
make help

# Install dependencies
make install

# Run test suite
make test

# Run linter
make lint

# Format code
make format

# Clean build artifacts
make clean
```

### Manual Commands

**Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Run tests**:
```bash
pytest tests/ -v
```

**Run linter**:
```bash
ruff check .
```

**Format code**:
```bash
ruff format .
```

**Clean artifacts**:
```bash
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -delete
find . -type d -name "*.egg-info" -exec rm -rf {} +
rm -rf .pytest_cache .ruff_cache
```

## Running the Pipeline

### Full Pipeline Execution

Execute all scripts in sequence:

```bash
# Step 1: Generate embeddings (15-20 minutes, costs ~$3-5)
python scripts/01_generate_embeddings.py

# Step 2: Train K-Means clustering (2-3 minutes)
python scripts/02_train_clustering.py

# Step 3: Evaluate cluster quality (1 minute)
python scripts/03_evaluate_clustering.py

# Step 4: Generate PCA visualizations (1 minute)
python scripts/04_visualize_clusters.py

# Step 5: Analyze and label clusters (1 minute)
python scripts/05_analyze_clusters.py

# Step 6: Initialize specialized agents (30 seconds)
python scripts/06_initialize_agents.py
```

### Individual Script Usage

**1. Generate Embeddings**:
```bash
python scripts/01_generate_embeddings.py
```
- Loads AG News dataset (120K train + 7.6K test)
- Generates 768-dim embeddings via Gemini API
- Caches to `data/embeddings/`
- **Runtime**: 15-20 minutes
- **Cost**: $3-5 USD

**2. Train Clustering**:
```bash
python scripts/02_train_clustering.py
```
- Loads cached embeddings
- Trains K-Means (k=4, random_state=42)
- Exports to `data/processed/cluster_assignments.csv`
- **Runtime**: 2-3 minutes

**3. Evaluate Clustering**:
```bash
python scripts/03_evaluate_clustering.py
```
- Computes silhouette score, Davies-Bouldin index
- Calculates cluster purity vs ground truth
- **Runtime**: 1 minute

**4. Visualize Clusters**:
```bash
python scripts/04_visualize_clusters.py
```
- Generates PCA 2D projection
- Saves 300 DPI PNG to `visualizations/cluster_pca.png`
- **Runtime**: 1 minute

**5. Analyze Clusters**:
```bash
python scripts/05_analyze_clusters.py
```
- Labels clusters with dominant categories
- Extracts representative documents
- Exports to `results/cluster_analysis.txt`
- **Runtime**: 1 minute

**6. Initialize Agents**:
```bash
python scripts/06_initialize_agents.py
```
- Creates 4 specialized agents (one per cluster)
- Computes context size reduction metrics
- **Runtime**: 30 seconds

### Re-running with Fresh Data

**Clear embedding cache** (forces regeneration):
```bash
rm -rf data/embeddings/train_embeddings.npy
rm -rf data/embeddings/train_metadata.json
python scripts/01_generate_embeddings.py
```

**Clear clustering outputs**:
```bash
rm -rf data/processed/cluster_*.{csv,npy,json}
python scripts/02_train_clustering.py
```

**Clear all generated data**:
```bash
rm -rf data/ results/ visualizations/
# Then re-run pipeline
```

## Testing

### Running Tests

**All tests**:
```bash
pytest tests/ -v
```

**Specific epic**:
```bash
pytest tests/epic1/ -v  # Project initialization tests
pytest tests/epic2/ -v  # Clustering tests
pytest tests/epic3/ -v  # Multi-agent tests
```

**Specific test file**:
```bash
pytest tests/epic1/test_config.py -v
```

**Specific test function**:
```bash
pytest tests/epic1/test_config.py::test_config_loads -v
```

**With coverage**:
```bash
pytest tests/ --cov=src/context_aware_multi_agent_system --cov-report=html
```

### Test Organization

```
tests/
├── epic1/                    # Epic 1: Project Initialization
│   ├── test_config.py       # Configuration system tests
│   ├── test_dataset.py      # Dataset loading tests
│   └── test_gemini_auth.py  # API authentication tests
├── epic2/                    # Epic 2: Clustering
│   ├── test_embeddings.py   # Embedding generation tests
│   ├── test_clustering.py   # K-Means clustering tests
│   └── test_metrics.py      # Evaluation metrics tests
└── epic3/                    # Epic 3: Multi-Agent
    ├── test_agent.py        # Specialized agent tests
    └── test_routing.py      # Query routing tests
```

### Test Fixtures

Common fixtures defined in `conftest.py`:
- `sample_config`: Mock configuration
- `sample_embeddings`: Small embedding arrays for testing
- `sample_clusters`: Pre-assigned cluster labels

## Code Quality

### Linting with Ruff

**Check code style**:
```bash
ruff check .
```

**Auto-fix issues**:
```bash
ruff check --fix .
```

**Format code**:
```bash
ruff format .
```

### Ruff Configuration

Defined in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "C4", "PT"]
ignore = []

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

**Selected rules**:
- **E**: pycodestyle errors
- **F**: pyflakes
- **W**: pycodestyle warnings
- **I**: isort (import sorting)
- **N**: pep8-naming
- **UP**: pyupgrade
- **B**: flake8-bugbear
- **A**: flake8-builtins
- **C4**: flake8-comprehensions
- **PT**: flake8-pytest-style

## Reproducibility

### Setting Random Seeds

All experiments use fixed seeds for reproducibility:

```python
from context_aware_multi_agent_system.utils.reproducibility import set_seed

# Set all random seeds (Python, NumPy)
set_seed(42)

# Now all random operations are deterministic
# - NumPy: np.random.rand(), etc.
# - Python: random.choice(), etc.
```

**Note**: scikit-learn algorithms use separate `random_state` parameters loaded from `config.yaml`.

### Configuration Versioning

- **config.yaml**: Committed to git, tracks all experimental parameters
- **requirements.txt**: Version-pinned dependencies
- **pyproject.toml**: Project metadata and tool configurations

### Data Versioning (Recommended)

For production, consider using DVC (Data Version Control):

```bash
# Install DVC
pip install dvc

# Initialize DVC
dvc init

# Track data directory
dvc add data/
git add data.dvc .gitignore
git commit -m "Track data with DVC"
```

## Common Development Tasks

### Adding a New Script

1. Create script in `scripts/`:
```bash
touch scripts/07_my_new_script.py
```

2. Add executable header:
```python
#!/usr/bin/env python3
"""
Description of what this script does.
"""
from context_aware_multi_agent_system.config import Config, Paths

def main():
    config = Config()
    paths = Paths()
    # Implementation...

if __name__ == "__main__":
    main()
```

3. Make executable (optional):
```bash
chmod +x scripts/07_my_new_script.py
```

### Adding a New Module

1. Create module file:
```bash
touch src/context_aware_multi_agent_system/my_module.py
```

2. Add module docstring and implementation:
```python
"""
Module description.

Classes:
    MyClass: Brief description

Functions:
    my_function: Brief description
"""

def my_function():
    """Detailed docstring."""
    pass
```

3. Update `__init__.py`:
```python
from .my_module import my_function

__all__ = ["my_function"]
```

### Adding Tests

1. Create test file:
```bash
touch tests/epic1/test_my_feature.py
```

2. Write tests:
```python
"""Tests for my_feature module."""

import pytest
from context_aware_multi_agent_system.my_module import my_function

def test_my_function():
    """Test my_function with valid input."""
    result = my_function()
    assert result is not None
```

3. Run tests:
```bash
pytest tests/epic1/test_my_feature.py -v
```

## Troubleshooting

### Issue: API Authentication Error

**Symptom**:
```
❌ Invalid API key
```

**Solution**:
1. Verify `.env` exists and contains `GEMINI_API_KEY`
2. Check API key is valid at [Google AI Studio](https://aistudio.google.com/app/apikey)
3. Ensure no extra spaces in `.env` file
4. Restart Python interpreter to reload environment

### Issue: Import Errors

**Symptom**:
```
ModuleNotFoundError: No module named 'context_aware_multi_agent_system'
```

**Solution**:
1. Verify virtual environment is activated: `which python`
2. Install in editable mode: `pip install -e .`
3. Or add to PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"`

### Issue: Embedding Generation Interrupted

**Symptom**:
```
^C KeyboardInterrupt
```

**Solution**:
Checkpoint system automatically saves progress. Just re-run:
```bash
python scripts/01_generate_embeddings.py
# ⏯️ Resuming from checkpoint: last processed index = 50000
```

### Issue: Cluster Quality Low

**Symptom**:
```
⚠️ Silhouette Score: 0.15 (Target: >0.3)
```

**Solutions**:
1. Try different `n_clusters` in `config.yaml`
2. Experiment with `init` methods: `"k-means++"` or `"random"`
3. Increase `max_iter` for better convergence
4. Check for embedding quality issues

### Issue: Memory Error

**Symptom**:
```
MemoryError: Unable to allocate array
```

**Solutions**:
1. Reduce `sample_size` in `config.yaml` (e.g., set to 10000)
2. Process in batches
3. Increase system RAM
4. Use smaller `batch_size` in embedding generation

## Development Workflow Summary

### Daily Development Cycle

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Pull latest changes
git pull origin main

# 3. Install any new dependencies
pip install -r requirements.txt

# 4. Make changes to code

# 5. Format code
make format

# 6. Run linter
make lint

# 7. Run tests
make test

# 8. Commit changes
git add .
git commit -m "Description of changes"
git push origin feature-branch
```

### Pre-Commit Checklist

Before committing code:

- ✅ All tests pass: `pytest tests/ -v`
- ✅ Code is formatted: `ruff format .`
- ✅ No lint errors: `ruff check .`
- ✅ Documentation updated (if needed)
- ✅ `.env` not included in commit
- ✅ Commit message is descriptive

---

**Last Updated**: 2025-11-09
**Guide Version**: 1.0
