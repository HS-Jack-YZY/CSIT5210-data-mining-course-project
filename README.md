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

### Quick Start: Generate Embeddings for AG News Dataset

Run the embedding generation script to process all 120K AG News documents:

```bash
python scripts/01_generate_embeddings.py
```

This script will:
- Load the AG News dataset (120K train + 7.6K test documents)
- Generate embeddings using Gemini Batch API ($0.075/1M tokens)
- Cache embeddings to `data/embeddings/` to avoid repeated API calls
- Track API usage, cost, and performance metrics
- Resume automatically if interrupted (checkpoint system)

**Expected Output:**
```
📊 Starting embedding generation for train split
📊 Total documents: 120000, Batch size: 100, Total batches: 1200
🔄 Processing batch 1/1200 (documents 0-99)
📊 Progress: 1000/120000 documents processed (83.3 docs/sec)
...
✅ Embedding generation complete!
💰 Estimated total cost: $3.25 USD
⏱️ Total execution time: 14.5 minutes
```

**Cost Optimization:**
- Batch API: $0.075/1M tokens (50% savings vs standard API)
- Caching: Subsequent runs use cached embeddings (0 cost)
- Expected cost for full dataset: $3-5 (well below $10 PRD limit)

### K-Means Clustering

After generating embeddings, train K-Means clustering to partition documents into semantic groups:

```bash
python scripts/02_train_clustering.py
```

This script will:
- Load cached embeddings from `data/embeddings/`
- Train K-Means clustering (k=4, random_state=42)
- Export cluster assignments to `data/processed/cluster_assignments.csv`
- Export cluster centroids to `data/processed/centroids.npy`
- Export clustering metadata to `data/processed/cluster_metadata.json`
- Validate cluster balance (no cluster <10% or >50% of data)

**Expected Output:**
```
📊 Starting K-Means clustering...
📂 Loading cached embeddings...
📊 Loaded 120000 embeddings (768-dim) from cache
📊 Fitting K-Means clustering...
✅ Clustering converged in 15 iterations
📊 Cluster sizes: [29825, 30138, 30013, 30024] (balanced)
✅ Clustering completed successfully
⏱️ Total execution time: 2m 15s
```

### Cluster Quality Evaluation

Evaluate clustering quality using standard metrics:

```bash
python scripts/03_evaluate_clustering.py
```

This script will:
- Load embeddings, cluster assignments, and centroids
- Calculate Silhouette Score (target >0.3)
- Calculate Davies-Bouldin Index (lower is better)
- Calculate Cluster Purity vs ground truth labels
- Generate confusion matrix
- Assess cluster balance

**Expected Output:**
```
📊 Computing cluster quality metrics...
✅ Cluster Quality Evaluation Complete
   - Silhouette Score: 0.2841 (Target: >0.3, ⚠️ Below target)
   - Davies-Bouldin Index: 1.85
   - Cluster Purity: 68.5% (Target: >70%, ⚠️ Below target)
   - Cluster Balance: Balanced
```

### PCA Cluster Visualization

Generate publication-quality 2D visualization of clusters using PCA dimensionality reduction:

```bash
python scripts/04_visualize_clusters.py
```

This script will:
- Apply PCA dimensionality reduction (768D → 2D)
- Generate scatter plot showing 4 semantic clusters with distinct colors
- Mark cluster centroids with star symbols
- Calculate variance explained by PC1 and PC2
- Export 300 DPI PNG visualization for reports
- Optionally generate interactive Plotly HTML visualization

**Expected Output:**
```
📊 Starting PCA cluster visualization...
📊 Loading embeddings and cluster labels...
✅ Loaded 120000 embeddings (768D)
📊 Applying PCA dimensionality reduction (768D → 2D)...
✅ PCA complete. Variance explained: 0.3%
📊 PC1 variance: 0.2%
📊 PC2 variance: 0.2%
⚠️ Low variance explained (0.3%), 2D projection may lose information
📊 Generating cluster scatter plot...
✅ Visualization saved: visualizations/cluster_pca.png (300 DPI)
✅ PCA Cluster Visualization Complete
   - Documents visualized: 120,000
   - Variance explained: 0.3% (PC1: 0.2%, PC2: 0.2%)
   - Output: visualizations/cluster_pca.png (300 DPI)
   - Execution time: 1.0s
```

**Output Files:**
- `visualizations/cluster_pca.png` - Static scatter plot (300 DPI, publication quality)
- `visualizations/cluster_pca.html` - Interactive Plotly visualization (optional, requires plotly)

**Variance Explained Interpretation:**
- >20%: Good projection, 2D captures main structure
- 10-20%: Acceptable for visualization, some information loss
- <10%: High-dimensional data, 2D is rough approximation

**Note**: Low variance (<20%) is expected for high-dimensional text embeddings. The visualization is still useful for demonstrating cluster separation and qualitative assessment.

### Embedding Generation (Programmatic)

Generate semantic embeddings using the Gemini API:

```python
from context_aware_multi_agent_system.config import Config, Paths
from context_aware_multi_agent_system.features import EmbeddingService, EmbeddingCache

# Initialize configuration
config = Config()
paths = Paths()

# Create embedding service with API key from .env
service = EmbeddingService(config.gemini_api_key)

# Test API connection
service.test_connection()  # Returns True if successful

# Generate single embedding
embedding = service.generate_embedding("Hello world")
print(embedding.shape)  # (768,)

# Generate batch embeddings (uses Gemini Batch API)
documents = ["First document", "Second document", "Third document"]
embeddings = service.generate_batch(documents, batch_size=100)
print(embeddings.shape)  # (3, 768)
print(embeddings.dtype)  # float32
```

### Embedding Cache

Save and load embeddings to avoid redundant API calls:

```python
from context_aware_multi_agent_system.features import EmbeddingCache
from context_aware_multi_agent_system.config import Paths
import numpy as np

# Initialize cache
paths = Paths()
cache = EmbeddingCache(paths.data_embeddings)

# Save embeddings
embeddings = np.random.rand(100, 768).astype(np.float32)
metadata = {
    "model": "gemini-embedding-001",
    "dimensions": 768,
    "num_documents": 100,
    "dataset": "ag_news",
    "split": "train"
}
cache.save(embeddings, "train", metadata)

# Load embeddings
embeddings, metadata = cache.load("train")
print(embeddings.shape)  # (100, 768)
print(metadata["model"])  # "gemini-embedding-001"

# Check if cache exists
if cache.exists("test"):
    embeddings, metadata = cache.load("test")
else:
    # Generate embeddings...
    pass

# Clear cache to force regeneration
cache.clear("train")
```

### Cache Management

The embedding cache stores generated embeddings to avoid repeated API calls:

**Cache Location:**
- Train embeddings: `data/embeddings/train_embeddings.npy`
- Test embeddings: `data/embeddings/test_embeddings.npy`
- Metadata: `data/embeddings/{split}_metadata.json`

**Force Regeneration:**
```bash
# Clear train cache
rm data/embeddings/train_embeddings.npy data/embeddings/train_metadata.json

# Clear test cache
rm data/embeddings/test_embeddings.npy data/embeddings/test_metadata.json

# Then re-run the script
python scripts/01_generate_embeddings.py
```

**Resume from Checkpoint:**

If embedding generation is interrupted, the checkpoint system automatically resumes:

```bash
# First run (interrupted at 50K documents)
python scripts/01_generate_embeddings.py
# ^C (Ctrl+C to interrupt)

# Second run (resumes from 50K)
python scripts/01_generate_embeddings.py
# ⏯️ Resuming from checkpoint: last processed index = 50000
```

Checkpoint files are stored at `data/embeddings/.checkpoint_{split}.json` and deleted automatically on successful completion.

### Troubleshooting

**API Authentication Error:**
```
❌ Invalid API key
```
Solution: Copy `.env.example` to `.env` and add your Gemini API key:
```bash
cp .env.example .env
echo "GEMINI_API_KEY=your_api_key_here" >> .env
```

**Rate Limiting:**
```
⚠️ Batch 123 failed after 3 retries, skipping
```
Solution: The script automatically retries with exponential backoff (4s, 8s, 16s). Failed batches are skipped and reported in the final summary. You can manually retry by running the script again.

**Network Interruption:**
```
💾 Saved checkpoint: processed 50000 documents
```
Solution: The checkpoint system saves progress after each batch. Simply re-run the script to resume from the last checkpoint.

**Cost Exceeds Target:**
```
⚠️ WARNING: Total cost $5.50 exceeds target of $5.00
```
Solution: This is unusual. Check your batch size configuration in `config.yaml` and verify you're using the batch API (`use_batch_api: true`).
```

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
