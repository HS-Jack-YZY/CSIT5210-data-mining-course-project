# Source Tree Analysis

## Project Structure Overview

```
context-aware-multi-agent-system/
├── data/                          # Data directory (gitignored)
│   ├── raw/                      # Raw AG News dataset
│   ├── embeddings/               # Cached Gemini embeddings
│   ├── interim/                  # Intermediate processing results
│   └── processed/                # Final processed data (clusters, centroids)
│
├── src/context_aware_multi_agent_system/  # Main source code package
│   ├── __init__.py              # Package initialization
│   ├── config.py                # [CORE] Configuration management system
│   │
│   ├── data/                    # Data loading layer
│   │   ├── __init__.py
│   │   └── load_dataset.py     # AG News dataset loader
│   │
│   ├── features/                # Feature engineering layer
│   │   ├── __init__.py
│   │   ├── embedding_service.py  # Gemini API embedding generation
│   │   └── embedding_cache.py    # Embedding caching system
│   │
│   ├── models/                  # Model layer
│   │   ├── __init__.py
│   │   ├── clustering.py        # K-Means clustering implementation
│   │   └── agent.py             # Specialized agent implementation
│   │
│   ├── evaluation/              # Evaluation and metrics layer
│   │   ├── __init__.py
│   │   ├── clustering_metrics.py  # Silhouette, Davies-Bouldin metrics
│   │   ├── cluster_analysis.py   # Cluster labeling and analysis
│   │   └── cost_calculator.py    # API cost calculation
│   │
│   ├── visualization/           # Visualization layer
│   │   ├── __init__.py
│   │   └── cluster_plots.py     # PCA visualization
│   │
│   └── utils/                   # Utility layer
│       ├── __init__.py
│       └── reproducibility.py   # Random seed management
│
├── scripts/                     # [ENTRY POINTS] Executable pipeline scripts
│   ├── 01_generate_embeddings.py   # Step 1: Generate Gemini embeddings
│   ├── 02_train_clustering.py      # Step 2: Train K-Means clustering
│   ├── 03_evaluate_clustering.py   # Step 3: Evaluate cluster quality
│   ├── 04_visualize_clusters.py    # Step 4: Generate PCA visualizations
│   ├── 05_analyze_clusters.py      # Step 5: Analyze and label clusters
│   └── 06_initialize_agents.py     # Step 6: Initialize specialized agents
│
├── tests/                       # Test suite
│   ├── epic1/                   # Epic 1 tests (project setup)
│   ├── epic2/                   # Epic 2 tests (clustering)
│   └── epic3/                   # Epic 3 tests (multi-agent)
│
├── docs/                        # Project documentation
│   ├── PRD.md                   # Product Requirements Document
│   ├── architecture.md          # System architecture
│   ├── epics.md                 # Epic breakdown
│   ├── tech-spec-epic-*.md      # Technical specifications per epic
│   ├── stories/                 # User story implementations
│   └── retrospectives/          # Epic retrospectives
│
├── results/                     # Experiment results (JSON outputs)
├── visualizations/              # Generated visualizations (PNG, HTML)
├── models/                      # Trained model artifacts
├── notebooks/                   # Jupyter notebooks (exploratory analysis)
├── reports/                     # Generated reports
│
├── config.yaml                  # [CONFIG] Centralized configuration
├── .env                        # [SECRET] API keys (not in git)
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata and tool configs
├── Makefile                    # Development commands
└── README.md                   # Project overview and usage guide
```

## Critical Directories Explained

### `/src/context_aware_multi_agent_system/` - Main Application Code

**Purpose**: Core implementation of the context-aware multi-agent classification system

**Architecture Pattern**: Layered architecture following data science best practices

**Layers**:
1. **Data Layer** (`data/`): Dataset loading and preprocessing
2. **Feature Layer** (`features/`): Embedding generation with caching
3. **Model Layer** (`models/`): Clustering algorithms and agent implementations
4. **Evaluation Layer** (`evaluation/`): Metrics, analysis, and cost calculation
5. **Visualization Layer** (`visualization/`): Chart and plot generation
6. **Utility Layer** (`utils/`): Cross-cutting concerns (reproducibility, logging)

### `/scripts/` - Executable Pipeline

**Purpose**: Sequential execution scripts for the ML pipeline

**Execution Order**:
1. `01_generate_embeddings.py` → Generates 768-dim embeddings via Gemini API
2. `02_train_clustering.py` → Trains K-Means (k=4) on embeddings
3. `03_evaluate_clustering.py` → Computes quality metrics (silhouette, purity)
4. `04_visualize_clusters.py` → Creates PCA 2D projections
5. `05_analyze_clusters.py` → Labels clusters with dominant categories
6. `06_initialize_agents.py` → Creates specialized agents per cluster

**Usage**: Run scripts in sequence after setup (see development-guide.md)

### `/tests/` - Test Suite

**Purpose**: Unit and integration tests organized by epic

**Structure**:
- `epic1/`: Project initialization tests (config, dataset loading, API auth)
- `epic2/`: Clustering tests (embeddings, K-Means, evaluation metrics)
- `epic3/`: Multi-agent tests (agent initialization, context reduction)

**Test Runner**: pytest (configured in pyproject.toml)

### `/data/` - Data Directory (Gitignored)

**Purpose**: Storage for datasets, embeddings, and processed outputs

**Created automatically on first run**

**Contents**:
- `raw/`: AG News dataset (120K train + 7.6K test)
- `embeddings/`: Cached Gemini embeddings (768-dim numpy arrays)
- `interim/`: Temporary processing artifacts
- `processed/`: Cluster assignments, centroids, metadata

**Size**: ~500MB (embeddings cache dominates)

### `/docs/` - Documentation Hub

**Purpose**: All project documentation following BMM methodology

**Key Documents**:
- **PRD.md**: Product requirements and success criteria
- **architecture.md**: System design and technical decisions
- **epics.md**: Epic breakdown with user stories
- **tech-spec-epic-*.md**: Detailed technical specifications per epic
- **stories/**: Individual user story markdown files with acceptance criteria
- **retrospectives/**: Epic completion retrospectives

## Entry Points

### Primary Entry Point: `scripts/`

All executable workflows start from scripts in the `/scripts` directory.

**Standard workflow**:
```bash
# 1. Setup (see development-guide.md)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add GEMINI_API_KEY

# 2. Execute pipeline
python scripts/01_generate_embeddings.py
python scripts/02_train_clustering.py
python scripts/03_evaluate_clustering.py
python scripts/04_visualize_clusters.py
python scripts/05_analyze_clusters.py
python scripts/06_initialize_agents.py
```

### Testing Entry Point: `pytest`

```bash
# Run all tests
pytest tests/

# Run specific epic tests
pytest tests/epic1/
pytest tests/epic2/
pytest tests/epic3/
```

### Development Entry Point: `Makefile`

```bash
make install    # Install dependencies
make test       # Run pytest
make lint       # Run Ruff linter
make format     # Format code with Ruff
make clean      # Remove build artifacts
```

## Integration Points

### External APIs
- **Gemini Embedding API**: `src/features/embedding_service.py`
  - Model: `gemini-embedding-001`
  - Authentication: API key from `.env`
  - Cost: $0.075/1M tokens (batch API)

### Data Persistence
- **Embedding Cache**: `data/embeddings/` (numpy arrays + JSON metadata)
- **Cluster Outputs**: `data/processed/` (CSV + numpy + JSON)
- **Experiment Results**: `results/` (JSON)
- **Visualizations**: `visualizations/` (PNG, HTML)

### Configuration
- **Application Config**: `config.yaml` (committed)
- **Secrets**: `.env` (not committed)
- **Access**: `from context_aware_multi_agent_system.config import Config, Paths`

## Key Design Patterns

### 1. Configuration-Driven Architecture
- Centralized config in `config.yaml`
- Environment-specific secrets in `.env`
- Managed by `config.py` singleton

### 2. Caching Strategy
- Embedding cache prevents redundant API calls
- Checkpoint system for resumable execution
- Metadata tracking for cache validation

### 3. Layered Data Pipeline
- Clear separation of concerns (data → features → models → evaluation)
- Each layer exposes clean APIs
- Testable in isolation

### 4. Reproducibility
- Fixed random seeds (`random_state=42`)
- Deterministic scikit-learn algorithms
- Version-pinned dependencies

## Summary

**Total Python Files**: 17 modules in `src/`
**Total Scripts**: 6 executable pipelines
**Test Modules**: 3 epic test suites
**Configuration Files**: 2 (config.yaml, .env)
**Documentation Files**: 20+ markdown files

**Architecture**: Modular layered architecture optimized for ML experimentation
**Primary Language**: Python 3.10+
**Key Dependencies**: scikit-learn, Gemini API, NumPy, Pandas
**Testing**: pytest with epic-based organization
**Code Quality**: Ruff (linting + formatting)
