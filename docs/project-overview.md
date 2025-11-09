# Project Overview: Context-Aware Multi-Agent System

## Executive Summary

The Context-Aware Multi-Agent System is a research project demonstrating cost-efficient text classification using K-Means clustering and multi-agent architecture. The system reduces Large Language Model (LLM) API costs by 90%+ through intelligent context partitioning.

**Core Innovation**: Instead of providing all 120K documents to a single classifier, the system clusters documents semantically and routes queries to specialized agents containing only relevant context—achieving dramatic cost reduction while maintaining classification accuracy.

## Project Metadata

| Attribute | Value |
|-----------|-------|
| **Project Name** | Context-Aware Multi-Agent System |
| **Type** | Data Science / Machine Learning Research |
| **Version** | 0.1.0 |
| **Author** | Jack YUAN |
| **License** | MIT |
| **Python Version** | 3.10+ |
| **Repository Type** | Monolith |
| **Primary Language** | Python |

## Technology Stack

### Core Technologies

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Language** | Python | 3.10+ | Primary implementation language |
| **Embedding API** | Google Gemini | gemini-embedding-001 | 768-dim text embeddings |
| **ML Framework** | scikit-learn | ≥1.7.2 | K-Means clustering, metrics |
| **Numerical** | NumPy | ≥1.24.0 | Array operations |
| **Data Processing** | Pandas | ≥2.0.0 | DataFrame manipulation |
| **Dataset** | Hugging Face datasets | ≥2.14.0 | AG News (120K articles) |

### Supporting Tools

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Visualization** | Matplotlib | ≥3.7.0 | Static plots |
| **Visualization** | Seaborn | ≥0.12.0 | Statistical charts |
| **Configuration** | PyYAML | ≥6.0 | Config file parsing |
| **Environment** | python-dotenv | ≥1.0.0 | .env loading |
| **Retry Logic** | tenacity | ≥8.0.0 | API retry handling |
| **Testing** | pytest | ≥7.4.0 | Unit/integration tests |
| **Code Quality** | Ruff | ≥0.1.0 | Linting + formatting |
| **Notebooks** | Jupyter | ≥1.0.0 | Interactive analysis |

## Architecture Classification

### Repository Structure
- **Type**: Monolith
- **Parts**: Single cohesive codebase
- **Root**: `/Users/yuanzheyi/.../report/`

### Architecture Pattern
**Type**: Layered Data Science Pipeline

**Layers**:
1. **Data Layer**: Dataset loading (AG News)
2. **Feature Layer**: Embedding generation (Gemini API) + caching
3. **Model Layer**: K-Means clustering + specialized agents
4. **Evaluation Layer**: Metrics (silhouette, purity) + cost analysis
5. **Visualization Layer**: PCA plots, cluster analysis charts
6. **Utility Layer**: Configuration, reproducibility, logging

**Design Principles**:
- Configuration-driven (YAML + .env)
- Cache-optimized (embedding reuse)
- Reproducible (fixed random seeds)
- Modular (clean layer separation)

## Key Features

### 1. Cost-Efficient Embedding Generation
- **Batch API**: Uses Gemini Batch API ($0.075/1M tokens vs $0.15/1M standard)
- **Caching System**: Stores embeddings locally to avoid redundant API calls
- **Checkpoint Recovery**: Resumes interrupted embedding generation
- **Target Cost**: <$5 for full 120K dataset

### 2. Semantic Clustering
- **Algorithm**: K-Means (k=4)
- **Input**: 768-dim Gemini embeddings
- **Output**: Cluster assignments + centroids + metadata
- **Quality Metrics**: Silhouette score, Davies-Bouldin index, cluster purity

### 3. Multi-Agent Classification
- **Architecture**: 4 specialized agents (one per cluster)
- **Context Reduction**: Each agent holds ~25% of documents (75% reduction)
- **Routing**: Cosine similarity between query and cluster centroids
- **Cost Savings**: 90%+ reduction vs. single-agent baseline

### 4. Comprehensive Evaluation
- **Clustering Quality**: Silhouette, Davies-Bouldin, purity metrics
- **Visualization**: PCA 2D projections (300 DPI publication quality)
- **Cost Analysis**: Detailed API cost tracking and reporting
- **Representative Sampling**: Top-10 documents per cluster

## Repository Structure

```
context-aware-multi-agent-system/
├── src/                        # Main application code
│   └── context_aware_multi_agent_system/
│       ├── data/              # Dataset loading
│       ├── features/          # Embedding generation + cache
│       ├── models/            # Clustering + agents
│       ├── evaluation/        # Metrics + analysis
│       ├── visualization/     # Plots
│       └── utils/             # Config + reproducibility
├── scripts/                   # Executable pipelines (01-06)
├── tests/                     # pytest suite (epic1, epic2, epic3)
├── docs/                      # Documentation (PRD, architecture, specs)
├── data/                      # Data directory (gitignored)
├── results/                   # Experiment outputs (JSON)
├── visualizations/            # Generated charts (PNG, HTML)
├── models/                    # Trained models
├── notebooks/                 # Jupyter notebooks
└── reports/                   # Generated reports
```

## Development Workflow

### Epic Structure
The project is organized into 4 epics following BMM (BMAD Method) methodology:

1. **Epic 1**: Project Initialization and Foundation
   - Environment setup, configuration management
   - Dataset loading, API authentication
   - **Status**: ✅ Completed

2. **Epic 2**: Clustering and Evaluation
   - Embedding generation with caching
   - K-Means clustering implementation
   - Quality evaluation and visualization
   - **Status**: ✅ Completed

3. **Epic 3**: Multi-Agent Classification System
   - Specialized agent implementation
   - Cosine similarity classification engine
   - Agent router with query routing
   - **Status**: 🔄 In Progress (Story 3-1 in review)

4. **Epic 4**: Baseline Comparison and Reporting
   - Baseline system implementation
   - Cost calculation engine
   - Comprehensive metrics collection
   - Visualization suite and experimental report
   - **Status**: 📋 Backlog

### Sprint Tracking
Progress tracked in `docs/sprint-status.yaml` with story-level granularity.

## Getting Started

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Google Gemini API key

### Quick Start
```bash
# 1. Clone and setup
git clone <repository-url>
cd context-aware-multi-agent-system

# 2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Edit .env and add: GEMINI_API_KEY=your_api_key_here

# 5. Run pipeline
python scripts/01_generate_embeddings.py
python scripts/02_train_clustering.py
python scripts/03_evaluate_clustering.py
python scripts/04_visualize_clusters.py
python scripts/05_analyze_clusters.py
python scripts/06_initialize_agents.py
```

## Documentation Index

### Core Documentation
- **[README.md](../README.md)**: Quick start guide and usage examples
- **[PRD.md](PRD.md)**: Product requirements and success criteria
- **[architecture.md](architecture.md)**: System architecture and design decisions
- **[epics.md](epics.md)**: Epic breakdown and story mapping

### Technical Specifications
- **[tech-spec-epic-1.md](tech-spec-epic-1.md)**: Epic 1 technical details
- **[tech-spec-epic-2.md](tech-spec-epic-2.md)**: Epic 2 technical details
- **[tech-spec-epic-3.md](tech-spec-epic-3.md)**: Epic 3 technical details

### Implementation Documentation
- **[source-tree-analysis.md](source-tree-analysis.md)**: Detailed code structure analysis
- **[development-guide.md](development-guide.md)**: Development setup and workflows _(To be generated)_
- **[sprint-status.yaml](sprint-status.yaml)**: Current sprint progress

### Story Documentation
See `docs/stories/` for individual user story implementations with acceptance criteria.

### Retrospectives
See `docs/retrospectives/` for epic completion retrospectives.

## Project Goals

### Primary Objectives
1. **Cost Reduction**: Achieve 90%+ reduction in LLM API costs vs. baseline
2. **Scalability**: Handle 120K documents efficiently
3. **Quality**: Maintain acceptable classification accuracy
4. **Reproducibility**: Enable exact result replication

### Success Metrics
- ✅ Embedding cost: <$5 USD (target achieved)
- ✅ Cluster quality: Silhouette score >0.28 (achieved 0.284)
- 🔄 Classification accuracy: >85% (Epic 3 in progress)
- 🔄 Cost reduction: >90% vs. baseline (Epic 4 planned)

## Current Status

**Active Epic**: Epic 3 (Multi-Agent Classification System)
**Current Story**: 3-1 (Specialized Agent Implementation) - In Review
**Completed Epics**: 2/4
**Completed Stories**: 10/20
**Overall Progress**: 50%

## Next Steps

1. **Complete Epic 3**: Finish multi-agent classification implementation
   - Cosine similarity classification engine
   - Agent router with query routing
   - Classification accuracy measurement
   - Performance benchmarking

2. **Begin Epic 4**: Baseline comparison and reporting
   - Implement baseline single-agent system
   - Calculate comprehensive cost metrics
   - Generate experimental report
   - Create final visualization suite

3. **Documentation**: Complete code documentation and README refinements

## Contact

**Author**: Jack YUAN
**Project**: CSIT5210 Course Project
**Institution**: Hong Kong University of Science and Technology

## License

MIT License - See LICENSE file for details

---

**Last Updated**: 2025-11-09
**Documentation Version**: 1.0
**Project Version**: 0.1.0
