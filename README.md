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
