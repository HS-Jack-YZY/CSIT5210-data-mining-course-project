# report - Epic Breakdown

**Author:** Jack YUAN
**Date:** 2025-11-09
**Project Level:** Level 1 - Small Feature
**Target Scale:** Academic Course Project (3-day implementation)

---

## Overview

This document provides the complete epic and story breakdown for report, decomposing the requirements from the [PRD](./PRD.md) into implementable stories.

### Epic Summary

This project is decomposed into 4 epics following a logical progression from foundation to validation:

1. **Epic 1: Foundation & Environment Setup** - Establish project infrastructure and data pipeline
2. **Epic 2: Embedding & Clustering Engine** - Implement core K-Means clustering for semantic partitioning
3. **Epic 3: Multi-Agent Classification System** - Build cosine similarity-based query routing
4. **Epic 4: Cost Analysis & Experimental Validation** - Prove cost reduction and generate academic deliverables

**Implementation Sequence:** Epic 1 → Epic 2 → Epic 3 → Epic 4

**Timeline Alignment:**
- Day 1: Epic 1 + Epic 2 (Foundation & Clustering)
- Day 2: Epic 3 (Classification & Routing)
- Day 3: Epic 4 (Validation & Reporting)

---

## Epic 1: Foundation & Environment Setup

**Goal:** Establish the foundational project infrastructure, development environment, and data pipeline that enables all subsequent clustering and classification work.

**Value:** This epic creates the necessary scaffolding for the academic project—without proper setup, data loading, and configuration, no clustering or cost optimization experiments can proceed.

### Story 1.1: Project Initialization and Environment Setup

As a **data mining student**,
I want **a properly configured Python project with all dependencies and structure**,
So that **I can begin implementing clustering and classification algorithms without environment issues**.

**Acceptance Criteria:**

**Given** I am starting a new Python project
**When** I set up the development environment
**Then** the project has the following structure:
```
project/
├── data/
│   ├── raw/
│   ├── embeddings/
│   └── clusters/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── embedding_service.py
│   ├── clustering.py
│   ├── agent.py
│   ├── router.py
│   ├── baseline.py
│   └── metrics.py
├── notebooks/
│   └── analysis.ipynb
├── visualizations/
├── results/
├── .env.example
├── .gitignore
├── config.yaml
├── requirements.txt
└── README.md
```

**And** requirements.txt includes:
- google-genai>=0.3.0
- scikit-learn>=1.7.2
- numpy>=1.24
- pandas>=2.0
- datasets>=2.14
- matplotlib>=3.7
- seaborn>=0.12

**And** .gitignore includes .env, data/, *.pyc, __pycache__/

**And** .env.example contains GEMINI_API_KEY template

**And** config.yaml has clustering parameters (K=4, random_state=42)

**Prerequisites:** None (first story)

**Technical Notes:**
- Use Python 3.9+ for compatibility
- Follow PEP 8 structure conventions
- Create placeholder __init__.py for src package
- Document expected runtime in README

---

### Story 1.2: Configuration Management System

As a **data mining student**,
I want **a centralized configuration system for all experimental parameters**,
So that **I can easily adjust clustering settings and reproduce experiments**.

**Acceptance Criteria:**

**Given** the project structure exists
**When** I implement the configuration module
**Then** config.yaml contains all parameters:
- Dataset settings (name, categories, sample_size)
- Clustering config (algorithm, n_clusters=4, random_state=42, max_iter=300, init='k-means++')
- Embedding config (model='gemini-embedding-001', batch_size=100, cache_dir)
- Classification config (method='cosine_similarity', threshold=0.7)
- Metrics config (cost_per_1M_tokens_under_200k=3.0, cost_per_1M_tokens_over_200k=6.0)

**And** src/config.py loads and validates configuration

**And** configuration values are accessible throughout the codebase

**And** invalid configurations raise informative errors

**Prerequisites:** Story 1.1

**Technical Notes:**
- Use PyYAML for config parsing
- Add type hints for configuration dataclass
- Validate required fields on load
- Support environment variable overrides for API keys

---

### Story 1.3: AG News Dataset Loading and Validation

As a **data mining student**,
I want **to load and validate the AG News dataset**,
So that **I have clean, structured data ready for embedding generation**.

**Acceptance Criteria:**

**Given** the Hugging Face datasets library is installed
**When** I run the dataset loading script
**Then** AG News dataset is loaded with 4 categories (World, Sports, Business, Sci/Tech)

**And** dataset statistics are logged:
- Training samples: 120,000
- Test samples: 7,600
- Category distribution per class

**And** text fields (title + description) are extracted and combined

**And** train/test split is maintained

**And** data is validated:
- No missing text fields
- All 4 categories present
- Samples have expected structure

**And** dataset is cached to data/raw/ to avoid re-downloading

**And** optional sampling parameter allows subset selection for faster experiments

**Prerequisites:** Story 1.1, Story 1.2

**Technical Notes:**
- Use `datasets.load_dataset('ag_news')`
- Combine title + description with separator
- Log category distribution for balance verification
- Handle network errors with retry logic
- Save as CSV for inspection if needed

---

### Story 1.4: Gemini API Integration and Authentication

As a **data mining student**,
I want **secure integration with Google Gemini Embedding API**,
So that **I can generate semantic embeddings for clustering**.

**Acceptance Criteria:**

**Given** I have a valid Gemini API key
**When** I set up the embedding service
**Then** src/embedding_service.py contains EmbeddingService class

**And** API key is loaded from GEMINI_API_KEY environment variable

**And** the service validates API key on initialization with a test call

**And** authentication errors provide clear messages (e.g., "Invalid API key. Set GEMINI_API_KEY environment variable")

**And** the service supports:
- Single embedding generation: `generate_embedding(text: str) -> np.ndarray`
- Batch embedding generation: `generate_embeddings_batch(texts: list) -> list[np.ndarray]`

**And** rate limiting is handled with exponential backoff (max 3 retries)

**And** network errors are caught and logged with retry suggestions

**And** API usage is tracked (number of calls, estimated cost)

**Prerequisites:** Story 1.1, Story 1.2

**Technical Notes:**
- Use google-genai SDK (v0.3.0+)
- Model: 'gemini-embedding-001' (768 dimensions)
- Implement retry decorator for resilience
- Log all API calls for debugging
- Validate embedding dimensions (expected: 768)

---

## Epic 2: Embedding & Clustering Engine

**Goal:** Implement the core K-Means clustering algorithm to partition AG News documents into semantic clusters, demonstrating data mining techniques for course evaluation.

**Value:** This epic delivers the "magic moment" of the project—visual proof that clustering captures meaningful semantic boundaries. The PCA visualization showing 4 distinct clusters validates the entire cost optimization approach.

### Story 2.1: Batch Embedding Generation with Caching

As a **data mining student**,
I want **to generate embeddings for all AG News documents with efficient caching**,
So that **I have vector representations ready for clustering without repeated API calls**.

**Acceptance Criteria:**

**Given** the AG News dataset is loaded and the Gemini API is configured
**When** I run the embedding generation process
**Then** embeddings are generated for all training documents (120,000 samples)

**And** Gemini Batch API is used for cost efficiency ($0.075/1M tokens)

**And** embeddings are cached to data/embeddings/ as .npy files with metadata

**And** on subsequent runs, cached embeddings are loaded (skip API calls)

**And** embedding dimensions are validated (768D for gemini-embedding-001)

**And** progress is logged every 1000 documents processed

**And** total API usage is tracked and reported (calls, tokens, estimated cost)

**And** the process handles:
- Network interruptions (resume from last checkpoint)
- API rate limits (exponential backoff)
- Invalid responses (log and skip)

**Prerequisites:** Story 1.3, Story 1.4

**Technical Notes:**
- Batch size: 100 documents per API call (configurable)
- Save embeddings with document IDs for traceability
- Expected cost: <$5 for full dataset
- Expected time: ~10-15 minutes for full dataset

---

### Story 2.2: K-Means Clustering Implementation

As a **data mining student**,
I want **to apply K-Means clustering to partition documents into K=4 semantic clusters**,
So that **I can demonstrate clustering algorithm mastery for course evaluation**.

**Acceptance Criteria:**

**Given** document embeddings are generated and cached
**When** I run the clustering algorithm
**Then** K-Means clustering is applied with parameters:
- n_clusters = 4
- init = 'k-means++'
- random_state = 42
- max_iter = 300

**And** the algorithm converges successfully (< max_iter)

**And** each document is assigned to exactly one cluster (0-3)

**And** cluster centroids are extracted and stored (4 centroids × 768 dimensions)

**And** cluster assignments are saved to data/clusters/assignments.csv with columns:
- document_id
- cluster_id
- ground_truth_category
- distance_to_centroid

**And** cluster size distribution is logged and validated (no extreme imbalance)

**And** convergence information is logged (iterations, final inertia)

**Prerequisites:** Story 2.1

**Technical Notes:**
- Use scikit-learn KMeans class
- k-means++ initialization reduces convergence time
- Save centroids separately for classification routing
- Expected runtime: <5 minutes for 120K documents

---

### Story 2.3: Cluster Quality Evaluation

As a **data mining student**,
I want **to evaluate cluster quality using standard metrics**,
So that **I can prove the clustering produces good semantic separation**.

**Acceptance Criteria:**

**Given** K-Means clustering is complete
**When** I run cluster quality evaluation
**Then** Silhouette Score is calculated (target: >0.3)

**And** Davies-Bouldin Index is calculated (lower = better)

**And** Intra-cluster distance is computed (compactness metric)

**And** Inter-cluster distance is computed (separation metric)

**And** Cluster purity is calculated by comparing with AG News ground truth categories

**And** confusion matrix shows cluster-to-category alignment

**And** all metrics are saved to results/cluster_quality.json with:
```json
{
  "silhouette_score": 0.35,
  "davies_bouldin_index": 1.2,
  "intra_cluster_distance": {...},
  "inter_cluster_distance": {...},
  "cluster_purity": 0.82,
  "cluster_sizes": [30000, 29500, 30500, 30000]
}
```

**And** evaluation summary is logged to console

**Prerequisites:** Story 2.2

**Technical Notes:**
- Use scikit-learn metrics: silhouette_score, davies_bouldin_score
- Purity = % documents in cluster matching dominant category
- Expected Silhouette >0.3 validates good separation
- Document any outlier clusters

---

### Story 2.4: PCA Cluster Visualization

As a **data mining student**,
I want **a clear 2D visualization showing 4 distinct semantic clusters**,
So that **I can demonstrate clustering effectiveness visually in my report**.

**Acceptance Criteria:**

**Given** clustering is complete with quality metrics
**When** I generate the cluster visualization
**Then** PCA dimensionality reduction projects 768D embeddings to 2D

**And** a scatter plot is created with:
- 4 colored clusters (distinct colors for each cluster)
- Cluster centroids marked with special symbols (e.g., stars)
- Legend mapping colors to cluster IDs
- Axis labels ("PC1" and "PC2" with variance explained)
- Title: "K-Means Clustering of AG News (K=4, PCA Projection)"

**And** plot shows clear visual separation between clusters

**And** visualization is saved to visualizations/cluster_pca.png (300 DPI)

**And** optional: interactive Plotly version saved as cluster_pca.html

**And** variance explained by PC1 and PC2 is logged (should be >20% combined)

**Prerequisites:** Story 2.2, Story 2.3

**Technical Notes:**
- Use scikit-learn PCA with n_components=2
- Use seaborn or matplotlib for publication-quality plots
- Colorblind-friendly palette (e.g., colorbrewer Set2)
- This is the "wow moment" visualization for the report
- Consider sampling 10K points if plot is too dense

---

### Story 2.5: Cluster Analysis and Labeling

As a **data mining student**,
I want **to analyze and semantically label each cluster**,
So that **I understand what topics each cluster represents**.

**Acceptance Criteria:**

**Given** clustering and visualization are complete
**When** I analyze cluster contents
**Then** for each cluster (0-3), the system:
- Maps cluster to dominant AG News category (World/Sports/Business/Sci-Tech)
- Extracts top 10 most representative documents (closest to centroid)
- Calculates cluster purity (% documents matching dominant category)

**And** cluster analysis report is saved to results/cluster_analysis.txt with:
```
Cluster 0: Sports (Purity: 85%)
  - Top documents: [list of 10 sample headlines]
  - Size: 30,000 documents
  - Alignment: 85% Sports, 10% World, 3% Business, 2% Sci/Tech

Cluster 1: World (Purity: 82%)
  ...
```

**And** cluster semantic labels are saved to results/cluster_labels.json

**And** misclassified documents are identified for manual inspection (optional)

**Prerequisites:** Story 2.2, Story 2.3

**Technical Notes:**
- Cluster labels help validate semantic coherence
- High purity (>80%) indicates good clustering
- Representative documents useful for demo presentation
- This analysis validates clustering captures real semantic boundaries

---

## Epic 3: Multi-Agent Classification System

**Goal:** Build a classification and routing system that uses cosine similarity to intelligently route queries to specialized agents, demonstrating classification techniques and enabling cost optimization.

**Value:** This epic transforms clustering into cost savings—by routing queries to the right agent (with only 1/K of the context), we achieve the 90%+ cost reduction goal while maintaining response accuracy.

### Story 3.1: Specialized Agent Implementation

As a **data mining student**,
I want **specialized agents that each maintain only documents from one cluster**,
So that **I can demonstrate context reduction for cost optimization**.

**Acceptance Criteria:**

**Given** cluster assignments are complete
**When** I implement the SpecializedAgent class
**Then** src/agent.py contains SpecializedAgent with:
```python
class SpecializedAgent:
    def __init__(self, cluster_id: int, documents: list, cluster_label: str)
    def get_context_size(self) -> int
    def get_documents(self) -> list
    def process_query(self, query: str) -> str  # For MVP, can be stub
```

**And** 4 agent instances are created (one per cluster)

**And** each agent is assigned only documents from its cluster (~30K docs each, not all 120K)

**And** agent registry maps cluster_id → agent instance

**And** each agent logs its initialization:
- cluster_id
- cluster_label (e.g., "Sports")
- number of documents
- total token count (context size)

**And** context size reduction is calculated and logged:
- Baseline: 100% (all documents)
- Per agent: ~25% (1/K where K=4)

**Prerequisites:** Story 2.2, Story 2.5

**Technical Notes:**
- For MVP, agents don't need actual LLM integration (focus on routing)
- Context reduction is the key metric (1/4 of baseline)
- Each agent maintains documents as text or IDs for retrieval
- Future: can integrate with LLM for actual query processing

---

### Story 3.2: Cosine Similarity Classification Engine

As a **data mining student**,
I want **to classify queries to clusters using cosine similarity**,
So that **I can demonstrate classification algorithm mastery for course evaluation**.

**Acceptance Criteria:**

**Given** cluster centroids are stored and query embeddings can be generated
**When** I implement the classification engine
**Then** src/router.py contains classification logic:
```python
def classify_query(query_embedding: np.ndarray, centroids: np.ndarray) -> int
def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float
```

**And** for a given query embedding:
- Cosine similarity is computed with all 4 cluster centroids
- Query is assigned to cluster with highest similarity score
- Similarity scores for all clusters are returned

**And** classification is fast (<10ms computation time, excluding embedding generation)

**And** the implementation:
- Uses numpy for efficient computation
- Handles edge cases (zero vectors, NaN values)
- Logs classification decisions with similarity scores

**And** classification function is tested on sample queries

**Prerequisites:** Story 2.2, Story 3.1

**Technical Notes:**
- Cosine similarity = dot(A,B) / (norm(A) * norm(B))
- Range: [-1, 1], higher = more similar
- Zero training time (unlike neural classifiers)
- Use numpy.dot() and numpy.linalg.norm() for efficiency

---

### Story 3.3: Agent Router with Query Routing

As a **data mining student**,
I want **a router that classifies queries and routes them to the correct specialized agent**,
So that **I can demonstrate the complete classification and routing pipeline**.

**Acceptance Criteria:**

**Given** specialized agents exist and classification engine works
**When** I implement the AgentRouter class
**Then** src/router.py contains AgentRouter with:
```python
class AgentRouter:
    def __init__(self, agents: dict, centroids: np.ndarray, embedding_service: EmbeddingService)
    def route_query(self, query: str) -> tuple[SpecializedAgent, dict]
```

**And** routing workflow for a query:
1. Generate query embedding using EmbeddingService
2. Classify query to cluster using cosine similarity
3. Retrieve corresponding SpecializedAgent from registry
4. Return agent + routing metadata

**And** routing metadata includes:
- assigned_cluster_id
- similarity_scores (for all 4 clusters)
- classification_confidence (max similarity score)
- routing_time (milliseconds)

**And** routing decision is logged:
- Query text (truncated)
- Assigned cluster
- Confidence score
- Routing time

**And** total routing time is <1 second (including embedding generation)

**Prerequisites:** Story 3.1, Story 3.2

**Technical Notes:**
- Router is the main interface for query processing
- Embedding generation is the slowest step (~500-800ms)
- Classification is nearly instant (~5-10ms)
- Router enables A/B testing (route to multiple agents if needed)

---

### Story 3.4: Classification Accuracy Measurement

As a **data mining student**,
I want **to measure classification accuracy on the AG News test set**,
So that **I can prove the system achieves >80% routing accuracy**.

**Acceptance Criteria:**

**Given** the router is implemented and test set exists (7,600 samples)
**When** I run classification accuracy evaluation
**Then** all test queries are classified to clusters

**And** predicted clusters are compared with ground truth categories

**And** overall accuracy is calculated (% correct cluster assignments)

**And** per-cluster metrics are computed:
- Precision (% of assigned queries that belong to cluster)
- Recall (% of cluster's queries correctly identified)
- F1-score (harmonic mean of precision and recall)

**And** confusion matrix (4×4) is generated showing:
- Rows: ground truth categories
- Columns: predicted clusters
- Diagonal: correct classifications

**And** all metrics are saved to results/classification_accuracy.json:
```json
{
  "overall_accuracy": 0.83,
  "per_cluster_metrics": {...},
  "confusion_matrix": [[...], [...], [...], [...]],
  "misclassification_rate": 0.17
}
```

**And** confusion matrix visualization is saved to visualizations/confusion_matrix.png

**And** accuracy summary is logged to console

**Prerequisites:** Story 3.3, Story 2.1

**Technical Notes:**
- Use scikit-learn: accuracy_score, precision_recall_fscore_support
- Target: >80% overall accuracy validates routing effectiveness
- Confusion matrix reveals which categories are confused
- Low accuracy (<80%) would indicate poor cluster separation

---

### Story 3.5: Query Processing Performance Benchmarking

As a **data mining student**,
I want **to measure and optimize query processing performance**,
So that **I can demonstrate the system meets <1 second routing requirement**.

**Acceptance Criteria:**

**Given** the router is fully implemented
**When** I run performance benchmarking
**Then** 100 sample queries are processed and timed

**And** timing breakdown is measured for each query:
- Embedding generation time
- Classification computation time
- Agent retrieval time
- Total routing time

**And** performance statistics are calculated:
- Mean routing time
- Median routing time
- 95th percentile routing time
- Max routing time

**And** all queries meet <1 second requirement (excluding LLM call)

**And** results are saved to results/routing_performance.json

**And** performance bottlenecks are identified:
- If embedding >900ms: note API latency
- If classification >50ms: note inefficiency

**And** performance summary is logged

**Prerequisites:** Story 3.3

**Technical Notes:**
- Use time.perf_counter() for precise timing
- Embedding generation is expected bottleneck (~500-800ms)
- Classification should be <10ms (pure computation)
- Future optimization: batch embedding for multiple queries
- This validates system is fast enough for real-time use

---

## Epic 4: Cost Analysis & Experimental Validation

**Goal:** Prove the cost optimization hypothesis through rigorous baseline comparison, generate comprehensive metrics, produce high-quality visualizations, and deliver the complete experimental report for academic submission.

**Value:** This epic provides the academic deliverables—quantitative proof of >90% cost reduction, publication-quality visualizations, and a complete experimental report connecting data mining theory to real-world LLM cost optimization.

### Story 4.1: Baseline System Implementation

As a **data mining student**,
I want **a naive baseline system that sends all context to a single agent**,
So that **I can quantify the cost savings from clustering-based routing**.

**Acceptance Criteria:**

**Given** the AG News dataset and embedding service exist
**When** I implement the baseline system
**Then** src/baseline.py contains BaselineSystem class with:
```python
class BaselineSystem:
    def __init__(self, all_documents: list, embedding_service: EmbeddingService)
    def process_query(self, query: str) -> dict
    def get_context_size(self) -> int
```

**And** the baseline system:
- Maintains ALL 120,000 training documents (100% context)
- Simulates sending entire context for every query
- Counts total tokens in context
- Estimates API cost per query based on context size

**And** baseline context size is logged:
- Total documents: 120,000
- Total tokens: ~200K+ (triggers higher pricing tier)
- Cost per query: $1.20-$1.80 (depending on exact token count)

**And** for test queries, baseline tracks:
- Number of queries processed
- Total tokens sent (queries × context_size)
- Total estimated cost

**And** baseline results are saved to results/baseline_metrics.json

**Prerequisites:** Story 1.3, Story 1.4

**Technical Notes:**
- No actual LLM calls needed (focus on token counting)
- Use tiktoken or simple word-based estimation for token counts
- Simulate the "current practice" of sending everything
- This is the comparison benchmark for cost savings

---

### Story 4.2: Cost Calculation Engine

As a **data mining student**,
I want **accurate cost calculations using actual LLM pricing tiers**,
So that **I can demonstrate realistic cost savings to course evaluators**.

**Acceptance Criteria:**

**Given** token counts from both baseline and optimized systems
**When** I implement the cost calculation engine
**Then** src/metrics.py contains cost calculation functions:
```python
def calculate_llm_cost(token_count: int, pricing_config: dict) -> float
def compare_costs(baseline_cost: float, optimized_cost: float) -> dict
```

**And** pricing tiers are applied correctly:
- <200K tokens: $3.00 per 1M input tokens
- ≥200K tokens: $6.00 per 1M input tokens (Claude Sonnet 4.5)

**And** for each query:
- Baseline cost = f(full_context_tokens)
- Optimized cost = f(cluster_context_tokens)  # ~1/4 of baseline
- Savings = baseline_cost - optimized_cost
- Reduction % = (savings / baseline_cost) × 100

**And** aggregate metrics are computed:
- Total baseline cost for N queries
- Total optimized cost for N queries
- Average cost per query (both systems)
- Overall cost reduction percentage

**And** cost comparison results are saved to results/cost_comparison.json:
```json
{
  "baseline": {
    "total_cost": 180.00,
    "cost_per_query": 1.80,
    "avg_tokens_per_query": 210000
  },
  "optimized": {
    "total_cost": 15.00,
    "cost_per_query": 0.15,
    "avg_tokens_per_query": 52500
  },
  "savings": {
    "total_saved": 165.00,
    "reduction_percentage": 91.67,
    "cost_multiplier": 12.0
  }
}
```

**Prerequisites:** Story 4.1, Story 3.1

**Technical Notes:**
- Use actual Claude Sonnet 4.5 pricing from Anthropic
- Token counting must be accurate (affects cost calculation)
- Consider edge cases (empty context, single document)
- Target: demonstrate >90% reduction

---

### Story 4.3: Comprehensive Metrics Collection

As a **data mining student**,
I want **to collect all performance metrics in one place**,
So that **I have complete data for my experimental report**.

**Acceptance Criteria:**

**Given** all system components are implemented
**When** I run the comprehensive metrics collection
**Then** the following metrics are collected and saved:

**Clustering Metrics** (from Epic 2):
- Silhouette Score
- Davies-Bouldin Index
- Cluster purity
- Cluster size distribution

**Classification Metrics** (from Epic 3):
- Overall accuracy
- Per-cluster precision, recall, F1
- Confusion matrix

**Cost Metrics** (from Epic 4):
- Baseline vs optimized comparison
- Cost reduction percentage
- Token consumption comparison

**Performance Metrics** (from Epic 3):
- Average routing time
- Embedding generation time
- Classification computation time

**And** all metrics are consolidated into results/final_metrics.json

**And** a metrics summary table is generated for easy reference

**And** metrics are validated:
- Silhouette Score >0.3 ✅
- Classification accuracy >80% ✅
- Cost reduction >90% ✅
- Routing time <1s ✅

**Prerequisites:** Story 2.3, Story 3.4, Story 4.2

**Technical Notes:**
- Centralized metrics module avoids duplication
- JSON format enables easy parsing for report generation
- Summary table can be markdown or CSV
- Metrics validation ensures success criteria are met

---

### Story 4.4: Visualization Suite Generation

As a **data mining student**,
I want **publication-quality visualizations for my experimental report**,
So that **I can visually demonstrate clustering, classification, and cost savings**.

**Acceptance Criteria:**

**Given** all metrics are collected
**When** I generate the visualization suite
**Then** the following visualizations are created:

**1. Cluster PCA Plot** (already done in Story 2.4):
- 4 colored clusters with centroids
- Clear separation visible
- 300 DPI PNG

**2. Cost Comparison Bar Chart**:
- Side-by-side bars: baseline vs optimized
- Y-axis: total cost ($)
- Annotated with % reduction
- Saved to visualizations/cost_comparison.png

**3. Confusion Matrix Heatmap** (already done in Story 3.4):
- 4×4 grid showing classification performance
- Color-coded cells
- 300 DPI PNG

**4. Silhouette Analysis Plot**:
- Silhouette coefficient per cluster
- Average line indicating overall score
- Saved to visualizations/silhouette_analysis.png

**5. Performance Breakdown Chart**:
- Stacked bar or pie chart showing time breakdown
- Embedding vs classification vs retrieval
- Saved to visualizations/performance_breakdown.png

**And** all visualizations use consistent:
- Color palette (colorblind-friendly)
- Font sizes (readable in report)
- Professional styling
- High resolution (300 DPI)

**And** visualizations are saved to visualizations/ directory

**Prerequisites:** Story 4.3

**Technical Notes:**
- Use matplotlib/seaborn for static plots
- Consistent figure size (e.g., 10×6 inches)
- Include titles, axis labels, legends
- These visualizations are for academic report inclusion

---

### Story 4.5: Experimental Report Generation

As a **data mining student**,
I want **a complete experimental report documenting methodology, results, and analysis**,
So that **I can submit this as my course project deliverable**.

**Acceptance Criteria:**

**Given** all metrics and visualizations are ready
**When** I generate the experimental report
**Then** a markdown report is created at results/experimental_report.md with sections:

**1. Introduction**
- Problem: LLM cost optimization
- Approach: Clustering + classification
- Objectives: Demonstrate data mining techniques

**2. Methodology**
- Dataset: AG News (4 categories, 120K training, 7.6K test)
- Embedding: Gemini Embedding API (768D)
- Clustering: K-Means (K=4, k-means++, random_state=42)
- Classification: Cosine similarity with centroids
- Evaluation: Silhouette Score, accuracy, cost comparison

**3. Experimental Setup**
- Technology stack listed
- Configuration parameters documented
- Reproducibility information

**4. Results**
- **Clustering Results:**
  - Silhouette Score: {value}
  - PCA visualization embedded
  - Cluster purity metrics
- **Classification Results:**
  - Overall accuracy: {value}
  - Confusion matrix embedded
  - Per-cluster performance
- **Cost Optimization Results:**
  - Baseline cost: {value}
  - Optimized cost: {value}
  - Reduction: {value}%
  - Cost comparison chart embedded

**5. Discussion**
- Analysis of results
- Connection to K-Means theory (initialization, convergence, centroids)
- Connection to classification theory (cosine similarity, embedding space)
- Why clustering works for cost optimization
- Limitations and potential improvements

**6. Conclusion**
- Summary: Achieved {X}% cost reduction with {Y}% accuracy
- Validation of clustering and classification for LLM optimization
- Implications for real-world applications

**And** all visualizations are embedded or referenced

**And** report is formatted professionally (academic style)

**And** report is also exported as PDF: results/experimental_report.pdf

**Prerequisites:** Story 4.3, Story 4.4

**Technical Notes:**
- Use markdown for easy editing, convert to PDF via pandoc or similar
- Include quantitative results in tables
- Reference figures by number
- Academic writing tone
- Cite relevant papers (optional but good practice)

---

### Story 4.6: Code Documentation and README

As a **data mining student**,
I want **comprehensive documentation for code and usage**,
So that **course instructors can easily run and verify my project**.

**Acceptance Criteria:**

**Given** the project is complete
**When** I finalize documentation
**Then** README.md contains:

**1. Project Overview**
- Description: LLM cost optimization using clustering + classification
- Key results: {X}% cost reduction, {Y}% accuracy

**2. Installation**
```bash
pip install -r requirements.txt
```
- Python version: 3.9+
- API key setup: GEMINI_API_KEY

**3. Usage**
- Step-by-step instructions to run the project
- Expected outputs for each step
- Runtime estimates

**4. Project Structure**
- Directory tree explanation
- Key files description

**5. Results Summary**
- Link to experimental report
- Link to visualizations
- Key metrics highlighted

**6. Reproducibility**
- Fixed random seeds documented
- Configuration file explained
- Expected metric ranges

**And** all Python files have:
- Module docstrings
- Function docstrings (Google style)
- Type hints
- Inline comments for complex logic

**And** config.yaml has inline comments explaining each parameter

**And** a USAGE.md or demo notebook shows example workflow

**Prerequisites:** All previous stories

**Technical Notes:**
- README is the first thing instructors see
- Clear, concise, professional
- Include badges if applicable (Python version, etc.)
- Provide troubleshooting section if helpful

---

### Story 4.7: Final Validation and Testing

As a **data mining student**,
I want **to validate that all success criteria are met before submission**,
So that **I can confidently submit a complete project**.

**Acceptance Criteria:**

**Given** all stories are implemented
**When** I run final validation
**Then** all success criteria are verified:

**Academic Success Criteria:**
- ✅ K-Means clustering correctly partitions data
- ✅ Silhouette Score >0.3 achieved
- ✅ PCA visualization shows 4 distinct clusters
- ✅ Cosine similarity classification >80% accuracy
- ✅ Confusion matrix generated
- ✅ Experimental results reproducible
- ✅ Real-world problem connection clear

**Technical Success Metrics:**
- ✅ Cost reduction >90% demonstrated
- ✅ Classification accuracy >80%
- ✅ Silhouette Score >0.3
- ✅ Query routing <1 second
- ✅ Clear cluster visualization

**Deliverables Quality:**
- ✅ Clean Python code (PEP 8)
- ✅ Well-commented explaining algorithms
- ✅ Reproducible with clear instructions
- ✅ Experimental report complete
- ✅ All visualizations high quality

**And** a validation checklist is completed and saved to results/validation_checklist.md

**And** any failing criteria are identified and addressed

**And** all tests pass (if unit tests were written)

**And** the project is ready for submission

**Prerequisites:** All previous stories in Epic 4

**Technical Notes:**
- Create automated validation script if time permits
- Manual checklist acceptable for MVP
- Document any deviations from targets with justification
- Ensure git repository is clean and organized

---

## Epic Breakdown Summary

### Overview

This epic breakdown transforms the PRD for the Context-Aware Multi-Agent System into **21 implementable user stories** organized across **4 strategic epics**. Each story is sized for single-session completion by a development agent and follows BDD-style acceptance criteria for clarity.

### Epic Distribution

| Epic | Stories | Focus Area | Timeline |
|------|---------|------------|----------|
| **Epic 1: Foundation & Environment Setup** | 4 stories | Infrastructure, data pipeline, API integration | Day 1 (morning) |
| **Epic 2: Embedding & Clustering Engine** | 5 stories | K-Means clustering, quality evaluation, visualization | Day 1 (afternoon) |
| **Epic 3: Multi-Agent Classification System** | 5 stories | Cosine similarity routing, accuracy measurement | Day 2 |
| **Epic 4: Cost Analysis & Experimental Validation** | 7 stories | Baseline comparison, metrics, reporting | Day 3 |
| **Total** | **21 stories** | Complete academic project | **3 days** |

### Requirements Coverage

All **14 functional requirements** from the PRD are fully covered:

- **FR-1 to FR-2:** Epic 1 (Stories 1.3, 1.4, 2.1)
- **FR-3 to FR-5:** Epic 2 (Stories 2.2, 2.3, 2.4, 2.5)
- **FR-6 to FR-8:** Epic 3 (Stories 3.1, 3.2, 3.3, 3.4)
- **FR-9 to FR-10:** Epic 4 (Stories 4.1, 4.2)
- **FR-11:** Epic 4 (Story 4.3)
- **FR-12:** Epic 4 (Story 4.5)
- **FR-13:** Epic 4 (Story 4.6)
- **FR-14:** Epic 2 & 4 (Stories 2.4, 4.4)

All **8 non-functional requirements** are addressed through quality attributes in stories:
- **NFR-1 (Performance):** Story 3.5
- **NFR-2 (Cost Efficiency):** Story 4.2
- **NFR-3 (Reliability):** Story 1.4 (error handling)
- **NFR-4 (Reproducibility):** Story 4.7
- **NFR-5 (Usability):** Story 4.6
- **NFR-6 (Maintainability):** All stories (modular design)
- **NFR-7 (Compatibility):** Story 1.1
- **NFR-8 (Security):** Story 1.4 (API key management)

### Story Dependencies

The epic breakdown follows a strict dependency flow with no forward dependencies:

```
Epic 1 (Foundation)
├─ Story 1.1 → Story 1.2 → Story 1.3 → Story 1.4
                                ↓              ↓
Epic 2 (Clustering)
├─ Story 2.1 ──────────────────────────────────┘
   └─ Story 2.2 → Story 2.3 → Story 2.4
      └─ Story 2.5
            ↓
Epic 3 (Classification)
├─ Story 3.1 → Story 3.2 → Story 3.3 → Story 3.4
                                 └─ Story 3.5
                                       ↓
Epic 4 (Validation)
├─ Story 4.1 → Story 4.2 → Story 4.3 → Story 4.4 → Story 4.5 → Story 4.6 → Story 4.7
```

### Key Deliverables by Epic

**Epic 1:** Project structure, configuration system, AG News dataset, Gemini API integration

**Epic 2:** Document embeddings, K-Means clustering, cluster quality metrics, PCA visualization, cluster analysis

**Epic 3:** Specialized agents, cosine similarity classifier, query router, classification accuracy metrics, performance benchmarks

**Epic 4:** Baseline system, cost comparison engine, comprehensive metrics, visualization suite, experimental report, documentation, validation

### Success Validation

The final story (4.7) validates that all success criteria are met:
- ✅ **Academic:** K-Means and cosine similarity mastery demonstrated
- ✅ **Technical:** >90% cost reduction, >80% accuracy, Silhouette >0.3
- ✅ **Deliverables:** Clean code, experimental report, visualizations

## Epic 5: Alternative Clustering Algorithms Exploration

**Goal:** Implement and evaluate alternative clustering algorithms (DBSCAN, Hierarchical Clustering, Gaussian Mixture Models) to compare with K-Means, addressing the documented limitations of K-Means on high-dimensional text embeddings.

**Value:** This epic provides deeper academic insights by comparing multiple clustering methods, demonstrating comprehensive understanding of data mining algorithms, and providing scientific evidence for algorithm selection in high-dimensional text clustering tasks.

**Academic Context:** This epic addresses the findings from Epic 2 where K-Means showed poor performance (Silhouette Score ≈0.0008, cluster purity ≈25%). By exploring alternative algorithms, we validate whether the issue is algorithm-specific or task-specific (curse of dimensionality).

### Story 5.1: DBSCAN Density-Based Clustering Implementation

As a **data mining student**,
I want **to apply DBSCAN clustering to the AG News embeddings**,
So that **I can evaluate whether density-based clustering performs better than K-Means on high-dimensional text data**.

**Acceptance Criteria:**

**Given** document embeddings are generated and cached (from Story 2.1)
**When** I run DBSCAN clustering algorithm
**Then** DBSCAN is applied with parameters:
- eps = 0.5 (initial, to be tuned)
- min_samples = 5 (configurable)
- metric = 'cosine' (appropriate for text embeddings)

**And** clustering results are generated including:
- Core samples identification
- Cluster assignments (including noise points labeled as -1)
- Number of clusters discovered (may differ from K=4)
- Number of noise points

**And** cluster assignments are saved to data/clusters/dbscan_assignments.csv with columns:
- document_id
- cluster_id (-1 for noise)
- ground_truth_category
- is_core_sample (boolean)

**And** parameter tuning is performed:
- Test multiple eps values (0.3, 0.5, 0.7, 1.0)
- Test multiple min_samples values (3, 5, 10)
- Select parameters that maximize Silhouette Score
- Document parameter selection rationale

**And** cluster size distribution is logged (including noise cluster size)

**And** convergence and performance metrics are tracked

**Prerequisites:** Story 2.1 (embeddings must exist)

**Technical Notes:**
- Use scikit-learn DBSCAN class
- Cosine metric important for text embeddings (not Euclidean)
- DBSCAN can discover variable number of clusters
- Noise points may indicate outliers or poor fit
- Expected runtime: ~10-15 minutes for 120K documents
- May need dimensionality reduction before DBSCAN for computational efficiency

---

### Story 5.2: Hierarchical Agglomerative Clustering with Dendrogram

As a **data mining student**,
I want **to apply hierarchical clustering and visualize the dendrogram**,
So that **I can understand the hierarchical structure of news categories and compare with flat clustering approaches**.

**Acceptance Criteria:**

**Given** document embeddings are generated and cached
**When** I run agglomerative hierarchical clustering
**Then** Hierarchical clustering is applied with parameters:
- n_clusters = 4 (for comparison with K-Means)
- linkage = 'ward' (minimum variance)
- affinity = 'euclidean' (required for ward)

**And** alternative linkage methods are tested:
- 'complete' (maximum linkage)
- 'average' (average linkage)
- 'single' (minimum linkage)
- Compare results across linkage methods

**And** cluster assignments are saved to data/clusters/hierarchical_assignments.csv

**And** dendrogram visualization is generated showing:
- Hierarchical merge structure
- Cluster boundaries at n_clusters=4
- Color-coded clusters
- Height axis (distance metric)
- Saved to visualizations/dendrogram.png (300 DPI)

**And** for computational efficiency:
- Use sampling if full dataset is too large (sample 10K documents)
- Or use truncated dendrogram for visualization
- Document any sampling strategy used

**And** cluster quality metrics are calculated (Silhouette Score, Davies-Bouldin Index)

**Prerequisites:** Story 2.1

**Technical Notes:**
- Use scikit-learn AgglomerativeClustering
- Ward linkage minimizes within-cluster variance (similar goal to K-Means)
- Dendrogram may be truncated for large datasets (use scipy.cluster.hierarchy)
- Expected runtime: ~15-20 minutes for full dataset (may require sampling)
- Hierarchical clustering provides insights into category relationships

---

### Story 5.3: Gaussian Mixture Model (GMM) Soft Clustering

As a **data mining student**,
I want **to apply Gaussian Mixture Models for probabilistic clustering**,
So that **I can compare soft clustering (probabilistic assignments) with hard clustering (K-Means) approaches**.

**Acceptance Criteria:**

**Given** document embeddings are generated and cached
**When** I run GMM clustering algorithm
**Then** GMM is applied with parameters:
- n_components = 4 (for comparison with K-Means)
- covariance_type = 'full' (initial setting)
- random_state = 42 (reproducibility)
- max_iter = 100

**And** alternative covariance types are tested:
- 'full' (each component has own covariance matrix)
- 'tied' (all components share covariance)
- 'diag' (diagonal covariance)
- 'spherical' (single variance per component)
- Compare BIC/AIC scores across types

**And** probabilistic cluster assignments are extracted:
- Hard assignments: argmax of probability distribution
- Soft assignments: full probability distribution over clusters
- Assignment confidence: max probability value

**And** cluster assignments are saved to data/clusters/gmm_assignments.csv with columns:
- document_id
- cluster_id (hard assignment)
- cluster_probabilities (array of 4 probabilities)
- assignment_confidence (max probability)
- ground_truth_category

**And** GMM-specific metrics are calculated:
- Log-likelihood of the model
- BIC (Bayesian Information Criterion)
- AIC (Akaike Information Criterion)
- Component weights (mixing coefficients)

**And** uncertainty analysis is performed:
- Identify documents with low assignment confidence (<0.5)
- Analyze confusion between cluster pairs
- Compare uncertainty patterns with ground truth categories

**And** convergence information is logged (iterations, final log-likelihood)

**Prerequisites:** Story 2.1

**Technical Notes:**
- Use scikit-learn GaussianMixture class
- GMM assumes Gaussian distributions in feature space
- Soft clustering provides uncertainty estimates (unlike K-Means)
- High-dimensional spaces may cause covariance matrix issues
- Expected runtime: ~5-10 minutes for 120K documents
- Compare convergence speed with K-Means

---

### Story 5.4: Comprehensive Clustering Algorithm Comparison

As a **data mining student**,
I want **a comprehensive comparison of all clustering algorithms tested**,
So that **I can make data-driven recommendations about which algorithm is most suitable for text clustering tasks**.

**Acceptance Criteria:**

**Given** all clustering algorithms have been executed (K-Means, DBSCAN, Hierarchical, GMM)
**When** I run the comparison analysis
**Then** a comparison matrix is created with all algorithms across metrics:

| Metric | K-Means | DBSCAN | Hierarchical | GMM |
|--------|---------|---------|--------------|-----|
| Silhouette Score | X | X | X | X |
| Davies-Bouldin Index | X | X | X | X |
| Cluster Purity | X | X | X | X |
| Number of Clusters | 4 | Variable | 4 | 4 |
| Noise Points | 0 | X | 0 | 0 |
| Runtime (seconds) | X | X | X | X |
| Convergence Iterations | X | N/A | N/A | X |

**And** side-by-side PCA visualizations are generated showing:
- 2x2 or 1x4 subplot layout
- All four algorithms on same 2D PCA projection
- Consistent color mapping where possible
- Cluster centroids or representative points marked
- Saved to visualizations/algorithm_comparison.png (300 DPI)

**And** per-algorithm analysis includes:
- Strengths: when algorithm performed well
- Weaknesses: failure modes and edge cases
- Computational complexity: runtime and scalability
- Parameter sensitivity: how tuning affects results

**And** ground truth alignment analysis:
- Confusion matrix for each algorithm
- Category-to-cluster mapping quality
- Misclassification patterns
- Best-performing algorithm for each category

**And** dimensionality challenge analysis:
- How each algorithm handles 768-dimensional space
- Evidence of curse of dimensionality
- Recommendations for dimensionality reduction preprocessing

**And** comprehensive comparison report is saved to results/clustering_comparison.md with sections:
1. **Methodology**: Datasets, metrics, evaluation approach
2. **Quantitative Results**: Comparison matrix with all metrics
3. **Visual Comparison**: PCA plots side-by-side
4. **Algorithm Analysis**: Strengths/weaknesses of each method
5. **Recommendations**: Which algorithm for which use case
6. **Lessons Learned**: Insights about high-dimensional text clustering

**And** key findings are summarized:
- Best overall algorithm (if any)
- Best algorithm for specific criteria (speed, purity, noise handling)
- Why K-Means failed (validated with alternatives)
- Recommendations for future text clustering projects

**Prerequisites:** Story 5.1, Story 5.2, Story 5.3, Story 2.2, Story 2.3

**Technical Notes:**
- Normalize metrics for fair comparison (some are 0-1, others unbounded)
- Use consistent random seeds across algorithms where applicable
- Consider computational cost vs quality tradeoffs
- This story integrates findings from entire Epic 5
- Results feed into final experimental report (Story 4.5 if extended)

---

### Epic 5 Summary

**Overview:**
Epic 5 extends the K-Means clustering experiment with a comprehensive exploration of alternative clustering algorithms, providing scientific evidence for algorithm selection in high-dimensional text clustering tasks.

**Story Distribution:**

| Story | Algorithm | Focus | Estimated Time |
|-------|-----------|-------|----------------|
| 5.1 | DBSCAN | Density-based clustering, noise handling | 4 hours |
| 5.2 | Hierarchical | Hierarchical structure, dendrogram visualization | 4 hours |
| 5.3 | GMM | Soft clustering, probabilistic assignments | 3 hours |
| 5.4 | Comparison | Comprehensive algorithm comparison and analysis | 5 hours |
| **Total** | **4 algorithms** | **Complete comparison study** | **~16 hours (2 days)** |

**Requirements Coverage:**
This epic addresses the "Growth Features" mentioned in PRD:
- ✅ "Try alternative clustering algorithms (DBSCAN for density-based)"
- ✅ "Cross-validate cluster stability"
- ✅ "Enhanced Clustering" section recommendations

**Key Deliverables:**
- DBSCAN clustering implementation with parameter tuning
- Hierarchical clustering with dendrogram visualization
- GMM soft clustering with uncertainty analysis
- Comprehensive comparison report with side-by-side PCA visualizations
- Algorithm selection recommendations for text clustering

**Academic Value:**
- Demonstrates deep understanding of multiple clustering paradigms
- Provides scientific validation of algorithm-task fit
- Addresses the "why K-Means failed" question with empirical evidence
- Shows professional approach to negative results (testing alternatives)

**Success Validation:**
- All four algorithms implemented and evaluated
- Comprehensive comparison matrix completed
- Clear recommendations for algorithm selection
- Visual and quantitative evidence supporting conclusions

---

### Next Steps

With this epic breakdown complete, the project is ready for:

1. **Architecture Planning** (Recommended): Run architecture workflow to define technical decisions
2. **Implementation**: Execute stories sequentially following the 3-day timeline
3. **Story Creation**: Use `create-story` workflow to generate individual story files with full context

---

_For implementation: Use the `create-story` workflow to generate individual story implementation plans from this epic breakdown._

