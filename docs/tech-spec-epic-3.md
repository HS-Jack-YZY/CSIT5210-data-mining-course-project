# Epic Technical Specification: Multi-Agent Classification System

Date: 2025-11-09
Author: Jack YUAN
Epic ID: 3
Status: Draft

---

## Overview

Epic 3 implements the **Multi-Agent Classification System**, which is the core innovation that transforms clustering results into cost savings. This epic builds upon the semantic clusters created in Epic 2 by implementing a cosine similarity-based classification engine that intelligently routes queries to specialized agents, each maintaining only 1/4 of the total document context.

The system consists of three key components: (1) **Specialized Agents** that each hold documents from a single cluster (~30K documents instead of 120K), (2) a **Cosine Similarity Classification Engine** that compares query embeddings against cluster centroids to determine routing, and (3) an **Agent Router** that orchestrates the end-to-end pipeline from query input to agent selection.

This epic directly demonstrates **classification algorithm mastery** for course evaluation by implementing cosine similarity in high-dimensional embedding space (768 dimensions) and achieving >80% routing accuracy. The performance metrics collected here—classification accuracy, routing time, and confusion matrix—provide quantitative proof that clustering-based routing works effectively for LLM cost optimization.

## Objectives and Scope

**In Scope:**

- Implement `SpecializedAgent` class to encapsulate cluster-specific document contexts
- Create 4 agent instances, each assigned documents from one semantic cluster (World, Sports, Business, Sci/Tech)
- Implement cosine similarity classification engine to route queries to appropriate clusters
- Build `AgentRouter` class that integrates embedding generation, classification, and agent selection
- Measure classification accuracy on AG News test set (7,600 queries with ground truth labels)
- Generate confusion matrix to visualize classification performance
- Benchmark query processing performance (target: <1 second per query)
- Demonstrate context reduction from 120K documents (baseline) to ~30K documents per agent (75% reduction)

**Out of Scope:**

- Actual LLM inference calls (agents are implementation stubs focusing on routing demonstration)
- Agent collaboration or multi-agent consensus mechanisms
- Advanced classification methods (K-NN, neural networks) - Epic 3 focuses on cosine similarity only
- Dynamic cluster reassignment or online learning
- Production-grade error handling beyond retry logic (this is academic proof-of-concept)

**Success Criteria:**

- Classification accuracy >80% on test set
- All 4 specialized agents created with correct context assignments
- Confusion matrix generated showing classification performance
- Query routing completes in <1 second (including embedding generation)
- Context size per agent verified at ~25% of baseline (1/K where K=4)

## System Architecture Alignment

Epic 3 aligns with the architecture document's **classification and routing module design**:

**Components Referenced:**
- `src/models/agent.py`: SpecializedAgent class implementation
- `src/models/router.py`: AgentRouter class with classification logic
- `src/evaluation/classification_metrics.py`: Accuracy, precision, recall, F1-score calculation
- `scripts/03_evaluate_classification.py`: Classification accuracy evaluation script

**Architecture Constraints:**
- Use cosine similarity from `sklearn.metrics.pairwise.cosine_similarity`
- Query embeddings generated via existing `EmbeddingService` from Epic 1
- Cluster centroids loaded from Epic 2 output (`data/processed/centroids.npy`)
- Classification must be deterministic (no randomness)
- All metrics exported as JSON following architecture patterns

**Data Dependencies:**
- Input: Cluster centroids (4 × 768 float32 array from Epic 2)
- Input: Cluster assignments (120K labels from Epic 2)
- Input: AG News test set (7,600 documents with ground truth categories)
- Output: Classification results (`data/processed/classification_results.csv`)
- Output: Confusion matrix visualization (`reports/figures/confusion_matrix.png`)
- Output: Routing performance metrics (`results/routing_performance.json`)

**Integration Points:**
- `EmbeddingService`: Generate query embeddings for classification
- Cluster centroids: Loaded from `data/processed/centroids.npy`
- Document assignments: Loaded from `data/processed/cluster_assignments.csv`

The architecture specifies that classification computation should be <10ms (excluding embedding generation ~500-800ms), resulting in total routing time <1 second per query.

## Detailed Design

### Services and Modules

**Module: `src/models/agent.py`**

| Class | Responsibility | Key Methods | Inputs | Outputs |
|-------|---------------|-------------|---------|---------|
| `SpecializedAgent` | Encapsulate cluster-specific context | `__init__(cluster_id, documents, cluster_label)` | cluster_id (int), documents (List[str]), cluster_label (str) | Agent instance |
| | | `get_context_size() -> int` | None | Total character count |
| | | `get_documents() -> List[str]` | None | Document list |
| | | `get_metadata() -> dict` | None | cluster_id, label, doc_count |

**Module: `src/models/router.py`**

| Class | Responsibility | Key Methods | Inputs | Outputs |
|-------|---------------|-------------|---------|---------|
| `AgentRouter` | Classify queries and route to agents | `__init__(agents, centroids, embedding_service)` | agents (Dict[int, SpecializedAgent]), centroids (np.ndarray), service | Router instance |
| | | `classify_query(query_embedding) -> Tuple[int, float]` | query_embedding (np.ndarray 768-dim) | cluster_id (int), confidence (float) |
| | | `route_query(query) -> Tuple[SpecializedAgent, dict]` | query (str) | selected_agent, routing_metadata |
| | | `compute_cosine_similarity(vec1, vec2) -> float` | vec1, vec2 (np.ndarray) | similarity score (float) |

**Module: `src/evaluation/classification_metrics.py`**

| Function | Responsibility | Inputs | Outputs |
|----------|---------------|--------|---------|
| `evaluate_classification_accuracy(predictions, ground_truth)` | Calculate overall accuracy | predictions (np.ndarray), ground_truth (np.ndarray) | accuracy (float) |
| `generate_confusion_matrix(predictions, ground_truth)` | Create confusion matrix | predictions, ground_truth | confusion_matrix (np.ndarray 4×4) |
| `calculate_per_cluster_metrics(predictions, ground_truth)` | Precision, recall, F1 per cluster | predictions, ground_truth | metrics_dict (Dict[int, dict]) |

**Agent Registry Pattern:**

```python
# Initialized in router
agents = {
    0: SpecializedAgent(cluster_id=0, documents=cluster_0_docs, cluster_label="Sports"),
    1: SpecializedAgent(cluster_id=1, documents=cluster_1_docs, cluster_label="World"),
    2: SpecializedAgent(cluster_id=2, documents=cluster_2_docs, cluster_label="Business"),
    3: SpecializedAgent(cluster_id=3, documents=cluster_3_docs, cluster_label="Sci/Tech")
}
```

### Data Models and Contracts

**SpecializedAgent Data Model:**

```python
@dataclass
class SpecializedAgentMetadata:
    cluster_id: int              # 0-3
    cluster_label: str           # e.g., "Sports"
    num_documents: int           # ~30,000 per agent
    total_characters: int        # Context size
    context_reduction_pct: float # ~75% (1 - 1/K)
```

**Classification Result Model:**

```python
@dataclass
class ClassificationResult:
    query_id: int
    query_text: str
    predicted_cluster: int       # 0-3
    ground_truth_category: int   # 0-3 (AG News label)
    confidence: float            # Cosine similarity score [0, 1]
    similarity_scores: List[float]  # All 4 cluster similarities
    routing_time_ms: float       # Milliseconds
```

**Routing Metadata Contract:**

```python
routing_metadata = {
    "assigned_cluster_id": int,
    "cluster_label": str,
    "similarity_scores": [float, float, float, float],  # For all 4 clusters
    "classification_confidence": float,  # Max similarity
    "embedding_time_ms": float,
    "classification_time_ms": float,
    "total_routing_time_ms": float
}
```

**Confusion Matrix Schema:**

```
               Predicted
               0    1    2    3
Ground  0  [[ a,   b,   c,   d ],
Truth   1   [ e,   f,   g,   h ],
        2   [ i,   j,   k,   l ],
        3   [ m,   n,   o,   p ]]

Where diagonal elements (a, f, k, p) are correct classifications.
```

### APIs and Interfaces

**SpecializedAgent API:**

```python
class SpecializedAgent:
    """
    Specialized agent maintaining documents from a single semantic cluster.
    Demonstrates context reduction for LLM cost optimization.
    """

    def __init__(
        self,
        cluster_id: int,
        documents: List[str],
        cluster_label: str
    ) -> None:
        """
        Initialize agent with cluster-specific context.

        Args:
            cluster_id: Cluster identifier (0-3)
            documents: List of documents assigned to this cluster
            cluster_label: Human-readable cluster name
        """

    def get_context_size(self) -> int:
        """Calculate total character count in agent's context."""
        return sum(len(doc) for doc in self.documents)

    def get_documents(self) -> List[str]:
        """Return all documents in agent's context."""
        return self.documents

    def get_metadata(self) -> dict:
        """Return agent metadata for logging/reporting."""
        return {
            "cluster_id": self.cluster_id,
            "cluster_label": self.cluster_label,
            "num_documents": len(self.documents),
            "context_size_chars": self.get_context_size()
        }
```

**AgentRouter API:**

```python
class AgentRouter:
    """
    Routes queries to specialized agents using cosine similarity classification.
    """

    def __init__(
        self,
        agents: Dict[int, SpecializedAgent],
        centroids: np.ndarray,
        embedding_service: EmbeddingService
    ) -> None:
        """
        Initialize router with agents, centroids, and embedding service.

        Args:
            agents: Dictionary mapping cluster_id → SpecializedAgent
            centroids: Cluster centroids (4, 768) from K-Means
            embedding_service: Service to generate query embeddings
        """

    def classify_query(
        self,
        query_embedding: np.ndarray
    ) -> Tuple[int, float]:
        """
        Classify query to cluster using cosine similarity.

        Args:
            query_embedding: Query embedding (768,) float32

        Returns:
            cluster_id: Assigned cluster (0-3)
            confidence: Max cosine similarity score [0, 1]
        """

    def route_query(
        self,
        query: str
    ) -> Tuple[SpecializedAgent, dict]:
        """
        End-to-end query routing pipeline.

        Args:
            query: Input query text

        Returns:
            selected_agent: SpecializedAgent for this query
            routing_metadata: Classification details and timing
        """

    @staticmethod
    def compute_cosine_similarity(
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1, vec2: Vectors of same dimensionality

        Returns:
            similarity: Cosine similarity [-1, 1], higher = more similar
        """
```

**Classification Metrics API:**

```python
def evaluate_classification_accuracy(
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """Calculate overall classification accuracy."""
    return accuracy_score(ground_truth, predictions)

def generate_confusion_matrix(
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> np.ndarray:
    """Generate 4×4 confusion matrix."""
    return confusion_matrix(ground_truth, predictions)

def calculate_per_cluster_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[int, dict]:
    """
    Calculate precision, recall, F1-score per cluster.

    Returns:
        {
            0: {"precision": 0.85, "recall": 0.82, "f1": 0.83},
            1: {...},
            ...
        }
    """
```

### Workflows and Sequencing

**Story 3.1: Specialized Agent Implementation**

```
1. Load cluster assignments from data/processed/cluster_assignments.csv
2. Load AG News training documents
3. For each cluster_id in [0, 1, 2, 3]:
   a. Filter documents where cluster_id == current_cluster
   b. Extract cluster_label from cluster analysis (Epic 2)
   c. Create SpecializedAgent(cluster_id, filtered_docs, cluster_label)
   d. Log agent initialization (cluster_id, label, num_docs, context_size)
4. Store agents in registry: Dict[int, SpecializedAgent]
5. Calculate context reduction:
   baseline_context = 120,000 documents
   agent_context = len(cluster_docs)  # ~30,000
   reduction_pct = 1 - (agent_context / baseline_context)  # ~75%
6. Verify all 4 agents created successfully
```

**Story 3.2: Cosine Similarity Classification Engine**

```
1. Load cluster centroids from data/processed/centroids.npy (shape: 4, 768)
2. Implement compute_cosine_similarity(vec1, vec2):
   a. Normalize vectors: vec1 / ||vec1||, vec2 / ||vec2||
   b. Compute dot product: np.dot(norm_vec1, norm_vec2)
   c. Return similarity score [-1, 1]
3. Implement classify_query(query_embedding):
   a. Compute similarities with all 4 centroids
   b. Find cluster_id with max similarity: argmax(similarities)
   c. Extract confidence = max(similarities)
   d. Return (cluster_id, confidence)
4. Test on sample queries with known ground truth
5. Verify classification time <10ms (numpy operations only)
```

**Story 3.3: Agent Router with Query Routing**

```
1. Initialize AgentRouter(agents, centroids, embedding_service)
2. Implement route_query(query):
   a. Start timer
   b. Generate query embedding via embedding_service
   c. Record embedding_time_ms
   d. Classify query using classify_query(query_embedding)
   e. Record classification_time_ms
   f. Retrieve agent from registry: agents[cluster_id]
   g. Build routing_metadata dict
   h. Log routing decision
   i. Return (selected_agent, routing_metadata)
3. Verify total routing time <1 second
4. Test routing pipeline end-to-end
```

**Story 3.4: Classification Accuracy Measurement**

```
1. Load AG News test set (7,600 samples)
2. For each test query:
   a. Generate embedding
   b. Classify to cluster
   c. Compare with ground truth category
   d. Record (predicted_cluster, ground_truth_category)
3. Calculate metrics:
   a. Overall accuracy = correct_predictions / total_predictions
   b. Per-cluster precision, recall, F1
   c. Generate 4×4 confusion matrix
4. Export results to:
   - results/classification_accuracy.json
   - visualizations/confusion_matrix.png
5. Verify accuracy >80%
```

**Story 3.5: Query Processing Performance Benchmarking**

```
1. Select 100 sample queries from test set
2. For each query:
   a. Measure embedding generation time
   b. Measure classification computation time
   c. Measure total routing time
   d. Record all timing breakdowns
3. Calculate statistics:
   - Mean, median, 95th percentile routing time
   - Embedding time distribution
   - Classification time distribution
4. Identify bottlenecks:
   - If embedding >900ms: note API latency
   - If classification >50ms: investigate inefficiency
5. Export to results/routing_performance.json
6. Verify all queries meet <1 second requirement
```

## Non-Functional Requirements

### Performance

**Query Classification Speed:**
- **Target**: Classification computation <10ms per query (cosine similarity only)
- **Constraint**: Total routing time <1 second (including embedding generation ~500-800ms)
- **Rationale**: Embedding API call is bottleneck; classification must be negligible overhead

**Batch Processing:**
- **Target**: Classify 7,600 test queries in <15 minutes total
- **Calculation**: 7,600 queries × 1 second = 127 minutes theoretical, optimized with batch embedding
- **Strategy**: Use batch embedding generation for test set evaluation

**Memory Usage:**
- **Agent Context**: Each agent stores ~30K documents (~15MB text data)
- **Total Memory**: 4 agents × 15MB + centroids (0.01MB) + embeddings cache ≈ 60MB
- **Acceptable**: Fits easily in 8GB RAM minimum requirement

### Security

**No Additional Security Requirements:**
- Epic 3 uses existing `EmbeddingService` with API key management from Epic 1
- No new external integrations or API calls
- Classification is purely computational (no network requests)
- Agent contexts stored in memory (no persistent storage of sensitive data)

### Reliability/Availability

**Classification Determinism:**
- **Requirement**: Same query must always route to same agent (deterministic classification)
- **Implementation**: Cosine similarity is deterministic (no randomness)
- **Verification**: Test query classified 10 times should yield identical results

**Error Handling:**
- **Scenario**: Embedding generation fails (network error)
- **Response**: Retry logic from Epic 1 `EmbeddingService` handles retries
- **Fallback**: If retries exhausted, log error and skip query (continue batch processing)

**Edge Cases:**
- **Zero vector**: If query embedding is all zeros, log warning and assign to cluster 0 (fallback)
- **NaN values**: Validate embeddings before classification; raise error if NaN detected
- **Tie similarity scores**: If two clusters have identical max similarity, choose lower cluster_id (deterministic tiebreaker)

### Observability

**Logging Requirements:**

```python
# Agent initialization
logger.info(f"📊 Initialized Agent {cluster_id} ({cluster_label}): {num_docs} documents, {context_size_chars:,} chars")

# Classification decision
logger.debug(f"🎯 Query classified to Cluster {cluster_id} (confidence: {confidence:.3f})")

# Routing completion
logger.info(f"✅ Query routed to Agent {cluster_id} ({cluster_label}) in {total_time_ms:.1f}ms")

# Accuracy evaluation
logger.info(f"📈 Classification Accuracy: {accuracy:.2%} ({correct}/{total} correct)")
```

**Metrics to Track:**

1. **Per-Query Metrics:**
   - Embedding generation time (ms)
   - Classification computation time (ms)
   - Total routing time (ms)
   - Assigned cluster_id
   - Confidence score

2. **Aggregate Metrics:**
   - Overall classification accuracy (%)
   - Per-cluster precision, recall, F1
   - Average routing time
   - 95th percentile routing time

3. **Performance Metrics:**
   - Queries processed per second
   - Embedding API call count
   - Classification throughput (queries/second)

**Metrics Export:**
- All metrics saved to `results/classification_metrics_{timestamp}.json`
- Confusion matrix visualization saved to `reports/figures/confusion_matrix.png`
- Performance statistics saved to `results/routing_performance.json`

## Dependencies and Integrations

**Python Package Dependencies:**

| Package | Version | Purpose | Usage in Epic 3 |
|---------|---------|---------|----------------|
| `scikit-learn` | ≥1.7.2 | ML algorithms and metrics | `cosine_similarity`, `accuracy_score`, `confusion_matrix`, `precision_recall_fscore_support` |
| `numpy` | ≥1.24.0 | Array operations | Centroid loading, embedding manipulation, argmax for classification |
| `pandas` | ≥2.0.0 | Data manipulation | Loading cluster assignments CSV, test set handling |
| `matplotlib` | ≥3.7.0 | Visualization | Confusion matrix heatmap |
| `seaborn` | ≥0.12.0 | Statistical plots | Enhanced confusion matrix styling |

**Internal Module Dependencies:**

```
Epic 3 Dependencies:
├── src/features/embedding_service.py (from Epic 1)
│   └── EmbeddingService.generate_embedding(query) → np.ndarray
├── data/processed/centroids.npy (from Epic 2)
│   └── Cluster centroids (4, 768) float32
├── data/processed/cluster_assignments.csv (from Epic 2)
│   └── document_id, cluster_id, ground_truth_category
└── data/processed/cluster_labels.json (from Epic 2)
    └── {0: "Sports", 1: "World", 2: "Business", 3: "Sci/Tech"}
```

**External API Dependencies:**

- **Google Gemini Embedding API** (via `EmbeddingService` from Epic 1)
  - Used for: Generating query embeddings in real-time
  - Rate limiting: Handled by existing retry logic
  - Cost: ~$0.15 per 1M tokens (standard API for single queries)

**Data Dependencies:**

| Input File | Source | Format | Shape/Schema |
|------------|--------|--------|--------------|
| `data/processed/centroids.npy` | Epic 2 Story 2.2 | numpy array | (4, 768) float32 |
| `data/processed/cluster_assignments.csv` | Epic 2 Story 2.2 | CSV | Columns: document_id, cluster_id, category |
| `data/processed/cluster_labels.json` | Epic 2 Story 2.5 | JSON | {cluster_id: label_string} |
| `data/raw/ag_news_test.csv` | Epic 1 Story 1.3 | CSV | 7,600 rows with text and labels |

**Integration Points:**

1. **EmbeddingService Integration:**
   ```python
   from src.features.embedding_service import EmbeddingService

   service = EmbeddingService(api_key=config.gemini_api_key)
   query_embedding = service.generate_embedding(query_text)
   # Returns: np.ndarray (768,) float32
   ```

2. **Centroids Loading:**
   ```python
   centroids = np.load(paths.data_processed / "centroids.npy")
   # Shape: (4, 768) float32
   # Rows correspond to clusters 0-3
   ```

3. **Cluster Assignments Loading:**
   ```python
   assignments_df = pd.read_csv(paths.data_processed / "cluster_assignments.csv")
   # Columns: document_id, cluster_id, ground_truth_category
   cluster_0_docs = assignments_df[assignments_df['cluster_id'] == 0]['document_id'].tolist()
   ```

**Version Compatibility:**

- All dependencies compatible with Python 3.10+
- scikit-learn 1.7.2 introduced performance improvements for `cosine_similarity` (10-15% faster)
- numpy float32 dtype ensures consistency with Epic 2 outputs
- No breaking changes expected in dependency minor versions

## Acceptance Criteria (Authoritative)

**Story 3.1: Specialized Agent Implementation**

AC-3.1.1: `SpecializedAgent` class implemented in `src/models/agent.py` with required methods
AC-3.1.2: 4 agent instances created, one per cluster (IDs 0-3)
AC-3.1.3: Each agent assigned only documents from its cluster (~30K docs each)
AC-3.1.4: Agent registry created as `Dict[int, SpecializedAgent]`
AC-3.1.5: Each agent logs initialization with cluster_id, label, num_documents, context_size
AC-3.1.6: Context size reduction calculated and verified at ~75% (1 - 1/4)
AC-3.1.7: All agents accessible via registry with correct cluster_id mapping

**Story 3.2: Cosine Similarity Classification Engine**

AC-3.2.1: `compute_cosine_similarity(vec1, vec2)` function implemented returning float [-1, 1]
AC-3.2.2: `classify_query(query_embedding)` returns (cluster_id, confidence) tuple
AC-3.2.3: Cosine similarity computed correctly with all 4 centroids
AC-3.2.4: Query assigned to cluster with highest similarity score
AC-3.2.5: Classification computation time <10ms verified on sample queries
AC-3.2.6: Function handles edge cases (zero vectors, NaN values) with appropriate errors
AC-3.2.7: Test cases pass for sample queries with known ground truth

**Story 3.3: Agent Router with Query Routing**

AC-3.3.1: `AgentRouter` class implemented in `src/models/router.py`
AC-3.3.2: Router initialized with agents, centroids, and embedding_service
AC-3.3.3: `route_query(query)` implements full pipeline: embed → classify → retrieve agent
AC-3.3.4: Routing metadata includes assigned_cluster_id, similarity_scores, confidence, timing
AC-3.3.5: Routing decision logged with query (truncated), cluster, confidence, time
AC-3.3.6: Total routing time <1 second verified (including embedding generation)
AC-3.3.7: End-to-end routing tested with sample queries

**Story 3.4: Classification Accuracy Measurement**

AC-3.4.1: All 7,600 test queries classified to clusters
AC-3.4.2: Predicted clusters compared with AG News ground truth categories
AC-3.4.3: Overall accuracy calculated and exceeds 80%
AC-3.4.4: Per-cluster metrics computed (precision, recall, F1) for all 4 clusters
AC-3.4.5: Confusion matrix (4×4) generated showing ground truth vs predicted
AC-3.4.6: Results saved to `results/classification_accuracy.json` with schema:
   ```json
   {
     "overall_accuracy": 0.83,
     "per_cluster_metrics": {...},
     "confusion_matrix": [[...], [...], [...], [...]]
   }
   ```
AC-3.4.7: Confusion matrix visualization saved to `reports/figures/confusion_matrix.png` (300 DPI)
AC-3.4.8: Accuracy summary logged to console

**Story 3.5: Query Processing Performance Benchmarking**

AC-3.5.1: 100 sample queries processed and timed
AC-3.5.2: Timing breakdown measured: embedding_time, classification_time, total_routing_time
AC-3.5.3: Performance statistics calculated: mean, median, 95th percentile routing time
AC-3.5.4: All queries meet <1 second total routing requirement
AC-3.5.5: Results saved to `results/routing_performance.json`
AC-3.5.6: Bottlenecks identified and documented (expected: embedding generation ~500-800ms)
AC-3.5.7: Performance summary logged to console

## Traceability Mapping

| Acceptance Criteria | Spec Section(s) | Component(s)/API(s) | Test Idea |
|---------------------|----------------|---------------------|-----------|
| AC-3.1.1 | Services and Modules, APIs | `SpecializedAgent` class | Unit test: instantiate agent, verify methods exist |
| AC-3.1.2 | Workflows: Story 3.1 | Agent registry | Integration test: verify 4 agents created |
| AC-3.1.3 | Data Models | `cluster_assignments.csv` | Assertion: each agent has ~30K docs |
| AC-3.1.4 | Services and Modules | `Dict[int, SpecializedAgent]` | Unit test: registry keys are [0, 1, 2, 3] |
| AC-3.1.5 | Observability | Logging | Manual: inspect logs for initialization messages |
| AC-3.1.6 | Workflows: Story 3.1 | Context size calculation | Assertion: reduction_pct ≈ 0.75 |
| AC-3.1.7 | Services and Modules | Agent registry | Unit test: agents[i] returns correct agent |
| AC-3.2.1 | APIs | `compute_cosine_similarity()` | Unit test: similarity(v, v) == 1.0 |
| AC-3.2.2 | APIs | `classify_query()` | Unit test: returns (int, float) |
| AC-3.2.3 | Workflows: Story 3.2 | Cosine similarity computation | Integration test: verify 4 similarity scores |
| AC-3.2.4 | Workflows: Story 3.2 | `argmax(similarities)` | Unit test: max similarity cluster selected |
| AC-3.2.5 | Performance | Classification speed | Benchmark: time 100 classifications, verify <10ms avg |
| AC-3.2.6 | Reliability | Edge case handling | Unit test: zero vector raises ValueError |
| AC-3.2.7 | Workflows: Story 3.2 | Sample query classification | Integration test: known queries classified correctly |
| AC-3.3.1 | Services and Modules | `AgentRouter` class | Unit test: instantiate router, verify attributes |
| AC-3.3.2 | APIs | Router initialization | Unit test: router initialized with correct parameters |
| AC-3.3.3 | Workflows: Story 3.3 | `route_query()` pipeline | Integration test: query → agent returned |
| AC-3.3.4 | Data Models | Routing metadata | Assertion: metadata contains required keys |
| AC-3.3.5 | Observability | Routing logs | Manual: inspect logs for routing decisions |
| AC-3.3.6 | Performance | Routing time | Benchmark: 100 queries, verify <1s each |
| AC-3.3.7 | Workflows: Story 3.3 | End-to-end routing | Integration test: route sample queries successfully |
| AC-3.4.1 | Workflows: Story 3.4 | Test set classification | Integration test: 7,600 queries classified |
| AC-3.4.2 | Workflows: Story 3.4 | Accuracy calculation | Assertion: predictions vs ground truth compared |
| AC-3.4.3 | Workflows: Story 3.4 | Overall accuracy | Assertion: accuracy > 0.80 |
| AC-3.4.4 | APIs | `calculate_per_cluster_metrics()` | Unit test: precision/recall/F1 per cluster |
| AC-3.4.5 | Data Models | Confusion matrix | Assertion: matrix shape == (4, 4) |
| AC-3.4.6 | Dependencies | JSON export | File test: verify JSON schema matches |
| AC-3.4.7 | Dependencies | matplotlib visualization | Visual test: confusion matrix PNG generated |
| AC-3.4.8 | Observability | Accuracy logging | Manual: inspect console output |
| AC-3.5.1 | Workflows: Story 3.5 | Benchmark script | Integration test: 100 queries timed |
| AC-3.5.2 | Workflows: Story 3.5 | Timing breakdown | Assertion: 3 timing metrics recorded per query |
| AC-3.5.3 | APIs | Performance statistics | Unit test: mean/median/p95 calculated |
| AC-3.5.4 | Performance | Routing time requirement | Assertion: all times < 1000ms |
| AC-3.5.5 | Dependencies | JSON export | File test: routing_performance.json exists |
| AC-3.5.6 | Workflows: Story 3.5 | Bottleneck identification | Manual: review timing breakdown analysis |
| AC-3.5.7 | Observability | Performance logging | Manual: inspect console summary |

## Risks, Assumptions, Open Questions

**Risks:**

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| **R-3.1**: Classification accuracy <80% due to poor cluster separation | High - invalidates cost optimization claim | Medium | MITIGATION: Epic 2 achieved Silhouette >0.3, indicating good separation. If accuracy low, review cluster purity from Story 2.5. |
| **R-3.2**: Embedding API latency >1s makes routing too slow | Medium - exceeds performance target | Low | MITIGATION: Batch embed test queries upfront. For real-time, use caching or async embedding generation. |
| **R-3.3**: Cluster-category mismatch (clusters don't align with AG News categories) | High - confuses classification interpretation | Medium | MITIGATION: Story 2.5 cluster analysis already mapped clusters to categories. Use this mapping for interpretation. |
| **R-3.4**: Agent implementation too simplistic (no actual LLM calls) | Low - sufficient for academic demo | N/A | ACCEPTANCE: Epic 3 focuses on routing demonstration, not LLM inference. Agents are intentionally stubs. |
| **R-3.5**: Confusion matrix reveals systematic misclassification (e.g., World vs Sci/Tech) | Medium - indicates semantic overlap | Medium | MITIGATION: Document in experimental report as limitation. Consider hierarchical clustering in future work. |

**Assumptions:**

| Assumption | Validation | Dependency |
|-----------|-----------|------------|
| **A-3.1**: Cluster centroids from Epic 2 are representative of cluster semantics | Silhouette Score >0.3 in Story 2.3 | Epic 2 Story 2.2 |
| **A-3.2**: AG News test set ground truth labels are accurate | Provided by Hugging Face datasets library (trusted source) | Epic 1 Story 1.3 |
| **A-3.3**: Cosine similarity is appropriate for 768-dimensional embeddings | Standard practice in NLP/embedding classification | Architecture ADR-002 |
| **A-3.4**: 4 clusters correspond roughly to 4 AG News categories | Validated by cluster purity analysis in Story 2.5 | Epic 2 Story 2.5 |
| **A-3.5**: Embedding service is reliable (<5% failure rate) | Retry logic from Epic 1 handles transient errors | Epic 1 Story 1.4 |
| **A-3.6**: Query embeddings have same dimensionality (768) as training embeddings | Gemini API guarantees consistent dimensions for same model | Epic 1 Story 1.4 |

**Open Questions:**

| Question | Context | Resolution Path |
|----------|---------|----------------|
| **Q-3.1**: What if two clusters have identical max similarity (tie)? | Rare edge case in cosine similarity | DECISION: Choose lower cluster_id (deterministic tiebreaker). Document in code. |
| **Q-3.2**: Should we implement confidence thresholding (route to multiple agents if low confidence)? | Could improve accuracy for ambiguous queries | OUT OF SCOPE: Epic 3 uses single-agent routing only. Note as future enhancement. |
| **Q-3.3**: How to handle queries that don't fit any cluster well (all similarities <0.3)? | Possible for out-of-distribution queries | DECISION: Route to highest similarity cluster anyway. Log low-confidence warning. |
| **Q-3.4**: Should agent context include only text or also metadata (IDs, categories)? | Affects agent implementation | DECISION: Agents store text only (List[str]). Metadata tracked separately for analysis. |
| **Q-3.5**: How to visualize confusion matrix (heatmap vs annotated grid)? | Affects report quality | DECISION: Use seaborn heatmap with annotations (numeric values in cells). |

## Test Strategy Summary

**Unit Testing (pytest):**

```python
# tests/test_agent.py
def test_specialized_agent_initialization()
def test_agent_get_context_size()
def test_agent_get_documents()
def test_agent_get_metadata()

# tests/test_router.py
def test_compute_cosine_similarity()
def test_classify_query_returns_correct_cluster()
def test_router_initialization()
def test_route_query_end_to_end()
def test_edge_case_zero_vector()
def test_edge_case_tie_similarity()

# tests/test_classification_metrics.py
def test_evaluate_classification_accuracy()
def test_generate_confusion_matrix()
def test_calculate_per_cluster_metrics()
```

**Integration Testing:**

1. **Agent Creation Integration:**
   - Load cluster assignments from Epic 2
   - Create all 4 agents
   - Verify correct document assignment
   - Verify context size reduction

2. **Classification Pipeline Integration:**
   - Load centroids from Epic 2
   - Generate sample query embedding
   - Classify query
   - Retrieve corresponding agent
   - Verify end-to-end flow

3. **Accuracy Evaluation Integration:**
   - Load test set
   - Classify all queries
   - Compare with ground truth
   - Generate confusion matrix
   - Export results to JSON

**Performance Testing:**

1. **Classification Speed Benchmark:**
   - Measure cosine similarity computation for 1000 queries
   - Verify average time <10ms

2. **Routing Throughput Benchmark:**
   - Route 100 queries end-to-end
   - Measure total time, embedding time, classification time
   - Verify <1 second per query

**Acceptance Testing:**

1. **Classification Accuracy:**
   - Run on full test set (7,600 queries)
   - Verify overall accuracy >80%
   - Verify no cluster has accuracy <70%

2. **Confusion Matrix Validation:**
   - Generate confusion matrix
   - Verify diagonal elements dominate (correct classifications)
   - Identify and document systematic errors

3. **Performance Validation:**
   - Measure routing time for 100 sample queries
   - Verify 95th percentile <1 second

**Manual Testing:**

1. **Sample Query Routing:**
   - Test queries from each category (World, Sports, Business, Sci/Tech)
   - Verify intuitive routing (Sports query → Sports agent)
   - Log confidence scores for review

2. **Visualization Quality:**
   - Generate confusion matrix heatmap
   - Verify 300 DPI, readable labels, professional formatting
   - Include in experimental report

**Edge Case Testing:**

| Edge Case | Test Approach | Expected Behavior |
|-----------|--------------|-------------------|
| Zero embedding vector | Unit test with np.zeros(768) | Raise ValueError with informative message |
| NaN values in embedding | Unit test with np.nan values | Raise ValueError before classification |
| Tie similarity scores | Unit test with identical centroids | Choose cluster_id=0 (deterministic) |
| Empty query string | Integration test | EmbeddingService handles error, propagate to router |
| Very long query (>1000 chars) | Integration test | Route normally (truncation handled by API) |

**Test Coverage Goals:**

- Unit test coverage: >80% for core functions
- Integration test coverage: All 5 stories end-to-end
- Performance benchmarks: All NFRs validated
- Acceptance criteria: 100% AC coverage (35 acceptance criteria)
