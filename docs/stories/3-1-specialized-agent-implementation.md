# Story 3.1: Specialized Agent Implementation

Status: ready-for-dev

## Story

As a **data mining student**,
I want **specialized agents that each maintain only documents from one cluster**,
So that **I can demonstrate context reduction for cost optimization**.

## Acceptance Criteria

### AC-3.1.1: SpecializedAgent Class Implementation

**Given** K-Means clustering is complete with 4 clusters
**When** I implement the SpecializedAgent class
**Then**:
- ✅ `SpecializedAgent` class is implemented in `src/models/agent.py`
- ✅ Class includes required methods: `__init__`, `get_context_size()`, `get_documents()`, `get_metadata()`
- ✅ Class follows architecture patterns: PascalCase class name, snake_case methods, type hints on all methods
- ✅ Google-style docstrings with usage examples for all methods
- ✅ Module includes `__all__` export list

**Validation:**
```python
from src.context_aware_multi_agent_system.models.agent import SpecializedAgent

# Verify class exists and has required methods
assert hasattr(SpecializedAgent, '__init__')
assert hasattr(SpecializedAgent, 'get_context_size')
assert hasattr(SpecializedAgent, 'get_documents')
assert hasattr(SpecializedAgent, 'get_metadata')
```

---

### AC-3.1.2: Four Agent Instances Created

**Given** cluster assignments exist in `data/processed/cluster_assignments.csv`
**When** I create agent instances for all clusters
**Then**:
- ✅ Exactly 4 agent instances are created (one per cluster, IDs 0-3)
- ✅ Agent registry is created as `Dict[int, SpecializedAgent]`
- ✅ Registry keys are [0, 1, 2, 3] matching cluster IDs
- ✅ Each agent is properly initialized with cluster-specific data
- ✅ Registry is accessible for routing operations

**Validation:**
```python
agents = create_agent_registry(cluster_assignments, documents, cluster_labels)

assert len(agents) == 4
assert set(agents.keys()) == {0, 1, 2, 3}
assert all(isinstance(agent, SpecializedAgent) for agent in agents.values())
```

---

### AC-3.1.3: Cluster-Specific Document Assignment

**Given** cluster assignments from Story 2.2 exist
**When** agents are initialized
**Then**:
- ✅ Each agent is assigned ONLY documents from its cluster
- ✅ Agent 0 contains ~30K documents (all cluster_id=0 documents)
- ✅ Agent 1 contains ~30K documents (all cluster_id=1 documents)
- ✅ Agent 2 contains ~30K documents (all cluster_id=2 documents)
- ✅ Agent 3 contains ~30K documents (all cluster_id=3 documents)
- ✅ Total documents across all agents = 120,000 (full training set)
- ✅ No document appears in multiple agents (strict partitioning)

**Validation:**
```python
# Load cluster assignments
assignments_df = pd.read_csv('data/processed/cluster_assignments.csv')

for cluster_id in range(4):
    expected_doc_ids = assignments_df[assignments_df['cluster_id'] == cluster_id]['document_id'].tolist()
    agent_doc_ids = [doc['id'] for doc in agents[cluster_id].get_documents()]

    assert set(agent_doc_ids) == set(expected_doc_ids)
    assert 25000 <= len(agent_doc_ids) <= 35000  # ~30K per cluster, allowing variance
```

---

### AC-3.1.4: Agent Registry Pattern

**Given** 4 agents are created
**When** the agent registry is built
**Then**:
- ✅ Registry is implemented as `Dict[int, SpecializedAgent]`
- ✅ Registry maps cluster_id (0-3) to agent instance
- ✅ Registry supports O(1) lookup by cluster_id
- ✅ Registry is immutable after initialization (prevents accidental modification)
- ✅ Registry can be serialized for persistence (optional)

**Validation:**
```python
registry = create_agent_registry(...)

# Type validation
assert isinstance(registry, dict)
assert all(isinstance(k, int) for k in registry.keys())
assert all(isinstance(v, SpecializedAgent) for v in registry.values())

# Lookup performance
agent = registry[2]  # O(1) access
assert agent.cluster_id == 2
```

---

### AC-3.1.5: Agent Initialization Logging

**Given** agents are being initialized
**When** each agent is created
**Then**:
- ✅ Each agent logs initialization with emoji-prefixed message
- ✅ Log includes: cluster_id, cluster_label, num_documents, context_size_chars
- ✅ Log format: `"📊 Initialized Agent {cluster_id} ({cluster_label}): {num_docs:,} documents, {context_size_chars:,} chars"`
- ✅ All 4 initialization logs are emitted during registry creation
- ✅ Logs use INFO level for visibility
- ✅ Context size is formatted with thousand separators for readability

**Validation:**
```python
# Example expected logs:
# INFO: 📊 Initialized Agent 0 (Sports): 30,245 documents, 12,345,678 chars
# INFO: 📊 Initialized Agent 1 (World): 29,876 documents, 11,987,543 chars
# INFO: 📊 Initialized Agent 2 (Business): 30,123 documents, 12,098,765 chars
# INFO: 📊 Initialized Agent 3 (Sci/Tech): 29,756 documents, 11,876,432 chars
```

---

### AC-3.1.6: Context Size Reduction Calculation

**Given** all 4 agents are initialized
**When** I calculate context size reduction
**Then**:
- ✅ Baseline context size is calculated (all 120,000 documents)
- ✅ Per-agent context size is calculated (~30,000 documents each)
- ✅ Context reduction percentage is computed: `1 - (agent_context / baseline_context)`
- ✅ Reduction is approximately 75% (1 - 1/4) for each agent
- ✅ Reduction metric is logged with clear messaging
- ✅ Results are saved to metadata for reporting

**Validation:**
```python
baseline_context = sum(len(doc) for doc in all_documents)  # ~48M characters

for cluster_id, agent in agents.items():
    agent_context = agent.get_context_size()
    reduction_pct = 1 - (agent_context / baseline_context)

    # Each agent should have ~25% of baseline (75% reduction)
    assert 0.70 <= reduction_pct <= 0.80  # 70-80% reduction
    assert 0.20 <= (agent_context / baseline_context) <= 0.30  # 20-30% of baseline

# Log example: "✅ Context Reduction: Agent 0 uses 25.2% of baseline (74.8% reduction)"
```

---

### AC-3.1.7: Agent Registry Accessibility

**Given** agent registry is created
**When** accessing agents for routing
**Then**:
- ✅ All agents are accessible via registry with correct cluster_id mapping
- ✅ `registry[0]` returns Agent 0 (Sports)
- ✅ `registry[1]` returns Agent 1 (World)
- ✅ `registry[2]` returns Agent 2 (Business)
- ✅ `registry[3]` returns Agent 3 (Sci/Tech)
- ✅ Invalid cluster_id raises KeyError with informative message
- ✅ Registry supports iteration: `for cluster_id, agent in registry.items()`

**Validation:**
```python
# Direct access
sports_agent = registry[0]
assert sports_agent.cluster_id == 0
assert sports_agent.cluster_label == "Sports"

# Iteration
for cluster_id, agent in registry.items():
    assert agent.cluster_id == cluster_id
    assert cluster_id in range(4)

# Error handling
try:
    invalid_agent = registry[5]
    assert False, "Should raise KeyError"
except KeyError as e:
    assert "Cluster ID 5 not found" in str(e) or "5" in str(e)
```

---

## Tasks / Subtasks

- [ ] Implement SpecializedAgent class in `src/models/agent.py` (AC: #3.1.1, #3.1.5)
  - [ ] Create `src/context_aware_multi_agent_system/models/agent.py` file
  - [ ] Import required modules: typing, logging, numpy, pandas
  - [ ] Define `SpecializedAgent` class with PascalCase naming
  - [ ] Implement `__init__(cluster_id: int, documents: List[dict], cluster_label: str)` method
    - [ ] Validate inputs: cluster_id in [0, 3], documents non-empty, cluster_label valid
    - [ ] Store cluster_id, documents, cluster_label as instance attributes
    - [ ] Log initialization with emoji-prefixed message (📊)
    - [ ] Calculate and store context_size_chars on initialization
  - [ ] Implement `get_context_size() -> int` method
    - [ ] Calculate total character count: `sum(len(doc['text']) for doc in self.documents)`
    - [ ] Return integer character count
  - [ ] Implement `get_documents() -> List[dict]` method
    - [ ] Return copy of documents list (prevent external modification)
  - [ ] Implement `get_metadata() -> dict` method
    - [ ] Return dict with: cluster_id, cluster_label, num_documents, context_size_chars
  - [ ] Add comprehensive Google-style docstrings to all methods
  - [ ] Add type hints to all method signatures
  - [ ] Add `__all__ = ['SpecializedAgent']` export list

- [ ] Create agent initialization script `scripts/06_initialize_agents.py` (AC: #3.1.2, #3.1.3, #3.1.4, #3.1.6, #3.1.7)
  - [ ] Import required modules: Config, Paths, logger, pandas, numpy, SpecializedAgent
  - [ ] Implement `set_seed(42)` at script start for reproducibility
  - [ ] Load configuration from `config.yaml`
  - [ ] Setup logging with emoji prefixes
  - [ ] Load cluster assignments from `data/processed/cluster_assignments.csv`
  - [ ] Load cluster labels from `results/cluster_labels.json` (from Story 2.5)
  - [ ] Load AG News training documents
  - [ ] Validate inputs: file existence, shape consistency, cluster_id range [0,3]
  - [ ] If files missing, raise FileNotFoundError with clear message and next steps
  - [ ] For each cluster_id in [0, 1, 2, 3]:
    - [ ] Filter documents where cluster_id == current_cluster
    - [ ] Extract cluster_label from cluster_labels.json
    - [ ] Create `SpecializedAgent(cluster_id, filtered_docs, cluster_label)`
    - [ ] Verify agent initialization logged (check logs)
  - [ ] Build agent registry as `Dict[int, SpecializedAgent]`
  - [ ] Validate registry: 4 agents, keys [0,1,2,3], all SpecializedAgent instances
  - [ ] Calculate baseline context size (all 120K documents)
  - [ ] For each agent:
    - [ ] Calculate agent context size
    - [ ] Calculate reduction percentage: `1 - (agent_context / baseline_context)`
    - [ ] Log context reduction: "✅ Context Reduction: Agent {id} uses {pct}% of baseline ({reduction}% reduction)"
  - [ ] Verify all agents accessible via registry
  - [ ] Test iteration over registry
  - [ ] Save agent metadata to `results/agent_metadata.json` (cluster_id, label, num_docs, context_size, reduction_pct)
  - [ ] Log completion summary:
    - [ ] Total agents created: 4
    - [ ] Average context reduction: ~75%
    - [ ] Metadata saved: results/agent_metadata.json

- [ ] Implement agent registry creation function (AC: #3.1.2, #3.1.4, #3.1.7)
  - [ ] Create `create_agent_registry()` function in `src/models/agent.py`
  - [ ] Function signature: `create_agent_registry(cluster_assignments_df: pd.DataFrame, documents: List[dict], cluster_labels: dict) -> Dict[int, SpecializedAgent]`
  - [ ] For each unique cluster_id in assignments:
    - [ ] Filter documents by cluster_id
    - [ ] Get cluster_label from cluster_labels dict
    - [ ] Create SpecializedAgent instance
    - [ ] Add to registry dict
  - [ ] Validate registry has exactly 4 entries
  - [ ] Validate keys are [0, 1, 2, 3]
  - [ ] Return immutable dict (optional: use MappingProxyType)
  - [ ] Add docstring with usage example

- [ ] Calculate and validate context reduction (AC: #3.1.6)
  - [ ] Implement `calculate_context_reduction()` helper function
  - [ ] Calculate baseline context size: sum of all document lengths
  - [ ] For each agent:
    - [ ] Get agent context size via `agent.get_context_size()`
    - [ ] Calculate reduction: `1 - (agent_size / baseline_size)`
    - [ ] Store reduction percentage
  - [ ] Validate average reduction is ~75% (tolerance ±5%)
  - [ ] Log reduction metrics for each agent
  - [ ] Return reduction statistics dict

- [ ] Test agent implementation (AC: #3.1.1 through #3.1.7)
  - [ ] Unit test: SpecializedAgent initialization with synthetic data
  - [ ] Unit test: `get_context_size()` returns correct character count
  - [ ] Unit test: `get_documents()` returns copy (not reference)
  - [ ] Unit test: `get_metadata()` returns correct schema
  - [ ] Unit test: Invalid inputs raise appropriate errors (cluster_id out of range, empty documents)
  - [ ] Integration test: Create 4 agents from actual cluster assignments
  - [ ] Integration test: Verify each agent has correct document count (~30K)
  - [ ] Integration test: Verify no document overlap between agents
  - [ ] Integration test: Verify total documents across agents = 120K
  - [ ] Integration test: Verify context reduction ~75% for each agent
  - [ ] Integration test: Verify agent registry accessibility (get by cluster_id)
  - [ ] Performance test: Agent creation time <5 seconds for all 4 agents
  - [ ] Logging test: Verify initialization logs emitted for all agents

- [ ] Update project documentation (AC: all)
  - [ ] Update README.md with agent initialization script usage
  - [ ] Document script usage: `python scripts/06_initialize_agents.py`
  - [ ] Document expected outputs: `results/agent_metadata.json`
  - [ ] Document agent registry pattern and usage
  - [ ] Add troubleshooting section for common errors
  - [ ] Document context reduction metric interpretation

## Dev Notes

### Architecture Alignment

This story implements the **Specialized Agent** component defined in the architecture (Epic 3, Story 3.1). It integrates with:

1. **Cookiecutter Data Science Structure**: Follows `src/models/` for agent logic, `scripts/` for orchestration, `results/` for metadata
2. **Story 2.2 Outputs**: Consumes cluster assignments from `data/processed/cluster_assignments.csv`
3. **Story 2.5 Outputs**: Uses cluster labels from `results/cluster_labels.json`
4. **Story 1.3 Outputs**: Uses AG News training documents
5. **Configuration System**: Uses `config.yaml` for agent parameters
6. **Epic 3 Foundation**: Establishes agent pattern for routing implementation (Stories 3.2, 3.3)

**Constraints Applied:**
- **Performance**: Agent initialization <5 seconds for all 4 agents (NFR-1 from PRD)
- **Reproducibility**: Fixed random_state=42 ensures deterministic document assignment
- **Logging**: Uses emoji-prefixed logging (📊, ✅, ⚠️, ❌) from `utils/logger.py`
- **Error Handling**: Validates input file existence and data schema before agent creation

**Architectural Patterns Followed:**
- **Initialization Order**: `set_seed → load config → setup logger → validate → execute`
- **Data Loading**: Check file exists → load → validate → process
- **File Naming**: snake_case for modules (`agent.py`), PascalCase for classes (`SpecializedAgent`)
- **Configuration Access**: No hardcoded values, all parameters from `config.yaml`
- **Type Hints**: All methods have full type annotations
- **Docstrings**: Google-style docstrings with usage examples

### Context Reduction Strategy

**Why Specialized Agents Enable Cost Optimization:**

**1. Context Size Reduction (Primary Value)**
- **Baseline**: Single agent maintains all 120,000 documents (~48M characters, ~200K+ tokens)
- **Optimized**: Each specialized agent maintains ~30,000 documents (~12M characters, ~50K tokens)
- **Reduction**: 75% context reduction per agent (1 - 1/4)
- **Cost Impact**: Baseline triggers >200K token tier ($6/M), optimized stays in <200K tier ($3/M)
- **Combined Effect**: 75% fewer tokens × 50% lower price tier = >90% cost reduction

**2. Routing Efficiency**
- Query routed to correct agent in <1 second (including embedding generation)
- No multi-agent consensus needed (single-agent response)
- Deterministic routing (cosine similarity classification)
- No context switching overhead

**3. Semantic Coherence**
- Each agent specializes in one topic (Sports, World, Business, Sci/Tech)
- Cluster purity ~25% (Story 2.5) indicates semantic clustering challenges
- Despite low purity, partitioning still achieves cost reduction
- Future: Higher purity would improve both accuracy and cost efficiency

**4. Scalability Pattern**
- K=4 demonstrates concept, easily extendable to K=8, K=16
- Linear scaling: K agents → 1/K context per agent
- Cost reduction scales with K: 1 - 1/K
- Example: K=8 → 87.5% reduction, K=16 → 93.75% reduction

**Expected Behavior:**
- Agent initialization completes in <5 seconds
- Each agent holds 25-35K documents (cluster size variance expected)
- Context reduction ~75% per agent (tolerance ±5% due to cluster size variance)
- Total documents across agents = 120,000 (strict partitioning, no overlap)
- Agent metadata saved for Epic 4 cost analysis

### Data Models and Contracts

**Input Data:**

```python
# Cluster Assignments (from Story 2.2)
Type: pd.DataFrame
Columns: document_id (int), cluster_id (int), ground_truth_category (int), distance_to_centroid (float)
Shape: (120000, 4)
Source: data/processed/cluster_assignments.csv
Validation: All cluster_id in [0, 3], no missing values

# Cluster Labels (from Story 2.5)
Type: JSON
Schema: {
  "clusters": {
    "0": {"label": "Sports", ...},
    "1": {"label": "World", ...},
    "2": {"label": "Business", ...},
    "3": {"label": "Sci/Tech", ...}
  }
}
Source: results/cluster_labels.json
Validation: All 4 cluster IDs present, labels non-empty

# AG News Training Documents
Type: List[dict]
Schema: [{"id": int, "text": str, "label": int}, ...]
Length: 120,000
Source: Hugging Face datasets (loaded via Story 1.3)
Validation: All documents have 'id' and 'text' fields
```

**Output Data:**

```python
# Agent Metadata JSON
Type: JSON file
Path: results/agent_metadata.json
Format: Structured JSON (indent=2)
Schema:
{
  "timestamp": str (ISO format),
  "n_agents": int (4),
  "baseline_context_size": int (total characters),
  "agents": {
    "0": {
      "cluster_id": int (0),
      "cluster_label": str ("Sports"),
      "num_documents": int (~30000),
      "context_size_chars": int (~12M),
      "context_size_tokens": int (~50K estimate),
      "reduction_percentage": float (0.75)
    },
    ...
  },
  "average_reduction": float (0.75)
}
Size: ~2-5 KB
```

**API Contracts:**

```python
class SpecializedAgent:
    def __init__(
        self,
        cluster_id: int,           # 0-3
        documents: List[dict],     # [{"id": int, "text": str, "label": int}, ...]
        cluster_label: str         # e.g., "Sports"
    ) -> None:
        """
        Initialize specialized agent with cluster-specific documents.

        Args:
            cluster_id: Cluster identifier (0-3)
            documents: List of documents assigned to this cluster
            cluster_label: Human-readable cluster name

        Raises:
            ValueError: If cluster_id not in [0, 3] or documents empty
        """

    def get_context_size(self) -> int:
        """
        Calculate total character count in agent's context.

        Returns:
            Total characters across all documents
        """

    def get_documents(self) -> List[dict]:
        """
        Return all documents in agent's context.

        Returns:
            Copy of documents list (prevents external modification)
        """

    def get_metadata(self) -> dict:
        """
        Return agent metadata for logging/reporting.

        Returns:
            Dict with keys: cluster_id, cluster_label, num_documents, context_size_chars
        """


def create_agent_registry(
    cluster_assignments_df: pd.DataFrame,
    documents: List[dict],
    cluster_labels: dict
) -> Dict[int, SpecializedAgent]:
    """
    Create registry of specialized agents for all clusters.

    Args:
        cluster_assignments_df: DataFrame with document_id and cluster_id
        documents: All training documents
        cluster_labels: Mapping from cluster_id to label string

    Returns:
        Dict mapping cluster_id (0-3) → SpecializedAgent

    Raises:
        ValueError: If cluster assignments incomplete or labels missing
    """
```

### Project Structure Notes

After completion, the following files will be created/modified:

**New Files:**
- `src/context_aware_multi_agent_system/models/agent.py` - SpecializedAgent class
- `scripts/06_initialize_agents.py` - Orchestration script for agent creation
- `results/agent_metadata.json` - Agent initialization metadata
- `tests/epic3/test_agent.py` - Unit tests for SpecializedAgent
- `tests/epic3/test_agent_initialization.py` - Integration tests for agent creation

**Modified Files:**
- `src/context_aware_multi_agent_system/models/__init__.py` - Export SpecializedAgent
- `README.md` - Add agent initialization documentation

**Expected Directory Structure:**
```
context-aware-multi-agent-system/
├── scripts/
│   ├── 01_generate_embeddings.py       # EXISTING: From Story 2.1
│   ├── 02_train_clustering.py          # EXISTING: From Story 2.2
│   ├── 03_evaluate_clustering.py       # EXISTING: From Story 2.3
│   ├── 04_visualize_clusters.py        # EXISTING: From Story 2.4
│   ├── 05_analyze_clusters.py          # EXISTING: From Story 2.5
│   └── 06_initialize_agents.py         # NEW: Agent initialization
├── data/
│   └── processed/                      # EXISTING: From Story 2.2
│       └── cluster_assignments.csv     # INPUT: Cluster labels
├── src/
│   └── context_aware_multi_agent_system/
│       └── models/                     # EXISTING: Created in Epic 2
│           ├── __init__.py             # MODIFIED: Export SpecializedAgent
│           ├── clustering.py           # EXISTING: From Story 2.2
│           └── agent.py                # NEW: SpecializedAgent class
├── results/                            # EXISTING: Created in Story 2.5
│   ├── cluster_labels.json             # INPUT: From Story 2.5
│   └── agent_metadata.json             # NEW: Agent initialization data
└── tests/
    └── epic3/                          # NEW: Epic 3 tests
        ├── __init__.py
        ├── test_agent.py               # NEW: Unit tests
        └── test_agent_initialization.py  # NEW: Integration tests
```

### Testing Standards

**Unit Tests:**
```python
# tests/epic3/test_agent.py

def test_specialized_agent_initialization():
    """Test agent initialization with valid inputs."""
    documents = [
        {"id": 0, "text": "Sample document 1", "label": 2},
        {"id": 1, "text": "Sample document 2", "label": 2}
    ]

    agent = SpecializedAgent(
        cluster_id=0,
        documents=documents,
        cluster_label="Sports"
    )

    assert agent.cluster_id == 0
    assert agent.cluster_label == "Sports"
    assert len(agent.get_documents()) == 2

def test_get_context_size():
    """Test context size calculation."""
    documents = [
        {"id": 0, "text": "12345", "label": 0},  # 5 chars
        {"id": 1, "text": "abcde", "label": 0}   # 5 chars
    ]

    agent = SpecializedAgent(0, documents, "Test")
    assert agent.get_context_size() == 10

def test_get_documents_returns_copy():
    """Test that get_documents returns copy, not reference."""
    documents = [{"id": 0, "text": "test", "label": 0}]
    agent = SpecializedAgent(0, documents, "Test")

    docs_copy = agent.get_documents()
    docs_copy.append({"id": 1, "text": "extra", "label": 0})

    # Original should be unchanged
    assert len(agent.get_documents()) == 1

def test_invalid_cluster_id():
    """Test that invalid cluster_id raises ValueError."""
    documents = [{"id": 0, "text": "test", "label": 0}]

    with pytest.raises(ValueError, match="cluster_id must be in range"):
        SpecializedAgent(cluster_id=5, documents=documents, cluster_label="Test")

def test_empty_documents():
    """Test that empty documents raise ValueError."""
    with pytest.raises(ValueError, match="documents cannot be empty"):
        SpecializedAgent(cluster_id=0, documents=[], cluster_label="Test")
```

**Integration Tests:**
```python
# tests/epic3/test_agent_initialization.py

def test_full_agent_initialization_pipeline():
    """Test creating all 4 agents from cluster assignments."""
    result = subprocess.run(['python', 'scripts/06_initialize_agents.py'],
                           capture_output=True)
    assert result.returncode == 0

    # Verify metadata file created
    assert Path('results/agent_metadata.json').exists()

    # Verify metadata schema
    with open('results/agent_metadata.json') as f:
        metadata = json.load(f)

    assert metadata['n_agents'] == 4
    assert len(metadata['agents']) == 4

    # Verify all agents present
    for cluster_id in range(4):
        agent_data = metadata['agents'][str(cluster_id)]
        assert agent_data['cluster_id'] == cluster_id
        assert agent_data['num_documents'] > 0
        assert 0.70 <= agent_data['reduction_percentage'] <= 0.80

def test_agent_registry_creation():
    """Test creating agent registry from actual data."""
    # Load data
    assignments_df = pd.read_csv('data/processed/cluster_assignments.csv')
    with open('results/cluster_labels.json') as f:
        cluster_labels_data = json.load(f)
        cluster_labels = {int(k): v['label'] for k, v in cluster_labels_data['clusters'].items()}

    # Load documents (mock or actual)
    documents = load_ag_news_documents()  # From Story 1.3

    # Create registry
    registry = create_agent_registry(assignments_df, documents, cluster_labels)

    # Validate registry
    assert len(registry) == 4
    assert set(registry.keys()) == {0, 1, 2, 3}

    # Validate document partitioning
    total_docs = sum(len(agent.get_documents()) for agent in registry.values())
    assert total_docs == 120000

def test_context_reduction_calculation():
    """Test that context reduction is ~75% for each agent."""
    registry = create_agent_registry(...)  # Setup

    baseline_size = 120000 * 400  # Approx total characters

    for agent in registry.values():
        agent_size = agent.get_context_size()
        reduction = 1 - (agent_size / baseline_size)
        assert 0.70 <= reduction <= 0.80  # 70-80% reduction
```

**Expected Test Coverage:**
- SpecializedAgent class: initialization, all methods, error handling
- Agent registry creation: 4 agents, correct partitioning, no overlap
- Context reduction: calculation, validation, logging
- Performance: agent creation <5 seconds
- Integration: full pipeline from cluster assignments to agent metadata

### Learnings from Previous Story

**From Story 2-5-cluster-analysis-and-labeling (Status: done):**

- ✅ **Cluster Outputs Available**: Use cluster results from Story 2.2 and Story 2.5
  - Cluster assignments: `data/processed/cluster_assignments.csv`
  - Cluster labels: `results/cluster_labels.json` (from Story 2.5)
  - Cluster purity: ~25% (Story 2.5 completion notes) - indicates room for improvement but functionality complete
  - Validation: Check files exist before loading

- ✅ **Cluster Purity Context**: Story 2.5 achieved 25.3% average purity
  - Below 70% target, but all functionality implemented correctly
  - Indicates K-Means may need optimization in future epic
  - Does not block Story 3.1 - agents work regardless of purity
  - Future: Higher purity would improve classification accuracy (Story 3.4)

- ✅ **Configuration Pattern**: Follow established config access pattern
  - Use `config.get("agent.context_limit")` if agent-specific config needed
  - Use `paths.results` for output directory
  - Use `paths.data_processed` for cluster assignments
  - Add agent section to `config.yaml` if needed

- ✅ **Logging Pattern**: Follow emoji-prefixed logging from previous stories
  - INFO: "📊 Loading cluster assignments and labels..."
  - SUCCESS: "✅ Loaded 4 cluster labels: Sports, World, Business, Sci/Tech"
  - INFO: "📊 Creating specialized agents..."
  - SUCCESS: "✅ Initialized Agent 0 (Sports): 30,245 documents, 12,345,678 chars"
  - SUCCESS: "✅ All 4 agents created successfully"
  - INFO: "📊 Calculating context reduction..."
  - SUCCESS: "✅ Average context reduction: 75.2%"
  - WARNING: "⚠️ Agent 2 has fewer documents than expected (28,500 vs ~30,000)"
  - ERROR: "❌ Agent creation failed: {error_message}"

- ✅ **Reproducibility Pattern**: Reuse set_seed() from previous stories
  - Call `set_seed(42)` at script start (for consistency)
  - Agent creation is deterministic (based on cluster assignments)
  - Ensures reproducible agent initialization

- ✅ **Error Handling Pattern**: Follow previous stories' error handling approach
  - Clear error messages with troubleshooting guidance
  - FileNotFoundError if cluster assignments missing: suggest running Story 2.2 script
  - FileNotFoundError if cluster labels missing: suggest running Story 2.5 script
  - ValueError for validation failures with helpful context
  - Provide actionable next steps

- ✅ **Type Hints and Docstrings**: Maintain documentation standards
  - All methods have full type hints
  - Google-style docstrings with usage examples
  - Example: `def get_context_size(self) -> int:`

- ✅ **Data Validation Pattern**: Follow validation approach
  - Pre-flight checks: file exists, shape correct, dtype correct, cluster_id range [0,3]
  - Fail-fast with clear error messages
  - Log validation success for debugging

- ✅ **Directory Creation**: Follow pattern for output directories
  - Use `Path.mkdir(parents=True, exist_ok=True)` to create directories
  - Create `results/` if it doesn't exist
  - No errors if directories already exist

- ✅ **Testing Pattern**: Follow comprehensive test approach
  - Create `tests/epic3/test_agent.py` for unit tests
  - Create `tests/epic3/test_agent_initialization.py` for integration tests
  - Map tests to acceptance criteria (AC-3.1.1, AC-3.1.2, etc.)
  - Use `pytest.raises()` for exception testing
  - Use pytest fixtures for test setup (temp directories, mock data)
  - Test both unit (small synthetic data) and integration (full dataset)

**Files to Reuse (DO NOT RECREATE):**
- `src/utils/logger.py` - Use for emoji-prefixed logging
- `src/utils/reproducibility.py` - Use `set_seed(42)` function
- `src/config.py` - Load config for agent parameters (if configured)
- `data/processed/cluster_assignments.csv` - Input from Story 2.2
- `results/cluster_labels.json` - Input from Story 2.5
- AG News training documents - From Story 1.3

**Key Services from Previous Stories:**
- **Config class** (Story 1.2): Configuration management with `get()` method
- **Paths class** (Story 1.2): Path resolution
- **set_seed()** (Story 1.1): Reproducibility enforcement
- **Logger** (Story 1.2): Emoji-prefixed structured logging
- **KMeansClustering** (Story 2.2): Cluster assignments available
- **ClusterAnalyzer** (Story 2.5): Cluster labels available

**Technical Debt from Previous Stories:**
- Story 2.5: Cluster purity 25.3% (below 70% target) - future optimization needed
  - Impact on Story 3.1: None - agents work regardless of purity
  - Impact on Story 3.4: May affect classification accuracy
  - Mitigation: Document purity limitation in agent metadata

**New Patterns to Establish:**
- **Agent Registry Pattern**: `Dict[int, SpecializedAgent]` for O(1) cluster_id lookup
- **Context Reduction Calculation**: `1 - (agent_context / baseline_context)`
- **Agent Initialization Logging**: "📊 Initialized Agent {id} ({label}): {num_docs:,} documents, {context_size:,} chars"
- **Agent Metadata Export**: JSON with cluster_id, label, num_docs, context_size, reduction_pct

### References

- [Source: docs/tech-spec-epic-3.md#Specialized Agent Implementation]
- [Source: docs/epics.md#Story 3.1 - Specialized Agent Implementation]
- [Source: docs/PRD.md#FR-5 - Multi-Agent Classification System]
- [Source: docs/architecture.md#Agent Components]
- [Source: stories/2-2-k-means-clustering-implementation.md#Cluster Assignments Available]
- [Source: stories/2-5-cluster-analysis-and-labeling.md#Cluster Labels Available]

## Change Log

### 2025-11-09 - Story Drafted
- **Version:** v1.0
- **Changes:**
  - ✅ Story created from epics.md and tech-spec-epic-3.md
  - ✅ All 7 acceptance criteria defined with validation examples
  - ✅ Tasks and subtasks mapped to ACs
  - ✅ Dev notes include architecture alignment and learnings from Story 2.5
  - ✅ References to source documents included
  - ✅ Context reduction strategy detailed with cost optimization rationale
  - ✅ Integration points with Epic 2 outputs documented
- **Status:** backlog → drafted

## Dev Agent Record

### Context Reference

- [3-1-specialized-agent-implementation.context.xml](3-1-specialized-agent-implementation.context.xml)

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
