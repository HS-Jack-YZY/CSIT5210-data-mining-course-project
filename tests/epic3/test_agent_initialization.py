"""
Integration tests for agent initialization script.

Tests cover:
- AC-3.1.2: Four agent instances created
- AC-3.1.3: Cluster-specific document assignment
- AC-3.1.4: Agent registry pattern
- AC-3.1.5: Agent initialization logging
- AC-3.1.6: Context size reduction calculation
- AC-3.1.7: Agent registry accessibility
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from context_aware_multi_agent_system.config import Paths
from context_aware_multi_agent_system.models.agent import (
    SpecializedAgent,
    create_agent_registry
)


class TestAgentRegistryCreation:
    """Test create_agent_registry function (AC-3.1.2, AC-3.1.4)."""

    def test_registry_creation_with_synthetic_data(self):
        """Test creating agent registry from synthetic cluster assignments."""
        # Create synthetic cluster assignments
        assignments_df = pd.DataFrame({
            'document_id': list(range(12)),
            'cluster_id': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        })

        # Create synthetic documents
        documents = [
            {"id": i, "text": f"Document {i} text", "label": i % 4}
            for i in range(12)
        ]

        # Create cluster labels
        cluster_labels = {
            0: "Sports",
            1: "World",
            2: "Business",
            3: "Sci/Tech"
        }

        # Create registry
        registry = create_agent_registry(
            cluster_assignments_df=assignments_df,
            documents=documents,
            cluster_labels=cluster_labels
        )

        # Validate registry structure (AC-3.1.4)
        assert isinstance(registry, dict)
        assert len(registry) == 4
        assert set(registry.keys()) == {0, 1, 2, 3}

        # Validate all values are SpecializedAgent instances
        for agent in registry.values():
            assert isinstance(agent, SpecializedAgent)

    def test_registry_document_partitioning(self):
        """Test that registry correctly partitions documents by cluster (AC-3.1.3)."""
        # Create cluster assignments
        assignments_df = pd.DataFrame({
            'document_id': [0, 1, 2, 3, 4, 5, 6, 7],
            'cluster_id': [0, 0, 1, 1, 2, 2, 3, 3]
        })

        documents = [
            {"id": i, "text": f"Doc {i}", "label": i % 4}
            for i in range(8)
        ]

        cluster_labels = {0: "A", 1: "B", 2: "C", 3: "D"}

        registry = create_agent_registry(assignments_df, documents, cluster_labels)

        # Verify each agent has correct documents
        assert len(registry[0].get_documents()) == 2
        assert len(registry[1].get_documents()) == 2
        assert len(registry[2].get_documents()) == 2
        assert len(registry[3].get_documents()) == 2

        # Verify no overlap between agents
        all_doc_ids = []
        for agent in registry.values():
            agent_doc_ids = [doc['id'] for doc in agent.get_documents()]
            all_doc_ids.extend(agent_doc_ids)

        # No duplicates
        assert len(all_doc_ids) == len(set(all_doc_ids))

    def test_registry_missing_cluster_raises_error(self):
        """Test that missing cluster IDs raise ValueError."""
        # Missing cluster 2 and 3
        assignments_df = pd.DataFrame({
            'document_id': [0, 1, 2, 3],
            'cluster_id': [0, 0, 1, 1]
        })

        documents = [{"id": i, "text": f"Doc {i}", "label": 0} for i in range(4)]
        cluster_labels = {0: "A", 1: "B"}

        with pytest.raises(ValueError, match="Cluster assignments incomplete"):
            create_agent_registry(assignments_df, documents, cluster_labels)

    def test_registry_missing_labels_raises_error(self):
        """Test that missing cluster labels raise ValueError."""
        assignments_df = pd.DataFrame({
            'document_id': [0, 1, 2, 3],
            'cluster_id': [0, 1, 2, 3]
        })

        documents = [{"id": i, "text": f"Doc {i}", "label": 0} for i in range(4)]

        # Missing label for cluster 3
        cluster_labels = {0: "A", 1: "B", 2: "C"}

        with pytest.raises(ValueError, match="Cluster labels missing"):
            create_agent_registry(assignments_df, documents, cluster_labels)

    def test_registry_type_validation(self):
        """Test registry type structure (AC-3.1.4)."""
        assignments_df = pd.DataFrame({
            'document_id': [0, 1, 2, 3],
            'cluster_id': [0, 1, 2, 3]
        })

        documents = [{"id": i, "text": f"Doc {i}", "label": 0} for i in range(4)]
        cluster_labels = {0: "A", 1: "B", 2: "C", 3: "D"}

        registry = create_agent_registry(assignments_df, documents, cluster_labels)

        # Validate types
        assert isinstance(registry, dict)
        assert all(isinstance(k, int) for k in registry.keys())
        assert all(isinstance(v, SpecializedAgent) for v in registry.values())


class TestAgentRegistryAccessibility:
    """Test agent registry accessibility (AC-3.1.7)."""

    @pytest.fixture
    def sample_registry(self):
        """Create sample registry for testing."""
        assignments_df = pd.DataFrame({
            'document_id': [0, 1, 2, 3],
            'cluster_id': [0, 1, 2, 3]
        })

        documents = [{"id": i, "text": f"Doc {i}", "label": i} for i in range(4)]

        cluster_labels = {
            0: "Sports",
            1: "World",
            2: "Business",
            3: "Sci/Tech"
        }

        return create_agent_registry(assignments_df, documents, cluster_labels)

    def test_direct_access_by_cluster_id(self, sample_registry):
        """Test direct access to agents by cluster_id (AC-3.1.7)."""
        sports_agent = sample_registry[0]
        assert sports_agent.cluster_id == 0
        assert sports_agent.cluster_label == "Sports"

        world_agent = sample_registry[1]
        assert world_agent.cluster_id == 1
        assert world_agent.cluster_label == "World"

        business_agent = sample_registry[2]
        assert business_agent.cluster_id == 2
        assert business_agent.cluster_label == "Business"

        scitech_agent = sample_registry[3]
        assert scitech_agent.cluster_id == 3
        assert scitech_agent.cluster_label == "Sci/Tech"

    def test_registry_iteration(self, sample_registry):
        """Test registry supports iteration (AC-3.1.7)."""
        cluster_ids = []
        labels = []

        for cluster_id, agent in sample_registry.items():
            assert agent.cluster_id == cluster_id
            assert cluster_id in range(4)
            cluster_ids.append(cluster_id)
            labels.append(agent.cluster_label)

        assert set(cluster_ids) == {0, 1, 2, 3}
        assert set(labels) == {"Sports", "World", "Business", "Sci/Tech"}

    def test_invalid_cluster_id_raises_keyerror(self, sample_registry):
        """Test that invalid cluster_id raises KeyError (AC-3.1.7)."""
        with pytest.raises(KeyError):
            _ = sample_registry[5]

        with pytest.raises(KeyError):
            _ = sample_registry[-1]


class TestContextReductionCalculation:
    """Test context size reduction calculation (AC-3.1.6)."""

    def test_context_reduction_calculation(self):
        """Test that context reduction is calculated correctly."""
        # Create balanced assignments (25% each)
        assignments_df = pd.DataFrame({
            'document_id': list(range(100)),
            'cluster_id': [i % 4 for i in range(100)]
        })

        # Each document has 100 characters
        documents = [
            {"id": i, "text": "a" * 100, "label": i % 4}
            for i in range(100)
        ]

        cluster_labels = {0: "A", 1: "B", 2: "C", 3: "D"}

        registry = create_agent_registry(assignments_df, documents, cluster_labels)

        # Calculate baseline context size
        baseline_context = sum(len(doc['text']) for doc in documents)
        assert baseline_context == 10000  # 100 docs × 100 chars

        # Each agent should have ~25% of baseline (75% reduction)
        for cluster_id, agent in registry.items():
            agent_context = agent.get_context_size()
            reduction_pct = 1 - (agent_context / baseline_context)

            # Each agent has 25 documents × 100 chars = 2500 chars
            assert agent_context == 2500
            assert reduction_pct == 0.75  # 75% reduction

    def test_context_reduction_with_variance(self):
        """Test context reduction with unbalanced cluster sizes."""
        # Unbalanced: 40%, 30%, 20%, 10%
        assignments_df = pd.DataFrame({
            'document_id': list(range(100)),
            'cluster_id': ([0] * 40 + [1] * 30 + [2] * 20 + [3] * 10)
        })

        documents = [
            {"id": i, "text": "x" * 100, "label": 0}
            for i in range(100)
        ]

        cluster_labels = {0: "A", 1: "B", 2: "C", 3: "D"}

        registry = create_agent_registry(assignments_df, documents, cluster_labels)

        baseline_context = 100 * 100  # 10000

        # Agent 0: 40 docs → 40% of baseline → 60% reduction
        assert registry[0].get_context_size() == 4000
        reduction_0 = 1 - (4000 / baseline_context)
        assert reduction_0 == 0.60

        # Agent 3: 10 docs → 10% of baseline → 90% reduction
        assert registry[3].get_context_size() == 1000
        reduction_3 = 1 - (1000 / baseline_context)
        assert reduction_3 == 0.90


class TestAgentInitializationScript:
    """Integration test for full agent initialization script (AC-3.1.2 through AC-3.1.7)."""

    @pytest.mark.slow
    def test_full_initialization_pipeline(self):
        """Test running complete agent initialization script."""
        paths = Paths()
        script_path = paths.project_root / "scripts" / "06_initialize_agents.py"

        # Check prerequisites exist
        if not (paths.data_processed / "cluster_assignments.csv").exists():
            pytest.skip("Cluster assignments not found - run clustering first")

        if not (paths.results / "cluster_labels.json").exists():
            pytest.skip("Cluster labels not found - run cluster analysis first")

        # Run script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True
        )

        # Verify script succeeded
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Verify metadata file created
        metadata_path = paths.results / "agent_metadata.json"
        assert metadata_path.exists()

        # Verify metadata schema
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        assert 'n_agents' in metadata
        assert metadata['n_agents'] == 4

        assert 'agents' in metadata
        assert len(metadata['agents']) == 4

        # Verify all agents present
        for cluster_id in range(4):
            agent_key = str(cluster_id)
            assert agent_key in metadata['agents']

            agent_data = metadata['agents'][agent_key]
            assert agent_data['cluster_id'] == cluster_id
            assert agent_data['num_documents'] > 0
            assert 'cluster_label' in agent_data
            assert 'context_size_chars' in agent_data
            assert 'reduction_percentage' in agent_data

            # Verify reduction is reasonable (70-80% for balanced clusters)
            reduction = agent_data['reduction_percentage']
            assert 0.70 <= reduction <= 0.80

        # Verify average reduction
        assert 'average_reduction' in metadata
        avg_reduction = metadata['average_reduction']
        assert 0.70 <= avg_reduction <= 0.80

        # Verify total documents
        assert metadata['total_documents'] == 120000


class TestDocumentConservation:
    """Test that documents are conserved across agents (AC-3.1.3)."""

    def test_total_documents_conservation(self):
        """Test that sum of documents across agents equals total."""
        assignments_df = pd.DataFrame({
            'document_id': list(range(1000)),
            'cluster_id': [i % 4 for i in range(1000)]
        })

        documents = [
            {"id": i, "text": f"Doc {i}", "label": i % 4}
            for i in range(1000)
        ]

        cluster_labels = {0: "A", 1: "B", 2: "C", 3: "D"}

        registry = create_agent_registry(assignments_df, documents, cluster_labels)

        # Count total documents across all agents
        total_docs = sum(len(agent.get_documents()) for agent in registry.values())

        assert total_docs == 1000

    def test_no_document_overlap(self):
        """Test that no document appears in multiple agents (AC-3.1.3)."""
        assignments_df = pd.DataFrame({
            'document_id': list(range(100)),
            'cluster_id': [i % 4 for i in range(100)]
        })

        documents = [
            {"id": i, "text": f"Doc {i}", "label": i % 4}
            for i in range(100)
        ]

        cluster_labels = {0: "A", 1: "B", 2: "C", 3: "D"}

        registry = create_agent_registry(assignments_df, documents, cluster_labels)

        # Collect all document IDs across agents
        all_doc_ids = []
        for agent in registry.values():
            agent_doc_ids = [doc['id'] for doc in agent.get_documents()]
            all_doc_ids.extend(agent_doc_ids)

        # No duplicates (strict partitioning)
        assert len(all_doc_ids) == len(set(all_doc_ids))

    def test_each_agent_has_only_its_cluster_documents(self):
        """Test that each agent contains only its cluster's documents (AC-3.1.3)."""
        assignments_df = pd.DataFrame({
            'document_id': list(range(20)),
            'cluster_id': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        })

        documents = [
            {"id": i, "text": f"Doc {i}", "label": 0}
            for i in range(20)
        ]

        cluster_labels = {0: "A", 1: "B", 2: "C", 3: "D"}

        registry = create_agent_registry(assignments_df, documents, cluster_labels)

        # Verify each agent has only documents from its cluster
        for cluster_id in range(4):
            expected_doc_ids = assignments_df[
                assignments_df['cluster_id'] == cluster_id
            ]['document_id'].tolist()

            agent_doc_ids = [
                doc['id'] for doc in registry[cluster_id].get_documents()
            ]

            assert set(agent_doc_ids) == set(expected_doc_ids)
