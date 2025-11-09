"""
Unit tests for SpecializedAgent class.

Tests cover:
- AC-3.1.1: SpecializedAgent class implementation
- Initialization validation
- Context size calculation
- Document retrieval
- Metadata generation
- Error handling
"""

import pytest
from src.context_aware_multi_agent_system.models.agent import SpecializedAgent


class TestSpecializedAgentInitialization:
    """Test SpecializedAgent initialization (AC-3.1.1)."""

    def test_valid_initialization(self):
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
        assert agent.context_size_chars > 0

    def test_initialization_with_all_cluster_ids(self):
        """Test initialization with all valid cluster IDs (0-3)."""
        documents = [{"id": 0, "text": "Test", "label": 0}]

        for cluster_id in range(4):
            agent = SpecializedAgent(
                cluster_id=cluster_id,
                documents=documents,
                cluster_label=f"Cluster{cluster_id}"
            )
            assert agent.cluster_id == cluster_id

    def test_invalid_cluster_id_too_low(self):
        """Test that cluster_id < 0 raises ValueError."""
        documents = [{"id": 0, "text": "test", "label": 0}]

        with pytest.raises(ValueError, match="cluster_id must be in range"):
            SpecializedAgent(cluster_id=-1, documents=documents, cluster_label="Test")

    def test_invalid_cluster_id_too_high(self):
        """Test that cluster_id > 3 raises ValueError."""
        documents = [{"id": 0, "text": "test", "label": 0}]

        with pytest.raises(ValueError, match="cluster_id must be in range"):
            SpecializedAgent(cluster_id=5, documents=documents, cluster_label="Test")

    def test_empty_documents_raises_error(self):
        """Test that empty documents list raises ValueError."""
        with pytest.raises(ValueError, match="documents cannot be empty"):
            SpecializedAgent(cluster_id=0, documents=[], cluster_label="Test")

    def test_empty_cluster_label_raises_error(self):
        """Test that empty cluster_label raises ValueError."""
        documents = [{"id": 0, "text": "test", "label": 0}]

        with pytest.raises(ValueError, match="cluster_label cannot be empty"):
            SpecializedAgent(cluster_id=0, documents=documents, cluster_label="")

    def test_whitespace_cluster_label_raises_error(self):
        """Test that whitespace-only cluster_label raises ValueError."""
        documents = [{"id": 0, "text": "test", "label": 0}]

        with pytest.raises(ValueError, match="cluster_label cannot be empty"):
            SpecializedAgent(cluster_id=0, documents=documents, cluster_label="   ")

    def test_cluster_label_stripped(self):
        """Test that cluster_label whitespace is stripped."""
        documents = [{"id": 0, "text": "test", "label": 0}]

        agent = SpecializedAgent(
            cluster_id=0,
            documents=documents,
            cluster_label="  Sports  "
        )

        assert agent.cluster_label == "Sports"


class TestGetContextSize:
    """Test get_context_size method (AC-3.1.1)."""

    def test_context_size_calculation(self):
        """Test context size calculation with known character counts."""
        documents = [
            {"id": 0, "text": "12345", "label": 0},  # 5 chars
            {"id": 1, "text": "abcde", "label": 0}   # 5 chars
        ]

        agent = SpecializedAgent(0, documents, "Test")
        assert agent.get_context_size() == 10

    def test_context_size_empty_strings(self):
        """Test context size with documents containing empty text."""
        documents = [
            {"id": 0, "text": "", "label": 0},
            {"id": 1, "text": "test", "label": 0}
        ]

        agent = SpecializedAgent(0, documents, "Test")
        assert agent.get_context_size() == 4

    def test_context_size_single_document(self):
        """Test context size with single document."""
        documents = [{"id": 0, "text": "Hello World", "label": 0}]

        agent = SpecializedAgent(0, documents, "Test")
        assert agent.get_context_size() == 11

    def test_context_size_cached(self):
        """Test that context size is cached on initialization."""
        documents = [{"id": 0, "text": "test", "label": 0}]

        agent = SpecializedAgent(0, documents, "Test")

        # Get context size twice - should return same cached value
        size1 = agent.get_context_size()
        size2 = agent.get_context_size()

        assert size1 == size2 == 4


class TestGetDocuments:
    """Test get_documents method (AC-3.1.1)."""

    def test_get_documents_returns_copy(self):
        """Test that get_documents returns copy, not reference."""
        documents = [{"id": 0, "text": "test", "label": 0}]
        agent = SpecializedAgent(0, documents, "Test")

        # Get documents and modify the returned list
        docs_copy = agent.get_documents()
        docs_copy.append({"id": 1, "text": "extra", "label": 0})

        # Original should be unchanged
        assert len(agent.get_documents()) == 1

    def test_get_documents_content(self):
        """Test that get_documents returns correct content."""
        documents = [
            {"id": 0, "text": "doc1", "label": 0},
            {"id": 1, "text": "doc2", "label": 1}
        ]
        agent = SpecializedAgent(0, documents, "Test")

        retrieved_docs = agent.get_documents()

        assert len(retrieved_docs) == 2
        assert retrieved_docs[0]["text"] == "doc1"
        assert retrieved_docs[1]["text"] == "doc2"


class TestGetMetadata:
    """Test get_metadata method (AC-3.1.1)."""

    def test_metadata_schema(self):
        """Test that metadata contains all required keys."""
        documents = [{"id": 0, "text": "test", "label": 0}]
        agent = SpecializedAgent(2, documents, "Business")

        metadata = agent.get_metadata()

        assert 'cluster_id' in metadata
        assert 'cluster_label' in metadata
        assert 'num_documents' in metadata
        assert 'context_size_chars' in metadata

    def test_metadata_values(self):
        """Test that metadata values are correct."""
        documents = [
            {"id": 0, "text": "Business news article", "label": 3},
            {"id": 1, "text": "Another business story", "label": 3}
        ]
        agent = SpecializedAgent(2, documents, "Business")

        metadata = agent.get_metadata()

        assert metadata['cluster_id'] == 2
        assert metadata['cluster_label'] == "Business"
        assert metadata['num_documents'] == 2
        assert metadata['context_size_chars'] == len("Business news article") + len("Another business story")

    def test_metadata_immutability(self):
        """Test that modifying metadata doesn't affect agent."""
        documents = [{"id": 0, "text": "test", "label": 0}]
        agent = SpecializedAgent(0, documents, "Test")

        metadata = agent.get_metadata()
        metadata['cluster_id'] = 999
        metadata['num_documents'] = 999

        # Original agent should be unchanged
        assert agent.cluster_id == 0
        assert len(agent.get_documents()) == 1


class TestAgentIntegration:
    """Integration tests for SpecializedAgent with realistic data."""

    def test_agent_with_realistic_documents(self):
        """Test agent with realistic AG News-style documents."""
        documents = [
            {
                "id": 0,
                "text": "The Lakers won the championship game last night with a score of 102-98.",
                "label": 1
            },
            {
                "id": 1,
                "text": "Olympic athletes prepare for the upcoming summer games in Paris.",
                "label": 1
            },
            {
                "id": 2,
                "text": "NFL season kicks off with exciting matchups across the country.",
                "label": 1
            }
        ]

        agent = SpecializedAgent(
            cluster_id=0,
            documents=documents,
            cluster_label="Sports"
        )

        assert agent.cluster_id == 0
        assert agent.cluster_label == "Sports"
        assert len(agent.get_documents()) == 3
        assert agent.get_context_size() > 100  # Realistic text length

        metadata = agent.get_metadata()
        assert metadata['num_documents'] == 3
        assert metadata['cluster_label'] == "Sports"

    def test_agent_performance_large_dataset(self):
        """Test agent performance with larger document set."""
        # Create 1000 documents
        documents = [
            {"id": i, "text": f"Document {i} " * 50, "label": i % 4}
            for i in range(1000)
        ]

        agent = SpecializedAgent(
            cluster_id=1,
            documents=documents,
            cluster_label="World"
        )

        assert len(agent.get_documents()) == 1000
        assert agent.get_context_size() > 0

        # Verify get_documents is reasonably fast (< 1ms for copy operation)
        import time
        start = time.time()
        _ = agent.get_documents()
        elapsed = time.time() - start
        assert elapsed < 0.01  # Should be very fast
