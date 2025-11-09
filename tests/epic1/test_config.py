"""
Unit and integration tests for configuration management system.

Tests the Config and Paths classes for correct loading, validation,
error handling, and directory management.
"""

import os
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

from src.context_aware_multi_agent_system.config import Config, Paths


class TestConfig:
    """Test suite for Config class."""

    def test_config_loads_without_errors(self):
        """Test AC-2: Config() initializes without errors."""
        config = Config()
        assert config is not None

    def test_config_get_with_dot_notation(self):
        """Test AC-2: config.get() with dot notation returns correct values."""
        config = Config()

        # Test clustering configuration
        assert config.get("clustering.n_clusters") == 4
        assert config.get("clustering.random_state") == 42
        assert config.get("clustering.max_iter") == 300
        assert config.get("clustering.init") == "k-means++"

        # Test embedding configuration
        assert config.get("embedding.model") == "gemini-embedding-001"
        assert config.get("embedding.output_dimensionality") == 768
        assert config.get("embedding.batch_size") == 100

        # Test dataset configuration
        assert config.get("dataset.name") == "ag_news"
        assert config.get("dataset.categories") == 4

        # Test classification configuration
        assert config.get("classification.method") == "cosine_similarity"
        assert config.get("classification.threshold") == 0.7

        # Test metrics configuration
        assert config.get("metrics.cost_per_1M_tokens_under_200k") == 3.0
        assert config.get("metrics.cost_per_1M_tokens_over_200k") == 6.0
        assert config.get("metrics.target_cost_reduction") == 0.90

    def test_config_get_returns_default_for_missing_key(self):
        """Test config.get() returns default value for non-existent keys."""
        config = Config()
        assert config.get("nonexistent.key") is None
        assert config.get("nonexistent.key", "default_value") == "default_value"

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key_12345"})
    def test_gemini_api_key_retrieves_from_env(self):
        """Test AC-2: config.gemini_api_key returns API key from .env."""
        config = Config()
        assert config.gemini_api_key == "test_api_key_12345"

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises_clear_error(self):
        """Test AC-3: Missing API key raises clear error with instructions."""
        config = Config()

        with pytest.raises(ValueError) as exc_info:
            _ = config.gemini_api_key

        error_message = str(exc_info.value)
        assert "GEMINI_API_KEY not found" in error_message
        assert "Copy .env.example to .env and add your API key" in error_message

    def test_config_validate_passes_for_valid_config(self):
        """Test AC-2: config.validate() returns True for valid config."""
        config = Config()
        assert config.validate() is True

    def test_all_sections_accessible(self):
        """Test AC-2: All sections accessible via properties."""
        config = Config()

        # Test all section properties
        assert "name" in config.dataset
        assert "algorithm" in config.clustering
        assert "model" in config.embedding
        assert "method" in config.classification
        assert "cost_per_1M_tokens_under_200k" in config.metrics

    def test_missing_config_file_raises_error(self):
        """Test AC-3: Missing config.yaml raises clear error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_path = Path(tmpdir) / "nonexistent_config.yaml"

            with pytest.raises(FileNotFoundError) as exc_info:
                Config(config_path=str(nonexistent_path))

            error_message = str(exc_info.value)
            assert "Configuration file not found" in error_message
            assert "nonexistent_config.yaml" in error_message

    def test_invalid_yaml_syntax_raises_error(self):
        """Test AC-3: Invalid YAML syntax raises clear error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write invalid YAML (unmatched brackets)
            f.write("invalid: [yaml syntax\n")
            invalid_config_path = f.name

        try:
            with pytest.raises(yaml.YAMLError) as exc_info:
                Config(config_path=invalid_config_path)

            error_message = str(exc_info.value)
            assert "Invalid YAML syntax" in error_message
        finally:
            os.unlink(invalid_config_path)

    def test_missing_required_field_raises_error(self):
        """Test AC-3: Missing required field raises ValueError with field name."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write config missing required field
            incomplete_config = {
                "dataset": {"name": "ag_news", "categories": 4},
                "clustering": {"algorithm": "kmeans"}  # Missing n_clusters
            }
            yaml.dump(incomplete_config, f)
            incomplete_config_path = f.name

        try:
            config = Config(config_path=incomplete_config_path)

            with pytest.raises(ValueError) as exc_info:
                config.validate()

            error_message = str(exc_info.value)
            assert "Required configuration field missing" in error_message
            assert "clustering.n_clusters" in error_message
        finally:
            os.unlink(incomplete_config_path)

    def test_invalid_data_type_raises_error(self):
        """Test AC-3: Invalid data type raises TypeError with expected type."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # Write config with wrong data type
            invalid_type_config = {
                "dataset": {"name": "ag_news", "categories": 4, "sample_size": None},
                "clustering": {
                    "algorithm": "kmeans",
                    "n_clusters": "four",  # Should be int, not string
                    "random_state": 42,
                    "max_iter": 300,
                    "init": "k-means++"
                },
                "embedding": {
                    "model": "gemini-embedding-001",
                    "batch_size": 100,
                    "cache_dir": "data/embeddings",
                    "output_dimensionality": 768
                },
                "classification": {"method": "cosine_similarity", "threshold": 0.7},
                "metrics": {
                    "cost_per_1M_tokens_under_200k": 3.0,
                    "cost_per_1M_tokens_over_200k": 6.0,
                    "target_cost_reduction": 0.90
                }
            }
            yaml.dump(invalid_type_config, f)
            invalid_type_path = f.name

        try:
            config = Config(config_path=invalid_type_path)

            with pytest.raises(TypeError) as exc_info:
                config.validate()

            error_message = str(exc_info.value)
            assert "Invalid type" in error_message
            assert "clustering.n_clusters" in error_message
            assert "expected int" in error_message
        finally:
            os.unlink(invalid_type_path)


class TestPaths:
    """Test suite for Paths class."""

    def test_all_path_attributes_exist(self):
        """Test AC-4: All path attributes exist."""
        paths = Paths()

        # Verify all required path attributes
        assert hasattr(paths, 'data')
        assert hasattr(paths, 'data_raw')
        assert hasattr(paths, 'data_embeddings')
        assert hasattr(paths, 'data_interim')
        assert hasattr(paths, 'data_processed')
        assert hasattr(paths, 'models')
        assert hasattr(paths, 'notebooks')
        assert hasattr(paths, 'reports')
        assert hasattr(paths, 'reports_figures')
        assert hasattr(paths, 'results')

    def test_directories_created_if_missing(self):
        """Test AC-4: Directories are created if missing."""
        paths = Paths()

        # Verify all directories exist (created by __init__)
        assert paths.data_raw.exists()
        assert paths.data_embeddings.exists()
        assert paths.data_interim.exists()
        assert paths.data_processed.exists()
        assert paths.models.exists()
        assert paths.notebooks.exists()
        assert paths.reports_figures.exists()
        assert paths.results.exists()

    def test_all_paths_are_absolute(self):
        """Test AC-4: All paths are absolute (not relative)."""
        paths = Paths()

        # Verify all paths are absolute
        assert paths.data.is_absolute()
        assert paths.data_raw.is_absolute()
        assert paths.data_embeddings.is_absolute()
        assert paths.data_interim.is_absolute()
        assert paths.data_processed.is_absolute()
        assert paths.models.is_absolute()
        assert paths.notebooks.is_absolute()
        assert paths.reports.is_absolute()
        assert paths.reports_figures.is_absolute()
        assert paths.results.is_absolute()

    def test_path_operations_use_pathlib(self):
        """Test AC-4: Path operations use pathlib.Path."""
        paths = Paths()

        # Verify all paths are Path objects
        assert isinstance(paths.data, Path)
        assert isinstance(paths.data_raw, Path)
        assert isinstance(paths.models, Path)
        assert isinstance(paths.reports, Path)
        assert isinstance(paths.results, Path)

    def test_repr_method_for_debugging(self):
        """Test Paths has useful __repr__ for debugging."""
        paths = Paths()
        repr_str = repr(paths)

        assert "Paths(" in repr_str
        assert "project_root=" in repr_str
        assert "data=" in repr_str
