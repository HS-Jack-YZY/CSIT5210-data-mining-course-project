"""
Configuration management system for the Context-Aware Multi-Agent News Classification System.

This module provides centralized configuration loading, validation, and path management
for all experimental parameters and project directories.

Classes:
    Config: Loads and validates configuration from config.yaml and environment variables
    Paths: Manages all project directory paths with automatic creation
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


class Config:
    """
    Centralized configuration manager.

    Loads configuration from config.yaml and environment variables (.env),
    validates all required fields, and provides dot-notation access to parameters.

    Usage:
        config = Config()
        n_clusters = config.get("clustering.n_clusters")  # Returns 4
        api_key = config.gemini_api_key  # Loads from .env
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Config by loading config.yaml and environment variables.

        Args:
            config_path: Path to config.yaml file (default: project_root/config.yaml)

        Raises:
            FileNotFoundError: If config.yaml doesn't exist
            yaml.YAMLError: If config.yaml has invalid syntax
        """
        # Load environment variables from .env file
        load_dotenv()

        # Determine config file path
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent.resolve()
            config_path = project_root / "config.yaml"
        else:
            config_path = Path(config_path)

        # Load YAML configuration
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please create config.yaml in the project root directory."
            )

        try:
            with open(config_path, 'r') as f:
                self._config: Dict[str, Any] = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Invalid YAML syntax in {config_path}:\n{e}\n"
                f"Please check the configuration file for syntax errors."
            )

        # Store config path for reference
        self._config_path = config_path

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve configuration value using dot notation.

        Args:
            key: Configuration key with dot notation (e.g., "clustering.n_clusters")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config.get("clustering.n_clusters")
            4
            >>> config.get("embedding.model")
            "gemini-embedding-001"
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def validate(self) -> bool:
        """
        Validate all required configuration fields and data types.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If required fields are missing
            TypeError: If field has invalid data type
        """
        # Define required fields with expected types
        required_fields = {
            "dataset.name": str,
            "dataset.categories": int,
            "clustering.algorithm": str,
            "clustering.n_clusters": int,
            "clustering.random_state": int,
            "clustering.max_iter": int,
            "clustering.init": str,
            "embedding.model": str,
            "embedding.batch_size": int,
            "embedding.cache_dir": str,
            "embedding.output_dimensionality": int,
            "classification.method": str,
            "classification.threshold": (int, float),
            "metrics.cost_per_1M_tokens_under_200k": (int, float),
            "metrics.cost_per_1M_tokens_over_200k": (int, float),
            "metrics.target_cost_reduction": (int, float),
        }

        # Validate each required field
        for field_path, expected_type in required_fields.items():
            value = self.get(field_path)

            # Check if field exists
            if value is None:
                raise ValueError(
                    f"Required configuration field missing: '{field_path}'\n"
                    f"Please add this field to {self._config_path}"
                )

            # Check data type
            if not isinstance(value, expected_type):
                expected_type_name = (
                    expected_type.__name__
                    if not isinstance(expected_type, tuple)
                    else " or ".join(t.__name__ for t in expected_type)
                )
                raise TypeError(
                    f"Invalid type for '{field_path}': expected {expected_type_name}, "
                    f"got {type(value).__name__}\n"
                    f"Please fix the configuration in {self._config_path}"
                )

        return True

    @property
    def gemini_api_key(self) -> str:
        """
        Retrieve GEMINI_API_KEY from environment variables.

        Returns:
            API key string

        Raises:
            ValueError: If GEMINI_API_KEY is not found in environment
        """
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Copy .env.example to .env and add your API key."
            )

        return api_key

    @property
    def dataset(self) -> Dict[str, Any]:
        """Access dataset configuration section."""
        return self._config.get("dataset", {})

    @property
    def clustering(self) -> Dict[str, Any]:
        """Access clustering configuration section."""
        return self._config.get("clustering", {})

    @property
    def embedding(self) -> Dict[str, Any]:
        """Access embedding configuration section."""
        return self._config.get("embedding", {})

    @property
    def classification(self) -> Dict[str, Any]:
        """Access classification configuration section."""
        return self._config.get("classification", {})

    @property
    def metrics(self) -> Dict[str, Any]:
        """Access metrics configuration section."""
        return self._config.get("metrics", {})


class Paths:
    """
    Centralized path management for all project directories.

    Provides absolute Path objects for all project directories and automatically
    creates them if they don't exist. Uses pathlib.Path for cross-platform compatibility.

    Usage:
        paths = Paths()
        data_file = paths.data_raw / "ag_news.csv"
        model_file = paths.models / "kmeans_model.pkl"
    """

    def __init__(self):
        """
        Initialize Paths and create all project directories.

        All paths are absolute (resolved) for consistency across different
        working directories.
        """
        # Project root is 3 levels up from this file: src/context_aware_multi_agent_system/config.py
        self.project_root = Path(__file__).parent.parent.parent.resolve()

        # Define all project directories
        self.data = self.project_root / "data"
        self.data_raw = self.data / "raw"
        self.data_embeddings = self.data / "embeddings"
        self.data_interim = self.data / "interim"
        self.data_processed = self.data / "processed"

        self.models = self.project_root / "models"
        self.notebooks = self.project_root / "notebooks"

        self.reports = self.project_root / "reports"
        self.reports_figures = self.reports / "figures"

        self.results = self.project_root / "results"
        self.src = self.project_root / "src"

        # Create all directories if they don't exist
        self._create_directories()

    def _create_directories(self) -> None:
        """
        Create all project directories if they don't exist.

        Uses mkdir(parents=True, exist_ok=True) for safe creation without errors.
        """
        directories = [
            self.data_raw,
            self.data_embeddings,
            self.data_interim,
            self.data_processed,
            self.models,
            self.notebooks,
            self.reports_figures,
            self.results,
            self.src,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Paths(\n"
            f"  project_root={self.project_root},\n"
            f"  data={self.data},\n"
            f"  models={self.models},\n"
            f"  reports={self.reports},\n"
            f"  results={self.results}\n"
            f")"
        )
