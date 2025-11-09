"""
Reproducibility utilities for ensuring deterministic behavior in experiments.

This module provides utilities to set random seeds globally for reproducible results
across numpy, Python's random module, and machine learning algorithms.

Functions:
    set_seed: Sets random seeds for numpy and Python's random module
"""

import random
import numpy as np


def set_seed(seed: int) -> None:
    """
    Set random seeds globally for reproducible experiments.

    This function sets the random seed for both numpy and Python's built-in random module
    to ensure deterministic behavior across multiple runs of the same experiment.

    Args:
        seed: Integer seed value (recommended: 42 for consistency with project configuration)

    Note:
        - This affects numpy.random and Python's random module
        - scikit-learn algorithms use separate random_state parameters (pass from config)
        - K-Means clustering requires random_state=42 parameter separately
        - Call this function at the beginning of your script/notebook for full reproducibility

    Example:
        >>> from context_aware_multi_agent_system.utils.reproducibility import set_seed
        >>> set_seed(42)
        >>> import numpy as np
        >>> result1 = np.random.rand(5)
        >>> set_seed(42)
        >>> result2 = np.random.rand(5)
        >>> assert np.allclose(result1, result2)  # Identical results

    Benefits:
        - Reproducible clustering: Same K-Means results across runs
        - Reproducible train/test splits
        - Consistent random sampling
        - Deterministic numpy array shuffling
        - Essential for academic research and experiment validation
    """
    # Set numpy random seed
    np.random.seed(seed)

    # Set Python's built-in random seed
    random.seed(seed)
