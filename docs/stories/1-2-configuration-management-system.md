# Story 1.2: Configuration Management System

Status: review

## Story

As a **data mining student**,
I want **a centralized configuration system for all experimental parameters**,
so that **I can easily adjust clustering settings and reproduce experiments**.

## Acceptance Criteria

### AC-1: Configuration File Created with All Parameters
**Given** the project structure exists from Story 1.1
**When** I create the configuration file
**Then** `config.yaml` exists in project root with the following sections:
- ✅ **Dataset configuration**: name, categories, sample_size
- ✅ **Clustering configuration**: algorithm, n_clusters=4, random_state=42, max_iter=300, init='k-means++'
- ✅ **Embedding configuration**: model='gemini-embedding-001', batch_size=100, cache_dir, output_dimensionality=768
- ✅ **Classification configuration**: method='cosine_similarity', threshold=0.7
- ✅ **Metrics configuration**: cost_per_1M_tokens_under_200k=3.0, cost_per_1M_tokens_over_200k=6.0, target_cost_reduction=0.90

### AC-2: Config Class Loads and Validates Configuration
**Given** `config.yaml` exists with valid parameters
**When** I initialize the Config class
**Then**:
- ✅ `Config()` initializes without errors
- ✅ `config.get("clustering.n_clusters")` returns `4`
- ✅ `config.get("embedding.model")` returns `"gemini-embedding-001"`
- ✅ `config.get("embedding.output_dimensionality")` returns `768`
- ✅ `config.gemini_api_key` returns API key from `.env` (or raises clear error if missing)
- ✅ `config.validate()` returns `True` for valid config
- ✅ All sections accessible: `dataset`, `clustering`, `embedding`, `classification`, `metrics`

### AC-3: Invalid Configuration Raises Informative Errors
**Given** `config.yaml` has invalid or missing values
**When** I load the configuration
**Then**:
- ✅ Missing required fields raise `ValueError` with specific field name
- ✅ Invalid data types raise `TypeError` with expected type information
- ✅ Error messages guide user to fix configuration
- ✅ API key missing from `.env` raises clear error: "GEMINI_API_KEY not found. Copy .env.example to .env and add your API key."

### AC-4: Paths System Creates and Manages All Project Directories
**Given** configuration is loaded
**When** I initialize the Paths class
**Then**:
- ✅ All path attributes exist: `data`, `data_raw`, `data_embeddings`, `data_interim`, `data_processed`, `models`, `notebooks`, `reports`, `reports_figures`, `results`
- ✅ Directories are created if missing (mkdir -p behavior with `parents=True, exist_ok=True`)
- ✅ All paths are absolute (not relative)
- ✅ Path operations use `pathlib.Path` for cross-platform compatibility

### AC-5: Reproducibility Utility Sets Random Seeds
**Given** I call `set_seed(42)`
**When** I run random operations
**Then**:
- ✅ `numpy.random.rand()` produces identical results across runs
- ✅ `random.random()` produces identical results across runs
- ✅ Subsequent operations using numpy/random are deterministic
- ✅ K-Means with `random_state=42` will produce identical clusters (validated in Epic 2)

## Tasks / Subtasks

- [x] Create config.yaml in project root (AC: #1)
  - [x] Add dataset configuration section (name: "ag_news", categories: 4, sample_size: null)
  - [x] Add clustering configuration section (algorithm: "kmeans", n_clusters: 4, random_state: 42, max_iter: 300, init: "k-means++")
  - [x] Add embedding configuration section (model: "gemini-embedding-001", batch_size: 100, cache_dir: "data/embeddings", output_dimensionality: 768)
  - [x] Add classification configuration section (method: "cosine_similarity", threshold: 0.7)
  - [x] Add metrics configuration section (cost_per_1M_tokens_under_200k: 3.0, cost_per_1M_tokens_over_200k: 6.0, target_cost_reduction: 0.90)
  - [x] Add inline comments explaining each parameter
  - [x] Validate YAML syntax

- [x] Implement Config class in src/context_aware_multi_agent_system/config.py (AC: #2, #3)
  - [x] Import required libraries: PyYAML, python-dotenv, pathlib, typing
  - [x] Implement __init__ method to load config.yaml using PyYAML
  - [x] Load environment variables using python-dotenv (load_dotenv())
  - [x] Implement get() method with dot notation support (e.g., "clustering.n_clusters")
  - [x] Implement validate() method checking all required fields and data types
  - [x] Implement gemini_api_key property to retrieve GEMINI_API_KEY from environment
  - [x] Add error handling for missing config file
  - [x] Add error handling for invalid YAML syntax with helpful messages
  - [x] Add error handling for missing required fields with specific field names
  - [x] Add error handling for missing API key with setup instructions
  - [x] Add type hints for all methods

- [x] Implement Paths class in src/context_aware_multi_agent_system/config.py (AC: #4)
  - [x] Import pathlib.Path
  - [x] Define project_root as absolute path
  - [x] Define all directory paths using Path objects:
    - data, data_raw, data_embeddings, data_interim, data_processed
    - models, notebooks, reports, reports_figures, results, src
  - [x] Implement __init__ or __post_init__ to create directories if missing
  - [x] Use Path.mkdir(parents=True, exist_ok=True) for safe directory creation
  - [x] Ensure all paths are absolute (Path.resolve())
  - [x] Add __repr__ method for debugging (optional)

- [x] Implement set_seed() utility in src/context_aware_multi_agent_system/utils/reproducibility.py (AC: #5)
  - [x] Import numpy, random
  - [x] Create set_seed(seed: int) function
  - [x] Set numpy random seed: np.random.seed(seed)
  - [x] Set Python random seed: random.seed(seed)
  - [x] Document that scikit-learn uses random_state parameter separately
  - [x] Add docstring explaining reproducibility benefits
  - [x] Add type hints

- [x] Test configuration system (AC: #2, #3, #4, #5)
  - [x] Test Config loads without errors
  - [x] Test config.get() with dot notation for all sections
  - [x] Test config.gemini_api_key retrieves from .env
  - [x] Test config.validate() passes for valid config
  - [x] Test missing config.yaml raises clear error
  - [x] Test invalid YAML syntax raises clear error
  - [x] Test missing required field raises clear error
  - [x] Test missing API key raises clear error with instructions
  - [x] Test Paths creates all directories
  - [x] Test all Paths attributes are absolute
  - [x] Test set_seed(42) produces identical numpy.random results
  - [x] Test set_seed(42) produces identical random.random results

- [x] Update documentation (AC: all)
  - [x] Add config.yaml usage to README.md
  - [x] Document configuration parameters in inline comments
  - [x] Update .env.example if needed (already has GEMINI_API_KEY)
  - [x] Add reproducibility information to README.md

## Dev Notes

### Architecture Alignment

This story implements **ADR-004** (Reproducibility) and **ADR-006** (Separate Configuration and Secrets) from [architecture.md](../architecture.md).

**Configuration Strategy:**
- `config.yaml` (committed to git) contains all experimental parameters
- `.env` (excluded from git) contains secrets (GEMINI_API_KEY)
- Environment variables override YAML values where applicable
- PyYAML for YAML parsing, python-dotenv for environment variable loading

**Paths Management:**
- Centralized Paths class ensures consistent directory access across all modules
- Uses pathlib.Path for cross-platform compatibility (Windows, macOS, Linux)
- Automatic directory creation prevents file I/O errors
- All paths are absolute to avoid working directory issues

**Reproducibility:**
- Fixed random seed (random_state=42) ensures identical clustering results
- set_seed() utility centralizes seed management for numpy and Python's random
- scikit-learn algorithms use random_state parameter separately (passed from config)

**Technology Stack:**
- PyYAML>=6.0 for config.yaml parsing (already installed in Story 1.1)
- python-dotenv>=1.0.0 for .env management (already installed in Story 1.1)
- pathlib (Python standard library, no install needed)

### Testing Standards

**Configuration Testing:**
- Manual validation: Load config.yaml, verify all sections present
- Automated testing: Test config.get() for each parameter
- Error scenario testing: Missing file, invalid syntax, missing fields
- API key testing: Verify .env loading, test missing key error message

**Paths Testing:**
- Directory creation: Verify all directories created on first run
- Path validation: Check all paths are absolute (Path.is_absolute())
- Cross-platform: Test on macOS (primary), consider Windows/Linux compatibility

**Reproducibility Testing:**
```python
# Test deterministic behavior
import numpy as np
from src.context_aware_multi_agent_system.utils.reproducibility import set_seed

set_seed(42)
result1 = np.random.rand(10)

set_seed(42)
result2 = np.random.rand(10)

assert np.allclose(result1, result2)  # Must be identical
```

### Project Structure Notes

After completion, the following files will be created:

**New Files:**
- `config.yaml` - Configuration file in project root (committed)
- `src/context_aware_multi_agent_system/config.py` - Config and Paths classes
- `src/context_aware_multi_agent_system/utils/reproducibility.py` - set_seed utility

**Modified Files:**
- `README.md` - Updated with configuration usage instructions

**Expected Project Structure:**
```
context-aware-multi-agent-system/
├── config.yaml                    # NEW: Centralized configuration
├── .env                           # Existing: Contains GEMINI_API_KEY
├── .env.example                   # Existing: Template from Story 1.1
├── src/
│   └── context_aware_multi_agent_system/
│       ├── config.py              # NEW: Config and Paths classes
│       └── utils/
│           ├── __init__.py        # Existing from Story 1.1
│           └── reproducibility.py # NEW: set_seed utility
├── data/                          # Existing from Story 1.1
│   ├── raw/
│   ├── embeddings/
│   ├── interim/
│   └── processed/
├── models/, notebooks/, reports/, results/  # All existing
└── ...
```

### Learnings from Previous Story

**From Story 1-1 (Status: done):**

- ✅ **Project Structure Established**: Use `src/context_aware_multi_agent_system/` as module root
  - All submodules initialized with __init__.py files
  - Virtual environment active with Python 3.12.7

- ✅ **Dependencies Installed**: All required packages available
  - PyYAML>=6.0 installed and ready for config.yaml parsing
  - python-dotenv>=1.0.0 installed for .env management
  - Correct import syntax: `from google import genai` (not `import google.genai`)

- ✅ **.env Pattern Established**:
  - .env file excluded from git via .gitignore
  - .env.example committed as template with GEMINI_API_KEY placeholder
  - Use this pattern: Load API key from environment, never hardcode

- ✅ **Security Patterns Configured**:
  - .gitignore includes .env, data/, *.pyc, __pycache__/, .ipynb_checkpoints/
  - Ensure config.yaml is committed but .env remains excluded

- ⚠️ **Use Absolute Paths**: Story 1.1 established directory structure; use pathlib.Path.resolve() for absolute paths

- ⚠️ **Code Review Finding**: Import statement corrected to `from google import genai`
  - This is the official SDK syntax, use in all future code

**Files to Reuse (DO NOT RECREATE):**
- `src/context_aware_multi_agent_system/__init__.py` - Module initialization exists
- `src/context_aware_multi_agent_system/utils/__init__.py` - Utils package initialized
- `.env.example` - Template already has GEMINI_API_KEY
- `.gitignore` - Security patterns already configured

**New Services Created in This Story:**
- `Config` class at `src/context_aware_multi_agent_system/config.py` - Use for all configuration access
- `Paths` class at `src/context_aware_multi_agent_system/config.py` - Use for all file paths
- `set_seed()` at `src/context_aware_multi_agent_system/utils/reproducibility.py` - Call at script startup

[Source: stories/1-1-project-initialization-and-environment-setup.md#Dev-Agent-Record]

### References

- [Source: docs/tech-spec-epic-1.md#AC-3 - Configuration System Functional]
- [Source: docs/tech-spec-epic-1.md#AC-4 - Paths System Functional]
- [Source: docs/tech-spec-epic-1.md#AC-5 - Reproducibility Utility Functional]
- [Source: docs/tech-spec-epic-1.md#Detailed Design - Services and Modules]
- [Source: docs/tech-spec-epic-1.md#Data Models and Contracts - Configuration Schema]
- [Source: docs/tech-spec-epic-1.md#APIs and Interfaces - Config API]
- [Source: docs/architecture.md#ADR-004 - Reproducibility via Fixed Seeds]
- [Source: docs/architecture.md#ADR-006 - Separate Configuration and Secrets]
- [Source: docs/epics.md#Story 1.2 - Configuration Management System]

## Dev Agent Record

### Context Reference

- [Story Context](1-2-configuration-management-system.context.xml) - Generated 2025-11-09

### Agent Model Used

claude-sonnet-4-5-20250929

### Debug Log References

**Implementation Plan:**
1. Created config.yaml with 5 configuration sections (dataset, clustering, embedding, classification, metrics)
2. Implemented Config class with YAML loading, dot notation access, validation, and API key retrieval
3. Implemented Paths class with automatic directory creation and absolute path management
4. Implemented set_seed() utility for reproducibility
5. Wrote comprehensive test suite (24 tests covering all acceptance criteria)
6. Updated README.md with configuration usage and reproducibility documentation

**Technical Decisions:**
- Used PyYAML for config parsing and python-dotenv for environment variable management
- Implemented dot notation access (e.g., config.get("clustering.n_clusters")) for clean API
- All paths use pathlib.Path for cross-platform compatibility
- Comprehensive error messages guide users to fix configuration issues
- Type hints on all methods for better IDE support

### Completion Notes List

✅ **All Acceptance Criteria Met:**
- AC-1: config.yaml created with all 5 sections and inline documentation
- AC-2: Config class loads, validates, and provides dot notation access to all parameters
- AC-3: Comprehensive error handling with informative messages for all error scenarios
- AC-4: Paths class manages all project directories with automatic creation and absolute paths
- AC-5: set_seed() utility ensures reproducibility for numpy and Python random operations

✅ **Test Results:** 24/24 tests passed
- 11 Config class tests (loading, validation, error handling, API key)
- 5 Paths class tests (directory creation, absolute paths, pathlib usage)
- 8 set_seed() tests (numpy/random reproducibility, K-Means validation)

✅ **Code Quality:**
- Full type hints on all functions and methods
- Comprehensive docstrings with examples
- Cross-platform path handling with pathlib.Path
- Security: API keys loaded from .env, never hardcoded or logged

### File List

**New Files:**
- config.yaml (project root)
- src/context_aware_multi_agent_system/config.py
- src/context_aware_multi_agent_system/utils/reproducibility.py
- tests/epic1/__init__.py
- tests/epic1/test_config.py
- tests/epic1/test_reproducibility.py

**Modified Files:**
- README.md (added Configuration and Reproducibility sections)

## Change Log

**2025-11-09** - v2.0 - Story implemented and ready for review
- ✅ Implemented all tasks and subtasks
- ✅ Created config.yaml with 5 configuration sections
- ✅ Implemented Config class with validation and dot notation access
- ✅ Implemented Paths class with automatic directory creation
- ✅ Implemented set_seed() reproducibility utility
- ✅ Wrote comprehensive test suite (24 tests, 100% passing)
- ✅ Updated README.md with configuration and reproducibility documentation
- ✅ All acceptance criteria validated and satisfied
- Status: review (implementation complete, awaiting code review)

**2025-11-09** - v1.0 - Story drafted
- Initial story creation by Scrum Master (Bob)
- All acceptance criteria extracted from Tech Spec Epic 1
- Tasks decomposed from requirements
- Previous story learnings incorporated
- Status: drafted (ready for story-context or story-ready workflow)
