# Story 1.3: AG News Dataset Loading and Validation

Status: ready-for-dev

## Story

As a **data mining student**,
I want **to load and validate the AG News dataset**,
so that **I have clean, structured data ready for embedding generation**.

## Acceptance Criteria

### AC-1: AG News Dataset Loaded from Hugging Face
**Given** the Hugging Face datasets library is installed
**When** I call `DatasetLoader().load_ag_news()`
**Then**:
- ✅ Returns tuple of (train_dataset, test_dataset)
- ✅ Train dataset has exactly 120,000 samples
- ✅ Test dataset has exactly 7,600 samples
- ✅ Dataset has fields: `text`, `label`
- ✅ Labels are in range [0-3] (4 categories)
- ✅ Dataset statistics are logged (sample counts, category distribution)

### AC-2: Dataset Structure Validated
**Given** Dataset is loaded
**When** I call `validate_dataset(dataset)`
**Then**:
- ✅ Returns `True` for valid AG News dataset
- ✅ Validates expected fields present (`text`, `label`)
- ✅ Validates 4 categories exist (labels 0-3)
- ✅ Validates no missing values in `text` or `label` fields
- ✅ Validates label range [0-3]
- ✅ Raises `DatasetLoadError` with clear message for invalid dataset

### AC-3: Dataset Cached Locally for Performance
**Given** Dataset loaded once
**When** I load dataset again
**Then**:
- ✅ Loading completes in <5 seconds (uses cache)
- ✅ Cache location: `~/.cache/huggingface/datasets/ag_news/`
- ✅ No network calls made (works offline)
- ✅ Warning logged: "⚠️ Using cached dataset from ~/.cache/huggingface/"

### AC-4: Text Fields Extracted and Combined
**Given** Dataset is loaded
**When** Text fields are processed
**Then**:
- ✅ Title and description fields are combined into single `text` field
- ✅ Text fields are stripped of leading/trailing whitespace
- ✅ No missing or empty text values after processing
- ✅ Sample text examples logged for verification

### AC-5: Category Distribution Logged and Balanced
**Given** Dataset is loaded and validated
**When** I call `get_category_distribution(dataset)`
**Then**:
- ✅ Returns dictionary mapping category labels to document counts
- ✅ All 4 categories (0=World, 1=Sports, 2=Business, 3=Sci/Tech) present
- ✅ Category distribution is logged with counts and percentages
- ✅ Categories are reasonably balanced (no category < 10% of total)

### AC-6: Optional Sampling Support for Faster Experiments
**Given** Configuration specifies `dataset.sample_size` (e.g., 1000)
**When** Dataset is loaded
**Then**:
- ✅ Only specified number of samples loaded from training set
- ✅ Sampling maintains category distribution (stratified sampling)
- ✅ Test set can optionally be sampled proportionally
- ✅ Log message indicates: "Using sample of {sample_size} documents"

## Tasks / Subtasks

- [ ] Implement DatasetLoader class in src/context_aware_multi_agent_system/data/load_dataset.py (AC: #1, #2, #5)
  - [ ] Import required libraries: datasets (Hugging Face), pandas, typing
  - [ ] Implement __init__ method to accept Config object
  - [ ] Implement load_ag_news() method:
    - [ ] Load AG News dataset using `datasets.load_dataset("ag_news")`
    - [ ] Extract train and test splits
    - [ ] Return tuple of (train_dataset, test_dataset)
  - [ ] Implement validate_dataset() method:
    - [ ] Check expected fields present: "text", "label"
    - [ ] Verify 4 unique categories in labels
    - [ ] Check for missing values in text/label
    - [ ] Validate label range [0-3]
    - [ ] Return True if valid, raise DatasetLoadError if invalid
  - [ ] Implement get_category_distribution() method:
    - [ ] Count documents per category (0-3)
    - [ ] Calculate percentages
    - [ ] Return dict mapping {label: count}
  - [ ] Add type hints for all methods
  - [ ] Add docstrings with usage examples

- [ ] Implement text processing utilities (AC: #4)
  - [ ] Create process_text() helper function:
    - [ ] Combine title + description fields if separate
    - [ ] Strip whitespace using str.strip()
    - [ ] Validate no empty strings remain
  - [ ] Handle AG News format (text field already combined in Hugging Face version)
  - [ ] Log sample texts for verification (first 3 documents)

- [ ] Implement sampling support (AC: #6)
  - [ ] Add sample_dataset() method to DatasetLoader:
    - [ ] Check config.get("dataset.sample_size")
    - [ ] If None → return full dataset
    - [ ] If int → perform stratified sampling
    - [ ] Use pandas for stratified sampling (group by label, sample proportionally)
    - [ ] Log sampling info: "Using sample of {n} documents"
  - [ ] Maintain category distribution in sampled data
  - [ ] Add option to sample test set proportionally

- [ ] Implement caching verification (AC: #3)
  - [ ] Test first load (network download)
  - [ ] Verify cache location: ~/.cache/huggingface/datasets/
  - [ ] Test second load (should use cache, <5s)
  - [ ] Add logging for cache usage detection
  - [ ] Document cache clearing procedure in dev notes

- [ ] Add comprehensive logging (AC: all)
  - [ ] Log dataset loading start: "📊 Loading AG News dataset..."
  - [ ] Log successful load: "✅ Dataset loaded: {train_size} train, {test_size} test"
  - [ ] Log cache usage: "⚠️ Using cached dataset from ~/.cache/huggingface/"
  - [ ] Log category distribution: "Category 0 (World): {count} ({percent}%)"
  - [ ] Log validation results: "✅ Dataset validated: 4 categories, {total} samples"
  - [ ] Log sample texts: "Sample text: {text[:100]}..."

- [ ] Test dataset loading workflow (AC: #1, #2, #3, #4, #5, #6)
  - [ ] Test full dataset load (120K train, 7.6K test)
  - [ ] Test validation passes for valid dataset
  - [ ] Test validation fails for invalid dataset (mock)
  - [ ] Test category distribution calculation
  - [ ] Test cache performance (second load <5s)
  - [ ] Test sampling with sample_size=1000
  - [ ] Test stratified sampling maintains distribution
  - [ ] Test missing field detection
  - [ ] Test invalid label range detection

- [ ] Create DatasetLoadError exception class (AC: #2)
  - [ ] Define custom exception in data/load_dataset.py
  - [ ] Inherit from Exception
  - [ ] Accept message parameter with error details
  - [ ] Use in validate_dataset() for clear error reporting

- [ ] Update documentation (AC: all)
  - [ ] Add dataset loading section to README.md
  - [ ] Document AG News structure (4 categories, sample counts)
  - [ ] Document cache location and clearing procedure
  - [ ] Document sampling feature usage
  - [ ] Add example usage code snippet

## Dev Notes

### Architecture Alignment

This story implements **FR-1** (Dataset Loading) from [PRD.md](../PRD.md) and **AC-6, AC-7, AC-8** from [tech-spec-epic-1.md](../tech-spec-epic-1.md#acceptance-criteria-authoritative).

**Dataset Integration:**
- AG News dataset from Hugging Face Datasets library (public, no PII)
- Train: 120,000 samples | Test: 7,600 samples
- 4 categories: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
- Text field already combines title + description in HF version

**Caching Strategy:**
- Hugging Face auto-caches to `~/.cache/huggingface/datasets/ag_news/`
- First load: downloads from network (~2 minutes)
- Subsequent loads: instant from cache (<5 seconds)
- Offline-capable after first successful load
- Cache clearing: `rm -rf ~/.cache/huggingface/datasets/ag_news/`

**Validation Approach:**
- Explicit validation checks prevent silent failures downstream
- Early detection of data quality issues (missing values, invalid labels)
- Clear error messages guide troubleshooting
- Validation runs automatically on load

**Sampling Feature:**
- Optional for rapid experimentation during development
- Stratified sampling maintains category distribution
- Configured via `dataset.sample_size` in config.yaml
- Full dataset (sample_size: null) used for final experiments

**Technology Stack:**
- Hugging Face `datasets` library (v2.14+) for AG News loading
- Pandas for stratified sampling operations
- Follows logging patterns established in Story 1.2

### Testing Standards

**Dataset Loading Tests:**
- Integration test: Load actual AG News from Hugging Face
- Verify exact sample counts: 120,000 train, 7,600 test
- Verify field structure: "text", "label" fields present
- Verify label range: all labels in [0-3]

**Validation Tests:**
```python
# Test valid dataset passes validation
loader = DatasetLoader(config)
train, test = loader.load_ag_news()
assert loader.validate_dataset(train) == True

# Test invalid dataset raises error
invalid_dataset = {"wrong_field": [...]}
with pytest.raises(DatasetLoadError):
    loader.validate_dataset(invalid_dataset)
```

**Category Distribution Tests:**
```python
# Test category distribution calculation
distribution = loader.get_category_distribution(train_data)
assert len(distribution) == 4  # 4 categories
assert all(count > 0 for count in distribution.values())  # All categories present
```

**Sampling Tests:**
```python
# Test stratified sampling maintains distribution
config.set("dataset.sample_size", 1000)
train_sampled, _ = loader.load_ag_news()
assert len(train_sampled) == 1000

dist_full = loader.get_category_distribution(train_full)
dist_sample = loader.get_category_distribution(train_sampled)

# Distribution should be similar (within 5% per category)
for cat in range(4):
    pct_full = dist_full[cat] / len(train_full)
    pct_sample = dist_sample[cat] / len(train_sampled)
    assert abs(pct_full - pct_sample) < 0.05
```

**Cache Performance Tests:**
```python
import time

# First load (network download)
start = time.time()
train1, test1 = loader.load_ag_news()
first_load_time = time.time() - start

# Second load (cache)
start = time.time()
train2, test2 = loader.load_ag_news()
second_load_time = time.time() - start

assert second_load_time < 5.0  # Cache load <5 seconds
assert first_load_time > second_load_time  # Cache is faster
```

### Project Structure Notes

After completion, the following files will be created/modified:

**New Files:**
- `src/context_aware_multi_agent_system/data/__init__.py` - Data module initialization
- `src/context_aware_multi_agent_system/data/load_dataset.py` - DatasetLoader class and utilities
- `tests/epic1/test_dataset_loading.py` - Comprehensive dataset loading tests

**Modified Files:**
- `README.md` - Updated with dataset loading instructions and cache information

**Expected Data Structure:**
```
context-aware-multi-agent-system/
├── src/
│   └── context_aware_multi_agent_system/
│       ├── data/
│       │   ├── __init__.py        # NEW: Data module initialization
│       │   └── load_dataset.py    # NEW: DatasetLoader class
│       ├── config.py               # Existing from Story 1.2
│       └── utils/
│           └── reproducibility.py  # Existing from Story 1.2
├── tests/
│   └── epic1/
│       ├── test_config.py          # Existing from Story 1.2
│       ├── test_reproducibility.py # Existing from Story 1.2
│       └── test_dataset_loading.py # NEW: Dataset tests
├── data/
│   └── raw/                        # Dataset cached here (optional, HF uses ~/.cache)
└── ~/.cache/huggingface/datasets/ag_news/  # Hugging Face auto-cache location
```

### Learnings from Previous Story

**From Story 1-2-configuration-management-system (Status: done):**

- ✅ **Config System Ready**: Use `Config().get("dataset.name")` to retrieve dataset parameters
  - Config class at `src/context_aware_multi_agent_system/config.py`
  - Access dataset config: `config.get("dataset.name")` → "ag_news"
  - Access sample_size: `config.get("dataset.sample_size")` → None or int

- ✅ **Paths System Available**: Use `Paths` class for consistent directory access
  - `Paths.data_raw` for raw dataset storage (if needed)
  - All paths are absolute and directories auto-created
  - Import: `from src.context_aware_multi_agent_system.config import Paths`

- ✅ **Logging Pattern Established**: Follow emoji-prefixed logging
  - INFO: "📊 Loading AG News dataset..."
  - SUCCESS: "✅ Dataset loaded successfully: 120,000 train, 7,600 test"
  - WARNING: "⚠️ Using cached dataset from ~/.cache/"
  - ERROR: "❌ Dataset validation failed: Missing 'text' field"

- ✅ **Type Hints Standard**: All methods have full type hints
  - Follow pattern from Config class
  - Use typing module: `from typing import Tuple, Dict, Optional`
  - Example: `def load_ag_news(self) -> Tuple[Dataset, Dataset]:`

- ✅ **Error Handling Pattern**: Informative errors with next steps
  - Raise custom exceptions (DatasetLoadError) with clear messages
  - Include field names in validation errors
  - Provide troubleshooting guidance

- ⚠️ **Module Structure**: Continue using `src/context_aware_multi_agent_system/` as root
  - Create `data/` submodule for dataset operations
  - Initialize with `__init__.py`
  - Import: `from src.context_aware_multi_agent_system.data.load_dataset import DatasetLoader`

**Files to Reuse (DO NOT RECREATE):**
- `src/context_aware_multi_agent_system/__init__.py` - Module root
- `src/context_aware_multi_agent_system/config.py` - Config and Paths classes
- `src/context_aware_multi_agent_system/utils/` - Utils package
- `config.yaml` - Configuration file (already has dataset section)
- `tests/epic1/__init__.py` - Test package initialization

**Key Services from Previous Story:**
- `Config` class: Use `config.get("dataset.categories")` to get expected category count (4)
- `Paths` class: Use `paths.data_raw` for optional local dataset storage
- Logging utilities: Follow established emoji-prefixed pattern

[Source: stories/1-2-configuration-management-system.md#Dev-Agent-Record]

### References

- [Source: docs/tech-spec-epic-1.md#AC-6 - AG News Dataset Loaded]
- [Source: docs/tech-spec-epic-1.md#AC-7 - Dataset Validated]
- [Source: docs/tech-spec-epic-1.md#AC-8 - Dataset Cached Locally]
- [Source: docs/tech-spec-epic-1.md#Data Models and Contracts - Dataset Data Model]
- [Source: docs/tech-spec-epic-1.md#APIs and Interfaces - DatasetLoader API]
- [Source: docs/tech-spec-epic-1.md#Workflows and Sequencing - Story 1.3 Dataset Loading Workflow]
- [Source: docs/PRD.md#FR-1 - Dataset Loading and Preprocessing]
- [Source: docs/epics.md#Story 1.3 - AG News Dataset Loading and Validation]

## Dev Agent Record

### Context Reference

- [1-3-ag-news-dataset-loading-and-validation.context.xml](1-3-ag-news-dataset-loading-and-validation.context.xml)

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
