# Story 2.1: Batch Embedding Generation with Caching

Status: review

## Story

As a **data mining student**,
I want **to generate embeddings for all AG News documents with efficient caching**,
So that **I have vector representations ready for clustering without repeated API calls**.

## Acceptance Criteria

### AC-1: Batch Embedding Generation with Gemini API

**Given** the AG News dataset is loaded and the Gemini API is configured
**When** I run the embedding generation process
**Then**:
- ✅ Embeddings are generated for all training documents (120,000 samples)
- ✅ Gemini Batch API is used for cost efficiency ($0.075/1M tokens vs standard $0.15/1M)
- ✅ Batch size is configurable (default: 100 documents per API call)
- ✅ Embedding dimensions are validated (768D for gemini-embedding-001)
- ✅ Progress is logged every 1000 documents processed
- ✅ Total API usage is tracked and reported (calls, tokens, estimated cost)

### AC-2: Embedding Cache Implementation

**Given** Embeddings are generated
**When** I save embeddings to cache
**Then**:
- ✅ Embeddings cached to `data/embeddings/train_embeddings.npy` as NumPy array
- ✅ Test embeddings cached to `data/embeddings/test_embeddings.npy`
- ✅ Metadata saved to `data/embeddings/train_metadata.json` including:
  - model name (gemini-embedding-001)
  - dimensions (768)
  - num_documents
  - timestamp
  - dataset name and split
  - api_calls count
  - estimated_cost
- ✅ Embeddings are float32 dtype for memory efficiency
- ✅ Shape validation: (120000, 768) for train, (7600, 768) for test

**And when** I run the script again
**Then**:
- ✅ Cached embeddings are loaded from disk (skip API calls)
- ✅ Log message: "⚠️ Using cached embeddings from {path}"
- ✅ Metadata is validated (dimensions, model compatibility)

### AC-3: Error Handling and Resilience

**Given** Network interruptions or API errors may occur
**When** The embedding generation encounters errors
**Then**:
- ✅ Network interruptions: Resume from last checkpoint (batch-level granularity)
- ✅ API rate limits: Automatic retry with exponential backoff (4s, 8s, 16s)
- ✅ Invalid responses: Log warning, skip failed batch, continue processing
- ✅ Checkpoint file created after each successful batch (resumable)
- ✅ Maximum 3 retry attempts per batch before skipping
- ✅ Final report includes: total batches, successful, failed, skipped

### AC-4: Performance and Cost Tracking

**Given** Embedding generation is running
**When** The process completes
**Then**:
- ✅ Total execution time logged (expected: ~10-15 minutes for 120K documents)
- ✅ Cost calculation accurate: tokens × pricing ($0.075/1M for batch API)
- ✅ Final summary displayed:
  - Total documents processed
  - Total API calls made
  - Total tokens consumed
  - Estimated cost (in USD)
  - Average time per batch
  - Total execution time
- ✅ Cost is below $5 for full dataset (PRD requirement)

## Tasks / Subtasks

- [x] Create embedding orchestration script `scripts/01_generate_embeddings.py` (AC: #1, #2, #3, #4)
  - [x] Import required modules: Config, Paths, EmbeddingService, EmbeddingCache, DatasetLoader
  - [x] Implement set_seed(42) at script start for reproducibility
  - [x] Load configuration from config.yaml
  - [x] Setup logging with emoji prefixes
  - [x] Implement checkpoint system:
    - [x] Checkpoint file: `data/embeddings/.checkpoint_{split}.json`
    - [x] Store: last_processed_index, timestamp, batch_size
    - [x] Load checkpoint if exists, resume from last index
  - [x] Implement batch processing loop:
    - [x] Load AG News dataset (train and test splits)
    - [x] Check for cached embeddings, load if exists
    - [x] If cache missing, generate embeddings:
      - [x] Process documents in batches (batch_size from config)
      - [x] Call EmbeddingService.generate_batch() for each batch
      - [x] Log progress every 1000 documents
      - [x] Save checkpoint after each batch
      - [x] Track API calls, tokens, estimated cost
    - [x] Save embeddings to cache after all batches complete
    - [x] Delete checkpoint file on successful completion
  - [x] Implement error handling:
    - [x] Try/except wrapper for each batch
    - [x] Retry with exponential backoff on network errors
    - [x] Log failed batches with document indices
    - [x] Continue processing even if some batches fail
  - [x] Display final summary with all metrics

- [x] Enhance EmbeddingService.generate_batch() method in `src/context_aware_multi_agent_system/features/embedding_service.py` (AC: #1, #3)
  - [x] Accept list of texts (batch) and batch_size parameter
  - [x] Split large batches if needed (Gemini API may have limits)
  - [x] Call Gemini API with batch request: `client.models.embed_content(model=model, contents=batch)`
  - [x] Extract embeddings from batch response
  - [x] Convert to numpy array with shape (batch_size, 768) dtype float32
  - [x] Validate all embeddings have shape (768,) and dtype float32
  - [x] Use @retry decorator for resilience (from Story 1.4)
  - [x] Return numpy array (batch_size, 768) float32

- [x] Implement checkpoint management utilities (AC: #3)
  - [x] Create `save_checkpoint(split, last_index, metadata)` function
    - [x] Write to `data/embeddings/.checkpoint_{split}.json`
    - [x] Include: last_processed_index, timestamp, batch_size, total_batches
  - [x] Create `load_checkpoint(split)` function
    - [x] Read from checkpoint file if exists
    - [x] Return last_processed_index or 0 if no checkpoint
  - [x] Create `delete_checkpoint(split)` function
    - [x] Remove checkpoint file on successful completion

- [x] Add cost calculation utility to `src/context_aware_multi_agent_system/evaluation/cost_calculator.py` (AC: #4)
  - [x] Implement `estimate_embedding_cost(num_tokens, use_batch_api=True)` function
  - [x] Pricing:
    - [x] Batch API: $0.075 per 1M tokens
    - [x] Standard API: $0.15 per 1M tokens
  - [x] Token estimation: ~10 tokens per document (title + description average)
  - [x] Return cost in USD with 4 decimal places

- [x] Update config.yaml with embedding generation parameters (AC: #1)
  - [x] Add embedding.batch_size: 100
  - [x] Add embedding.cache_enabled: true
  - [x] Add embedding.use_batch_api: true
  - [x] Add embedding.checkpoint_enabled: true
  - [x] Verify embedding.model: "gemini-embedding-001"
  - [x] Verify embedding.cache_dir: "data/embeddings"

- [x] Test embedding generation pipeline (AC: #1, #2, #3, #4)
  - [x] Test full pipeline on small sample (1000 documents)
  - [x] Test cache loading (run twice, verify second run uses cache)
  - [x] Test checkpoint resume (interrupt mid-batch, verify resume)
  - [x] Test batch API cost calculation
  - [x] Test error handling (mock API failure, verify retry)
  - [x] Test final summary output (verify all metrics present)
  - [x] Validate embedding shape and dtype
  - [x] Validate metadata completeness

- [x] Update project documentation (AC: all)
  - [x] Update README.md with embedding generation instructions
  - [x] Document script usage: `python scripts/01_generate_embeddings.py`
  - [x] Document cache management (clear cache, force regeneration)
  - [x] Document checkpoint system behavior
  - [x] Add troubleshooting section for API errors
  - [x] Document expected cost and runtime

## Dev Notes

### Architecture Alignment

This story implements **FR-2** (Embedding Generation) from [PRD.md](../PRD.md) and integrates with **Epic 2** requirements from [tech-spec-epic-2.md](../tech-spec-epic-2.md).

**Batch Embedding Strategy:**
- **Batch Size**: 100 documents per API call (configurable in config.yaml)
- **API Endpoint**: Gemini Batch Embedding API (`embed_content` with content list)
- **Cost Optimization**: Batch API pricing ($0.075/1M tokens) = 50% savings vs standard API
- **Expected Cost**: ~$3-5 for 120K documents (well below $10 PRD limit)
- **Expected Time**: 10-15 minutes for full dataset (network-dependent)

**Caching Strategy:**
- **Format**: NumPy `.npy` files (compact binary format, fast I/O)
- **Metadata**: JSON sidecar files (human-readable provenance)
- **Cache Directory**: `data/embeddings/` (excluded from git via .gitignore)
- **Cache Validation**: Check shape, dtype, model compatibility before use
- **Cache Invalidation**: Manual deletion required if model changes

**Checkpoint System:**
- **Purpose**: Resume interrupted embedding generation without reprocessing
- **Granularity**: Batch-level (save after each successful batch)
- **Format**: JSON checkpoint file with last processed index
- **Cleanup**: Automatic deletion on successful completion
- **Use Case**: Network interruptions, API quota exhaustion, user cancellation

**Error Handling:**
- **Network Errors**: Retry with exponential backoff (4s, 8s, 16s), max 3 attempts
- **API Rate Limits**: Automatic backoff handled by retry decorator (from Story 1.4)
- **Invalid Responses**: Log warning, skip batch, continue processing
- **Partial Failures**: Final report shows success/failure counts per batch

**Technology Stack:**
- **google-genai** (≥0.3.0): Gemini API SDK with batch embedding support
- **numpy** (≥1.24): Embedding array storage and operations (float32 dtype)
- **tenacity** (≥8.0): Retry decorator for resilience (reused from Story 1.4)
- Follows logging patterns established in Story 1.2 and 1.4

### Testing Standards

**Embedding Generation Tests:**
```python
# Test batch embedding generation
service = EmbeddingService(config.gemini_api_key)
documents = ["doc1", "doc2", "doc3"]
embeddings = service.generate_batch(documents, batch_size=3)
assert embeddings.shape == (3, 768)
assert embeddings.dtype == np.float32

# Test cache save/load roundtrip
cache = EmbeddingCache(paths.data_embeddings)
cache.save(embeddings, "train", metadata)
loaded_embeddings, loaded_metadata = cache.load("train")
assert np.allclose(embeddings, loaded_embeddings)
```

**Checkpoint Tests:**
```python
# Test checkpoint save and resume
save_checkpoint("train", last_index=5000, metadata)
checkpoint = load_checkpoint("train")
assert checkpoint["last_processed_index"] == 5000

# Test checkpoint cleanup
delete_checkpoint("train")
assert not checkpoint_path.exists()
```

**Cost Calculation Tests:**
```python
# Test cost estimation
cost = estimate_embedding_cost(num_tokens=1_200_000, use_batch_api=True)
assert cost == 0.09  # $0.075 per 1M tokens

cost_standard = estimate_embedding_cost(num_tokens=1_200_000, use_batch_api=False)
assert cost_standard == 0.18  # $0.15 per 1M tokens
```

**Expected Test Coverage:**
- Batch embedding generation: shape, dtype, batch size handling
- Cache operations: save, load, validation, cache hit/miss
- Checkpoint system: save, load, resume, cleanup
- Error handling: network failures, retry logic, partial failures
- Cost calculation: batch API, standard API, token estimation

### Project Structure Notes

After completion, the following files will be created/modified:

**New Files:**
- `scripts/01_generate_embeddings.py` - Orchestration script for embedding generation
- `src/context_aware_multi_agent_system/evaluation/cost_calculator.py` - Cost estimation utilities
- `data/embeddings/train_embeddings.npy` - Cached train embeddings (120000, 768) float32
- `data/embeddings/test_embeddings.npy` - Cached test embeddings (7600, 768) float32
- `data/embeddings/train_metadata.json` - Train embedding metadata
- `data/embeddings/test_metadata.json` - Test embedding metadata

**Modified Files:**
- `src/context_aware_multi_agent_system/features/embedding_service.py` - Enhanced generate_batch() method
- `config.yaml` - Added embedding generation parameters
- `README.md` - Added embedding generation documentation

**Expected Directory Structure:**
```
context-aware-multi-agent-system/
├── scripts/
│   └── 01_generate_embeddings.py       # NEW: Embedding orchestration
├── data/
│   └── embeddings/                     # NEW: Embedding cache
│       ├── train_embeddings.npy        # NEW: 120K train embeddings
│       ├── test_embeddings.npy         # NEW: 7.6K test embeddings
│       ├── train_metadata.json         # NEW: Train metadata
│       ├── test_metadata.json          # NEW: Test metadata
│       └── .checkpoint_train.json      # TEMP: Checkpoint (deleted on success)
├── src/context_aware_multi_agent_system/
│   ├── features/
│   │   ├── embedding_service.py        # MODIFIED: Enhanced batch method
│   │   └── embedding_cache.py          # EXISTING: From Story 1.4
│   └── evaluation/
│       └── cost_calculator.py          # NEW: Cost estimation
└── config.yaml                         # MODIFIED: Embedding params
```

### Learnings from Previous Story

**From Story 1-4-gemini-api-integration-and-authentication (Status: done):**

- ✅ **EmbeddingService Available**: Use existing `EmbeddingService` class from Story 1.4
  - Located at: `src/context_aware_multi_agent_system/features/embedding_service.py`
  - Has `generate_embedding(text)` method for single embeddings
  - Needs enhancement: `generate_batch(texts, batch_size)` method for batch processing
  - Already includes retry logic with exponential backoff (reuse decorator)

- ✅ **EmbeddingCache Available**: Use existing `EmbeddingCache` class from Story 1.4
  - Located at: `src/context_aware_multi_agent_system/features/embedding_cache.py`
  - Has `save(embeddings, split, metadata)` method
  - Has `load(split)` method
  - Has `exists(split)` method
  - Already implements dtype validation (float32)
  - Already implements metadata injection (timestamp)

- ✅ **Config Integration**: Follow established pattern from Story 1.4
  - Use `config.get("embedding.model")` for model name
  - Use `config.get("embedding.batch_size")` for batch size
  - Use `config.gemini_api_key` property for API authentication
  - Use `paths.data_embeddings` for cache directory

- ✅ **Logging Pattern**: Follow emoji-prefixed logging from Story 1.4
  - INFO: "📊 Processing batch X/Y (documents {start}-{end})"
  - SUCCESS: "✅ Generated embeddings for {count} documents"
  - WARNING: "⚠️ Using cached embeddings from {path}"
  - WARNING: "⚠️ Batch {batch_id} failed after 3 retries, skipping"
  - ERROR: "❌ Embedding generation failed: {error_message}"

- ✅ **Retry Logic**: Reuse tenacity decorator from Story 1.4
  - Max 3 attempts per API call
  - Exponential backoff: 4s, 8s, 16s
  - Retry on network errors and rate limiting
  - Do NOT retry on ValueError (validation errors)
  - Log each retry attempt

- ✅ **Error Handling**: Follow custom exception pattern from Story 1.4
  - Create `EmbeddingGenerationError` (similar to `AuthenticationError`)
  - Include helpful error messages with troubleshooting guidance
  - Never expose API key in error messages
  - Provide context: which batch failed, what to do next

- ✅ **Type Hints and Docstrings**: Maintain standards from Story 1.4
  - All methods have full type hints
  - Google-style docstrings with usage examples
  - Example: `def generate_batch(self, documents: List[str], batch_size: int = 100) -> np.ndarray:`

- ✅ **Testing Pattern**: Follow Story 1.4 comprehensive test approach
  - Create `tests/epic2/test_batch_embedding_generation.py`
  - Map tests to acceptance criteria (AC-1, AC-2, AC-3, AC-4)
  - Use pytest.raises() for exception testing
  - Use pytest fixtures for test setup (temp cache, mock API)
  - Mock API calls to avoid actual network requests

**Files to Reuse (DO NOT RECREATE):**
- `src/context_aware_multi_agent_system/features/embedding_service.py` - Enhance with batch method
- `src/context_aware_multi_agent_system/features/embedding_cache.py` - Use as-is for caching
- `src/context_aware_multi_agent_system/config.py` - Add new config parameters
- `src/context_aware_multi_agent_system/utils/logger.py` - Use for logging
- `src/context_aware_multi_agent_system/utils/reproducibility.py` - Use set_seed(42)

**Key Services from Previous Stories:**
- **EmbeddingService** (Story 1.4): Gemini API wrapper with retry logic
- **EmbeddingCache** (Story 1.4): Cache save/load with validation
- **Config class** (Story 1.2): Configuration management
- **Paths class** (Story 1.2): Path resolution
- **DatasetLoader** (Story 1.3): AG News dataset loading

**Technical Debt from Story 1.4:**
- None affecting this story - Story 1.4 is complete and approved

**Review Findings from Story 1.4 to Apply:**
- ✅ Use comprehensive docstrings with usage examples
- ✅ Add type hints to all method signatures
- ✅ Include explicit validation checks with informative error messages
- ✅ Log all major operations for debugging
- ✅ Write tests covering all acceptance criteria
- ✅ Mock external API calls in tests

[Source: stories/1-4-gemini-api-integration-and-authentication.md#Dev-Agent-Record]

### References

- [Source: docs/tech-spec-epic-2.md#Story 2.1 - Batch Embedding Generation]
- [Source: docs/epics.md#Story 2.1 - Batch Embedding Generation with Caching]
- [Source: docs/PRD.md#FR-2 - Embedding Generation]
- [Source: docs/architecture.md#Embedding Service - Google Gemini Embedding API]
- [Source: stories/1-4-gemini-api-integration-and-authentication.md#EmbeddingService and EmbeddingCache]

## Dev Agent Record

### Context Reference

- [2-1-batch-embedding-generation-with-caching.context.xml](2-1-batch-embedding-generation-with-caching.context.xml) - Generated 2025-11-09

### Agent Model Used

claude-sonnet-4-5-20250929

### Debug Log References

- Successfully implemented batch embedding generation using Gemini Batch API
- All 99 tests passing (82 from Epic 1, 17 from Epic 2)
- Fixed 3 legacy tests to match new Batch API behavior

### Completion Notes List

- Implemented complete batch embedding generation system with checkpoint support and cost tracking
- Enhanced EmbeddingService.generate_batch() to use true Gemini Batch API (saves 50% cost)
- Created comprehensive test suite with 17 new tests covering all acceptance criteria
- Updated README with detailed usage instructions, cache management, and troubleshooting guide
- All acceptance criteria met: batch processing, caching, error handling, cost tracking

###File List

**New Files:**
- scripts/01_generate_embeddings.py
- src/context_aware_multi_agent_system/evaluation/cost_calculator.py
- tests/epic2/__init__.py
- tests/epic2/test_batch_embedding_generation.py

**Modified Files:**
- src/context_aware_multi_agent_system/features/embedding_service.py
- config.yaml
- README.md
- tests/epic1/test_embedding_service.py
