# Product Brief: Context-Aware Agent Routing for LLM Cost Optimization

**Date:** 2025-11-08
**Author:** Jack YUAN
**Context:** Academic Course Project (Data Mining)

---

## Executive Summary

This project demonstrates how clustering and classification techniques can dramatically reduce Large Language Model (LLM) API costs when processing long-context conversations. By intelligently routing user queries to specialized agents that maintain only relevant context subsets, the system aims to achieve 90%+ cost reduction while maintaining response quality.

The approach addresses a critical inefficiency in current LLM usage: when context length exceeds 200K tokens, API costs double (from $3/1M to $6/1M tokens for Claude Sonnet 4.5), yet most user queries only need a small fraction of that context. This project applies data mining techniques—specifically clustering and classification—to solve this cost optimization problem.

**Key Innovation:** Segment long conversations into semantic clusters, route new queries only to the relevant context subset, eliminating wasteful token consumption.

**Target Outcome:** Validate 90%+ LLM API cost reduction through experimental demonstration using AG News dataset.

---

## Core Vision

### Problem Statement

Current LLM-based conversational systems suffer from a critical cost inefficiency: **context length penalty**.

**The Problem:**
- Long conversations accumulate context that grows to 200K+ tokens
- Every new user input sends the ENTIRE context to the LLM, even if the question only relates to a small portion
- Cost scaling is non-linear: >200K tokens = 2x price increase (e.g., Claude Sonnet 4.5: $3/1M → $6/1M tokens)
- Most queries (~90%) only need 10-20% of the full context to generate accurate responses

**Real-World Impact:**
- A 300K token conversation costs $1.80 per query (vs $0.18 if only relevant 30K subset used)
- 10x cost multiplier due to irrelevant context transmission
- Compounds rapidly: 100 queries = $180 vs $18 with intelligent routing

**Root Cause:**
LLMs process context holistically without semantic segmentation—they don't distinguish between "relevant" and "irrelevant" context for a given query. Current systems lack a mechanism to identify and route queries to contextually appropriate subsets.

### Proposed Solution

A **multi-agent system with clustering-based context partitioning and classification-based query routing**.

**Two-Phase Approach:**

**Phase 1 - Initialization (Offline):**
1. Take a long-context document/conversation
2. Generate semantic embeddings using Google Gemini Embedding API
3. Apply K-Means clustering to partition content into K semantic categories
4. Assign each cluster to a specialized agent with ONLY that context subset
5. Store cluster centroids for runtime classification

**Phase 2 - Runtime (Online):**
1. User submits a new query
2. Generate query embedding
3. Classify query to nearest cluster using cosine similarity
4. Route to the specialized agent for that cluster
5. Agent processes query with reduced context (1/K of original size)

**Key Mechanism:**
- **Clustering** ensures semantic coherence within each agent's context
- **Classification** enables fast (<1 second) routing decisions
- **Agent specialization** eliminates irrelevant context from API calls

**Expected Outcome:**
- 90-98% reduction in LLM API calls to full context
- Each agent operates on ~1/K context size
- Maintains accuracy by preserving semantic relevance

### Key Differentiators

This approach is distinguished by:

1. **Data Mining Focus:** Explicitly demonstrates clustering (K-Means) and classification (cosine similarity) techniques required for course
2. **Simplicity Over Complexity:** No heavyweight agent frameworks—clean Python implementation easy to explain and replicate
3. **Cost-Driven Design:** Optimizes for the most expensive resource (LLM tokens), not just performance
4. **Practical Validation:** Uses real benchmark dataset (AG News) to prove concept with measurable metrics
5. **Rapid Prototyping:** 3-day implementation timeline achievable with proven components

**Compared to alternatives:**
- **vs. RAG (Retrieval-Augmented Generation):** Proactive clustering vs reactive retrieval; lower latency
- **vs. Prompt Compression:** Semantic routing vs lossy compression; maintains full context fidelity
- **vs. Context Pruning:** Systematic clustering vs heuristic pruning; better semantic preservation

---

## Target Users

### Primary Users

**Academic Evaluators (Course Instructors)**
- **Situation:** Assessing student understanding of clustering and classification algorithms in practical applications
- **Current Workflow:** Review project reports demonstrating data mining techniques
- **Specific Needs:**
  - Clear demonstration of K-Means clustering application
  - Measurable classification performance metrics
  - Reproducible experimental results
  - Connection to real-world problem (LLM cost optimization)
- **Success Criteria:** Project clearly shows mastery of course concepts through working implementation

**Secondary Audience: Developers exploring LLM cost optimization**
- Though this is a course project, the concept addresses real challenges faced by developers building LLM-powered applications with long conversation histories

---

## MVP Scope

### Core Features

The minimal viable demonstration must include:

1. **Document Embedding Generation**
   - Load AG News dataset (4-category news classification)
   - Generate embeddings using Google Gemini Embedding API
   - Store embeddings for clustering

2. **K-Means Clustering Implementation**
   - Apply K-Means (K=4) to partition documents into semantic clusters
   - Evaluate cluster quality using Silhouette Score
   - Visualize clusters (PCA 2D projection)

3. **Specialized Agent Creation**
   - Create 4 agents, each holding documents from one cluster
   - Implement simple agent class with context subset

4. **Cosine Similarity Classification**
   - Classify new queries to nearest cluster centroid
   - Route query to corresponding agent

5. **Baseline Comparison System**
   - Implement naive approach (all context to single agent)
   - Measure API calls and token consumption

6. **Cost Metrics Measurement**
   - Track: Number of LLM calls, total tokens, cost savings %
   - Compare baseline vs optimized system
   - Demonstrate >90% reduction

7. **Experimental Report**
   - Document methodology, results, visualizations
   - Explain clustering and classification techniques used

### MVP Success Criteria

**Technical Metrics:**
- ✅ Achieve >90% reduction in LLM API calls vs baseline
- ✅ Maintain >80% classification accuracy
- ✅ Query classification time <1 second
- ✅ Silhouette Score >0.3 (good cluster separation)

**Deliverables:**
- ✅ Working Python code (clean, commented)
- ✅ Experimental report with graphs and metrics
- ✅ Demo showing cost comparison (baseline vs optimized)
- ✅ Clear explanation connecting to data mining course concepts

### Out of Scope for MVP

**Explicitly excluded to meet 3-day timeline:**
- ❌ Production-ready deployment or scalability optimizations
- ❌ Advanced agent frameworks (LangChain, CrewAI, AutoGen)
- ❌ Multiple dataset testing beyond AG News
- ❌ Hyperparameter tuning for optimal K selection
- ❌ Real-time conversation handling (focus on batch processing)
- ❌ UI/UX for end users (command-line demo sufficient)
- ❌ Advanced clustering algorithms (DBSCAN, Hierarchical)
- ❌ Alternative classification methods (neural networks, KNN)

---

## Technical Preferences

Based on technical research completed:

**Technology Stack:**
- **Embedding Model:** Google Gemini Embedding API (`gemini-embedding-001`)
  - Rationale: Top MTEB benchmark performance, batch API for cost efficiency ($0.075/1M tokens)
- **Clustering Algorithm:** K-Means (scikit-learn)
  - Rationale: Fast, well-documented, perfect for spherical clusters in embedding space
- **Classification Method:** Cosine Similarity + Nearest Neighbor
  - Rationale: Zero training time, <1ms inference, natural fit for embeddings
- **Dataset:** AG News (4 categories, 120K training samples)
  - Rationale: Clear semantic boundaries, moderate size, standard benchmark
- **Agent Architecture:** Simple Python classes (no framework)
  - Rationale: Full control, minimal dependencies, easy to explain
- **LLM:** Google Gemini 2.5 Flash
  - Rationale: Course requirement, cost-efficient for demonstration

**Supporting Libraries:**
- numpy, pandas (data manipulation)
- scikit-learn (clustering, metrics, evaluation)
- google-genai (Gemini API SDK)
- matplotlib/seaborn (visualization)

---

## Timeline

**Total Duration:** 3 days

**Day 1: Data Preparation & Clustering**
- Load AG News dataset
- Generate embeddings using Gemini Batch API
- Apply K-Means clustering (K=4)
- Evaluate cluster quality (Silhouette Score)
- Visualize clusters

**Day 2: Agent System & Classification**
- Implement SpecializedAgent and AgentRouter classes
- Build baseline system (single agent with full context)
- Test classification accuracy
- Measure token consumption comparison

**Day 3: Experimentation & Reporting**
- Run comparative experiments (baseline vs optimized)
- Collect cost metrics and performance data
- Generate visualizations (cluster plots, cost charts)
- Write experimental report
- Prepare demo

---

## Risks and Assumptions

### Key Assumptions

1. **Cluster Quality:** AG News categories will naturally separate into 4 semantic clusters
   - Mitigation: Dataset chosen specifically for clear category boundaries

2. **Classification Accuracy:** Cosine similarity will achieve >80% routing accuracy
   - Mitigation: If low, can tune K or use alternative dataset

3. **API Availability:** Gemini Embedding API will remain accessible during implementation
   - Mitigation: Fallback to Sentence-BERT (local, free) if needed

4. **Cost Reduction:** Routing will actually reduce token consumption in practice
   - Validation: Baseline comparison will empirically verify savings

### Known Limitations

1. **Academic Scope:** This is a proof-of-concept, not production system
2. **Dataset Simplification:** AG News has clean boundaries; real conversations may be messier
3. **K Selection:** Using K=4 to match dataset; real systems would need dynamic K selection
4. **Stateless Agents:** No conversation memory across queries (acceptable for demo)

---

## Supporting Materials

This Product Brief was created based on comprehensive technical research:

**Research Foundation:**
- Technical Research Report: "Multi-Agent System for Cost-Efficient Text Classification" (2025-11-08)
- 30+ verified sources including official documentation, benchmarks, and 2025 academic research
- Technology evaluation across 6 domains: embeddings, clustering, classification, datasets, architectures, frameworks

**Key Research Findings:**
- Gemini Embedding API: Top MTEB ranking, batch processing for cost efficiency
- K-Means: Standard choice for text clustering with embeddings
- AG News: Optimal dataset for 3-day timeline and clear demonstration
- Academic evidence: LLM routing can reduce costs by 90-98% (arXiv 2025, Helicone 2025)

**Cost Optimization Evidence:**
- Industry research validates clustering + routing approach for cost savings
- QC-Opt framework and Cast AI products use similar intelligent routing
- Token compression + routing: 90% cost reduction without performance loss (Helicone 2025)

---

_This Product Brief captures the vision and requirements for a data mining course project demonstrating clustering and classification techniques for LLM cost optimization._

_It was created through collaborative discovery and reflects the unique needs of this academic project._

_Next: PRD workflow will transform this brief into detailed planning artifacts._
