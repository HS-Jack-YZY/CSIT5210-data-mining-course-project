# Project Documentation Index

## Project Overview

- **Name**: Context-Aware Multi-Agent System
- **Type**: Data Science / Machine Learning Research Project
- **Repository**: Monolith (single cohesive codebase)
- **Primary Language**: Python 3.10+
- **Architecture Pattern**: Layered Data Science Pipeline

## Quick Reference

| Aspect | Details |
|--------|---------|
| **Tech Stack** | Python, scikit-learn, Gemini API, NumPy, Pandas |
| **ML Framework** | scikit-learn (K-Means clustering) |
| **Embedding Service** | Google Gemini API (gemini-embedding-001, 768-dim) |
| **Dataset** | AG News (120K articles, 4 categories) |
| **Testing** | pytest (epic-based organization) |
| **Code Quality** | Ruff (linting + formatting) |
| **Entry Points** | `scripts/01_*.py` through `scripts/06_*.py` |
| **Architecture** | data → features → models → evaluation → visualization |

## Generated Documentation

### Core Documentation
- **[Project Overview](./project-overview.md)** - Executive summary, tech stack, architecture, current status
- **[Source Tree Analysis](./source-tree-analysis.md)** - Detailed code structure, entry points, design patterns
- **[Development Guide](./development-guide.md)** - Setup, configuration, build commands, testing

### Architecture and Design
- **[Architecture](./architecture.md)** - System architecture and technical decisions (existing)
- **[PRD](./PRD.md)** - Product Requirements Document (existing)
- **[Epics](./epics.md)** - Epic breakdown and user story mapping (existing)

### Technical Specifications
- **[Tech Spec - Epic 1](./tech-spec-epic-1.md)** - Project Initialization and Environment Setup
- **[Tech Spec - Epic 2](./tech-spec-epic-2.md)** - Clustering and Evaluation
- **[Tech Spec - Epic 3](./tech-spec-epic-3.md)** - Multi-Agent Classification System

### Implementation Documentation
- **[Sprint Status](./sprint-status.yaml)** - Current sprint progress and story status
- **[User Stories](./stories/)** - Individual story implementations with acceptance criteria
  - 11 completed/in-progress stories across Epics 1-3
- **[Retrospectives](./retrospectives/)** - Epic completion retrospectives
  - Epic 1 Retrospective (2025-11-09)
  - Epic 2 Retrospective (2025-11-09)

### Research and Analysis
- **[Product Brief](./product-brief-report-2025-11-08.md)** - Initial product vision and market analysis
- **[Technical Research](./research-technical-2025-11-08.md)** - Technical feasibility research
- **[Implementation Readiness Report](./implementation-readiness-report-2025-11-09.md)** - Solutioning gate check results

## Existing Documentation

### Project Root
- **[README.md](../README.md)** - Quick start guide, usage examples, API documentation
- **[LICENSE](../LICENSE)** - MIT License
- **[Makefile](../Makefile)** - Development command shortcuts
- **[requirements.txt](../requirements.txt)** - Python dependencies (version-pinned)
- **[pyproject.toml](../pyproject.toml)** - Project metadata and tool configurations
- **[config.yaml](../config.yaml)** - Centralized experimental configuration
- **[.env.example](../.env.example)** - Environment variable template

### Configuration Files
- **config.yaml** - All experimental parameters (dataset, clustering, embedding, metrics)
- **.env** - API keys and secrets (not in git, use .env.example as template)

## Getting Started

### For New Developers

1. **Read First**:
   - [README.md](../README.md) - Understand project purpose and quick start
   - [project-overview.md](./project-overview.md) - Get comprehensive overview
   - [development-guide.md](./development-guide.md) - Set up development environment

2. **Setup Environment**:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env  # Add GEMINI_API_KEY
   ```

3. **Explore Codebase**:
   - [source-tree-analysis.md](./source-tree-analysis.md) - Understand code structure
   - [architecture.md](./architecture.md) - Learn system design

4. **Run Pipeline**:
   ```bash
   python scripts/01_generate_embeddings.py
   python scripts/02_train_clustering.py
   python scripts/03_evaluate_clustering.py
   # ... continue with remaining scripts
   ```

### For Project Planning

1. **Review Requirements**:
   - [PRD.md](./PRD.md) - Product requirements and success criteria
   - [epics.md](./epics.md) - Epic breakdown and story mapping

2. **Check Current Status**:
   - [sprint-status.yaml](./sprint-status.yaml) - Current sprint progress
   - [project-overview.md](./project-overview.md) - High-level status summary

3. **Plan Next Steps**:
   - Review remaining stories in [epics.md](./epics.md)
   - Check acceptance criteria in [stories/](./stories/)
   - Coordinate with technical specifications

### For Understanding Implementation

1. **Architecture**:
   - [architecture.md](./architecture.md) - System design and decisions
   - [source-tree-analysis.md](./source-tree-analysis.md) - Code organization

2. **Technical Details**:
   - [tech-spec-epic-1.md](./tech-spec-epic-1.md) - Foundation and setup
   - [tech-spec-epic-2.md](./tech-spec-epic-2.md) - Clustering implementation
   - [tech-spec-epic-3.md](./tech-spec-epic-3.md) - Multi-agent system

3. **Code**:
   - Browse `src/context_aware_multi_agent_system/` modules
   - Review `scripts/` for pipeline execution
   - Check `tests/` for usage examples

## Project Status

### Current Sprint

**Active Epic**: Epic 3 - Multi-Agent Classification System
**Status**: 🔄 In Progress

**Completed Epics**:
- ✅ Epic 1: Project Initialization and Environment Setup
- ✅ Epic 2: Clustering and Evaluation

**In Progress**:
- 🔄 Epic 3 (50% complete): Multi-Agent Classification System
  - ✅ Story 3-1: Specialized Agent Implementation (in review)
  - 📋 Story 3-2: Cosine Similarity Classification Engine (backlog)
  - 📋 Story 3-3: Agent Router with Query Routing (backlog)
  - 📋 Story 3-4: Classification Accuracy Measurement (backlog)
  - 📋 Story 3-5: Query Processing Performance Benchmarking (backlog)

**Upcoming**:
- 📋 Epic 4: Baseline Comparison and Reporting (not started)

### Key Metrics

- **Completed Stories**: 10/20 (50%)
- **Completed Epics**: 2/4 (50%)
- **Code Coverage**: TBD
- **Documentation Coverage**: 95%

## Directory Structure

```
docs/
├── index.md                           # This file (master index)
├── project-overview.md                # Executive summary
├── source-tree-analysis.md            # Code structure analysis
├── development-guide.md               # Developer setup guide
├── architecture.md                    # System architecture
├── PRD.md                            # Product requirements
├── epics.md                          # Epic breakdown
├── tech-spec-epic-1.md               # Epic 1 technical spec
├── tech-spec-epic-2.md               # Epic 2 technical spec
├── tech-spec-epic-3.md               # Epic 3 technical spec
├── sprint-status.yaml                # Sprint tracking
├── product-brief-report-2025-11-08.md         # Product vision
├── research-technical-2025-11-08.md           # Technical research
├── implementation-readiness-report-2025-11-09.md  # Gate check
├── project-scan-report.json          # Documentation workflow state
├── stories/                          # User story implementations
│   ├── 1-1-project-initialization-and-environment-setup.md
│   ├── 1-2-configuration-management-system.md
│   ├── 1-3-ag-news-dataset-loading-and-validation.md
│   ├── 1-4-gemini-api-integration-and-authentication.md
│   ├── 2-1-batch-embedding-generation-with-caching.md
│   ├── 2-2-k-means-clustering-implementation.md
│   ├── 2-3-cluster-quality-evaluation.md
│   ├── 2-4-pca-cluster-visualization.md
│   ├── 2-5-cluster-analysis-and-labeling.md
│   ├── 3-1-specialized-agent-implementation.md
│   └── validation-report-1-4-20251109.md
└── retrospectives/                   # Epic retrospectives
    ├── epic-1-retro-2025-11-09.md
    └── epic-2-retro-2025-11-09.md
```

## Contributing

### Development Workflow

1. **Check sprint status**: Review [sprint-status.yaml](./sprint-status.yaml)
2. **Pick a story**: Select from backlog in [epics.md](./epics.md)
3. **Read tech spec**: Review relevant [tech-spec-epic-*.md](./tech-spec-epic-1.md)
4. **Implement**: Follow acceptance criteria in story file
5. **Test**: Add tests in `tests/epic{n}/`
6. **Document**: Update story file with implementation notes
7. **Review**: Follow code review guidelines

### Documentation Updates

When updating code:
- ✅ Update relevant technical specification if architecture changes
- ✅ Update story file if implementation deviates from plan
- ✅ Add inline code comments for complex logic
- ✅ Update README.md if user-facing changes
- ✅ Run `make format` and `make lint` before committing

## Links and Resources

### Internal Documentation
- **Main README**: [../README.md](../README.md)
- **Configuration**: [../config.yaml](../config.yaml)
- **Dependencies**: [../requirements.txt](../requirements.txt)

### External Resources
- **Google Gemini API**: [https://ai.google.dev/](https://ai.google.dev/)
- **AG News Dataset**: [https://huggingface.co/datasets/ag_news](https://huggingface.co/datasets/ag_news)
- **scikit-learn Docs**: [https://scikit-learn.org/](https://scikit-learn.org/)
- **pytest Documentation**: [https://docs.pytest.org/](https://docs.pytest.org/)
- **Ruff Documentation**: [https://docs.astral.sh/ruff/](https://docs.astral.sh/ruff/)

## Search and Navigation Tips

### Finding Information

**Looking for...**
- **Setup instructions**: See [development-guide.md](./development-guide.md)
- **Architecture details**: See [architecture.md](./architecture.md)
- **Code structure**: See [source-tree-analysis.md](./source-tree-analysis.md)
- **Requirements**: See [PRD.md](./PRD.md)
- **Epic breakdown**: See [epics.md](./epics.md)
- **Story details**: See [stories/](./stories/)
- **Current progress**: See [sprint-status.yaml](./sprint-status.yaml)
- **API usage**: See [README.md](../README.md)
- **Configuration options**: See [config.yaml](../config.yaml)

### Quick Actions

```bash
# View project overview
cat docs/project-overview.md

# Check current sprint status
cat docs/sprint-status.yaml

# Read a specific story
cat docs/stories/3-1-specialized-agent-implementation.md

# View architecture
cat docs/architecture.md

# Check development setup
cat docs/development-guide.md
```

---

**Documentation Generated**: 2025-11-09
**Documentation Version**: 1.0
**Project Version**: 0.1.0
**Last Updated By**: BMM document-project workflow (exhaustive scan)

## Workflow Metadata

This documentation index was generated by the BMM document-project workflow:
- **Workflow Mode**: initial_scan
- **Scan Level**: exhaustive (all source files read)
- **Project Classification**: Data science/ML monolith project
- **Documentation Requirements**: Based on data project type template
- **State File**: [project-scan-report.json](./project-scan-report.json)
