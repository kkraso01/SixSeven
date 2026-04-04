# SixSeven: Agentic Debate Simulation Framework

SixSeven is a research framework for simulating debates between LLM-based agents. It enables reproducible experiments between divergent worldviews (e.g., Conspiracy theory proponent vs. Scientific consensus) using a multi-agent orchestration pipeline and a results-based output hierarchy.

This project was developed as part of the **MAI623-NLP** course for the **MSc in Artificial Intelligence** at the **University of Cyprus (UCY)**.

The framework supports post-debate analysis, including sentiment tracking, BERT-based emotion detection, and rhetorical marker analysis.

## Core Features
- **Multi-Agent Orchestration**: Structured debates between a Conspiracy Advocate (CA), a Scientific Advocate (SA), and a neutral Moderator (MA).
- **Structured Knowledge Representation**: Agent turns and moderator recaps use Pydantic models for data integrity and validation.
- **Rhetorical and Emotional Auditing**: Integrated NLP signals for assessing persuasion dynamics and rhetorical tactics.
- **Information Retrieval**: Optional real-time search via the DuckDuckGo Search Provider.
- **Batch Experiments**: Support for large-scale simulations with automated resume and rate-limit handling.

---

## Installation

SixSeven requires Python 3.11 or later.

### Method A: Pip (Standard)
Installation using `requirements.txt`:
```bash
# Clone the repository
git clone https://github.com/kkraso01/SixSeven.git && cd SixSeven

# Setup virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies and the local package
pip install -r requirements.txt && pip install -e .
```

### Method B: Poetry
Installation using `pyproject.toml`:
```bash
# Clone the repository
git clone https://github.com/kkraso01/SixSeven.git && cd SixSeven

# Install dependencies
poetry install
```

---

## Configuration

The system is configured via `config/config.ini`. Copy the template to begin:
```bash
cp config/config.example.ini config/config.ini
```

### Configuration Sections
- **[api]**: Settings for LLM providers (Ollama, OpenAI, Google Gemini).
- **[models]**: Model assignments and temperature settings per role.
- **[debate]**: Parameters for round limits, word constraints, and search.
- **[analysis]**: Settings for research audits, including the BERT emotion model and lexicons.

---

## Execution

### Single Debate Simulation
- **Standard**: `python cli/main.py`
- **Poetry**: `poetry run sixseven`

### Research Audit
Processes raw JSON memory states to generate sentiment trajectories, emotional distributions, and rhetorical summaries.
- **Standard**: `python cli/analyze_results.py --dir results/batches/ollama/raw`
- **Poetry**: `poetry run sixseven-analyze --dir results/batches/ollama/raw`

### Topic Selection Utility
Browse and select conspiracy topics from the built-in library.
- **Standard**: `python cli/view_topics.py all`
- **Poetry**: `poetry run sixseven-topics all`

### Batch Experiments
Execute large-scale experiment suites across all 20 topics and various model configurations.

#### 1. Ollama (Local)
No API limits. Runs continuously on local hardware.
- **Standard**: `python cli/batch_ollama.py`
- **Poetry**: `poetry run sixseven-batch-ollama`

#### 2. Gemini (Cloud)
Requires an API key. Subject to provider rate limits and daily quotas. This runner includes automated resume and backoff handling.
- **Standard**: `python cli/batch_gemini.py`
- **Poetry**: `poetry run sixseven-batch-gemini`

---

## Project Structure and Results

- **cli/**: Command-line entry points.
- **config/**: Configuration templates and active settings.
- **docs/**: Technical and research guides.
- **results/**: Output root.
    - **raw/**: Single-run JSON memory states and reports.
    - **batches/**: Organized experiment suites (e.g., `ollama/`, `gemini/`).
    - **transcripts/**: Markdown debate summaries for single runs.
    - **analysis/**: Automated metrics and plots for single runs.
- **src/debate/**: Core library.
- **tests/**: Unit and validation test suite.

---

## Documentation
1. [Architecture and Dependency Injection](docs/ARCHITECTURE.md)
2. [Batch Experiment Guide](docs/BATCH_GUIDE.md)
3. [MAI623-Group Project Instructions](docs/SixSeven-AgenticDebate&PersuasionBetweenConspiracy&ScientificModels.pdf)
