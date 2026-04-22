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

Runtime settings are configured via `config/config.ini`. Copy the template to begin:
```bash
cp config/config.example.ini config/config.ini
```

### Configuration Sections
- **[api]**: Settings for LLM providers (Ollama, OpenAI, Google Gemini).
- **[models]**: Optional role-model overrides. These are only applied when `load_from_ini = true`.
- **[debate]**: Parameters for round limits, word constraints, and search.
- **[analysis]**: Settings for research audits, including the BERT emotion model and lexicons.

### Model and Topic Pools
- **`config/model_pool.json`**: Primary source of available models and provider mappings. Default role models are inferred from this file.
- **`config/topics.json`**: Primary topic catalog used by the debate and batch pipelines.
- **`config/config.ini`**: Runtime flags and system settings. It can also override role models when `load_from_ini = true`.

### Configuration Flow
1. The application loads runtime settings from `config/config.ini`.
2. Default role models are inferred from `config/model_pool.json`.
3. If `[models] load_from_ini = true`, then `moderator_model`, `conspiracy_model`, and `scientific_model` from `config.ini` override the pool defaults.
4. Topics are loaded from `config/topics.json` by default.
5. Both pools also support environment-variable overrides:
   - `DEBATE_MODEL_POOL_FILE`
   - `DEBATE_TOPICS_FILE`

---

## Execution

### Single Debate Simulation
- **Standard**: `python cli/main.py`
- **Poetry**: `poetry run sixseven`

### Research Audit
Runs the post-debate analysis pipeline. By default, this executes the four custom analyzers sequentially in-process and writes outputs under `results/analysis/`.

#### Default Flow
No arguments are required. The default custom suite uses:
- **Input Runs**: `old_artifacts`
- **Artifacts**: `old_artifacts`
- **Output Root**: `results`
- **Executed Analyzers**: `debate_analysis -> topic_analysis -> role_analysis -> llm_analysis`

- **Standard**: `python cli/analyze_results.py`
- **Module**: `python -m cli.analyze_results`
- **Poetry**: `poetry run sixseven-analyze`

#### Important Note
The default custom suite reads from `old_artifacts` on purpose. The machine used for debate generation needed roughly two days of continuous running to populate the newer refactored outputs under `results/batches/ollama`, and those long runs were affected by network and scheduling interruptions. As a result, the refactored batch artifact directories are not reliably populated yet.

Because of that limitation, the analysis scripts currently default to the older per-run, pre-refactor artifact schema in `old_artifacts`, which remains the canonical input source for the analysis pipeline unless you explicitly override it with CLI flags.

#### Supported Flags
- `--dir`: Input directory. Defaults to `old_artifacts` for the custom suite and `results/raw` for the legacy advanced analysis path.
- `--out`: Root directory for analysis output. Defaults to `results`.
- `--artifacts`: Artifacts directory used by the role analyzer. Defaults to `old_artifacts`.
- `--no-emotion`: Disables transformer-based emotion extraction where supported.
- `--max-runs`: Maximum number of runs for custom analyzers that support run limiting.
- `--overwrite-existing`: Recomputes existing outputs for custom analyzers that support overwrite behavior.
- `--stop-on-error`: Stops the custom suite on the first analyzer failure.
- `--advanced-analysis`: Runs the older built-in advanced analysis flow instead of the default custom suite.
- `--custom-suite`: Explicitly selects the custom suite path. This is already the default.
- `--run`: Only used by the legacy advanced analysis path to target a single run.

#### Analyzer Flow
When `cli/analyze_results.py` runs with defaults, the flow is:
1. Parse CLI arguments.
2. Default to the custom suite path.
3. Call the central runner in `src/debate/analysis/analysis_runner.py`.
4. Execute `debate_analysis`, `topic_analysis`, `role_analysis`, and `llm_analysis` sequentially via imported `main(argv)` functions.
5. Adapt shared CLI inputs into analyzer-specific flags.
6. Write outputs under `results/analysis/debate_analysis`, `results/analysis/llm_analysis`, `results/analysis/role_analysis`, and `results/analysis/topic_analysis`.

#### Legacy Analysis
The older analysis path is still available for direct per-run or batch analysis over `results/raw/run_*`.
- **Standard**: `python cli/analyze_results.py --advanced-analysis`
- **Poetry**: `poetry run sixseven-analyze --advanced-analysis`

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
- **config/**: Runtime settings, model pools, and topic catalogs.
- **docs/**: Technical and research guides.
- **results/**: Output root.
    - **raw/**: Single-run JSON memory states and reports.
    - **batches/**: Organized experiment suites (e.g., `ollama/`, `gemini/`).
    - **transcripts/**: Markdown debate summaries for single runs.
    - **analysis/**: Post-run outputs from the default custom analysis suite and legacy advanced analysis.
- **src/debate/**: Core library.
    - **analysis/**: Post-run analysis logic and metrics.
    - **core/**: Foundational modules shared across the core library.
    - **resources/**: Static resources and data files (e.g., lexicons).        
    - **simulator/**: Consolidated debate orchestration and simulation engines.
- **src/debate_sim/**: Legacy simulation package snapshot and related package structure.
- **tests/**: Unit and validation test suite.

## Documentation
1. [Architecture and Dependency Injection](docs/ARCHITECTURE.md)
2. [Batch Experiment Guide](docs/BATCH_GUIDE.md)
3. [MAI623-Group Project Instructions](docs/SixSeven-AgenticDebate&PersuasionBetweenConspiracy&ScientificModels.pdf)
