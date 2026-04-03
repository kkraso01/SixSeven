# SixSeven: Agentic Debate Simulator

> An end-to-end research framework that simulates structured debates between LLM-based agents. SixSeven enables reproducible experiments across multiple models and topics, utilizing a "Research-First" output hierarchy to analyze persuasion dynamics and linguistic patterns.

**Status**: Architecture Refactored | 100% Test Coverage | Documentation Consolidated

---

## Quick Start

### 1. Installation
The project uses a standard `src/` layout. For full isolation, use a virtual environment.

```bash
# Clone and enter
git clone https://github.com/kkraso01/SixSeven.git && cd SixSeven

# Option A: Pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Option B: Poetry
python -m pip install poetry
poetry install
```

### 2. Configure Models
Copy the example configuration and set your preferred models (defaulting to local Ollama).
```bash
cp config/config.example.ini config/config.ini
```

### 3. Run a Single Debate
Execute the default experiment (mRNA Vaccine safety) using your configured models.
```bash
PYTHONPATH=src python cli/main.py
```

### 4. Advanced Research Mode
Run the research-grade batch analysis to generate sentiment, emotion, and rhetorical reports.
```bash
# This will process ALL runs in results/raw/ using BERT-based emotion detection
PYTHONPATH=src python cli/analyze_results.py
```

---

## Research-First Results
SixSeven uses a structured **`results/`** hierarchy to keep experiment data organized and analysis-ready.

*   **`results/raw/`**: Canonical JSON data, full memory states, and CSV logs for every run.
*   **`results/transcripts/`**: Human-readable Markdown summaries of every debate.
*   **`results/analysis/`**: Automated metrics and visualizations (confidence shifts, stance trajectories, emotion distribution).

---

## Project Navigation

```text
SixSeven/
├── cli/                  # Entry points for single & batch runs
├── config/               # Configuration (.ini) templates
├── docs/                 # Detailed Technical guides
├── results/              # Output hierarchy (raw, transcripts, analysis)
├── src/debate/           # Library core
│   ├── core/             # Container, Protocols, Schemas
│   ├── simulator/        # Engine, IO, Providers, Prompts
│   └── analysis/         # Metrics, Plots, Feature Extraction
└── tests/                # Unit & Validation suites
```

---

## Documentation Index

For deep dives into the engine or research methodologies:

1. [Architecture & DI](docs/ARCHITECTURE.md) - How the system is built.
2. [Research Methodology](docs/RESEARCH_METRICS.md) - How we measure persuasion.
3. [Batch Experiments](docs/BATCH_GUIDE.md) - Scaling your simulations.

---

## License
MIT
