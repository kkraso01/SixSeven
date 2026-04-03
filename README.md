# SixSeven: Agentic Debate Simulator

> An end-to-end research framework that simulates structured debates between LLM-based agents. SixSeven enables reproducible experiments across multiple models and topics, utilizing a "Research-First" output hierarchy to analyze persuasion dynamics and linguistic patterns.

**Status**: 🟢 Architecture Refactored | 🧪 100% Test Coverage | 📚 Documentation Consolidated

---

## 🚀 Quick Start

### 1. Installation
The project uses a standard `src/` layout. For full isolation, use a virtual environment.

```bash
# Clone and enter
git clone https://github.com/kkraso01/SixSeven.git && cd SixSeven

# Option A: Pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Option B: Poetry
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

---

## 📂 Research-First Results
SixSeven uses a structured **`results/`** hierarchy to keep experiment data organized and analysis-ready.

*   **`results/raw/`**: Canonical JSON data, full memory states, and CSV logs for every run.
*   **`results/transcripts/`**: Human-readable Markdown summaries of every debate.
*   **`results/analysis/`**: Automated metrics and PNG visualizations (confidence shifts, stance trajectories, emotion distribution).

---

## 🏗️ Core Architecture

The system is built on a **Modular Pillar** design to ensure maintainability and scalability:

*   **`src/debate/core`**: The foundational "Brain." Contains Pydantic schemas, dependency injection protocols, and the central service container.
*   **`src/debate/simulator`**: The Orchestration Engine. Manages the 3-agent feedback loop (Conspiracy Advocate vs. Scientific Advocate, supervised by a Moderator).
*   **`src/debate/analysis`**: The Research Pipeline. Extracts linguistic features, calculates persuasion metrics, and generates reports.

### 🔌 Dependency Injection
Every major subsystem (LLM provider, web search, file export) is bound by **Protocols**. This allows you to swap a local Ollama model for Gemini, or DuckDuckGo search for Tavily, without touching the core logic.

> [!TIP]
> For a deep-dive into the technical design and protocols, see [**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md).

---

## 📊 Batch Experiments

SixSeven is designed for large-scale comparative research. It supports running hundreds of debates with automatic completion tracking and resume capability.

```bash
# 1. Run local experiments (No API costs, handles ~120 debates)
python cli/batch_ollama.py

# 2. Run cloud experiments (Google Gemini API, handles rate-limiting gracefully)
python cli/batch_gemini.py

# 3. View topic list
python cli/view_topics.py
```

See [**docs/BATCH_GUIDE.md**](docs/BATCH_GUIDE.md) for strategy and optimization tips.

---

## 🗺️ Project Navigation

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

## 📚 Documentation Index

| Guide | Purpose |
| :--- | :--- |
| [**ARCHITECTURE.md**](docs/ARCHITECTURE.md) | Technical deep-dive, DI protocols, and data-flow diagrams. |
| [**BATCH_GUIDE.md**](docs/BATCH_GUIDE.md) | Comprehensive guide to running 300+ experiment batches. |
| [**CONTRIBUTING.md**](.agentic-instructions.md) | Instructions for extending agents or adding metrics. |

---

**Team**: SixSeven | **Repository**: [kkraso01/SixSeven](https://github.com/kkraso01/SixSeven)
