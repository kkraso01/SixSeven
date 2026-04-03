# SixSeven: Agentic Debate Simulator

> An end-to-end system that simulates structured debates between LLM-based agents arguing opposing positions on conspiracy theories, moderated by a control agent. Runs reproducible experiments across multiple models and topics, logs all interactions in a canonical debate format, and analyzes persuasion dynamics.

**Team**: SixSeven | **Status**:  Code Complete (Core Systems Ready)

---

## Documentation Guide

This README combines all key information. For detailed content, see:

### Core Documentation
| Document | Purpose |
|----------|----------|
| [**ARCHITECTURE.md**](docs/ARCHITECTURE.md) | Technical architecture and design details |
| [**BATCH_GUIDE.md**](docs/BATCH_GUIDE.md) | How to run 120-debate batch experiments |

### Reference
| Document | Purpose |
|----------|----------|
| [**.agentic-instructions.md**](.agentic-instructions.md) | Complete reference for agentic coding platforms |

---

## Quick Start

### Installation

**Using Pip:**
```bash
pip install -r requirements.txt
pip install -e .
```

**Using Poetry:**
```bash
poetry install
```

### Run Single Debate
```bash
python cli/main.py
```
Creates one debate on a random conspiracy topic, outputs to `artifacts/`.

### Run Batch Experiments
```bash
# Ollama batch first (no API limits, 8-12 hours)
python cli/batch_ollama.py

# Gemini batch when API quota available (rate-limited, with resume capability)
python cli/batch_gemini.py
```

### View Available Topics
```bash
python cli/view_topics.py summary
```

---

## Project Overview

### What It Does
- **Orchestrates debates** between 3 LLM agents (conspiracy proponent, scientific opponent, moderator)
- **Supports multiple models**: Google Gemini (cloud), Ollama-compatible models (local/remote)
- **Uses free search**: DuckDuckGo (no API key needed)
- **Logs canonically**: All debates in standard JSON schema
- **Analyzes automatically**: Persuasion metrics, language features, emotion analysis
- **Exports results**: JSON, CSV, Markdown, and PNG visualizations

### Key Capabilities
 Moderator-controlled debate flow  
 20 conspiracy theory topics dataset  
 8 model configurations per batch for comparison  
 Batch runner for 300+ experiments  
 Automatic analysis pipeline  
 CSV export for external analysis  

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full project details.

---

## System Requirements

### Prerequisites
- **Python 3.11+**
- **Ollama** running locally (for local models)
  ```bash
  ollama serve
  ollama pull <your-model>  # e.g. llama3, gemma3:27b, etc.
  ```
- **Gemini API key** (optional, for cloud experiments)

### Dependencies
See `requirements.txt` for full list. Key packages:
- `pydantic` >= 2.0 (schemas)
- `instructor` (structured LLM outputs via DI architecture)
- `google-genai` (Gemini API)
- `ddgs` (DuckDuckGo search)
- `scikit-learn` (analysis), `matplotlib` (plots)

---

##  Configuration

### Via config.ini
```ini
[api]
base_url = http://localhost:11434
api_mode = ollama
gemini_api_key = <your-key-here>

[models]
moderator_model = <your-model>
conspiracy_model = <your-model>
scientific_model = <your-model>

[debate]
rounds = 5
word_limit = 180
max_tokens = 600
```

**Configuration file location**: `config/config.ini` (copy from `config/config.example.ini`)\n**For complete configuration reference**, see [docs/ARCHITECTURE.md - Configuration](docs/ARCHITECTURE.md#-stage-1-configuration-configini).

### Via Environment Variables
```bash
export DEBATE_BASE_URL="http://localhost:11434"
export DEBATE_MODERATOR_MODEL="<your-model>"
export DEBATE_CONSPIRACY_MODEL="<your-model>"
export DEBATE_SCIENTIFIC_MODEL="<your-model>"
export DEBATE_ROUNDS="5"
export DEBATE_WORD_LIMIT="180"
export DEBATE_MAX_TOKENS="600"
export DEBATE_OUTPUT_DIR="artifacts"
export DEBATE_RUN_ANALYSIS="true"
```

Full configuration reference: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#-stage-1-configuration-configini)

---

## Running Experiments

### Single Debate
```bash
python cli/main.py
```

### Batch Experiments

**320 debates across 2 batches of 8 model configurations each:**

Each batch tests all permutations of 2 models across 3 roles (moderator, conspiracy, scientific).
Configure your models in `cli/batch_ollama.py` and `cli/batch_gemini.py`.

**Strategy**: Split into two batches to respect Gemini API limits

```bash
# Batch 1: Ollama models (160 experiments, 8-12 hours)
python cli/batch_ollama.py
# Output: <output_dir>/all_debates_ollama.csv

# Batch 2: Gemini models (160 experiments, rate-limited)
python cli/batch_gemini.py
# Output: <output_dir>/all_debates_gemini.csv
```

**For detailed batch strategy, optimization tips, and per-batch configurations**, see [docs/BATCH_GUIDE.md](docs/BATCH_GUIDE.md).

---

## Output Structure

Each run creates a timestamped directory: `artifacts/run_YYYYMMDD_HHMMSS/`

```
run_20240101_120000/
├── transcript.md              # Human-readable debate log
├── debate_log.csv             # Canonical debate format
├── memory.json                # Agent internal state
├── metrics.csv                # Quantitative metrics
├── final_report.json          # Complete debate record
├── experiment_metadata.json   # Configuration & metadata
├── run_config.json            # Exact parameters used
├── analysis_report.md         # Analysis findings (markdown)
├── analysis_report.json       # Analysis findings (structured)
└── figures/
    ├── confidence_trajectory.png
    ├── emotion_distribution.png
    ├── language_features.png
    └── [other visualizations]
```

### Canonical Debate Log Format
```json
{
  "debate_id": "modelA-v-modelB_topic_20240101_120000",
  "claim": "COVID-19 vaccines contain microchips.",
  "round": 1,
  "speaker_role": "proponent|opponent|moderator",
  "utterance": "...",
  "stance": "pro|con|neutral",
  "confidence": 0.8,
  "tool_used": "none|duckduckgo",
  "tool_query": "...",
  "reply_to_turn": 0
}
```

---

## Analysis Pipeline

### Automatic Metrics (Generated per Debate)
- **Persuasion**: Stance stability, confidence shifts, winner determination
- **Language**: Uncertainty markers, moral framing, modality (strong/weak)
- **Emotion**: NRC emotion lexicon analysis
- **Comparison**: Human vs. LLM patterns, model-vs-model analysis

### Using the Analysis API
```python
from debate.analysis.analysis_runner import analyze_run, analyze_all

# Analyze single debate
report = analyze_run("artifacts/run_20240101_120000")
print(report.metrics)

# Analyze all debates
aggregate = analyze_all("artifacts")
print(aggregate.summary_statistics)
```

Analysis runs automatically (disable with `DEBATE_RUN_ANALYSIS=0`).

---

## Dependency Injection

The orchestrator uses **Protocol-based dependency injection** for all major subsystems. This makes the system testable and extensible without modifying core logic.

### Default Usage (unchanged)
```python
from debate import DebateConfig, run_debate

config = DebateConfig.from_ini("config/config.ini")
run_debate(topic, motion, rounds, config)  # auto-wires defaults
```

### Explicit Service Wiring
```python
from debate import DebateConfig, DebateServices, build_default_services, run_debate

config = DebateConfig.from_ini("config/config.ini")
services = build_default_services(config)
run_debate(topic, motion, rounds, config, services=services)
```

### Custom Implementations
```python
from debate import DebateServices, run_debate
from debate.core.container import FilePromptLoader, FileArtifactExporter

# Swap in a custom search provider or mock LLM
services = DebateServices(
    llm=my_mock_llm,
    search=my_tavily_provider,
    prompts=FilePromptLoader(),
    exporter=FileArtifactExporter(),
    analyzer=None,  # skip analysis
)
run_debate(topic, motion, rounds, config, services=services)
```

### Available Protocols
| Protocol | Purpose | Default |
|---|---|---|
| `LLMClient` | Low-level LLM calls | `OllamaClient` |
| `StructuredLLMService` | LLM with retry logic | `StructuredLLM` |
| `SearchProvider` | Web search | `DuckDuckGoSearchProvider` |
| `PromptLoader` | Template loading | `FilePromptLoader` |
| `ArtifactExporter` | Output writing | `FileArtifactExporter` |
| `DebateAnalyzer` | Post-run analysis | `DefaultDebateAnalyzer` |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#-dependency-injection-architecture) for full details.

---

## � Development & Iteration

### Adding a New Conspiracy Topic
1. Edit `src/debate/core/topics.py`
2. Add entry to `CONSPIRACY_TOPICS` list
3. Test: `python cli/view_topics.py summary`
4. Re-run batch experiments

### Updating Agent Prompts
1. Edit `src/debate/simulator/prompts/{conspiracy.md, scientific.md, moderator.md}`
2. **Important**: Keep final line: `Return ONLY valid JSON matching the schema. No extra text.`
3. Test: `python cli/main.py`

### Changing Debate Parameters
1. Edit `config/config.ini` or environment variables
2. Test single debate: `python main.py`
3. Run validation: `python tests/validation/validate_run.py`

### Exporting Results
```python
import pandas as pd

# Combine batch results
ollama_df = pd.read_csv("artifacts/all_debates_ollama.csv")
gemini_df = pd.read_csv("artifacts/all_debates_gemini.csv")
combined = pd.concat([ollama_df, gemini_df], ignore_index=True)
combined.to_csv("artifacts/all_debates_combined.csv", index=False)
```

---

## Validation

### Validate Run Output
```bash
python tests/validation/validate_run.py artifacts/run_20240101_120000
```
Checks JSON schemas, debate log format, file integrity.

### Validate Configuration & Memory
```bash
python tests/validation/validate_memory.py
```
Analyzes debate memory architecture and configuration.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Connection refused" for Ollama | Ensure `ollama serve` is running, check endpoint in `config.ini` |
| Gemini API errors | Verify API key, check daily quota (1,500 req/day), rate limit (15 req/min) |
| Schema validation errors | Check `src/debate/core/schemas.py`, verify LLM output is valid JSON |
| Analysis won't run | Check `memory.json` exists, download NLTK data: `python -m nltk.downloader punkt wordnet` |
| Memory overflow | Reduce `max_tokens`, split batch into smaller runs |

Full troubleshooting: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#troubleshooting) (Coming soon or refer to internal docs)

---

## Project Structure

**Reorganized for better maintainability:**

```
SixSeven/
├──  cli/                          # CLI entry points
│   ├── main.py                      # Single debate runner
│   ├── base_batch.py                # BaseBatchRunner (shared batch logic + DI wiring)
│   ├── batch_ollama.py              # Ollama batch (160 experiments)
│   ├── batch_gemini.py              # Gemini batch (160 experiments, with resume)
│   └── view_topics.py               # Topic browser
│
├──   config/                      # Configuration
│   ├── config.ini                   # Your settings (git-ignored)
│   └── config.example.ini           # Example configuration
│
├──  tests/                        # Test suite (47 tests)
│   ├── test_unit/
│   │   ├── test_imports.py          # Import validator
│   │   ├── test_config.py           # DebateConfig tests
│   │   ├── test_schemas.py          # Pydantic model tests
│   │   ├── test_memory.py           # Memory state tests
│   │   ├── test_metrics.py          # Analysis metrics tests
│   │   ├── test_evaluation.py       # Evaluation function tests
│   │   └── test_search.py           # Web search tests
│   └── validation/                  # Validation tools
│       ├── validate_run.py          # Debate run validator
│       └── validate_memory.py       # Memory architecture validator
│
├──  docs/                         # Documentation
│   ├── README.md                    # (Redirects to root README)
│   ├── ARCHITECTURE.md              # Technical design and deep dive
│   └── BATCH_GUIDE.md               # Batch experiment guide
│
├──  src/debate/               # Core application (library)
│   ├── core/                        # Foundational modules
│   │   ├── config.py                # Config loading
│   │   ├── errors.py                # Shared exceptions
│   │   ├── logging.py               # Centralised logging setup
│   │   ├── protocols.py             # Protocol ABCs (DI interfaces)
│   │   ├── container.py             # Service container (wiring logic)
│   │   ├── schemas.py               # Pydantic models
│   │   └── topics.py                # 20 conspiracy theory topics
│   ├── simulator/                   # Consolidated simulation engine
│   │   ├── engine/                  # Orchestration & evaluation
│   │   ├── providers/               # LLM, Search, & Instructor backends
│   │   ├── io/                      # Output generation & CSV export
│   │   └── prompts/                 # Agent role templates
│   └── analysis/                    # Analysis pipeline
│
├──  artifacts/                    # Experiment runs (configurable)
└── pyproject.toml                   # Project metadata & Poetry config
```

---

## References & Resources

- **CrewAI**: https://www.crewai.com/
- **LangGraph**: LangChain graph-based orchestration
- **Tavily**: Enhanced search API (alternative to DuckDuckGo)
- **Intelligence Squared Debates**: https://tisjune.github.io/research/iq2
- **NRC Emotion Lexicon**: https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
- **Moral Framing (EMFD)**: https://github.com/medianeuroscience/emfd

---

## Documentation Notes

- **Documentation**: All core documentation is now located in the root `README.md` and the `docs/` directory.
- **Current Status**: All core systems ready, validation passing

**Repository**: [kkraso01/SixSeven](https://github.com/kkraso01/SixSeven)  
**Branch**: suprdev | **Default**: main  
**Team**: SixSeven
