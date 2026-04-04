# SixSeven Architecture

Complete end-to-end technical walkthrough of the debate simulation system.

## Architecture Overview

The project follows a **pipeline architecture** with these stages:

```
Config Loading -> Batch Orchestration -> Debate Simulation -> 
Export & Logging -> Optional Analysis -> Results Archive
```

---

## Stage 1: Configuration (config/config.ini)

**Purpose**: Centralize all runtime parameters

**Key sections**:

### [api]: Connection settings
- `base_url`: Points to Ollama or Gemini API 
- `api_mode`: Sets provider ("ollama", "openai", or "gemini")
- `gemini_api_key`: Required for Gemini mode

### [models]: LLM assignments
- Three roles get distinct models: `moderator_model`, `conspiracy_model`, `scientific_model`
- Each has independent `*_temperature`: controls randomness (0.2 = deterministic, 0.6 = creative)

### [debate]: Debate mechanics
- `rounds`: Max number of rounds (can end early if moderator decides)
- `word_limit`: Soft constraint per response
- `max_tokens`: Hard LLM generation limit
- `seed`: Reproducibility (empty = random)
- `max_search_rounds`: Max agent research rounds per turn
- `thinking_budget`: Extra tokens reserved for thinking-capable models
- `num_ctx`: Context window size for Ollama-compatible backends

### [output]: Storage paths
- `output_dir`: Where run artifacts get saved (e.g., `results/`)

### [history]: Context window management
- `history_mode`: "global_full" (all agents see full transcript) vs "per_agent" vs "memory_only"
- `history_trim`: How to reduce context size ("none", "rounds", "messages", "chars")
- `max_rounds_in_history`: Keep only last N rounds to fit context window
- `include_memory_summary`: Adds compact scoreboard/state summary
- `max_messages_in_history`: Keep only last N messages
- `max_chars_in_history`: Keep only last N characters
- `summarize_if_trimmed`: Add a moderator summary when history is trimmed
- `highlight_opponent_last`: Re-inject the opponent's last message to keep it salient

### [analysis]: Post-debate analysis
- `run_analysis`: Enable/disable automated metrics
- `analysis_shift_threshold`: Minimum confidence point shift to flag "persuasion"
- `analysis_similarity_method`: For detecting argument redundancy
- `emotion_model`: Model name for advanced emotion analysis (BERT-based)
- `overwrite`: Whether advanced analysis overwrites existing outputs
- `max_runs`: Limit for advanced analysis batch mode
- `uncertainty_lexicon`, `strong_modality_lexicon`, `weak_modality_lexicon`: Lexicon overrides

**Loading mechanism** (`config.py`):
```python
DebateConfig.from_ini("config/config.ini")  # Reads INI, falls back to hardcoded defaults
# OR
DebateConfig.from_env()  # Legacy alias for from_ini(); GEMINI_API_KEY is read if set
```

---

## Stage 2: Batch Orchestration

**Two batch runners** for different use cases:

### cli/batch_ollama.py & cli/batch_gemini.py

Both batch runners inherit from `cli/base_batch.py -> BaseBatchRunner`, which
encapsulates shared logic: topic loading, completion indexing, DI wiring via
`build_default_services(config)`, CSV aggregation, and progress reporting.

**OllamaBatchRunner** (cli/batch_ollama.py) - Local/Free
- Runs completely on Ollama (local/university server)
- **No API limits** -> can run 24/7
- 2 model configs x 20 topics = **40 debates**
- ~3-5 hours runtime

**GeminiBatchRunner** (cli/batch_gemini.py) - Cloud/API
- Uses Google Gemini (gemini-3-flash-preview) for configured roles (requires API key)
- **Rate limited** -> 5 req/min, 20 req/day (free tier)
- 8 model configs x 20 topics = **160 debates**
- **Auto-resume capability**: Detects rate limits + retries with exponential backoff

**Both runners**:
1. Load 20 conspiracy topics from `topics.py`
2. For each topic x model config:
   - Create `DebateConfig` instance
   - Call `run_debate(topic, motion, rounds, config)`
   - Collect `ExperimentResult`  
3. Export combined CSV: `all_debates_ollama.csv` or `all_debates_gemini.csv` (aggregates `run_*` directories found directly under the batch output directory)

---

## Stage 3: Core Debate Simulation (orchestrator.py)

This is where the core debate logic runs. Here's the flow:

### Initialization
```python
debate_id = "debate_xyz123"         # Unique ID per debate
memory = initial_memory(topic, motion)  # Empty scoreboard

# Wire services via dependency injection (or auto-build defaults)
if services is None:
    services = build_default_services(config)
llm = services.llm          # StructuredLLMService protocol
search = services.search     # SearchProvider protocol
prompts = services.prompts   # PromptLoader protocol

conversation_history = []    # Global transcript
```

> **Dependency Injection**: All concrete implementations (OllamaClient, DuckDuckGo,
> filesystem prompts, etc.) are now behind Protocol interfaces and injected via a
> `DebateServices` container. See [DI Architecture](#dependency-injection-architecture) below.

### Round Loop (`while round_number < max_rounds`)

   - System: CA role prompt ("You are the conspiracy proponent...")
   - System: Debate rules

2. **Call LLM**:
   ```python
   ca_turn = llm.call(AgentTurn, messages, 
                      model=config.conspiracy_model,
                      temperature=config.conspiracy_temperature,
                      max_tokens=config.max_tokens)
   ```

   - `claim`: Their main argument
   - `reasons`: List of supporting points
   - `confidence`: Updated belief (0-100) - can decrease if persuaded!
   - `tactic_used`: Rhetoric type ("appeal to authority", "reductio ad absurdum", etc.)
   - `question_to_opponent`: Pushback they want addressed
   - `what_changes_mind`: Evidence type that would flip them
   - Optional `search`: {"should_search": bool, "search_query": "..."}

4. **Handle search** (if requested):
   - Perform DuckDuckGo search with CA's query
   - Log to memory BUT **do NOT show to opponent** (search remains private)

5. **Update memory & conversation history**:
   - Append AgentTurn to memory's `debate_log`
   - Update CA's agent state (confidence, what_changes_mind)
   - Add to `conversation_history` (formatted as assistant message)

#### Phase 1b: Scientific Advocate (SA) Turn
- **Identical to CA** but:
  - Uses `ScientificTurn` schema (adds `clarify`, `evaluate_gaps`, `alternative_hypotheses`, `discriminating_tests`)
  - Gets `config.scientific_model` and `config.scientific_temperature`
  - Sees CA's last message highlighted in their context

#### Phase 2: Moderator Analysis
1. **Recap generation**: Moderator analyzes the round
   ```python
   recap = llm.call(ModeratorRecap, messages)
   ```
   Returns:
   - `summary_agreements`: Common ground found
   - `summary_disagreements`: Key dividing points
   - `detected_fallacies_or_moves`: Rhetorical tactics used
   - `civility_score`, `epistemic_quality_score`, `bridge_building_score` (0-5)
   - `confidence_updates`: {"CA_delta": +/-N, "SA_delta": +/-N} (how much the moderator thinks each shifted)

   - Adjust CA/SA confidence based on moderator's delta
   - Update scoreboard with quality metrics

#### Phase 3: Moderator Decision (Early Termination)
- Moderator decides: **should debate continue to next round?**
- Considers:
  - Total confidence shift from initial positions
  - This round's quality scores
  - Whether threshold (20+ confidence point shift) reached
- If `should_stop=True` -> break loop early

### After All Rounds

Generate final report:
```python
Contains winner prediction, key moments, persuasion analysis.

---

## Stage 4: Export & Logging
1. **transcript.md** - Readable markdown with formatted turns (stored as `<output_dir>/transcripts/run_<id>.md`)
2. **memory.json** - Full MemoryState object (all agent states, scoreboard, debate log)
3. **final_report.json** - Moderator's summary & predictions
4. **run_config.json** - Exact config used (model names, temperatures, prompt hashes, analysis settings)
6. **debate_log.csv** - Canonical CSV format:
   ```csv
    debate_id, claim, round, speaker_role, utterance, stance, confidence, tactic_used, tool_used, tool_query, reply_to_turn
   ```

**File structure**:
```text
results/  # or <output_dir>/
├── raw/                      # Single-run data
│   └── run_20260213_154520/
├── batches/                  # Experiment suites
│   ├── ollama/
│   │   ├── raw/
│   │   │   └── run_20260213_154620/
│   │   └── all_debates_ollama.csv
│   └── gemini/
│       ├── raw/
│       │   └── run_20260213_154720/
│       └── all_debates_gemini.csv
├── analysis/                 # Visualizations and metrics
└── transcripts/              # Human-readable markdown
```

---

## Stage 5: Analysis Pipeline (Optional)

Enabled by `run_analysis = true` in config.

**analyze_run()** generates:

1. **Stance trajectory** - Confidence over time per agent (plot + data)
2. **Quality scores** - Civility, epistemics, bridge-building per round (plot)
3. **Tactic histogram** - Frequency of each rhetorical move (plot)
4. **Persuasion moments** - When confidence shifted >threshold (analysis)
5. **Redundancy detection** - Repeated arguments (TFIDF similarity)
6. **Safety flags** - Polarization alerts, bad-faith indicators

**Output**:
- **analysis_report.json** - All metrics above
- **plots/** subdirectory with PNG plots

### Advanced Analysis CLI (cli/analyze_results.py)

An optional research-grade analyzer performs sentiment, rhetorical marker, and emotion analysis:

- Reads raw runs from `results/raw` by default and writes to `results/analysis/<run_id>/`
- Outputs `enriched_debate_log.csv`, `advanced_report.json`, and plots under `plots/`
- Uses TextBlob for sentiment and a BERT-based model for emotion detection

Note: The advanced analyzer currently expects legacy CSV columns (`speaker`, `claim`) in
`debate_log.csv`; the canonical exporter writes `speaker_role` and `utterance` instead.

---

## Dependency Injection Architecture

The orchestrator no longer instantiates concrete classes directly. Instead, every
major subsystem is defined as a **Protocol** (structural typing) and injected via
a lightweight `DebateServices` container.

### Protocol Definitions (`core/protocols.py`)

| Protocol | Responsibility | Default Implementation |
|---|---|---|
| `LLMClient` | Low-level LLM calls -> Pydantic objects | `OllamaClient` |
| `StructuredLLMService` | High-level LLM with retry/recovery | `StructuredLLM` |
| `SearchProvider` | Web search + result formatting | `DuckDuckGoSearchProvider` |
| `PromptLoader` | Load & interpolate prompt templates | `FilePromptLoader` |
| `ArtifactExporter` | Write transcripts, memory, CSVs | `FileArtifactExporter` |
| `DebateAnalyzer` | Post-debate analysis pipeline | `DefaultDebateAnalyzer` |

### Service Container (`core/container.py`)

```python
@dataclass
class DebateServices:
    llm: StructuredLLMService      # Protocol (StructuredLLM)
    search: SearchProvider          # Protocol (DuckDuckGoSearchProvider)
    prompts: PromptLoader           # Protocol (FilePromptLoader)
    exporter: ArtifactExporter      # Protocol (FileArtifactExporter)
    analyzer: Optional[DebateAnalyzer] = None  # Protocol (DefaultDebateAnalyzer)
```

> [!NOTE]
> The `core/container.py` file is the only place that imports concrete adapter implementations. All other components communicate strictly via protocols.

### Factory Wiring

```python
# Default wiring - the only place that knows about concrete classes
services = build_default_services(config)

# Custom wiring for tests or alternative backends
services = DebateServices(
    llm=my_mock_llm,
    search=my_tavily_provider,
    prompts=FilePromptLoader(),
    exporter=FileArtifactExporter(),
    analyzer=None,  # skip analysis
)
```

### How the Orchestrator Uses Services

```python
def run_debate(topic, motion, rounds, config, services=None):
    if services is None:
        services = build_default_services(config)  # auto-wire defaults
    llm = services.llm
    search = services.search
    prompts = services.prompts
    # ... all LLM calls go through llm.call()
    # ... all searches go through search.search()
    # ... all prompts go through prompts.load()
    # ... artifacts written via services.exporter.write()
    # ... analysis via services.analyzer.analyze()
```

### Benefits
- **Testability**: Swap any subsystem with a mock or stub
- **Flexibility**: Plug in Tavily search, S3 export, or a new LLM provider
- **No framework dependency**: Pure stdlib `typing.Protocol` - no DI container library
- **Backward compatible**: `run_debate(topic, motion, rounds, config)` still works

---

## Data Flow Summary

```
config.ini  
    |
DebateConfig(loaded)
    |
build_default_services(config)  ->  DebateServices container
    |                                ├── llm: StructuredLLMService
Batch Runner (for each topic/model)  ├── search: SearchProvider
    |                                ├── prompts: PromptLoader
run_debate(config, services)         ├── exporter: ArtifactExporter
    |                                └── analyzer: DebateAnalyzer
Round Loop:
    ├── CA Turn (services.llm + services.search if requested)
    ├── SA Turn (services.llm + services.search if requested) 
    ├── Moderator Recap (services.llm analyzes)
    └── Moderator Decision (continue?)
    |
Memory State (conversation_history + debate_log)
    |
services.exporter.write()
    ├── transcript.md
    ├── memory.json
    ├── debate_log.csv
    └── final_report.json
    |
(Optional) services.analyzer.analyze()
    ├── metrics.csv
    └── plots/*.png
    |
Batch Runner aggregates:
    └── all_debates_gemini.csv (160 experiments when run directories are discoverable under the batch output root)
```

---

## Key Design Decisions

### Three agents per debate
- **Conspiracy Advocate (CA)**: Proponent - argues conspiracy theory
- **Scientific Advocate (SA)**: Opponent - argues scientific consensus
- **Moderator (MA)**: Neutral - controls flow, measures persuasion

### Moderator-controlled flow
- CA/SA don't directly interact; moderator manages continuation
- Prevents degeneration into unproductive loops
- Can pause if persuasion threshold reached

### Confidence as persuasion proxy
- Agents' belief shifts (0-100 scale) measure success
- Moderator tracks deltas: has each agent been persuaded?
- Enables quantitative measurement of debate quality

### Private searches
- Agents can search for evidence but results aren't visible to opponent
- Tests information-seeking behavior in isolation
- Prevents search "spoofing" (making up search results)

### Dependency injection via Protocols
- All major subsystems (LLM, search, export, prompts, analysis) are behind `Protocol` interfaces
- `DebateServices` container aggregates all dependencies
- `build_default_services(config)` wires production implementations
- Orchestrator never imports concrete classes directly -> easily testable and extensible

### Structured LLM outputs
- Pydantic schemas force valid debate turn format
- Automatic validation + retry on malformed JSON
- Enables robust multi-agent pipelines

### Resumable batch runs
- Index completion status at startup
- Skip already-completed experiments
- Retry on rate limits with exponential backoff

### Context window management (history trimming)
- Keeps large debates within token limits
- Options: trim by rounds, messages, or characters
- Preserves system messages (rules, role prompts)

### Canonical logging
- All debates recorded in standardized CSV format
- `debate_log.csv` enables cross-debate analysis
- Batch runners aggregate into `all_debates_gemini.csv` (scans run_* under the batch output directory)

---

## Running the Full Pipeline

### Quick start (single debate)
```bash
python cli/main.py
```

### Batch experiments
```bash
python cli/batch_ollama.py    # ~8-12 hours, no API needed
python cli/batch_gemini.py    # rate-limited, requires API key + quota
```

### Analysis
```python
from debate import analyze_run, analyze_all

analyze_run("results/raw/run_20260213_154520")  # Single run
analyze_all("results")  # All runs + aggregate report
```

---

## Project Structure

```text
SixSeven/
├── cli/                         # CLI entry points
│   ├── main.py                     # Single debate runner
│   ├── base_batch.py               # BaseBatchRunner (shared batch logic + DI wiring)
│   ├── analyze_results.py           # Advanced analysis CLI
│   ├── batch_ollama.py             # Ollama batch (40 experiments)
│   ├── batch_gemini.py             # Gemini batch (160 experiments, with resume)
│   └── view_topics.py              # Browse conspiracy topics
│
├── config/                     # Configuration files
│   ├── config.ini                  # Your settings (git-ignored)
│   └── config.example.ini          # Example configuration
│
├── tests/                       # Test suite (47 tests)
│   ├── test_unit/
│   │   ├── test_imports.py         # Import validation
│   │   ├── test_config.py          # DebateConfig tests
│   │   ├── test_schemas.py         # Pydantic model tests
│   │   ├── test_memory.py          # Memory state tests
│   │   ├── test_metrics.py         # Analysis metrics tests
│   │   ├── test_evaluation.py      # Evaluation function tests
│   │   └── test_search.py          # Search functionality tests
│   └── validation/
│       ├── validate_run.py         # Debate run validation
│       └── validate_memory.py      # Memory architecture validation
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md             # This file (technical design)
│   └── BATCH_GUIDE.md              # Batch experiment guide
│
├── src/debate/                  # Core library
│   ├── __init__.py                 # Package exports
│   ├── core/                       # Foundational modules
│   │   ├── config.py               # Config loading
│   │   ├── errors.py               # Shared exceptions
│   │   ├── logging.py              # Centralised logging
│   │   ├── protocols.py            # Protocol definitions (DI interfaces)
│   │   ├── container.py            # Service wiring container
│   │   ├── schemas.py              # Pydantic data models
│   │   └── topics.py               # 20 conspiracy theory topics
│   │
│   ├── simulator/                  # Consolidated simulation logic
│   │   ├── engine/                 # Orchestration & evaluation
│   │   │   ├── orchestrator.py     # Main run_debate() entry point
│   │   │   ├── evaluation.py       # Metrics & score calculation
│   │   │   └── memory.py           # State management & debate history
│   │   │
│   │   ├── providers/              # Component backends
│   │   │   ├── instructor.py       # Structured LLM orchestration
│   │   │   ├── llm_client.py       # API clients (Ollama/Gemini)
│   │   │   └── search.py           # DuckDuckGo search adapter
│   │   │
│   │   ├── io/                     # Output & persistence
│   │   │   ├── writer.py           # Multi-format artifact saver
│   │   │   ├── csv_export.py       # Canonical CSV format
│   │   │   └── templates.py        # Markdown transcript templates
│   │   │
│   │   └── prompts/                # Role-playing templates
│   │       ├── conspiracy.md       # CA persona
│   │       ├── scientific.md       # SA persona
│   │       ├── moderator.md        # Moderator persona
│   │       └── moderator_decision.md # Early stop logic
│   │
│   └── analysis/                   # Post-run analysis logic
│       ├── analysis_runner.py      # Batch analysis orchestrator
│       ├── features.py             # Feature extraction
│       ├── metrics.py              # Numerical analysis
│       ├── plots.py                # Visualizations
│       ├── report_models.py        # Report schemas
│       └── report_writer.py        # File output saving
│
├── results/                    # Experiment results (batches + analysis)
├── README.md                   # Project overview and usage
└── pyproject.toml                   # Project metadata & Poetry config
```

---

## Configuration Flow

```python
# Load from config/config.ini
config = DebateConfig.from_ini("config/config.ini")

# Wire services (production defaults)
services = build_default_services(config)

# Run with injected services
run_debate(topic, motion, rounds, config, services=services)

# Essential fields used in run_debate():
config.base_url              # Where to send API requests
config.api_mode              # "ollama", "openai", or "gemini"
config.gemini_api_key        # If using Gemini

config.moderator_model       # LLM for moderator
config.conspiracy_model      # LLM for CA
config.scientific_model      # LLM for SA

config.moderator_temperature # Randomness (moderator)
config.conspiracy_temperature
config.scientific_temperature

config.rounds                # Max debate rounds
config.word_limit            # Per-turn limit
config.max_tokens            # LLM generation max
config.max_search_rounds     # Max research rounds per turn
config.thinking_budget       # Extra tokens for thinking-capable models
config.num_ctx               # Context window size for Ollama-compatible backends

config.output_dir            # Where to save results (default: "results")

config.history_mode          # "global_full", "per_agent", or "memory_only"
config.history_trim          # "none", "rounds", "messages", or "chars"
config.max_rounds_in_history # Trim if needed
config.max_messages_in_history
config.max_chars_in_history
config.summarize_if_trimmed
config.highlight_opponent_last

config.run_analysis          # Post-debate metrics
config.analysis_shift_threshold
config.analysis_similarity_method
```

---

## Typical Experiment Lifecycle

### Day 1: Ollama Batch
```bash
# Run locally, overnight, no API limits
python cli/batch_ollama.py
# Output: results/batches/ollama/all_debates_ollama.csv (40 experiments, log rows per turn)
```

### Day 2: Gemini Batch
```bash
# Run with Gemini API during daytime (monitor quota)
python cli/batch_gemini.py
# If rate-limited: auto-resumes on next run
# Output: results/batches/gemini/all_debates_gemini.csv (160 experiments, log rows per turn)
```

### Day 3: Analysis
```python
# Combine and analyze
import pandas as pd
ollama = pd.read_csv("results/batches/ollama/all_debates_ollama.csv")
gemini = pd.read_csv("results/batches/gemini/all_debates_gemini.csv")
combined = pd.concat([ollama, gemini])

# Generate aggregate analysis
from debate import analyze_all
analyze_all("results")
# Output: results/analysis/aggregate/ + plots/
```

---

## Multi-LLM Support

The system abstracts LLM backends:

```python
# Ollama (local)
- Uses OpenAI-compatible /v1 endpoint
- No API keys needed
- Free, offline-capable

# Gemini (cloud)
- Google's generative AI API
- Requires API key
- Rate-limited (free tier)

# OpenAI-compatible endpoint
- Same /v1 schema as Ollama
- Used when api_mode="openai" and base_url points to a compatible server
```

---

## Debugging & Monitoring

### Check current run progress
```bash
# Batch runner prints progress to console
# Check output_dir for partial results
ls -la results/batches/ollama/raw/run_*/
ls -la results/batches/gemini/raw/run_*/
```

### Inspect a single debate
```bash
# View readable transcript (runs are nested under type/transcripts)
cat results/batches/ollama/transcripts/run_20260213_154520.md

# Check raw memory state
cat results/batches/ollama/raw/run_20260213_154520/memory.json | python -m json.tool
```

### Resume failed batch
```bash
# Batch runners skip already-completed experiments
python cli/batch_gemini.py
# Only runs new topic+config combinations
```

---

End-to-end: from `config/config.ini` through orchestration, simulation, export, and analysis. The system is built for reproducibility (seeding), resumability (completion index), and observability (comprehensive logging.
