# Changelog

All notable changes to this project are documented here. This project adheres to [Semantic Versioning](https://semver.org/).

For quick start, installation, troubleshooting, and common tasks, see:
- **[README.md](README.md)** — Quick start, installation, configuration, commands, troubleshooting
- **[BATCH_GUIDE.md](BATCH_GUIDE.md)** — Batch experiment strategy, API estimates, model configurations

---

## [0.3.0] - 2026-02-14

**Status**: Import Path Migration & Package Configuration

### Changed

#### Import Path Migration
- **All imports** changed from `from src.debate_sim` → `from debate_sim` across 16 Python files (cli/, tests/)
- This follows the standard Python **src layout** convention where `src/` is a directory container, not a package
- Affected files: `cli/main.py`, `cli/base_batch.py`, `cli/batch_gemini.py`, `cli/batch_ollama.py`, `cli/view_topics.py`, plus all test files

#### Package Configuration
- Added `packages = [{include = "debate_sim", from = "src"}]` to `pyproject.toml` so the package is properly installable
- `debate_sim` is now importable after `pip install -e .`

#### Type Safety
- Made `generate()` and `call()` generic with `TypeVar("T", bound=BaseModel)`
- Replaced `conint()` with `Annotated[int, Field()]` (Pydantic v2)
- Added explicit type annotations, `cast()` calls, and `Literal` type aliases

#### Code Linting
- Fixed import sorting, typing modernization, import ordering, raise-from, unused vars, whitespace
- Removed manual `sys.path` hacks from `test_imports.py` and `validate_memory.py`

### Removed
- `sys.path` manipulation in `tests/test_unit/test_imports.py` and `tests/validation/validate_memory.py` (no longer needed)

---

## [0.2.0] - 2025-06-27

**Status**: Maintainability & Code Quality Improvements

### Added

#### Testing
- `tests/test_unit/test_config.py` — 6 tests for `DebateConfig` (defaults, INI parsing, quote stripping, invalid values)
- `tests/test_unit/test_schemas.py` — 13 tests for Pydantic validators and `_rename_keys`/`_fix_search_field` helpers
- `tests/test_unit/test_memory.py` — 9 tests for memory state management (immutability, confidence updates, scoreboard)
- `tests/test_unit/test_metrics.py` — 8 tests for analysis metrics (stance summary, tactic counting, safety flags)
- `tests/test_unit/test_evaluation.py` — 4 tests for evaluation functions (stance shift, metrics table)

#### Centralised Logging
- `src/debate_sim/core/logging.py` — `setup_logging()` with idempotent guard, configurable directory/file/level, file + console handlers

#### Developer Tooling
- `pyproject.toml` — Added dev tool configuration sections

### Changed

#### Code Quality
- **Extracted constants from magic numbers**: `ROLE_TO_AGENT` (metrics.py), `DEFAULT_INITIAL_CONFIDENCE` (models.py), `SEARCH_MAX_RESULTS`/`SEARCH_MAX_CHARS`/`MESSAGES_PER_ROUND`/`FINAL_REPORT_MAX_TOKENS`/`LLM_MAX_RETRIES` (orchestrator.py), `_HTTP_TIMEOUT`/`_OPENAI_READ_TIMEOUT`/`_OPENAI_CONNECT_TIMEOUT`/`_GEMINI_MIN_TOKENS`/`_RATE_LIMIT_EXTRA_DELAY`/`_RATE_LIMIT_FALLBACK_DELAY` (ollama_client.py), `EXCERPT_MAX_CHARS` (features.py)
- **Extracted shared validator helpers**: `_rename_keys()` and `_fix_search_field()` in `schemas.py` — replaced 6 inline rename loops
- **Replaced ~30 `print()` calls** with `logging.getLogger(__name__)` in `orchestrator.py`
- **Consolidated logging setup**: `cli/main.py` and `cli/base_batch.py` now use `setup_logging()` instead of duplicate `logging.basicConfig()`
- **Fixed type hints**: `Optional[str] = None` instead of `str = None`, `Optional[Path]` instead of `Path | None`
- **Added module & function docstrings** across 5+ modules (metrics.py, evaluation.py, models.py, report_models.py, features.py)

#### Bug Fixes
- **Config path**: Default changed from `"config.ini"` to `"config/config.ini"` (matches actual file location)
- **Quoted INI values**: `get_str()` now strips surrounding quotes (fixes `output_dir = "kalamaras-artifacts"` being parsed with literal quotes)

#### Naming
- **Renamed `PersuasionMoment`** in `report_models.py` → `PersuasionFlag` to disambiguate from the LLM-output `PersuasionMoment` in `schemas.py`

### Removed
- Duplicate dependencies: `duckduckgo-search` (kept `ddgs`), `google-generativeai` (kept `google-genai`)
- Relaxed `requires-python` from `>=3.13` to `>=3.11` (no 3.13-specific features used)

---

## [1.0.0] - 2026-02-14

**Status**: ✅ Code Complete (Core Systems Ready)

### Added

#### Dependency Injection Architecture
- `src/debate_sim/core/protocols.py` — 6 Protocol ABCs for extensible design:
  - `LLMClient` — Low-level LLM calls
  - `StructuredLLMService` — LLM with structured output and retry logic
  - `SearchProvider` — Web search abstraction
  - `PromptLoader` — Template file loading
  - `ArtifactExporter` — Debate artifact writing
  - `DebateAnalyzer` — Analysis pipeline interface
- `src/debate_sim/core/container.py` — `DebateServices` container dataclass with 4 adapter implementations and `build_default_services(config)` factory
- `src/debate_sim/core/errors.py` — Shared exception types (`LLMResponseError`)
- `cli/base_batch.py` — `BaseBatchRunner` base class providing shared batch logic (topic loading, completion index, DI wiring, CSV aggregation)

### Changed

#### Refactoring for Dependency Injection
- `src/debate_sim/debate/orchestrator.py` — `run_debate()` now accepts optional `services: DebateServices`; all concrete class imports replaced with protocol-based injection
- `src/debate_sim/__init__.py` — Exports `DebateServices`, `build_default_services`, and all 6 protocols
- `src/debate_sim/core/__init__.py` — Added DI exports; fixed pre-existing `DebateTopic` import bug
- `cli/main.py` — Wires `build_default_services(config)` and passes to `run_debate()`
- `src/debate_sim/export/csv_export.py` — Fixed pre-existing broken import path

#### Code Reduction
- `batch_ollama.py` — Reduced from 482 → 125 lines (thin `OllamaBatchRunner` subclass with shared DI wiring)
- `batch_gemini.py` — Reduced from 601 → 267 lines (`GeminiBatchRunner` with rate-limit retry and resume capability)
- `LLMResponseError` — Extracted from `ollama_client.py` → `core/errors.py` (re-exported from `ollama_client` for backward compatibility)
- `instructor_wrapper.py` — Now imports `LLMResponseError` from `core.errors` directly

### Removed

#### Dead Code Cleanup
- `src/debate_sim/debate/protocol.py` — Unused `TACTICS`, `PROMPT_DIR`, `load_prompt()` superseded by `FilePromptLoader` in DI layer
- `src/debate_sim/memory/store.py` — `save_memory()` never called; export handled by `export/writer.py`

#### Unused Functions
- `orchestrator.py`: `_format_transcript()`, `_agent_messages()`, `_memory_summary()` → superseded by `_build_agent_messages_with_history()` and `_compact_memory_summary()`
- `evaluation.py`: `update_metrics()` → never called
- `search_tool.py`: `generate_conspiracy_query()`, `generate_scientific_query()` → never called

#### Unused Imports & Variables
- `orchestrator.py`: `import json`, `from collections import Counter`, `StructuredLLMService` import, dead `tactic_counts` variable
- `evaluation.py`: `from collections import Counter`
- `container.py`: unused `Type` (from typing) and `BaseModel` (from pydantic)
- `cli/main.py`: dead `logger` variable
- `cli/view_topics.py`: unused `get_topic_by_id` import

### Fixed

- `tests/test_unit/test_imports.py`:
  - Moved from `tests/` to `tests/test_unit/` (better organization)
  - Fixed imports: `from src.debate_sim.config` → `from src.debate_sim.core.config`
  - Fixed imports: `from src.debate_sim.schemas` → `from src.debate_sim.core.schemas`
  - Removed import of deleted `debate.protocol`
- `tests/validation/validate_run.py`: Fixed path `from debate_sim.` → `from src.debate_sim.`

### Notes on Backward Compatibility
- **Fully backward compatible**: Calling `run_debate(topic, motion, rounds, config)` without `services` parameter auto-wires default implementations
- Explicit service wiring available through `DebateServices` container for custom implementations


---

## Verification & Status

### Test Results
- ✅ 47/47 unit tests pass
- ✅ 0 type errors across 29 source files
- ✅ All lint checks passed

### Core Features Verified
- ✅ Moderator-controlled debate flow with decision-making
- ✅ Multi-model support (Ollama local, Google Gemini API, DuckDuckGo search)
- ✅ Canonical debate log format with JSON schema
- ✅ 20 conspiracy theory topics dataset
- ✅ Batch experiment runner (configurable per runner, 20–160 experiments)
- ✅ CSV export and analysis pipeline
- ✅ Protocol-based dependency injection (fully extensible)

---

## Dependencies

### Pinned Versions (February 14, 2026)
```
pydantic==2.12.5
instructor==1.14.5
httpx==0.28.1
openai==2.16.0
ddgs>=9.10.0,<10.0.0
google-genai>=1.63.0,<2.0.0
matplotlib==3.10.8
scikit-learn==1.8.0
jinja2==3.1.6
rich==14.3.2
```
All versions locked in `requirements.txt`. See [README.md - Installation](README.md#-installation) for setup.

---

## Known Issues

### ⚠️ Non-Critical Warnings
- **Issue**: `FutureWarning` from instructor about google.generativeai deprecation
  - **Status**: Package still works; warning from instructor library
  - **Impact**: None — functionality unaffected
  - **Action**: Waiting for instructor package update

### 📌 Not Implemented (By Design)
- Language analysis features (explicitly excluded per user request)
- Advanced statistical analysis (future work)

---

## References

- Repository: [kkraso01/SixSeven](https://github.com/kkraso01/SixSeven)
- Branch: `suprdev` (default: `main`)
- Keep a Changelog: [https://keepachangelog.com/](https://keepachangelog.com/)
- Semantic Versioning: [https://semver.org/](https://semver.org/)

