# Multi-Agent Debate Simulator

A structured, reproducible debate simulator for NLP research that runs fully on a **local Ollama server** and uses **Instructor + Pydantic** schemas for all agent outputs.

## Requirements
- Python 3.11+
- Ollama running locally (default `http://localhost:11434`)

## Setup
```bash
pip install -e .
```

Start Ollama and pull a model (example):
```bash
ollama serve
ollama pull llama3.1:8b
```

## Running from Python
```python
from main import run_experiment

result = run_experiment()
print(result.run_dir)
```

## Analysis (Python API)
```python
from debate_sim.analysis.analysis_runner import analyze_all, analyze_run

report = analyze_run("artifacts/run_20240101_120000")
aggregate = analyze_all("artifacts")
```

By default, analysis runs automatically after each debate. Disable with `DEBATE_RUN_ANALYSIS=0`.

## Configuration
Configuration comes from environment variables (defaults shown):
- `DEBATE_BASE_URL` (default: `http://localhost:11434`)
- `DEBATE_API_MODE` (default: `ollama`, or `openai` for `/v1/chat/completions`)
- `DEBATE_MODERATOR_MODEL` (default: `llama3.1:8b`)
- `DEBATE_CONSPIRACY_MODEL` (default: `llama3.1:8b`)
- `DEBATE_SCIENTIFIC_MODEL` (default: `llama3.1:8b`)
- `DEBATE_ROUNDS` (default: `3`)
- `DEBATE_WORD_LIMIT` (default: `180`)
- `DEBATE_MAX_TOKENS` (default: `600`)
- `DEBATE_OUTPUT_DIR` (default: `artifacts`)
- `DEBATE_SEED` (optional)
- `DEBATE_RUN_ANALYSIS` (default: `True`)
- `DEBATE_ANALYSIS_SHIFT_THRESHOLD` (default: `5`)
- `DEBATE_ANALYSIS_SIMILARITY_METHOD` (default: `tfidf`)

## Outputs
Each run creates a timestamped directory under `artifacts/` with:
- `transcript.md`
- `memory.json`
- `final_report.json`
- `metrics.csv` (if metrics exist)
- `analysis_report.md`
- `analysis_report.json`
- `figures/*.png`
- `run_config.json`

## Validation
```bash
python scripts/validate_run.py artifacts/run_20240101_120000
```

## Prompt Variants
Edit files in `src/debate_sim/prompts/` to iterate on agent behavior. Keep the final line:
> Return ONLY valid JSON matching the schema. No extra text.
