
# Split Batch Experiment Guide

## Why Split?

To avoid hitting Gemini API rate limits (5 req/min, 20 req/day on free tier), the batch experiments are split into two separate runs:

1. **Ollama Batch** — No API limits, can run 24/7
2. **Gemini Batch** — Respects API limits, retries with exponential backoff

Both runners inherit from `cli/base_batch.py → BaseBatchRunner`, which
handles shared orchestration: topic loading, completion indexing (auto-resume),
DI wiring via `build_default_services(config)`, CSV aggregation, and progress
reporting.

---

## Batch Statistics

### Ollama Batch (`cli/batch_ollama.py`)
- **Models**: Two Ollama-compatible models (configured via `config.ini`, wired in the batch script)
- **Configurations**: 2 model combos (original + swapped CA/SA)
- **Experiments**: 20 topics × 2 configs = **40 experiments**
- **API Calls**: ~800 (5 rounds × ~4 calls/round × 40 experiments; depends on config/search)
- **Time**: 3-5 hours (depends on hardware)
- **Output**: `results/batches/ollama/all_debates_ollama.csv`
- **Features**: Auto-resume on interruption (completion index)

### Gemini Batch (`cli/batch_gemini.py`)
- **Models**: Gemini (gemini-3-flash-preview) + Ollama models (mixed configs)
- **Configurations**: 8 model combos (all permutations across 3 roles)
- **Experiments**: 20 topics × 8 configs = **160 experiments**
- **API Calls**: ~1,200-1,400 (varies by how many roles use Gemini)
- **Time**: Multiple days (free-tier rate limits: 5 req/min, 20 req/day)
- **Output**: `results/batches/gemini/all_debates_gemini.csv`
- **Features**: Auto-resume, per-minute retry with backoff, daily quota detection

### Combined Total
- **200 experiments** (40 Ollama + 160 Gemini)
- **~1,000 total debate rounds** (5 rounds each in Gemini; Ollama uses config.ini rounds)
- **Two separate CSV files** that can be analyzed together

---

## Architecture

```
BaseBatchRunner (cli/base_batch.py)
├── Topic loading (from config/topics.json via core/topics.py)
├── Completion index (_build_completion_index)
├── DI wiring (build_default_services → DebateServices)
├── CSV aggregation (export_all_debates_to_csv)
└── Progress/summary reporting
    │
    ├── OllamaBatchRunner (cli/batch_ollama.py)
    │   └── No retry logic needed (no API limits)
    │
    └── GeminiBatchRunner (cli/batch_gemini.py)
        ├── Per-minute rate-limit retry (exponential backoff)
        ├── Daily quota detection → graceful batch stop
        └── Retry delay extraction from error messages

Note: The combined CSV aggregation scans `run_*` under `<output_dir>/raw`, which
matches the current batch layout.
```

---

## How to Run

### Step 1: Run Ollama Batch First (No API Needed)

```bash
python cli/batch_ollama.py
```

**All 2 model configurations** (original + swapped CA/SA):
Dynamically loaded from the `ollama` section of `config/model_pool.json`.

Edit `config/model_pool.json` to change the specific model names and permutations used for this batch.

**Output:**
- `<output_dir>/batch_summary_ollama.json`
- `<output_dir>/batch_report_ollama.md`
- `<output_dir>/all_debates_ollama.csv`
- `<output_dir>/raw/run_<id>/...` and `<output_dir>/transcripts/run_<id>.md`

**Estimated time:** 8-12 hours (no rate limits)

### Step 2: Run Gemini Batch When API Quota Available

```bash
python cli/batch_gemini.py
```

**Requirements:**
-  Gemini API key in `config/config.ini`
-  API quota available (free tier: 20 req/day, 5 req/min)

**All 8 model configurations** (as defined in `config/model_pool.json`):
The permutations involve mixing `gemini` models with `ollama` models across the Moderator, CA, and SA roles.
- `gemini-3-flash-all` (3 API calls/round)
- `gemma3-27b-all` (0 API calls/round)
- Various hybrid combinations (1-2 API calls/round)

Edit the `gemini` section of `config/model_pool.json` to configure these specific permutations and model names.

**Output:**
- `results/batches/gemini/batch_summary_gemini.json`
- `results/batches/gemini/batch_report_gemini.md`
- `results/batches/gemini/all_debates_gemini.csv`
- `results/batches/gemini/raw/run_<id>/...` and `results/batches/gemini/transcripts/run_<id>.md`

**Estimated time:** Multiple days on free tier (rate-limited), 5-8 hours on paid tier

**Rate-limit behaviour:**
- Per-minute limit → automatic retry with exponential backoff (never gives up)
- Daily quota exhaustion → saves progress and stops; re-run tomorrow to continue

### Step 3: Analyze Combined Results

Both CSV files have the same format, so you can combine them:

```python
import pandas as pd

# Load both datasets
ollama_df = pd.read_csv("results/batches/ollama/all_debates_ollama.csv")
gemini_df = pd.read_csv("results/batches/gemini/all_debates_gemini.csv")

# Combine them
all_debates = pd.concat([ollama_df, gemini_df], ignore_index=True)

# Export combined
all_debates.to_csv("all_debates_combined.csv", index=False)

# Analyze by model type
ollama_debates = all_debates[all_debates.debate_id.str.contains("gemma|qwen|dolphin")]
gemini_debates = all_debates[all_debates.debate_id.str.contains("gemini")]

Note: The canonical CSV uses `speaker_role` and `utterance` columns (not `speaker` or `claim`).
```

---

## Expected API Usage

### Ollama (University Server)
- **Per experiment**: ~20 API calls (5 rounds × ~4 calls/round; depends on config/search)
- **Total**: 40 experiments × 20 = **~800 calls**
- **Rate limit**: None
- **Cost**: Free

### Gemini (Google API)
- **Per experiment**: 0-20 API calls (depends on how many roles use Gemini)
- **Total**: ~1,200-1,400 calls (varies by config)
- **Free-tier limits**: 5 req/min, 20 req/day
- **Paid-tier limits**: Depends on your plan
- **Cost**: Free tier available; check your usage

---

## Optimization Tips

### If You Hit Rate Limits

**Option 1: Reduce topics**
```python
# In cli/batch_gemini.py, change topics line
from debate.core.topics import get_sample_topics
topics = get_sample_topics(10)  # Only 10 topics instead of 20
# 10 × 8 = 80 experiments = ~600 API calls
```

**Option 2: Fewer model configs**
Keep only the configs you care about by editing `config/model_pool.json`. Simply remove or comment out configurations from the `gemini` list:
```json
{
  "gemini": [
    {
      "name": "gemini-all",
      "moderator": "gemini-3-flash-preview",
      "conspiracy": "gemini-3-flash-preview",
      "scientific": "gemini-3-flash-preview",
      "api_mode": "gemini"
    }
    // Remove the other 7 combos to speed up execution
  ]
}
```

**Option 3: Split across days**
Just re-run the same command — the completion index skips already-finished
experiments automatically:
```bash
# Day 1: runs until quota exhausted, saves progress
python cli/batch_gemini.py

# Day 2: picks up where it left off
python cli/batch_gemini.py
```

---

## Recommended Workflow

1. **Night 1**: Run `python cli/batch_ollama.py` overnight (8-12 hours)
2. **Morning**: Check results in `all_debates_ollama.csv`
3. **Day 2+**: Run `python cli/batch_gemini.py` (auto-resumes across days)
4. **When complete**: Combine and analyze both datasets

---

## Output Structure

```
results/batches/ollama/               Ollama batch (default output_dir)
├── batch_summary_ollama.json           Summary with stats
├── batch_report_ollama.md              Detailed report
├── all_debates_ollama.csv              40 Ollama debates
├── raw/run_20260214_120000/            Individual debate 1 (raw)
├── transcripts/run_20260214_120000.md  Individual debate 1 (transcript)
└── ...

results/batches/gemini/                Gemini batch (default output_dir)
├── batch_summary_gemini.json           Summary with stats
├── batch_report_gemini.md              Detailed report
├── all_debates_gemini.csv              160 Gemini debates
├── raw/run_20260214_180000/            Individual debate 1 (raw)
├── transcripts/run_20260214_180000.md  Individual debate 1 (transcript)
└── ...
```

---

## Checking Progress

### During Ollama Batch
```bash
# Count completed experiments (raw runs)
ls -d results/batches/ollama/raw/run_* | wc -l

# Check batch summary
cat results/batches/ollama/batch_summary_ollama.json | python -m json.tool | grep successful
```

### During Gemini Batch
```bash
# Watch for rate limit errors
cat results/batches/gemini/batch_summary_gemini.json | python -m json.tool | grep -i "fail\|quota"

# Monitor progress
ls -d results/batches/gemini/raw/run_* | wc -l
```

---

## Troubleshooting

**If Ollama batch fails:**
- Check your Ollama endpoint is reachable (default: `http://localhost:11434`)
- Verify models are pulled: `ollama list`

**If Gemini batch fails:**
- Verify API key in `config/config.ini`
- Check API quota: https://aistudio.google.com/app/apikey
- Check `batch_summary_gemini.json` for rate limit errors
- Try reducing topics or configs (see Optimization Tips above)
- Re-run the same command — it auto-resumes from where it stopped

**If both batches succeed but individual experiments fail:**
- Check individual run folders for transcripts
- Look at `final_report.json` for error details
- Review `memory.json` to see where debate stopped

---

## Next Steps After Completion

1. Combine CSVs (`all_debates_ollama.csv` + `all_debates_gemini.csv`)
2. Analyze confidence trajectories
3. Compare model behaviours across configurations
4. Identify which topics changed minds most
5. Examine tactic usage patterns
6. Look for censorship differences across models
