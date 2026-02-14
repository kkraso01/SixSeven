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
- **Models**: Two Ollama-compatible models (configured in batch script)
- **Configurations**: 8 model combos (all permutations of 2 models across 3 roles)
- **Experiments**: 20 topics × 8 configs = **160 experiments**
- **API Calls**: ~3,200 (all to Ollama endpoint, no limits)
- **Time**: 8-12 hours
- **Output**: `<output_dir>/all_debates_ollama.csv`
- **Features**: Auto-resume on interruption (completion index)

### Gemini Batch (`cli/batch_gemini.py`)
- **Models**: One Gemini model + one Ollama model (mixed configs)
- **Configurations**: 8 model combos (all permutations across 3 roles)
- **Experiments**: 20 topics × 8 configs = **160 experiments**
- **API Calls**: ~1,200-1,400 (varies by how many roles use Gemini)
- **Time**: Multiple days (free-tier rate limits: 5 req/min, 20 req/day)
- **Output**: `<output_dir>/all_debates_gemini.csv`
- **Features**: Auto-resume, per-minute retry with backoff, daily quota detection

### Combined Total
- **320 experiments** (160 Ollama + 160 Gemini)
- **~1,600 total debate rounds** (5 rounds each)
- **Two separate CSV files** that can be analyzed together

---

## Architecture

```
BaseBatchRunner (cli/base_batch.py)
├── Topic loading (CONSPIRACY_TOPICS from core/topics.py)
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
```

---

## How to Run

### Step 1: Run Ollama Batch First (No API Needed)

```bash
python cli/batch_ollama.py
```

**All 8 model configurations** (all permutations of Model A × Model B across 3 roles):
1. model-A-all — Model A for all 3 roles
2. model-B-all — Model B for all 3 roles
3. A-mod-B-agents — A moderator, B agents
4. B-mod-A-agents — B moderator, A agents
5. A-mod-ca_B-sa — A mod+CA, B SA
6. A-mod-sa_B-ca — A mod+SA, B CA
7. B-mod-sa_A-ca — B mod+SA, A CA
8. B-mod-ca_A-sa — B mod+CA, A SA

Edit `cli/batch_ollama.py` to set your specific model names in `runner.add_model_config()` calls.

**Output:**
- `<output_dir>/batch_summary_ollama.json`
- `<output_dir>/all_debates_ollama.csv`
- Individual run folders with transcripts

**Estimated time:** 8-12 hours (no rate limits)

### Step 2: Run Gemini Batch When API Quota Available

```bash
python cli/batch_gemini.py
```

**Requirements:**
- ✅ Gemini API key in `config/config.ini`
- ✅ API quota available (free tier: 20 req/day, 5 req/min)

**All 8 model configurations** (same permutation pattern, with Gemini + Ollama models):
1. gemini-all — Gemini for all 3 roles (3 API calls/round)
2. ollama-all — All Ollama (0 API calls — baseline comparison)
3. gemini-mod-ollama-agents — Gemini moderator, Ollama agents (1 API call/round)
4. ollama-mod-gemini-agents — Ollama moderator, Gemini agents (2 API calls/round)
5. gemini-mod-ca_ollama-sa — Gemini mod+CA, Ollama SA (2 API calls/round)
6. gemini-mod-sa_ollama-ca — Gemini mod+SA, Ollama CA (2 API calls/round)
7. ollama-mod-sa_gemini-ca — Ollama mod+SA, Gemini CA (1 API call/round)
8. ollama-mod-ca_gemini-sa — Ollama mod+CA, Gemini SA (1 API call/round)

Edit `cli/batch_gemini.py` to set your specific model names.

**Output:**
- `<output_dir>/batch_summary_gemini.json`
- `<output_dir>/all_debates_gemini.csv`
- Individual run folders with transcripts

**Estimated time:** Multiple days on free tier (rate-limited), 5-8 hours on paid tier

**Rate-limit behaviour:**
- Per-minute limit → automatic retry with exponential backoff (never gives up)
- Daily quota exhaustion → saves progress and stops; re-run tomorrow to continue

### Step 3: Analyze Combined Results

Both CSV files have the same format, so you can combine them:

```python
import pandas as pd

# Load both datasets
ollama_df = pd.read_csv("<output_dir>/all_debates_ollama.csv")
gemini_df = pd.read_csv("<output_dir>/all_debates_gemini.csv")

# Combine them
all_debates = pd.concat([ollama_df, gemini_df], ignore_index=True)

# Export combined
all_debates.to_csv("all_debates_combined.csv", index=False)

# Analyze by model type
ollama_debates = all_debates[all_debates.debate_id.str.contains("gemma|qwen|dolphin")]
gemini_debates = all_debates[all_debates.debate_id.str.contains("gemini")]
```

---

## Expected API Usage

### Ollama (University Server)
- **Per experiment**: ~20 API calls (5 rounds × ~4 calls/round)
- **Total**: 160 experiments × 20 = **~3,200 calls**
- **Rate limit**: None
- **Cost**: Free

### Gemini (Google API)
- **Per experiment**: 0-20 API calls (depends on how many roles use Gemini)
- **Total**: ~1,200-1,400 calls (varies by config)
- **Free-tier limits**: 5 req/min, 20 req/day
- **Paid-tier limits**: 15 req/min, 1,500 req/day
- **Cost**: Free tier available; check your usage

---

## Optimization Tips

### If You Hit Rate Limits

**Option 1: Reduce topics**
```python
# In cli/batch_gemini.py, change topics line
from debate_sim.core.topics import get_sample_topics
topics = get_sample_topics(10)  # Only 10 topics instead of 20
# 10 × 8 = 80 experiments = ~600 API calls
```

**Option 2: Fewer model configs**
Keep only the configs you care about:
```python
model_configs = [
    runner.add_model_config(
        name="gemini-all",
        moderator="<your-gemini-model>",
        conspiracy="<your-gemini-model>",
        scientific="<your-gemini-model>",
        api_mode="gemini",
    ),
]
# 20 × 1 = 20 experiments = ~400 API calls
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
<output_dir>/                           Ollama batch
├── batch_summary_ollama.json           Summary with stats
├── all_debates_ollama.csv              160 Ollama debates
├── run_20260214_120000/                Individual debate 1
├── run_20260214_120500/                Individual debate 2
└── ...

<output_dir>/                           Gemini batch
├── batch_summary_gemini.json           Summary with stats
├── all_debates_gemini.csv              160 Gemini debates
├── run_20260214_180000/                Individual debate 1
└── ...
```

---

## Checking Progress

### During Ollama Batch
```bash
# Count completed experiments
ls -d <output_dir>/run_* | wc -l

# Check batch summary
cat <output_dir>/batch_summary_ollama.json | python -m json.tool | grep successful
```

### During Gemini Batch
```bash
# Watch for rate limit errors
cat <output_dir>/batch_summary_gemini.json | python -m json.tool | grep -i "fail\|quota"

# Monitor progress
ls -d <output_dir>/run_* | wc -l
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
