# Split Batch Experiment Guide

## Why Split?

To avoid hitting Gemini API rate limits (15 req/min, 1,500 req/day), the batch experiments are split into two separate runs:

1. **Ollama Batch** - No API limits, can run 24/7
2. **Gemini Batch** - Respects API limits, runs with delays

##  Batch Statistics

### Ollama Batch (`batch_runner_ollama.py`)
- **Models**: gemma3:27b, dolphin-yi-34b (both on university server)
- **Configurations**: 3 model configs
- **Experiments**: 20 topics  3 configs = **60 experiments**
- **API Calls**: ~1,200 (all to university server, no limits)
- **Time**: 3-5 hours
- **Output**: `all_debates_ollama.csv`

### Gemini Batch (`batch_runner_gemini.py`)
- **Models**: gemini-2.5-pro (Google API)
- **Configurations**: 3 model configs (including mixed)
- **Experiments**: 20 topics  3 configs = **60 experiments**
- **API Calls**: ~1,200 (to Gemini API)
- **Time**: 5-8 hours (with rate limiting)
- **Output**: `all_debates_gemini.csv`

### Combined Total
- **120 experiments**
- **240 total debate rounds** (5 rounds each)
- **Two separate CSV files** that can be analyzed together

##  How to Run

### Step 1: Run Ollama Batch First (No API Needed)

```powershell
python batch_runner_ollama.py
```

**Models tested:**
-  gemma3-27b-all (Gemma3 for all roles)
-  dolphin-yi-34b-all (Dolphin Yi uncensored for all roles)
-  gemma-mod-dolphin-agents (Mixed: Gemma moderator, Dolphin agents)

**Output:**
- `artifacts/batch_summary_ollama.json`
- `artifacts/all_debates_ollama.csv`
- Individual run folders with transcripts

**Estimated time:** 3-5 hours (no rate limits)

### Step 2: Run Gemini Batch When API Quota Available

```powershell
python batch_runner_gemini.py
```

**Requirements:**
-  Gemini API key in config.ini
-  API quota available (1,500 calls/day)

**Models tested:**
-  gemini-2.5-pro-all (Gemini 2.5 Pro for all roles)
-  gemini-pro-mod-gemma-agents (Gemini moderator, Gemma agents)
-  gemma-mod-gemini-pro-agents (Gemma moderator, Gemini agents)

**Output:**
- `artifacts/batch_summary_gemini.json`
- `artifacts/all_debates_gemini.csv`
- Individual run folders with transcripts

**Estimated time:** 5-8 hours (with rate limiting)

### Step 3: Analyze Combined Results

Both CSV files have the same format, so you can:

```python
import pandas as pd

# Load both datasets
ollama_df = pd.read_csv("artifacts/all_debates_ollama.csv")
gemini_df = pd.read_csv("artifacts/all_debates_gemini.csv")

# Combine them
all_debates = pd.concat([ollama_df, gemini_df], ignore_index=True)

# Export combined
all_debates.to_csv("artifacts/all_debates_combined.csv", index=False)

# Analyze by model type
ollama_debates = all_debates[all_debates.debate_id.str.contains("gemma|dolphin")]
gemini_debates = all_debates[all_debates.debate_id.str.contains("gemini")]
```

##  Expected API Usage

### Ollama (University Server)
- **Per experiment**: ~20 API calls (5 rounds  ~4 calls/round)
- **Total**: 60 experiments  20 = **1,200 calls**
- **Rate limit**: None
- **Cost**: Free

### Gemini (Google API)
- **Per experiment**: ~20 API calls (5 rounds  ~4 calls/round)
- **Total**: 60 experiments  20 = **1,200 calls**
- **Rate limit**: 15 req/min, 1,500 req/day
- **Fits in quota**:  Just under daily limit (1,200 < 1,500)
- **Cost**: Free tier (check your usage)

##  Optimization Tips

### If You Hit Rate Limits

**Option 1: Reduce topics**
```python
# In batch_runner_gemini.py, line ~280
from src.debate_sim.topics import get_sample_topics
topics = get_sample_topics(10)  # Only 10 topics instead of 20
# 10  3 = 30 experiments = 600 API calls
```

**Option 2: Remove mixed configs**
```python
# Keep only the main Gemini config
model_configs = [
    runner.add_model_config(
        name="gemini-2.5-pro-all",
        moderator="gemini-2.5-pro",
        conspiracy="gemini-2.5-pro",
        scientific="gemini-2.5-pro",
        api_mode="gemini",
    ),
]
# 20  1 = 20 experiments = 400 API calls
```

**Option 3: Split Gemini batch into 2 days**
```python
# Day 1: First 10 topics
topics = CONSPIRACY_TOPICS[:10]

# Day 2: Last 10 topics  
topics = CONSPIRACY_TOPICS[10:]
```

##  Recommended Workflow

1. **Night 1**: Run `batch_runner_ollama.py` overnight (3-5 hours)
2. **Morning**: Check results in `all_debates_ollama.csv`
3. **Day 2**: Run `batch_runner_gemini.py` during the day (monitor progress)
4. **Evening**: Combine and analyze both datasets

This avoids API quota issues and gives you time to check intermediate results.

##  Output Structure

```
artifacts/
 batch_summary_ollama.json       Ollama batch results
 batch_summary_gemini.json       Gemini batch results
 all_debates_ollama.csv          60 Ollama debates
 all_debates_gemini.csv          60 Gemini debates
 run_20260207_120000/            Individual debate 1
 run_20260207_120500/            Individual debate 2
 ...                             120 total run folders
```

##  Checking Progress

### During Ollama Batch
```powershell
# Count completed experiments
Get-ChildItem artifacts/run_* | Measure-Object

# Check last experiment
Get-Content artifacts/batch_summary_ollama.json | Select-String "successful"
```

### During Gemini Batch
```powershell
# Watch for rate limit errors
Get-Content artifacts/batch_summary_gemini.json | Select-String "failed"

# Monitor progress
Get-ChildItem artifacts/run_* | Measure-Object
```

##  Troubleshooting

**If Ollama batch fails:**
- Check university server connection: `https://chatucy.cs.ucy.ac.cy/v1`
- Verify VPN if required
- Check if dolphin-yi-34b is available on server

**If Gemini batch fails:**
- Verify API key in config.ini
- Check API quota: https://aistudio.google.com/app/apikey
- Look for rate limit errors in batch_summary_gemini.json
- Try reducing topics or configs

**If both batches succeed but experiments fail:**
- Check individual run folders for transcripts
- Look at final_report.json for error details
- Review memory.json to see where debate stopped

##  Next Steps After Completion

1. Combine CSVs
2. Analyze confidence trajectories
3. Compare model behaviors (Gemma vs Gemini vs Dolphin)
4. Identify which topics changed minds most
5. Examine tactic usage patterns
6. Look for censorship differences (Dolphin Yi vs others)

Good luck with your experiments! 
