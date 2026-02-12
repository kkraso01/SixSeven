"""Batch experiment runner - GEMINI MODELS (requires API key)."""

from __future__ import annotations

import os
import json
import re
import time
import warnings
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Force UTF-8 on Windows to prevent charmap codec errors
os.environ.setdefault("PYTHONUTF8", "1")

# Suppress FutureWarning from instructor's deprecated google.generativeai import
warnings.filterwarnings("ignore", category=FutureWarning, module="instructor.providers.gemini.client")

from src.debate_sim import DebateConfig, run_debate
from src.debate_sim.topics import CONSPIRACY_TOPICS, DebateTopic, get_sample_topics
from src.debate_sim.export.csv_export import export_all_debates_to_csv


@dataclass
class ModelConfig:
    """Configuration for a model combination."""
    name: str
    moderator: str
    conspiracy: str
    scientific: str
    api_mode: str = "openai"
    base_url: str = "https://chatucy.cs.ucy.ac.cy/v1"
    gemini_api_key: Optional[str] = None


@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    topic_id: str
    model_config_name: str
    run_dir: str
    success: bool
    error: Optional[str] = None
    rounds_completed: int = 0


class BatchRunner:
    """Run systematic experiments across topics and models."""
    
    def __init__(
        self,
        output_dir: str = "artifacts",
        rounds: int = 5,
        word_limit: int = 180,
        base_config: Optional[DebateConfig] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rounds = rounds
        self.word_limit = word_limit
        self.base_config = base_config or DebateConfig.from_ini()
        self.results: List[ExperimentResult] = []
        self._rate_limit_hit = False
        self._base_wait = 20  # base wait seconds on rate limit (free tier = 5 req/min)
        # Build index of already-completed experiments once
        self._completed: dict[tuple[str, str], str] = {}  # (topic_id, config_name) -> run_dir
        self._build_completion_index()

    def _build_completion_index(self) -> None:
        """Scan artifact dirs once and index completed (topic_id, model_config) pairs."""
        count = 0
        for run_dir in sorted(self.output_dir.glob("run_*")):
            metadata_file = run_dir / "experiment_metadata.json"
            if not metadata_file.exists():
                continue
            try:
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                key = (metadata.get("topic_id", ""), metadata.get("model_config", ""))
                if key[0] and key[1]:
                    self._completed[key] = str(run_dir)
                    count += 1
            except Exception:
                pass  # corrupt metadata — will be re-run
        if count:
            print(f"  Completion index: {count} experiments already done — will be skipped.")

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        """Check if an exception is a Gemini API rate-limit / quota error."""
        err_str = str(error).lower()
        rate_limit_keywords = [
            "resource exhausted", "resourceexhausted",
            "rate limit", "rate_limit", "ratelimit",
            "quota", "429", "too many requests",
            "requests per minute", "tokens per minute",
        ]
        return any(kw in err_str for kw in rate_limit_keywords)

    @staticmethod
    def _is_daily_quota_error(error: Exception) -> bool:
        """Check if this is a daily (not per-minute) quota exhaustion."""
        err_str = str(error).lower()
        daily_keywords = [
            "per_day", "per day", "daily", "requests_per_day",
        ]
        return any(kw in err_str for kw in daily_keywords)

    @staticmethod
    def _extract_retry_delay(error: Exception) -> float:
        """Try to extract retry delay from error message (e.g. 'retry in 9.4s')."""
        err_str = str(error)
        # Look for patterns like "retry in 9.415487767s" or retry_delay { seconds: 9 }
        match = re.search(r'retry in ([\d.]+)s', err_str, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r'seconds:\s*(\d+)', err_str)
        if match:
            return float(match.group(1))
        return 0.0
    
    def add_model_config(
        self,
        name: str,
        moderator: str,
        conspiracy: str,
        scientific: str,
        api_mode: str = "openai",
        base_url: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ) -> ModelConfig:
        """Helper to create a model configuration."""
        return ModelConfig(
            name=name,
            moderator=moderator,
            conspiracy=conspiracy,
            scientific=scientific,
            api_mode=api_mode,
            base_url=base_url or self.base_config.base_url,
            gemini_api_key=gemini_api_key or self.base_config.gemini_api_key,
        )
    
    def run_experiment(
        self,
        topic: DebateTopic,
        model_config: ModelConfig,
    ) -> ExperimentResult:
        """Run a single debate experiment."""
        print(f"\n{'='*80}")
        print(f"Experiment: {topic.id}  {model_config.name}")
        print(f"{'='*80}")
        
        # Fast O(1) check against pre-built index
        key = (topic.id, model_config.name)
        if key in self._completed:
            prev_dir = self._completed[key]
            print(f"  Skipping - already completed: {prev_dir}")
            result = ExperimentResult(
                topic_id=topic.id,
                model_config_name=model_config.name,
                run_dir=prev_dir,
                success=True,
                rounds_completed=self.rounds,
            )
            self.results.append(result)
            return result
        
        # Retry loop — always wait & retry on per-minute rate limits, stop only on daily quota
        attempt = 0
        while True:
            attempt += 1
            try:
                result = self._do_run_experiment(topic, model_config, key)
                self.results.append(result)
                self.export_summary()
                return result

            except Exception as e:
                if self._is_rate_limit_error(e):
                    # Daily quota = stop entirely
                    if self._is_daily_quota_error(e):
                        print(f"\n{'!'*80}")
                        print(f"  DAILY QUOTA EXHAUSTED — stopping batch")
                        print(f"{'!'*80}")
                        print(f"  {e}")
                        print(f"  Progress saved. Re-run tomorrow.")
                        print(f"{'!'*80}\n")
                        self._rate_limit_hit = True
                        result = ExperimentResult(
                            topic_id=topic.id,
                            model_config_name=model_config.name,
                            run_dir="",
                            success=False,
                            error=f"DAILY_QUOTA: {e}",
                        )
                        self.results.append(result)
                        self.export_summary()
                        return result

                    # Per-minute rate limit → wait and retry (never give up)
                    retry_delay = self._extract_retry_delay(e)
                    wait = max(retry_delay + 5, self._base_wait)
                    print(f"\n  Rate limit hit (attempt {attempt}), waiting {wait:.0f}s...")
                    time.sleep(wait)
                    continue  # retry this experiment
                else:
                    # Non-rate-limit error — fail this experiment, move on
                    print(f" Experiment failed: {e}")
                    result = ExperimentResult(
                        topic_id=topic.id,
                        model_config_name=model_config.name,
                        run_dir="",
                        success=False,
                        error=str(e),
                    )
                    self.results.append(result)
                    self.export_summary()
                    return result

    def _do_run_experiment(
        self,
        topic: DebateTopic,
        model_config: ModelConfig,
        key: tuple[str, str],
    ) -> ExperimentResult:
        """Actually run a single debate (no retry logic here — caller handles that)."""
        config = DebateConfig(
            base_url=model_config.base_url,
            api_mode=model_config.api_mode,
            gemini_api_key=model_config.gemini_api_key,
            moderator_model=model_config.moderator,
            conspiracy_model=model_config.conspiracy,
            scientific_model=model_config.scientific,
            moderator_temperature=self.base_config.moderator_temperature,
            conspiracy_temperature=self.base_config.conspiracy_temperature,
            scientific_temperature=self.base_config.scientific_temperature,
            max_tokens=self.base_config.max_tokens,
            rounds=self.rounds,
            word_limit=self.word_limit,
            seed=self.base_config.seed,
            output_dir=str(self.output_dir),
            run_analysis=self.base_config.run_analysis,
            analysis_shift_threshold=self.base_config.analysis_shift_threshold,
            analysis_similarity_method=self.base_config.analysis_similarity_method,
            history_mode=self.base_config.history_mode,
            include_memory_summary=self.base_config.include_memory_summary,
            history_trim=self.base_config.history_trim,
            max_rounds_in_history=self.base_config.max_rounds_in_history,
            max_messages_in_history=self.base_config.max_messages_in_history,
            max_chars_in_history=self.base_config.max_chars_in_history,
            summarize_if_trimmed=self.base_config.summarize_if_trimmed,
            highlight_opponent_last=self.base_config.highlight_opponent_last,
        )

        # Run debate
        bundle = run_debate(
            topic=topic.topic,
            motion=topic.motion,
            rounds=self.rounds,
            config=config,
        )

        # Save experiment metadata
        metadata = {
            "topic_id": topic.id,
            "topic_category": topic.category,
            "topic_description": topic.description,
            "model_config": model_config.name,
            "models": {
                "moderator": model_config.moderator,
                "conspiracy": model_config.conspiracy,
                "scientific": model_config.scientific,
            },
            "api_mode": model_config.api_mode,
            "timestamp": datetime.now().isoformat(),
        }

        metadata_path = bundle.run_dir / "experiment_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # Update index so a re-run within the same session also skips
        self._completed[key] = str(bundle.run_dir)

        print(f" Experiment completed: {bundle.run_dir}")

        return ExperimentResult(
            topic_id=topic.id,
            model_config_name=model_config.name,
            run_dir=str(bundle.run_dir),
            success=True,
            rounds_completed=self.rounds,
        )
    
    def run_batch(
        self,
        topics: List[DebateTopic],
        model_configs: List[ModelConfig],
    ) -> List[ExperimentResult]:
        """Run all combinations of topics and model configurations."""
        print(f"\n{'='*80}")
        print(f"BATCH EXPERIMENT - GEMINI MODELS")
        print(f"{'='*80}")
        print(f"Topics: {len(topics)}")
        print(f"Model Configs: {len(model_configs)}")
        print(f"Total Experiments: {len(topics) * len(model_configs)}")
        print(f"  Watch API rate limits: 5 req/min, 20 req/day (free tier)")
        print(f"Press Ctrl+C at any time to stop gracefully")
        print(f"{'='*80}\n")
        
        total = len(topics) * len(model_configs)
        completed = 0
        
        try:
            # Process all topics for each model config before moving to next config
            for model_config in model_configs:
                print(f"\n{'='*80}")
                print(f"MODEL CONFIG: {model_config.name}")
                print(f"{'='*80}")
                for topic in topics:
                    completed += 1
                    print(f"\nProgress: {completed}/{total} | Config: {model_config.name} | Topic: {topic.id}")
                    self.run_experiment(topic, model_config)
                    if self._rate_limit_hit:
                        # Daily quota exhausted — stop the batch
                        print(f"\n{'='*80}")
                        print(f"  BATCH STOPPED — DAILY QUOTA EXHAUSTED")
                        print(f"{'='*80}")
                        print(f"Completed: {completed}/{total} experiments")
                        print(f"Progress saved. Re-run when quota resets.")
                        print(f"{'='*80}\n")
                        break
                if self._rate_limit_hit:
                    break
        except KeyboardInterrupt:
            print(f"\n\n{'='*80}")
            print(f"  BATCH INTERRUPTED BY USER")
            print(f"{'='*80}")
            print(f"Completed: {completed}/{total} experiments")
            print(f"Partial results will be saved.")
            print(f"{'='*80}\n")
        
        # Export summary
        self.export_summary()
        
        # Export all debates to single CSV
        combined_csv = self.output_dir / "all_debates_gemini.csv"
        export_all_debates_to_csv(self.output_dir, combined_csv)
        
        print(f"\n{'='*80}")
        print(f"BATCH COMPLETED OR INTERRUPTED")
        print(f"{'='*80}")
        print(f"To resume: Simply run this script again - completed experiments will be skipped")
        print(f"{'='*80}\n")
        
        return self.results
    
    def export_summary(self) -> None:
        """Export batch experiment summary with detailed tracking."""
        # Group results by model config for detailed reporting
        by_config = {}
        for r in self.results:
            if r.model_config_name not in by_config:
                by_config[r.model_config_name] = []
            by_config[r.model_config_name].append(r)
        
        # Detailed per-config breakdown
        config_summaries = {}
        for config_name, results in by_config.items():
            config_summaries[config_name] = {
                "total": len(results),
                "successful": sum(1 for r in results if r.success),
                "failed": sum(1 for r in results if not r.success),
                "topics_completed": [r.topic_id for r in results if r.success],
                "topics_failed": [r.topic_id for r in results if not r.success],
            }
        
        summary = {
            "batch_type": "gemini_only",
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(self.results),
            "successful": sum(1 for r in self.results if r.success),
            "failed": sum(1 for r in self.results if not r.success),
            "by_config": config_summaries,
            "results": [
                {
                    "topic_id": r.topic_id,
                    "model_config": r.model_config_name,
                    "success": r.success,
                    "run_dir": r.run_dir,
                    "error": r.error,
                    "rounds_completed": r.rounds_completed,
                }
                for r in self.results
            ],
        }
        
        summary_path = self.output_dir / "batch_summary_gemini.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        
        # Also create a detailed markdown report
        self._create_detailed_report(summary, by_config)
    
    def _create_detailed_report(self, summary: dict, by_config: dict) -> None:
        """Create a detailed markdown report of batch progress."""
        report_lines = [
            "# Gemini Batch Experiment Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## Overall Summary",
            f"- **Total Experiments**: {summary['total_experiments']}",
            f"- **Successful**: {summary['successful']}",
            f"- **Failed**: {summary['failed']}",
            f"- **Success Rate**: {summary['successful']/summary['total_experiments']*100:.1f}%" if summary['total_experiments'] > 0 else "- **Success Rate**: N/A",
            f"\n## Progress by Model Configuration",
        ]
        
        for config_name, config_data in summary['by_config'].items():
            report_lines.extend([
                f"\n### {config_name}",
                f"- Total: {config_data['total']}",
                f"- Successful: {config_data['successful']}",
                f"- Failed: {config_data['failed']}",
                f"- Success Rate: {config_data['successful']/config_data['total']*100:.1f}%" if config_data['total'] > 0 else "- Success Rate: N/A",
            ])
            
            if config_data['topics_completed']:
                report_lines.append(f"\n**Completed Topics ({len(config_data['topics_completed'])}):**")
                for topic in config_data['topics_completed']:
                    report_lines.append(f"-  {topic}")
            
            if config_data['topics_failed']:
                report_lines.append(f"\n**Failed Topics ({len(config_data['topics_failed'])}):**")
                for topic in config_data['topics_failed']:
                    report_lines.append(f"-  {topic}")
        
        if summary['failed'] > 0:
            report_lines.append(f"\n## Failed Experiments Details")
            for result in summary['results']:
                if not result['success']:
                    report_lines.extend([
                        f"\n### {result['topic_id']}  {result['model_config']}",
                        f"- **Error**: {result['error']}",
                    ])
        
        report_path = self.output_dir / "batch_report_gemini.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        
        print(f"\n Batch summary saved to: {self.output_dir / 'batch_summary_gemini.json'}")
        print(f" Detailed report saved to: {report_path}")
        print(f"   Successful: {summary['successful']}/{summary['total_experiments']}")
        print(f"   Failed: {summary['failed']}/{summary['total_experiments']}")


def main():
    """Run Gemini-only batch experiments (requires API key)."""
    runner = BatchRunner(
        output_dir="artifacts_gemini",
        rounds=5,
        word_limit=180,
    )
    
    # Get models from config.ini
    config = runner.base_config
    
    # Check API key
    if not config.gemini_api_key:
        print("\n ERROR: Gemini API key not configured!")
        print("   Add your API key to config.ini:")
        print("   [api]")
        print("   gemini_api_key = your-api-key-here")
        return
    
    print(f"\n Gemini API key configured")
    print(f"   API mode: {config.api_mode}")
    
    # GEMINI MODELS ONLY - Watch rate limits!
    # ALL 8 COMBINATIONS of Gemini-3-Flash-Preview and Gemma3-27b
    model_configs = [
        # 1. All Gemini 3 Flash Preview (3 API calls per round)
        runner.add_model_config(
            name="gemini-3-flash-all",
            moderator="gemini-3-flash-preview",
            conspiracy="gemini-3-flash-preview",
            scientific="gemini-3-flash-preview",
            api_mode="gemini",
        ),
        
        # 2. All Gemma3:27b (0 API calls - all Ollama)
        runner.add_model_config(
            name="gemma3-27b-all",
            moderator=config.moderator_model,
            conspiracy=config.conspiracy_model,
            scientific=config.scientific_model,
            api_mode="openai",
        ),
        
        # 3. Gemini moderator, Gemma agents (1 API call per round)
        runner.add_model_config(
            name="gemini-flash-mod-gemma-agents",
            moderator="gemini-3-flash-preview",
            conspiracy=config.conspiracy_model,
            scientific=config.scientific_model,
            api_mode="gemini",
        ),
        
        # 4. Gemma moderator, Gemini agents (2 API calls per round)
        runner.add_model_config(
            name="gemma-mod-gemini-flash-agents",
            moderator=config.moderator_model,
            conspiracy="gemini-3-flash-preview",
            scientific="gemini-3-flash-preview",
            api_mode="openai",
        ),
        
        # 5. Gemini mod+CA, Gemma SA (2 API calls per round)
        runner.add_model_config(
            name="gemini-mod-ca_gemma-sa",
            moderator="gemini-3-flash-preview",
            conspiracy="gemini-3-flash-preview",
            scientific=config.scientific_model,
            api_mode="gemini",
        ),
        
        # 6. Gemini mod+SA, Gemma CA (2 API calls per round)
        runner.add_model_config(
            name="gemini-mod-sa_gemma-ca",
            moderator="gemini-3-flash-preview",
            conspiracy=config.conspiracy_model,
            scientific="gemini-3-flash-preview",
            api_mode="gemini",
        ),
        
        # 7. Gemma mod+SA, Gemini CA (1 API call per round)
        runner.add_model_config(
            name="gemma-mod-sa_gemini-ca",
            moderator=config.moderator_model,
            conspiracy="gemini-3-flash-preview",
            scientific=config.scientific_model,
            api_mode="openai",
        ),
        
        # 8. Gemma mod+CA, Gemini SA (1 API call per round)
        runner.add_model_config(
            name="gemma-mod-ca_gemini-sa",
            moderator=config.moderator_model,
            conspiracy=config.conspiracy_model,
            scientific="gemini-3-flash-preview",
            api_mode="openai",
        ),
    ]
    
    # Use ALL conspiracy topics (20 topics)
    topics = CONSPIRACY_TOPICS
    
    # Run batch experiments
    # 20 topics  8 model configs = 160 experiments
    # WARNING: API limits apply! Estimated:
    #   - Gemini-all: 20 topics  5 rounds  3 calls = 300 API calls
    #   - Gemini mod/agents: varies by config (100-200 calls each)
    #   - Total: ~1,200-1,400 API calls
    # Daily limit: 1,500 requests/day
    # Rate limit: 5 requests/minute, 20 requests/day (free tier)
    print(f"\n  API USAGE WARNING:")
    print(f"   Total experiments: 160")
    print(f"   Estimated API calls: 1,200-1,400")
    print(f"   Daily limit: 20 requests/day (free tier)")
    print(f"   This will take MANY days due to rate limiting (5 req/min, 20 req/day)")
    print(f"   Press Ctrl+C to cancel, or wait 10 seconds to continue...")
    
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        print("\n Batch run cancelled by user")
        return
    
    results = runner.run_batch(topics, model_configs)
    
    print(f"\n Gemini batch complete!")
    print(f"   Experiments: {len(results)}")
    print(f"   Results saved to: {runner.output_dir}")
    print(f"   Combined CSV: all_debates_gemini.csv")
    print(f"\n Combine with Ollama results: all_debates_ollama.csv + all_debates_gemini.csv")


if __name__ == "__main__":
    main()
