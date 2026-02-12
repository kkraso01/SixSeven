"""Batch experiment runner - GEMINI MODELS (requires API key)."""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
        self.rounds = rounds
        self.word_limit = word_limit
        self.base_config = base_config or DebateConfig.from_ini()
        self.results: List[ExperimentResult] = []
    
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
        
        try:
            # Create config for this experiment
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
            
            result = ExperimentResult(
                topic_id=topic.id,
                model_config_name=model_config.name,
                run_dir=str(bundle.run_dir),
                success=True,
                rounds_completed=self.rounds,
            )
            
            print(f" Experiment completed: {bundle.run_dir}")
            
        except Exception as e:
            print(f" Experiment failed: {e}")
            result = ExperimentResult(
                topic_id=topic.id,
                model_config_name=model_config.name,
                run_dir="",
                success=False,
                error=str(e),
            )
        
        self.results.append(result)
        return result
    
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
        print(f"  Watch API rate limits: 15 req/min, 1500 req/day")
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
        
        return self.results
    
    def export_summary(self) -> None:
        """Export batch experiment summary."""
        summary = {
            "batch_type": "gemini_only",
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(self.results),
            "successful": sum(1 for r in self.results if r.success),
            "failed": sum(1 for r in self.results if not r.success),
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
        print(f"\n Batch summary saved to: {summary_path}")
        print(f"   Successful: {summary['successful']}/{summary['total_experiments']}")
        print(f"   Failed: {summary['failed']}/{summary['total_experiments']}")


def main():
    """Run Gemini-only batch experiments (requires API key)."""
    runner = BatchRunner(
        output_dir="artifacts",
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
    # ALL 8 COMBINATIONS of Gemini-2.5-Pro and Gemma3-27b
    model_configs = [
        # 1. All Gemini 2.5 Pro (3 API calls per round)
        runner.add_model_config(
            name="gemini-2.5-pro-all",
            moderator="gemini-2.5-pro",
            conspiracy="gemini-2.5-pro",
            scientific="gemini-2.5-pro",
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
            name="gemini-pro-mod-gemma-agents",
            moderator="gemini-2.5-pro",
            conspiracy=config.conspiracy_model,
            scientific=config.scientific_model,
            api_mode="gemini",
        ),
        
        # 4. Gemma moderator, Gemini agents (2 API calls per round)
        runner.add_model_config(
            name="gemma-mod-gemini-pro-agents",
            moderator=config.moderator_model,
            conspiracy="gemini-2.5-pro",
            scientific="gemini-2.5-pro",
            api_mode="openai",
        ),
        
        # 5. Gemini mod+CA, Gemma SA (2 API calls per round)
        runner.add_model_config(
            name="gemini-mod-ca_gemma-sa",
            moderator="gemini-2.5-pro",
            conspiracy="gemini-2.5-pro",
            scientific=config.scientific_model,
            api_mode="gemini",
        ),
        
        # 6. Gemini mod+SA, Gemma CA (2 API calls per round)
        runner.add_model_config(
            name="gemini-mod-sa_gemma-ca",
            moderator="gemini-2.5-pro",
            conspiracy=config.conspiracy_model,
            scientific="gemini-2.5-pro",
            api_mode="gemini",
        ),
        
        # 7. Gemma mod+SA, Gemini CA (1 API call per round)
        runner.add_model_config(
            name="gemma-mod-sa_gemini-ca",
            moderator=config.moderator_model,
            conspiracy="gemini-2.5-pro",
            scientific=config.scientific_model,
            api_mode="openai",
        ),
        
        # 8. Gemma mod+CA, Gemini SA (1 API call per round)
        runner.add_model_config(
            name="gemma-mod-ca_gemini-sa",
            moderator=config.moderator_model,
            conspiracy=config.conspiracy_model,
            scientific="gemini-2.5-pro",
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
    # Rate limit: 15 requests/minute
    print(f"\n  API USAGE WARNING:")
    print(f"   Total experiments: 160")
    print(f"   Estimated API calls: 1,200-1,400")
    print(f"   Daily limit: 1,500 requests/day")
    print(f"   This will take 8-12 hours due to rate limiting (15 req/min)")
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
