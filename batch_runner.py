"""Batch experiment runner for systematic model and topic combinations."""

from __future__ import annotations

import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
        rounds: int = 3,
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
            metadata_path.write_text(json.dumps(metadata, indent=2))
            
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
        print(f"BATCH EXPERIMENT")
        print(f"{'='*80}")
        print(f"Topics: {len(topics)}")
        print(f"Model Configs: {len(model_configs)}")
        print(f"Total Experiments: {len(topics) * len(model_configs)}")
        print(f"{'='*80}\n")
        
        total = len(topics) * len(model_configs)
        completed = 0
        
        for topic in topics:
            for model_config in model_configs:
                completed += 1
                print(f"\nProgress: {completed}/{total}")
                self.run_experiment(topic, model_config)
        
        # Export summary
        self.export_summary()
        
        # Export all debates to single CSV
        combined_csv = self.output_dir / "all_debates.csv"
        export_all_debates_to_csv(self.output_dir, combined_csv)
        
        return self.results
    
    def export_summary(self) -> None:
        """Export batch experiment summary."""
        summary = {
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
        
        summary_path = self.output_dir / "batch_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n Batch summary saved to: {summary_path}")
        print(f"   Successful: {summary['successful']}/{summary['total_experiments']}")
        print(f"   Failed: {summary['failed']}/{summary['total_experiments']}")


def main():
    """Example usage of batch runner."""
    runner = BatchRunner(
        output_dir="artifacts",
        rounds=10,
        word_limit=180,
    )
    
    # Get models from config.ini
    config = runner.base_config
    
    # Define model configurations to test
    # Test all combinations: Gemma3 (Ollama) vs Gemini 2.5 Pro (strong) vs Uncensored LLM
    model_configs = [
        # 1. Gemma3:27b for all roles (from config.ini - Ollama)
        runner.add_model_config(
            name="gemma3-27b-all",
            moderator=config.moderator_model,
            conspiracy=config.conspiracy_model,
            scientific=config.scientific_model,
            api_mode=config.api_mode,
        ),
        
        # 2. Gemini 2.5 Pro for all roles (strong model)
        runner.add_model_config(
            name="gemini-2.5-pro-all",
            moderator="gemini-2.5-pro",
            conspiracy="gemini-2.5-pro",
            scientific="gemini-2.5-pro",
            api_mode="gemini",
        ),
        
        # 3. Uncensored LLM for all roles (Dolphin Yi 34B)
        runner.add_model_config(
            name="dolphin-yi-34b-all",
            moderator="dolphin-yi-34b",
            conspiracy="dolphin-yi-34b",
            scientific="dolphin-yi-34b",
            api_mode="openai",
        ),
        
        # 4. Mixed: Gemini 2.5 Pro moderator with Gemma agents
        runner.add_model_config(
            name="gemini-pro-mod-gemma-agents",
            moderator="gemini-2.5-pro",
            conspiracy=config.conspiracy_model,
            scientific=config.scientific_model,
            api_mode="gemini",
        ),
        
        # 5. Mixed: Gemma moderator with Gemini 2.5 Pro agents
        runner.add_model_config(
            name="gemma-mod-gemini-pro-agents",
            moderator=config.moderator_model,
            conspiracy="gemini-2.5-pro",
            scientific="gemini-2.5-pro",
            api_mode="openai",
        ),
        
        # 6. Mixed: Gemini 2.5 Pro moderator with Dolphin Yi 34B agents
        runner.add_model_config(
            name="gemini-pro-mod-dolphin-agents",
            moderator="gemini-2.5-pro",
            conspiracy="dolphin-yi-34b",
            scientific="dolphin-yi-34b",
            api_mode="gemini",
        ),
    ]
    
    # Use ALL conspiracy topics (20 topics)
    topics = CONSPIRACY_TOPICS
    
    # Run batch experiments
    results = runner.run_batch(topics, model_configs)
    
    print(f"\n Batch experiment complete!")
    print(f"   Results: {len(results)} experiments")
    print(f"   Check {runner.output_dir} for all outputs")


if __name__ == "__main__":
    main()
