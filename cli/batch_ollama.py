"""Batch experiment runner - OLLAMA MODELS ONLY (no API limits).

Thin subclass of :class:`BaseBatchRunner` — no retry logic needed
because Ollama has no API rate limits.
"""

from __future__ import annotations

from cli.base_batch import BaseBatchRunner
from debate_sim.core.topics import CONSPIRACY_TOPICS


class OllamaBatchRunner(BaseBatchRunner):
    """Ollama batch runner — no API limits, can run continuously."""

    def _print_batch_info(self) -> None:
        print("No API limits - can run continuously!")


def main():
    """Run Ollama-only batch experiments (no API limits)."""
    runner = OllamaBatchRunner(
        batch_label="ollama",
        output_dir="kalamaras-artifacts",
        rounds=10,
        word_limit=300,
    )

    # Get models from config.ini
    config = runner.base_config

    # Fixed moderator from config.ini, swap conspiracy & scientific roles
    mod = config.moderator_model
    model_a = config.conspiracy_model   # default conspiracy advocate
    model_b = config.scientific_model   # default scientific advocate

    model_configs = [
        # 1. Original assignment: A=conspiracy, B=scientific
        runner.add_model_config(
            name=f"{model_a}-CA_{model_b}-SA",
            moderator=mod,
            conspiracy=model_a,
            scientific=model_b,
            api_mode="openai",
        ),
        # 2. Swapped: B=conspiracy, A=scientific
        runner.add_model_config(
            name=f"{model_b}-CA_{model_a}-SA",
            moderator=mod,
            conspiracy=model_b,
            scientific=model_a,
            api_mode="openai",
        ),
    ]

    # Use ALL conspiracy topics (20 topics)
    topics = CONSPIRACY_TOPICS

    # Run batch experiments
    # 20 topics × 2 model configs = 40 experiments
    results = runner.run_batch(topics, model_configs)

    print("\n Ollama batch complete!")
    print(f"   Experiments: {len(results)}")
    print(f"   Results saved to: {runner.output_dir}")
    print("   Combined CSV: all_debates_ollama.csv")


if __name__ == "__main__":
    main()
