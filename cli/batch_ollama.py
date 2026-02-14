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
        rounds=5,
        word_limit=180,
    )

    # Get models from config.ini
    config = runner.base_config

    # OLLAMA MODELS ONLY - No API limits, can run 24/7
    # ALL 8 COMBINATIONS of Gemma3-27b and qwen3:1.7b
    model_configs = [
        # 1. All Gemma3:27b
        runner.add_model_config(
            name="gemma3-27b-all",
            moderator=config.moderator_model,
            conspiracy=config.conspiracy_model,
            scientific=config.scientific_model,
            api_mode="openai",
        ),
        # 2. All Dolphin Yi 34B
        runner.add_model_config(
            name="qwen3:1.7b-all",
            moderator="qwen3:1.7b",
            conspiracy="qwen3:1.7b",
            scientific="qwen3:1.7b",
            api_mode="openai",
        ),
        # 3. Gemma3 moderator, Dolphin agents
        runner.add_model_config(
            name="gemma-mod-dolphin-agents",
            moderator=config.moderator_model,
            conspiracy="qwen3:1.7b",
            scientific="qwen3:1.7b",
            api_mode="openai",
        ),
        # 4. Dolphin moderator, Gemma3 agents
        runner.add_model_config(
            name="dolphin-mod-gemma-agents",
            moderator="qwen3:1.7b",
            conspiracy=config.conspiracy_model,
            scientific=config.scientific_model,
            api_mode="openai",
        ),
        # 5. Gemma3 mod+CA, Dolphin SA
        runner.add_model_config(
            name="gemma-mod-ca_dolphin-sa",
            moderator=config.moderator_model,
            conspiracy=config.conspiracy_model,
            scientific="qwen3:1.7b",
            api_mode="openai",
        ),
        # 6. Gemma3 mod+SA, Dolphin CA
        runner.add_model_config(
            name="gemma-mod-sa_dolphin-ca",
            moderator=config.moderator_model,
            conspiracy="qwen3:1.7b",
            scientific=config.scientific_model,
            api_mode="openai",
        ),
        # 7. Dolphin mod+SA, Gemma3 CA
        runner.add_model_config(
            name="dolphin-mod-sa_gemma-ca",
            moderator="qwen3:1.7b",
            conspiracy=config.conspiracy_model,
            scientific="qwen3:1.7b",
            api_mode="openai",
        ),
        # 8. Dolphin mod+CA, Gemma3 SA
        runner.add_model_config(
            name="dolphin-mod-ca_gemma-sa",
            moderator="qwen3:1.7b",
            conspiracy="qwen3:1.7b",
            scientific=config.scientific_model,
            api_mode="openai",
        ),
    ]

    # Use ALL conspiracy topics (20 topics)
    topics = CONSPIRACY_TOPICS

    # Run batch experiments
    # 20 topics  8 model configs = 160 experiments
    # No API limits - can complete in 8-12 hours
    results = runner.run_batch(topics, model_configs)

    print("\n Ollama batch complete!")
    print(f"   Experiments: {len(results)}")
    print(f"   Results saved to: {runner.output_dir}")
    print("   Combined CSV: all_debates_ollama.csv")
    print("\n Next step: Run batch_runner_gemini.py when you have API quota available")


if __name__ == "__main__":
    main()
