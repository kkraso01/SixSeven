"""Batch experiment runner - OLLAMA MODELS ONLY (no API limits).

Thin subclass of :class:`BaseBatchRunner` — no retry logic needed
because Ollama has no API rate limits.
"""

from __future__ import annotations

from cli.base_batch import BaseBatchRunner
from debate.core.model_pool import ModelPoolError, get_batch_model_configs
from debate.core.topics import CONSPIRACY_TOPICS


class OllamaBatchRunner(BaseBatchRunner):
    """Ollama batch runner — no API limits, can run continuously."""

    def _print_batch_info(self) -> None:
        print("No API limits - can run continuously!")


def main():
    """Run Ollama-only batch experiments (no API limits)."""
    runner = OllamaBatchRunner(
        batch_label="ollama",
    )

    try:
        pool_configs = get_batch_model_configs("ollama")
    except ModelPoolError as exc:
        print(f"\n ERROR: invalid model_pool.json configuration for ollama batch: {exc}")
        return

    model_configs = [
        runner.add_model_config(
            name=cfg["name"],
            moderator=cfg["moderator"],
            conspiracy=cfg["conspiracy"],
            scientific=cfg["scientific"],
            api_mode=cfg["api_mode"],
        )
        for cfg in pool_configs
    ]

    # Use ALL conspiracy topics (20 topics)
    topics = CONSPIRACY_TOPICS

    # Run batch experiments with generated permutations from model_pool.json
    results = runner.run_batch(topics, model_configs)

    print("\n Ollama batch complete!")
    print(f"   Experiments: {len(results)}")
    print(f"   Results saved to: {runner.output_dir}")
    print("   Combined CSV: all_debates_ollama.csv")


if __name__ == "__main__":
    main()
