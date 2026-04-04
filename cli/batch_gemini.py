"""Batch experiment runner - GEMINI MODELS (requires API key).

Subclass of :class:`BaseBatchRunner` that adds:
- Per-minute rate-limit retry with exponential back-off
- Daily quota detection and graceful batch abort
"""

from __future__ import annotations

import re
import time

from debate.providers.llm_client import (
    _RATE_LIMIT_EXTRA_DELAY,
    _RATE_LIMIT_FALLBACK_DELAY,
)

from cli.base_batch import BaseBatchRunner, ExperimentResult, ModelConfig
from debate.core.model_pool import ModelPoolError, get_batch_model_configs
from debate.core.topics import CONSPIRACY_TOPICS, DebateTopic


class GeminiBatchRunner(BaseBatchRunner):
    """Gemini batch runner — adds rate-limit retry and daily-quota detection."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rate_limit_hit = False
        self._base_wait = (
            _RATE_LIMIT_FALLBACK_DELAY  # base wait seconds on rate limit (free tier = 5 req/min)
        )

    # ── Rate-limit helpers ─────────────────────────────────────────

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        """Check if an exception is a Gemini API rate-limit / quota error."""
        err_str = str(error).lower()
        rate_limit_keywords = [
            "resource exhausted",
            "resourceexhausted",
            "rate limit",
            "rate_limit",
            "ratelimit",
            "quota",
            "429",
            "too many requests",
            "requests per minute",
            "tokens per minute",
        ]
        return any(kw in err_str for kw in rate_limit_keywords)

    @staticmethod
    def _is_daily_quota_error(error: Exception) -> bool:
        """Check if this is a daily (not per-minute) quota exhaustion."""
        err_str = str(error).lower()
        daily_keywords = ["per_day", "per day", "daily", "requests_per_day"]
        return any(kw in err_str for kw in daily_keywords)

    @staticmethod
    def _extract_retry_delay(error: Exception) -> float:
        """Try to extract retry delay from error message (e.g. 'retry in 9.4s')."""
        err_str = str(error)
        match = re.search(r"retry in ([\d.]+)s", err_str, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"seconds:\s*(\d+)", err_str)
        if match:
            return float(match.group(1))
        return 0.0

    # ── Overrides ──────────────────────────────────────────────────

    def _should_stop_batch(self) -> bool:
        return self._rate_limit_hit

    def _print_batch_info(self) -> None:
        print("  Watch API rate limits: 5 req/min, 20 req/day (free tier)")

    def run_experiment(
        self,
        topic: DebateTopic,
        model_config: ModelConfig,
    ) -> ExperimentResult:
        """Run a single experiment with rate-limit retry (never gives up on per-minute limits)."""
        print(f"\n{'=' * 80}")
        print(f"Experiment: {topic.id}  {model_config.name}")
        print(f"{'=' * 80}")

        skip = self._check_skip(topic, model_config)
        if skip is not None:
            return skip

        key = (topic.id, model_config.name)
        attempt = 0
        while True:
            attempt += 1
            try:
                result = self._execute_experiment(topic, model_config, key)
                self.results.append(result)
                self.export_summary()
                return result

            except Exception as e:
                if self._is_rate_limit_error(e):
                    # Daily quota = stop entirely
                    if self._is_daily_quota_error(e):
                        print(f"\n{'!' * 80}")
                        print("  DAILY QUOTA EXHAUSTED — stopping batch")
                        print(f"{'!' * 80}")
                        print(f"  {e}")
                        print("  Progress saved. Re-run tomorrow.")
                        print(f"{'!' * 80}\n")
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
                    wait = max(retry_delay + _RATE_LIMIT_EXTRA_DELAY, self._base_wait)
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


def main():
    """Run Gemini-only batch experiments (requires API key)."""
    runner = GeminiBatchRunner(
        batch_label="gemini",
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

    print("\n Gemini API key configured")
    print(f"   API mode: {config.api_mode}")

    try:
        pool_configs = get_batch_model_configs("gemini")
    except ModelPoolError as exc:
        print(f"\n ERROR: invalid model_pool.json configuration for gemini batch: {exc}")
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

    total_experiments = len(topics) * len(model_configs)

    # Run batch experiments
    print("\n  API USAGE WARNING:")
    print(f"   Total experiments: {total_experiments}")
    print("   Generated from model_pool.json permutations")
    print("   Estimated API calls: depends on rounds and number of Gemini roles per config")
    print("   Daily limit: 20 requests/day (free tier)")
    print("   This will take MANY days due to rate limiting (5 req/min, 20 req/day)")
    print("   Press Ctrl+C to cancel, or wait 10 seconds to continue...")

    try:
        time.sleep(10)
    except KeyboardInterrupt:
        print("\n Batch run cancelled by user")
        return

    results = runner.run_batch(topics, model_configs)

    print("\n Gemini batch complete!")
    print(f"   Experiments: {len(results)}")
    print(f"   Results saved to: {runner.output_dir}")
    print("   Combined CSV: all_debates_gemini.csv")
    print("\n Combine with Ollama results: all_debates_ollama.csv + all_debates_gemini.csv")


if __name__ == "__main__":
    main()
