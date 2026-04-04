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
    # - Gemini-all: 20 topics  5 rounds  3 calls = 300 API calls
    # - Gemini mod/agents: varies by config (100-200 calls each)
    # - Total: ~1,200-1,400 API calls
    # Daily limit: 1,500 requests/day
    # Rate limit: 5 requests/minute, 20 requests/day (free tier)
    print("\n  API USAGE WARNING:")
    print("   Total experiments: 160")
    print("   Estimated API calls: 1,200-1,400")
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
