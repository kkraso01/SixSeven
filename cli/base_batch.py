"""Shared base for batch experiment runners — DI-wired, deduplicated."""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from debate.io.csv_export import export_all_debates_to_csv

from debate import DebateConfig, run_debate
from debate.core.container import build_default_services
from debate.core.logging import setup_logging
from debate.core.topics import DebateTopic

# Force UTF-8 on Windows to prevent charmap codec errors
os.environ.setdefault("PYTHONUTF8", "1")

# Suppress FutureWarning from instructor's deprecated google.generativeai import
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="instructor.providers.gemini.client"
)

# Configure logging once at module level
logger = logging.getLogger(__name__)
setup_logging()

# ── Shared data classes ────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Configuration for a model combination."""

    name: str
    moderator: str
    conspiracy: str
    scientific: str
    api_mode: str = "openai"
    base_url: str = "https://chatucy.cs.ucy.ac.cy/v1"
    gemini_api_key: str | None = None


@dataclass
class ExperimentResult:
    """Result of a single experiment."""

    topic_id: str
    model_config_name: str
    run_dir: str
    success: bool
    error: str | None = None
    rounds_completed: int = 0


# ── Base runner ────────────────────────────────────────────────────


class BaseBatchRunner:
    """Run systematic experiments across topics and models.

    Subclasses only need to override behaviour such as retry policies or
    stop-batch conditions.  All shared orchestration, indexing, export,
    and DI wiring lives here.
    """

    def __init__(
        self,
        *,
        batch_label: str,
        output_dir: str | None = None,
        rounds: int | None = None,
        word_limit: int | None = None,
        base_config: DebateConfig | None = None,
    ):
        self.batch_label = batch_label
        self.base_config = base_config or DebateConfig.from_ini("config/config.ini")

        # Default to results/batches/<label> if no output dir provided
        if output_dir:
            self.output_dir = Path(output_dir)
        elif self.base_config.output_dir:
            # If config has an output dir, put batches inside it
            self.output_dir = Path(self.base_config.output_dir) / "batches" / batch_label
        else:
            # Fallback to current dir if all else fails (safety)
            self.output_dir = Path("results/batches") / batch_label

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rounds = rounds if rounds is not None else self.base_config.rounds
        self.word_limit = word_limit if word_limit is not None else self.base_config.word_limit
        self.results: list[ExperimentResult] = []
        # Index of already-completed experiments: (topic_id, config_name) -> run_dir
        self._completed: dict[tuple[str, str], str] = {}
        self._build_completion_index()

    # ── Completion index ───────────────────────────────────────────

    def _build_completion_index(self) -> None:
        """Scan artifact dirs once and index completed (topic_id, model_config) pairs.

        Now scans inside the 'raw/' subdirectory to align with the new hierarchy.
        """
        count = 0
        raw_dir = self.output_dir / "raw"
        if not raw_dir.exists():
            return

        for run_dir in sorted(raw_dir.glob("run_*")):
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

    # ── Helpers ────────────────────────────────────────────────────

    def add_model_config(
        self,
        name: str,
        moderator: str,
        conspiracy: str,
        scientific: str,
        api_mode: str = "openai",
        base_url: str | None = None,
        gemini_api_key: str | None = None,
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

    def _build_experiment_config(self, model_config: ModelConfig) -> DebateConfig:
        """Build a fully-populated :class:`DebateConfig` for one experiment."""
        return DebateConfig(
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
            max_search_rounds=self.base_config.max_search_rounds,
            thinking_budget=self.base_config.thinking_budget,
            num_ctx=self.base_config.num_ctx,
        )

    # ── Core execution ─────────────────────────────────────────────

    def _check_skip(self, topic: DebateTopic, model_config: ModelConfig) -> ExperimentResult | None:
        """Return a cached result if already completed, else *None*."""
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
        return None

    def _execute_experiment(
        self,
        topic: DebateTopic,
        model_config: ModelConfig,
        key: tuple[str, str],
    ) -> ExperimentResult:
        """Run debate, save metadata, update index.  May raise on failure."""
        config = self._build_experiment_config(model_config)
        services = build_default_services(config)

        bundle = run_debate(
            topic=topic.topic,
            motion=topic.motion,
            rounds=self.rounds,
            config=config,
            services=services,
            topic_description=topic.description,
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

    def run_experiment(
        self,
        topic: DebateTopic,
        model_config: ModelConfig,
    ) -> ExperimentResult:
        """Run a single debate experiment (skip-check + basic error handling).

        Override in subclasses to add retry or rate-limit logic.
        """
        print(f"\n{'=' * 80}")
        print(f"Experiment: {topic.id}  {model_config.name}")
        print(f"{'=' * 80}")

        skip = self._check_skip(topic, model_config)
        if skip is not None:
            return skip

        key = (topic.id, model_config.name)
        try:
            result = self._execute_experiment(topic, model_config, key)
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
        self.export_summary()
        return result

    # ── Batch orchestration ────────────────────────────────────────

    def _should_stop_batch(self) -> bool:
        """Return *True* to abort the batch early.  Override for rate-limit stops."""
        return False

    def _print_batch_info(self) -> None:
        """Print extra info lines in the batch header.  Override for custom messages."""
        pass

    def run_batch(
        self,
        topics: list[DebateTopic],
        model_configs: list[ModelConfig],
    ) -> list[ExperimentResult]:
        """Run all combinations of topics and model configurations."""
        total = len(topics) * len(model_configs)

        print(f"\n{'=' * 80}")
        print(f"BATCH EXPERIMENT - {self.batch_label.upper()}")
        print(f"{'=' * 80}")
        print(f"Topics: {len(topics)}")
        print(f"Model Configs: {len(model_configs)}")
        print(f"Total Experiments: {total}")
        self._print_batch_info()
        print("Press Ctrl+C at any time to stop gracefully")
        print(f"{'=' * 80}\n")

        completed = 0
        try:
            for model_config in model_configs:
                print(f"\n{'=' * 80}")
                print(f"MODEL CONFIG: {model_config.name}")
                print(f"{'=' * 80}")
                for topic in topics:
                    completed += 1
                    print(
                        f"\nProgress: {completed}/{total} "
                        f"| Config: {model_config.name} | Topic: {topic.id}"
                    )
                    self.run_experiment(topic, model_config)
                    if self._should_stop_batch():
                        print(f"\n{'=' * 80}")
                        print(f"  BATCH STOPPED — {self.batch_label.upper()}")
                        print(f"{'=' * 80}")
                        print(f"Completed: {completed}/{total} experiments")
                        print("Progress saved. Re-run to continue.")
                        print(f"{'=' * 80}\n")
                        break
                if self._should_stop_batch():
                    break
        except KeyboardInterrupt:
            print(f"\n\n{'=' * 80}")
            print("  BATCH INTERRUPTED BY USER")
            print(f"{'=' * 80}")
            print(f"Completed: {completed}/{total} experiments")
            print("Partial results will be saved.")
            print(f"{'=' * 80}\n")

        self.export_summary()

        combined_csv = self.output_dir / f"all_debates_{self.batch_label}.csv"
        # The aggregator now scans the 'raw/' subdirectory
        export_all_debates_to_csv(self.output_dir / "raw", combined_csv)

        print(f"\n{'=' * 80}")
        print("BATCH COMPLETED OR INTERRUPTED")
        print(f"{'=' * 80}")
        print("To resume: Simply run this script again - completed experiments will be skipped")
        print(f"{'=' * 80}\n")

        return self.results

    # ── Export / reporting ─────────────────────────────────────────

    def export_summary(self) -> None:
        """Export batch experiment summary with detailed tracking."""
        by_config: dict[str, list[ExperimentResult]] = {}
        for r in self.results:
            by_config.setdefault(r.model_config_name, []).append(r)

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
            "batch_type": self.batch_label,
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

        summary_path = self.output_dir / f"batch_summary_{self.batch_label}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._create_detailed_report(summary, by_config)

    def _create_detailed_report(
        self, summary: dict, by_config: dict[str, list[ExperimentResult]]
    ) -> None:
        """Create a detailed markdown report of batch progress."""
        label = self.batch_label.capitalize()
        report_lines = [
            f"# {label} Batch Experiment Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Overall Summary",
            f"- **Total Experiments**: {summary['total_experiments']}",
            f"- **Successful**: {summary['successful']}",
            f"- **Failed**: {summary['failed']}",
            (
                f"- **Success Rate**: "
                f"{summary['successful'] / summary['total_experiments'] * 100:.1f}%"
                if summary["total_experiments"] > 0
                else "- **Success Rate**: N/A"
            ),
            "\n## Progress by Model Configuration",
        ]

        for config_name, config_data in summary["by_config"].items():
            report_lines.extend(
                [
                    f"\n### {config_name}",
                    f"- Total: {config_data['total']}",
                    f"- Successful: {config_data['successful']}",
                    f"- Failed: {config_data['failed']}",
                    (
                        f"- Success Rate: "
                        f"{config_data['successful'] / config_data['total'] * 100:.1f}%"
                        if config_data["total"] > 0
                        else "- Success Rate: N/A"
                    ),
                ]
            )

            if config_data["topics_completed"]:
                report_lines.append(
                    f"\n**Completed Topics ({len(config_data['topics_completed'])}):**"
                )
                for topic in config_data["topics_completed"]:
                    report_lines.append(f"-  {topic}")

            if config_data["topics_failed"]:
                report_lines.append(f"\n**Failed Topics ({len(config_data['topics_failed'])}):**")
                for topic in config_data["topics_failed"]:
                    report_lines.append(f"-  {topic}")

        if summary["failed"] > 0:
            report_lines.append("\n## Failed Experiments Details")
            for result in summary["results"]:
                if not result["success"]:
                    report_lines.extend(
                        [
                            f"\n### {result['topic_id']}  {result['model_config']}",
                            f"- **Error**: {result['error']}",
                        ]
                    )

        report_path = self.output_dir / f"batch_report_{self.batch_label}.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")

        print(
            f"\n Batch summary saved to: "
            f"{self.output_dir / f'batch_summary_{self.batch_label}.json'}"
        )
        print(f" Detailed report saved to: {report_path}")
        print(f"   Successful: {summary['successful']}/{summary['total_experiments']}")
        print(f"   Failed: {summary['failed']}/{summary['total_experiments']}")
