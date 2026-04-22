from __future__ import annotations

import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, cast

from .features import load_run_inputs
from .language_analysis import language_use_summary_from_logs
from .metrics import (
    persuasion_moments,
    quality_summary_from_scores,
    redundancy_summary,
    safety_flags,
    stance_summary_from_logs,
    tactic_summary_from_logs,
)
from .plots import (
    plot_aggregate_histogram,
    plot_aggregate_scatter,
    plot_quality_scores,
    plot_stance_trajectory,
    plot_tactic_histogram,
)
from .report_models import (
    AggregateReport,
    AnalysisReport,
    RunCaseSummary,
)
from .report_writer import write_aggregate_report, write_analysis_report


@dataclass
class AnalysisSettings:
    shift_threshold: int = 5
    similarity_method: str = "tfidf"
    uncertainty_lexicon: list[str] | None = None
    strong_modality_lexicon: list[str] | None = None
    weak_modality_lexicon: list[str] | None = None


@dataclass
class CustomAnalyzerRunResult:
    name: str
    success: bool
    error: str | None = None
    show_in_summary: bool = True


def _runner_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log_custom_analyzer_event(
    analyzer_name: str,
    event: str,
    *,
    output_dir: Path | None = None,
    duration_seconds: float | None = None,
    detail: str | None = None,
) -> None:
    parts = [f"[analysis_runner] {_runner_timestamp()}", analyzer_name, event]
    if output_dir is not None:
        parts.append(f"output={output_dir}")
    if duration_seconds is not None:
        parts.append(f"duration={duration_seconds:.2f}s")
    if detail:
        parts.append(detail)
    print(" | ".join(parts))


def run_custom_analyzers(
    results_root: str = "results",
    input_runs_dir: str = "old_artifacts",
    artifacts_dir: str = "old_artifacts",
    skip_role_emotion: bool = False,
    skip_topic_analysis_emotion: bool = False,
    overwrite_existing: bool = False,
    max_runs: int | None = None,
    topic_analysis_emotion_model: str = "j-hartmann/emotion-english-distilroberta-base",
    topic_analysis_batch_size: int = 16,
    topic_analysis_max_length: int = 256,
    topic_analysis_uncertainty_lexicon: str | None = None,
    topic_analysis_strong_modality_lexicon: str | None = None,
    topic_analysis_weak_modality_lexicon: str | None = None,
    topic_analysis_nrc_lexicon: str | None = None,
    topic_analysis_emfd_lexicon: str | None = None,
    continue_on_error: bool = True,
) -> list[CustomAnalyzerRunResult]:
    """Run the custom analyzers sequentially in-process.

    The analyzers keep their own behavior and are executed one after another.
    """
    output_root = Path(results_root)
    debate_output_dir = output_root / "analysis" / "debate_analysis"
    llm_output_dir = output_root / "analysis" / "llm_analysis"
    role_output_dir = output_root / "analysis" / "role_analysis"
    topic_analysis_output_dir = output_root / "analysis" / "topic_analysis"
    topic_analysis_plots_dir = topic_analysis_output_dir / "plots"
    topic_analysis_visuals_dir = topic_analysis_output_dir / "top_words_visuals"

    from .debate_analysis.debate_analysis_pipeline import main as debate_analysis_main
    from .llm_analysis.plot_combined_emotion_trajectory import main as llm_combined_emotion_plots_main
    from .llm_analysis.run_analysis import main as llm_analysis_main
    from .role_analysis.role_analyzer import main as role_analysis_main
    from .topic_analysis.topic_analysis_pipeline import main as topic_analysis_main
    from .topic_analysis.topic_analysis_plots import main as topic_analysis_plots_main
    from .topic_analysis.visualize_top_words import main as topic_analysis_visuals_main

    debate_argv = [
        "--input-runs",
        str(Path(input_runs_dir)),
        "--output-analysis",
        str(debate_output_dir),
    ]
    if overwrite_existing:
        debate_argv.append("--overwrite-existing")

    llm_argv = [
        "--input-runs",
        str(Path(input_runs_dir)),
        "--output-analysis",
        str(llm_output_dir),
    ]
    if max_runs is not None:
        llm_argv.extend(["--max-runs", str(max_runs)])
    if overwrite_existing:
        llm_argv.append("--overwrite-existing")

    llm_combined_emotion_plots_argv = [
        "--output-root",
        str(llm_output_dir),
    ]

    role_argv = [
        "--artifacts",
        str(Path(artifacts_dir)),
        "--output-root",
        str(role_output_dir),
    ]
    if skip_role_emotion:
        role_argv.append("--skip-emotion")

    topic_analysis_argv = [
        "--artifacts",
        str(Path(input_runs_dir)),
        "--output_dir",
        str(topic_analysis_output_dir),
        "--emotion_model",
        topic_analysis_emotion_model,
        "--batch_size",
        str(topic_analysis_batch_size),
        "--max_length",
        str(topic_analysis_max_length),
    ]
    if topic_analysis_uncertainty_lexicon:
        topic_analysis_argv.extend(["--uncertainty_lexicon", topic_analysis_uncertainty_lexicon])
    if topic_analysis_strong_modality_lexicon:
        topic_analysis_argv.extend(["--strong_modality_lexicon", topic_analysis_strong_modality_lexicon])
    if topic_analysis_weak_modality_lexicon:
        topic_analysis_argv.extend(["--weak_modality_lexicon", topic_analysis_weak_modality_lexicon])
    if topic_analysis_nrc_lexicon:
        topic_analysis_argv.extend(["--nrc_lexicon", topic_analysis_nrc_lexicon])
    if topic_analysis_emfd_lexicon:
        topic_analysis_argv.extend(["--emfd_lexicon", topic_analysis_emfd_lexicon])
    if skip_topic_analysis_emotion:
        topic_analysis_argv.append("--skip-emotion")

    topic_analysis_plots_argv = [
        "--input_dir",
        str(topic_analysis_output_dir),
        "--output_dir",
        str(topic_analysis_plots_dir),
    ]

    topic_analysis_visuals_argv = [
        "--input_txt",
        str(topic_analysis_plots_dir / "top_words_by_topic_and_role.txt"),
        "--output_dir",
        str(topic_analysis_visuals_dir),
    ]

    runners: list[tuple[str, Callable[[list[str] | None], None], list[str], Path, bool]] = [
        ("debate_analysis", debate_analysis_main, debate_argv, debate_output_dir, True),
        ("topic_analysis", topic_analysis_main, topic_analysis_argv, topic_analysis_output_dir, True),
        ("role_analysis", role_analysis_main, role_argv, role_output_dir, True),
        ("llm_analysis", llm_analysis_main, llm_argv, llm_output_dir, True),
        (
            "llm_analysis_combined_emotion_plots",
            llm_combined_emotion_plots_main,
            llm_combined_emotion_plots_argv,
            llm_output_dir,
            False,
        ),
        ("topic_analysis_plots", topic_analysis_plots_main, topic_analysis_plots_argv, topic_analysis_plots_dir, False),
        (
            "topic_analysis_top_words_visuals",
            topic_analysis_visuals_main,
            topic_analysis_visuals_argv,
            topic_analysis_visuals_dir,
            False,
        ),
    ]

    results: list[CustomAnalyzerRunResult] = []
    _log_custom_analyzer_event(
        "custom_suite",
        "started",
        output_dir=output_root / "analysis",
        detail=f"analyzers={len(runners)}",
    )
    for name, func, argv, output_dir, show_in_summary in runners:
        started_at = perf_counter()
        _log_custom_analyzer_event(name, "started", output_dir=output_dir)
        try:
            func(argv)
            elapsed = perf_counter() - started_at
            _log_custom_analyzer_event(
                name,
                "finished",
                output_dir=output_dir,
                duration_seconds=elapsed,
            )
            results.append(CustomAnalyzerRunResult(name=name, success=True, show_in_summary=show_in_summary))
        except Exception:
            elapsed = perf_counter() - started_at
            error_trace = traceback.format_exc()
            final_line = error_trace.splitlines()[-1] if error_trace else "unknown error"
            _log_custom_analyzer_event(
                name,
                "failed",
                output_dir=output_dir,
                duration_seconds=elapsed,
                detail=final_line,
            )
            results.append(
                CustomAnalyzerRunResult(
                    name=name,
                    success=False,
                    error=error_trace,
                    show_in_summary=show_in_summary,
                )
            )
            if not continue_on_error:
                break

    completed = sum(1 for result in results if result.success)
    failed = len(results) - completed
    summarized = sum(1 for result in results if result.show_in_summary)
    _log_custom_analyzer_event(
        "custom_suite",
        "finished",
        output_dir=output_root / "analysis",
        detail=f"succeeded={completed} failed={failed} summarized={summarized}",
    )

    return results


def analyze_run(run_dir: str) -> AnalysisReport:
    """Analyze a single debate run and write reports/plots.

    Args:
        run_dir: Path to the raw run data (e.g., 'results/raw/run_2023...').

    Returns:
        The generated AnalysisReport object.
    """
    path = Path(run_dir)
    # Determine the project root (results/) from the raw run path
    # Expected structure: results/raw/run_ID -> results/
    results_root = path.parent.parent
    run_id = path.name

    # Set analysis target directory: results/analysis/run_ID
    analysis_dir = results_root / "analysis" / run_id
    analysis_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_run_inputs(path)
    settings = _settings_from_run_config(inputs.run_config)

    stance_summary = stance_summary_from_logs(inputs.memory, settings.shift_threshold)
    tactic_summary = tactic_summary_from_logs(inputs.memory.debate_log)
    quality_summary = quality_summary_from_scores(
        inputs.metrics.civility,
        inputs.metrics.epistemic_quality,
        inputs.metrics.bridge_building,
    )
    redundancy = redundancy_summary(inputs.memory, settings.similarity_method)
    persuasion = persuasion_moments(inputs.memory, inputs.transcript, settings.shift_threshold)
    safety = safety_flags(inputs.memory)
    language_use = language_use_summary_from_logs(
        inputs.memory.debate_log,
        uncertainty_lexicon=settings.uncertainty_lexicon,
        strong_modality_lexicon=settings.strong_modality_lexicon,
        weak_modality_lexicon=settings.weak_modality_lexicon,
    )

    # Plots go into results/analysis/run_ID/plots/
    figures_dir = analysis_dir / "plots"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = []
    if stance_summary.per_round_confidence["CA"] or stance_summary.per_round_confidence["SA"]:
        figure_paths.append(
            plot_stance_trajectory(figures_dir, stance_summary.per_round_confidence)
        )
    if any(len(values) > 0 for values in quality_summary.per_round.values()):
        figure_paths.append(plot_quality_scores(figures_dir, quality_summary.per_round))
    if tactic_summary.counts.get("CA"):
        figure_paths.append(plot_tactic_histogram(figures_dir, tactic_summary.counts.get("CA", {})))

    report = AnalysisReport(
        run_id=inputs.run_id,
        topic=inputs.memory.topic,
        motion=inputs.memory.motion,
        rounds_completed=inputs.memory.round,
        stance_summary=stance_summary,
        tactic_summary=tactic_summary,
        quality_summary=quality_summary,
        redundancy_summary=redundancy,
        persuasion_moments=persuasion,
        language_use=language_use,
        safety_flags=safety,
        limitations=_default_limitations(single_run=True),
        run_config=inputs.run_config,
        # Relativize figure paths to the analysis directory
        figures=[str(Path(fig).relative_to(analysis_dir)) for fig in figure_paths],
    )
    return write_analysis_report(analysis_dir, report)


def analyze_all(results_root: str) -> AggregateReport:
    """Analyze all runs found in the 'raw/' sub-directory.

    Args:
        results_root: The project output root (e.g., 'results/').

    Returns:
        The generated AggregateReport object.
    """
    root = Path(results_root)
    raw_root = root / "raw"

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_root}")

    run_dirs = sorted(
        [path for path in raw_root.iterdir() if path.is_dir() and path.name.startswith("run_")]
    )
    reports: list[AnalysisReport] = []

    # Analysis target for individual runs
    analysis_root = root / "analysis"

    for run_dir in run_dirs:
        run_id = run_dir.name
        report_path = analysis_root / run_id / "analysis_report.json"

        if report_path.exists():
            reports.append(
                AnalysisReport.model_validate_json(report_path.read_text(encoding="utf-8"))
            )
        else:
            reports.append(analyze_run(str(run_dir)))

    grouped_by = "model_signature"
    groups: dict[str, list[str]] = {}
    for report in reports:
        model_signature = None
        if report.run_config:
            cfg = cast(dict[str, Any], report.run_config)
            models = cast(dict[str, Any], cfg.get("models", {}))
            model_signature = "/".join(
                [
                    str(models.get("moderator", "")),
                    str(models.get("conspiracy", "")),
                    str(models.get("scientific", "")),
                ]
            )
        if not model_signature:
            model_signature = "unknown"
        groups.setdefault(model_signature, []).append(report.run_id)

    net_ca_shifts = [float(report.stance_summary.net_shift.get("CA", 0)) for report in reports]
    civility_means = [report.quality_summary.aggregates["civility"].mean for report in reports]
    epistemic_means = [
        report.quality_summary.aggregates["epistemic_quality"].mean for report in reports
    ]
    bridge_means = [report.quality_summary.aggregates["bridge_building"].mean for report in reports]
    tactic_diversity = [float(report.tactic_summary.diversity.get("CA", 0)) for report in reports]
    uncertainty_rates = [report.language_use.uncertainty.overall_rate_per_1000 for report in reports]

    summaries = [
        RunCaseSummary(
            run_id=report.run_id,
            ca_net_shift=report.stance_summary.net_shift.get("CA", 0),
            civility_mean=report.quality_summary.aggregates["civility"].mean,
            tactic_diversity_ca=report.tactic_summary.diversity.get("CA", 0),
            model_signature=groups_key_for_report(report, groups),
        )
        for report in reports
    ]

    best_cases = sorted(summaries, key=lambda item: abs(item.ca_net_shift), reverse=True)[:3]
    worst_cases = sorted(summaries, key=lambda item: item.civility_mean)[:3]

    # Aggregate reports live in results/analysis/aggregate/
    aggregate_dir = analysis_root / "aggregate"
    figures_dir = aggregate_dir / "plots"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = []
    if net_ca_shifts:
        figure_paths.append(plot_aggregate_histogram(figures_dir, net_ca_shifts))
    if civility_means:
        figure_paths.append(plot_aggregate_scatter(figures_dir, civility_means, net_ca_shifts))

    aggregate = AggregateReport(
        runs_analyzed=len(reports),
        grouped_by=grouped_by,
        distributions={
            "ca_net_shift": net_ca_shifts,
            "civility_mean": civility_means,
            "epistemic_mean": epistemic_means,
            "bridge_mean": bridge_means,
            "tactic_diversity_ca": tactic_diversity,
            "uncertainty_rate_per_1000": uncertainty_rates,
        },
        best_cases=best_cases,
        worst_cases=worst_cases,
        # Relativize figure paths to the aggregate analysis directory
        figures=[str(Path(fig).relative_to(aggregate_dir)) for fig in figure_paths],
        limitations=_default_limitations(single_run=False),
        groups=groups,
    )
    return write_aggregate_report(aggregate_dir, aggregate)


def groups_key_for_report(report: AnalysisReport, groups: dict[str, list[str]]) -> str | None:
    for key, runs in groups.items():
        if report.run_id in runs:
            return key
    return None


def _default_limitations(single_run: bool) -> list[str]:
    limitations = [
        "Deterministic heuristics may miss nuance in persuasion signals.",
        "Transcript parsing relies on consistent formatting in transcripts.",
        "Confidence trajectories are derived from agent self-reported values.",
    ]
    if not single_run:
        limitations.append("Cross-run comparisons assume consistent prompt structure.")
    return limitations


def _settings_from_run_config(run_config: dict[str, object] | None) -> AnalysisSettings:
    if not run_config:
        return AnalysisSettings()
    cfg = cast(dict[str, Any], run_config)
    analysis_cfg = cast(dict[str, Any], cfg.get("analysis", {}))
    return AnalysisSettings(
        shift_threshold=int(analysis_cfg.get("shift_threshold", 5)),
        similarity_method=str(analysis_cfg.get("similarity_method", "tfidf")),
        uncertainty_lexicon=analysis_cfg.get("uncertainty_lexicon"),
        strong_modality_lexicon=analysis_cfg.get("strong_modality_lexicon"),
        weak_modality_lexicon=analysis_cfg.get("weak_modality_lexicon"),
    )
