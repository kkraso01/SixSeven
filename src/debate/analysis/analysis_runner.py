from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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
