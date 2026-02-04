from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .features import load_run_inputs
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


def analyze_run(run_dir: str) -> AnalysisReport:
    path = Path(run_dir)
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

    figures_dir = path / "figures"
    figure_paths = []
    if stance_summary.per_round_confidence["CA"] or stance_summary.per_round_confidence["SA"]:
        figure_paths.append(plot_stance_trajectory(figures_dir, stance_summary.per_round_confidence))
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
        safety_flags=safety,
        limitations=_default_limitations(single_run=True),
        run_config=inputs.run_config,
    )
    report = report.model_copy(
        update={"figures": [str(Path(fig).relative_to(path)) for fig in figure_paths]}
    )
    return write_analysis_report(path, report)


def analyze_all(artifacts_root: str) -> AggregateReport:
    root = Path(artifacts_root)
    run_dirs = sorted([path for path in root.iterdir() if path.is_dir() and path.name.startswith("run_")])
    reports: List[AnalysisReport] = []
    for run_dir in run_dirs:
        report_path = run_dir / "analysis_report.json"
        if report_path.exists():
            reports.append(AnalysisReport.model_validate_json(report_path.read_text()))
        else:
            reports.append(analyze_run(str(run_dir)))

    grouped_by = "model_signature"
    groups: Dict[str, List[str]] = {}
    for report in reports:
        model_signature = None
        if report.run_config:
            models = report.run_config.get("models", {})
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

    net_ca_shifts = [report.stance_summary.net_shift.get("CA", 0) for report in reports]
    civility_means = [report.quality_summary.aggregates["civility"].mean for report in reports]
    epistemic_means = [report.quality_summary.aggregates["epistemic_quality"].mean for report in reports]
    bridge_means = [report.quality_summary.aggregates["bridge_building"].mean for report in reports]
    tactic_diversity = [report.tactic_summary.diversity.get("CA", 0) for report in reports]

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

    aggregate_dir = root / "aggregate"
    figures_dir = aggregate_dir / "figures"
    figure_paths = []
    if net_ca_shifts:
        figure_paths.append(plot_aggregate_histogram(figures_dir, net_ca_shifts))
    if civility_means:
        figure_paths.append(plot_aggregate_scatter(figures_dir, civility_means, net_ca_shifts))

    report = AggregateReport(
        runs_analyzed=len(reports),
        grouped_by=grouped_by,
        distributions={
            "ca_net_shift": net_ca_shifts,
            "civility_mean": civility_means,
            "epistemic_mean": epistemic_means,
            "bridge_mean": bridge_means,
            "tactic_diversity_ca": tactic_diversity,
        },
        best_cases=best_cases,
        worst_cases=worst_cases,
        figures=[str(Path(fig).relative_to(aggregate_dir)) for fig in figure_paths],
        limitations=_default_limitations(single_run=False),
        groups=groups,
    )
    return write_aggregate_report(aggregate_dir, report)


def groups_key_for_report(report: AnalysisReport, groups: Dict[str, List[str]]) -> str | None:
    for key, runs in groups.items():
        if report.run_id in runs:
            return key
    return None


def _default_limitations(single_run: bool) -> List[str]:
    limitations = [
        "Deterministic heuristics may miss nuance in persuasion signals.",
        "Transcript parsing relies on consistent formatting in transcripts.",
        "Confidence trajectories are derived from agent self-reported values.",
    ]
    if not single_run:
        limitations.append("Cross-run comparisons assume consistent prompt structure.")
    return limitations


def _settings_from_run_config(run_config: Dict[str, object] | None) -> AnalysisSettings:
    if not run_config:
        return AnalysisSettings()
    analysis_cfg = run_config.get("analysis", {})
    return AnalysisSettings(
        shift_threshold=int(analysis_cfg.get("shift_threshold", 5)),
        similarity_method=str(analysis_cfg.get("similarity_method", "tfidf")),
    )
