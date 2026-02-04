from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .report_models import AggregateReport, AnalysisReport


def write_analysis_report(run_dir: Path, report: AnalysisReport) -> AnalysisReport:
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "analysis_report.json"
    report_path.write_text(report.model_dump_json(indent=2))

    markdown_path = run_dir / "analysis_report.md"
    markdown_path.write_text(_render_markdown(report))
    return report


def write_aggregate_report(output_dir: Path, report: AggregateReport) -> AggregateReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "aggregate_report.json"
    report_path.write_text(report.model_dump_json(indent=2))

    markdown_path = output_dir / "aggregate_report.md"
    markdown_path.write_text(_render_aggregate_markdown(report))
    return report


def _render_markdown(report: AnalysisReport) -> str:
    stance = report.stance_summary
    tactic = report.tactic_summary
    quality = report.quality_summary

    lines: List[str] = [
        "# Analysis Report",
        "",
        "## Overview",
        f"- Run ID: {report.run_id}",
        f"- Topic: {report.topic}",
        f"- Motion: {report.motion}",
        f"- Rounds Completed: {report.rounds_completed}",
    ]

    if report.run_config:
        lines.append("- Models: " + json.dumps(report.run_config.get("models", {})))

    lines.extend(
        [
            "",
            "## Stance Trajectory Summary",
            "| Agent | Start | End | Net Shift |",
            "| --- | --- | --- | --- |",
        ]
    )
    for agent in ["CA", "SA"]:
        lines.append(
            f"| {agent} | {stance.start_confidence.get(agent, 0)} | "
            f"{stance.end_confidence.get(agent, 0)} | {stance.net_shift.get(agent, 0)} |"
        )
    if stance.shift_events:
        lines.extend(["", "### Stance Shift Events"])
        for event in stance.shift_events:
            lines.append(
                f"- Round {event.round} {event.agent} Delta {event.delta} "
                f"({event.confidence_before} -> {event.confidence_after})"
            )

    lines.extend(["", "## Persuasion Moments"])
    if report.persuasion_moments:
        for moment in report.persuasion_moments:
            lines.append(
                f"- Round {moment.round} {moment.affected_agent} (Delta {moment.delta}): "
                f"{moment.why_flagged} - \"{moment.excerpt}\""
            )
    else:
        lines.append("- None detected with current thresholds.")

    lines.extend(["", "## Rhetorical Tactic Usage", "### CA Tactics", "| Tactic | Count |", "| --- | --- |"])
    for tactic_name, count in tactic.counts.get("CA", {}).items():
        lines.append(f"| {tactic_name} | {count} |")
    lines.append(f"- Diversity: {tactic.diversity.get('CA', 0)}")
    lines.append(f"- Entropy: {tactic.entropy.get('CA', 0.0):.2f}")

    lines.extend(["", "### SA Tactics", "| Tactic | Count |", "| --- | --- |"])
    for tactic_name, count in tactic.counts.get("SA", {}).items():
        lines.append(f"| {tactic_name} | {count} |")
    lines.append(f"- Diversity: {tactic.diversity.get('SA', 0)}")
    lines.append(f"- Entropy: {tactic.entropy.get('SA', 0.0):.2f}")

    lines.extend(["", "## Dialogue Quality", "| Metric | Mean | Min | Max | Trend |", "| --- | --- | --- | --- | --- |"])
    for metric, aggregate in quality.aggregates.items():
        trend = quality.trends.get(metric, 0.0)
        lines.append(
            f"| {metric} | {aggregate.mean:.2f} | {aggregate.min} | {aggregate.max} | {trend:.2f} |"
        )
    if any(values for values in quality.per_round.values()):
        lines.extend(["", "### Per-round Scores", "| Round | Civility | Epistemic | Bridge |", "| --- | --- | --- | --- |"])
        rounds = max(len(values) for values in quality.per_round.values() if values)
        for idx in range(rounds):
            civility = quality.per_round["civility"][idx] if idx < len(quality.per_round["civility"]) else 0
            epistemic = (
                quality.per_round["epistemic_quality"][idx]
                if idx < len(quality.per_round["epistemic_quality"])
                else 0
            )
            bridge = (
                quality.per_round["bridge_building"][idx]
                if idx < len(quality.per_round["bridge_building"])
                else 0
            )
            lines.append(f"| {idx + 1} | {civility} | {epistemic} | {bridge} |")

    lines.extend(["", "## Redundancy / Repetition"])
    lines.append(f"- Method: {report.redundancy_summary.method}")
    lines.append(f"- Overall redundancy: {report.redundancy_summary.overall:.2f}")
    for agent, value in report.redundancy_summary.by_agent.items():
        lines.append(f"- {agent} redundancy: {value:.2f}")

    if report.safety_flags:
        lines.extend(["", "## Safety / Civility Flags"])
        for flag in report.safety_flags:
            lines.append(f"- {flag}")

    lines.extend(["", "## Limitations"])
    for item in report.limitations:
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def _render_aggregate_markdown(report: AggregateReport) -> str:
    lines: List[str] = [
        "# Aggregate Analysis Report",
        "",
        f"- Runs analyzed: {report.runs_analyzed}",
        f"- Grouped by: {report.grouped_by}",
        "",
        "## Distributions",
    ]
    for key, values in report.distributions.items():
        mean = sum(values) / len(values) if values else 0.0
        lines.append(f"- {key}: mean {mean:.2f} (n={len(values)})")

    lines.extend(["", "## Best Cases"])
    for case in report.best_cases:
        lines.append(
            f"- {case.run_id}: CA net shift {case.ca_net_shift}, "
            f"civility mean {case.civility_mean:.2f}, "
            f"tactic diversity {case.tactic_diversity_ca}"
        )

    lines.extend(["", "## Worst Cases"])
    for case in report.worst_cases:
        lines.append(
            f"- {case.run_id}: CA net shift {case.ca_net_shift}, "
            f"civility mean {case.civility_mean:.2f}, "
            f"tactic diversity {case.tactic_diversity_ca}"
        )

    if report.groups:
        lines.extend(["", "## Groups"])
        for group, runs in report.groups.items():
            lines.append(f"- {group}: {len(runs)} runs")

    lines.extend(["", "## Limitations"])
    for item in report.limitations:
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"
