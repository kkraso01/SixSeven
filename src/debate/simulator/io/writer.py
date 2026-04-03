from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from debate.core.schemas import DebateLogItem, FinalReport, MemoryState, ModeratorRecap

from .csv_export import export_debate_log_to_csv
from .templates import render_transcript


@dataclass
class ExportBundle:
    """Paths to all exported artifacts for a single debate run."""

    run_dir: Path
    transcript_path: Path
    memory_path: Path
    final_report_path: Path
    metrics_path: Path | None
    debate_log_csv_path: Path | None


def write_artifacts(
    output_dir: Path,
    memory: MemoryState,
    final_report: FinalReport,
    recaps: list[ModeratorRecap],
    ca_sa_log: list[DebateLogItem],
    metrics_table: list[dict],
    topic: str,
    motion: str,
    run_config: dict | None = None,
) -> ExportBundle:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = run_dir / "transcript.md"
    transcript = render_transcript(topic, motion, ca_sa_log, recaps)
    transcript_path.write_text(transcript, encoding="utf-8")

    memory_path = run_dir / "memory.json"
    memory_path.write_text(json.dumps(memory.model_dump(), indent=2), encoding="utf-8")

    final_report_path = run_dir / "final_report.json"
    final_report_path.write_text(json.dumps(final_report.model_dump(), indent=2), encoding="utf-8")

    if run_config:
        run_config_path = run_dir / "run_config.json"
        run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    metrics_path = None
    if metrics_table:
        metrics_path = run_dir / "metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics_table[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_table)

    # Export canonical debate log as CSV
    debate_log_csv_path = run_dir / "debate_log.csv"
    export_debate_log_to_csv(ca_sa_log, debate_log_csv_path)

    return ExportBundle(
        run_dir=run_dir,
        transcript_path=transcript_path,
        memory_path=memory_path,
        final_report_path=final_report_path,
        metrics_path=metrics_path,
        debate_log_csv_path=debate_log_csv_path,
    )
