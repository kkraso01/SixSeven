"""CSV export functionality for canonical debate logs."""

from __future__ import annotations

import csv
from pathlib import Path

from debate.core.schemas import DebateLogItem


def export_debate_log_to_csv(
    logs: list[DebateLogItem],
    output_path: Path,
) -> None:
    """
    Export debate logs to CSV format for easy analysis.

    Args:
        logs: List of DebateLogItem objects
        output_path: Path where CSV file should be written
    """
    if not logs:
        return

    # Define CSV columns in canonical order
    fieldnames = [
        "debate_id",
        "claim",
        "round",
        "speaker_role",
        "utterance",
        "stance",
        "confidence",
        "tactic_used",
        "tool_used",
        "tool_query",
        "reply_to_turn",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in logs:
            writer.writerow(
                {
                    "debate_id": item.debate_id,
                    "claim": item.claim,
                    "round": item.round,
                    "speaker_role": item.speaker_role,
                    "utterance": item.utterance,
                    "stance": item.stance,
                    "confidence": item.confidence if item.confidence is not None else "",
                    "tactic_used": item.tactic_used or "",
                    "tool_used": item.tool_used,
                    "tool_query": item.tool_query or "",
                    "reply_to_turn": item.reply_to_turn if item.reply_to_turn is not None else "",
                }
            )


def export_all_debates_to_csv(
    artifacts_dir: Path,
    output_path: Path,
) -> int:
    """
    Export all debate logs from artifacts directory to a single CSV.

    Args:
        artifacts_dir: Directory containing run_* subdirectories
        output_path: Path where combined CSV should be written

    Returns:
        Number of debates exported
    """
    from debate.core.schemas import MemoryState

    all_logs: list[DebateLogItem] = []
    debate_count = 0

    # Find all run directories
    if not artifacts_dir.exists():
        print(f"Artifacts directory not found: {artifacts_dir}")
        return 0

    run_dirs = sorted(
        [d for d in artifacts_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    )

    for run_dir in run_dirs:
        memory_path = run_dir / "memory.json"
        if not memory_path.exists():
            continue

        try:
            memory = MemoryState.model_validate_json(memory_path.read_text(encoding="utf-8"))
            all_logs.extend(memory.debate_log)
            debate_count += 1
        except Exception as e:
            print(f"Error loading {memory_path}: {e}")
            continue

    if all_logs:
        export_debate_log_to_csv(all_logs, output_path)
        print(f"Exported {len(all_logs)} log entries from {debate_count} debates to {output_path}")

    return debate_count
