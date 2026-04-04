from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path


def _clean_text(paragraphs: list[str] | None) -> str:
    if not paragraphs:
        return ""
    parts = [p.strip() for p in paragraphs if isinstance(p, str) and p.strip()]
    return "\n\n".join(parts)


def _summary_excerpt(summary: str, max_len: int = 240) -> str:
    text = " ".join(summary.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "..."


def _map_role_and_stance(speakertype: str) -> tuple[str, str]:
    st = (speakertype or "").strip().lower()
    if st == "for":
        return "proponent", "pro"
    if st == "against":
        return "opponent", "con"
    return "moderator", "neutral"


def _safe_run_suffix(debate_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", debate_id).strip("_")


def _render_transcript(topic: str, motion: str, logs: list[dict]) -> str:
    role_map = {
        "proponent": "Conspiracy Advocate (CA)",
        "opponent": "Scientific Advocate (SA)",
        "moderator": "Moderator",
    }
    lines = ["# Debate Transcript", f"\nTopic: {topic}", f"Motion: {motion}", ""]
    if logs:
        lines.append(f"Debate ID: {logs[0]['debate_id']}")
        lines.append("")

    grouped: dict[int, list[dict]] = {}
    for item in logs:
        grouped.setdefault(item["round"], []).append(item)

    for round_num in sorted(grouped):
        lines.append(f"\n## Round {round_num}")
        for item in grouped[round_num]:
            display = role_map.get(item["speaker_role"], item["speaker_role"])
            lines.append(f"\n### {display}")
            lines.append(f"Claim: {item['utterance']}")
    lines.append("")
    return "\n".join(lines)


def _write_csv(logs: list[dict], output_path: Path) -> None:
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in logs:
            row = dict(item)
            row["confidence"] = "" if row["confidence"] is None else row["confidence"]
            row["tactic_used"] = row["tactic_used"] or ""
            row["tool_query"] = row["tool_query"] or ""
            row["reply_to_turn"] = "" if row["reply_to_turn"] is None else row["reply_to_turn"]
            writer.writerow(row)


def build_artifacts(dataset_path: Path, output_root: Path) -> None:
    debates = json.loads(dataset_path.read_text(encoding="utf-8"))

    topics_output: list[dict] = []
    all_logs: list[dict] = []

    output_root.mkdir(parents=True, exist_ok=True)

    # Remove legacy layout from previous extractor versions.
    for legacy in (output_root / "raw", output_root / "transcripts"):
        if legacy.exists() and legacy.is_dir():
            shutil.rmtree(legacy)

    for debate_id, debate in sorted(debates.items()):
        title = (debate.get("title") or "").strip()
        if not title:
            continue

        topic_entry = {
            "id": debate_id,
            "category": "iq2_real_debate",
            "topic": title,
            "motion": title,
            "description": _summary_excerpt((debate.get("summary") or "").strip())
            if (debate.get("summary") or "").strip()
            else f"IQ2 debate: {title}",
        }
        topics_output.append(topic_entry)

        logs: list[dict] = []
        for turn in debate.get("transcript") or []:
            utterance = _clean_text(turn.get("paragraphs") or [])
            if not utterance:
                continue

            role, stance = _map_role_and_stance(turn.get("speakertype", ""))
            segment = turn.get("segment")
            round_num = int(segment) + 1 if isinstance(segment, int) else 1

            logs.append(
                {
                    "debate_id": debate_id,
                    "claim": title,
                    "round": round_num,
                    "speaker_role": role,
                    "utterance": utterance,
                    "stance": stance,
                    "confidence": None,
                    "tactic_used": None,
                    "tool_used": "none",
                    "tool_query": None,
                    "reply_to_turn": None,
                }
            )

        all_logs.extend(logs)

        run_id = f"run_iq2_{_safe_run_suffix(debate_id)}"
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # debate_log.csv
        _write_csv(logs, run_dir / "debate_log.csv")

        # memory.json compatible with MemoryState schema
        memory = {
            "topic": title,
            "motion": title,
            "round": max((item["round"] for item in logs), default=0),
            "agent_states": {
                "CA": {
                    "confidence": 55,
                    "values": ["skepticism", "narrative cohesion"],
                    "preferred_tactics": ["pattern-seeking"],
                    "rejected_frames": ["authority-only"],
                    "what_changes_mind": "Verifiable evidence that directly contradicts the core claim.",
                },
                "SA": {
                    "confidence": 55,
                    "values": ["falsifiability", "empirical rigor"],
                    "preferred_tactics": ["evidence"],
                    "rejected_frames": ["anecdote-only"],
                    "what_changes_mind": "Reproducible empirical evidence supporting the core claim.",
                },
            },
            "debate_log": logs,
            "scoreboard": {
                "stance_shift": {"CA_delta": 0, "SA_delta": 0},
                "bridge_score": 0,
                "civility_score": 0,
                "epistemic_quality": 0,
            },
            "moderator_notes": [],
            "next_round_strategy": {
                "CA": "Continue argumentation based on transcript evidence.",
                "SA": "Continue rebuttal based on transcript evidence.",
            },
        }
        (run_dir / "memory.json").write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")

        # final_report.json compatible with FinalReport schema
        max_round = max((item["round"] for item in logs), default=0)
        final_report = {
            "topic": title,
            "motion": title,
            "rounds_completed": max_round,
            "stance_trajectory": {"CA": [55], "SA": [55]},
            "tactic_counts": {},
            "key_persuasion_moments": [],
            "outcome_summary": "Imported historical IQ2 debate transcript; no simulated persuasion trajectory computed.",
            "limitations": [
                "Confidence, tactics, and search usage are unavailable in source transcript and are left unset.",
                "Rounds are mapped from IQ2 segments (intro/discussion/conclusion) rather than simulator-generated rounds.",
            ],
        }
        (run_dir / "final_report.json").write_text(
            json.dumps(final_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # run_config + metadata for near-compatibility with existing artifact expectations
        run_config = {
            "source": "iq2",
            "models": {
                "moderator": "human_moderator",
                "conspiracy": "human_for_team",
                "scientific": "human_against_team",
            },
            "api_mode": "offline_dataset",
            "max_tokens": None,
            "word_limit": None,
            "rounds": max_round,
        }
        (run_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

        metadata = {
            "topic_id": debate_id,
            "topic_category": "iq2_real_debate",
            "topic_description": topic_entry["description"],
            "model_config": "iq2_human_transcript",
            "models": run_config["models"],
            "api_mode": run_config["api_mode"],
        }
        (run_dir / "experiment_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # metrics.csv placeholder with simulator-compatible columns.
        metrics_headers = [
            "round",
            "bridge_score",
            "civility_score",
            "epistemic_quality",
            "CA_delta",
            "SA_delta",
        ]
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_headers)
            writer.writeheader()

        transcript_text = _render_transcript(title, title, logs)
        (run_dir / "transcript.md").write_text(transcript_text, encoding="utf-8")

    topics_path = output_root / "topics_iq2_llm.json"
    topics_path.write_text(json.dumps(topics_output, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_csv(all_logs, output_root / "all_debates_iq2.csv")

    print(f"Wrote {len(topics_output)} topics to {topics_path}")
    print(f"Wrote {len(all_logs)} log entries to {output_root / 'all_debates_iq2.csv'}")
    print(f"Created per-debate artifacts in {output_root}")


def main() -> None:
    dataset_dir = Path(__file__).resolve().parent
    dataset_path = dataset_dir / "iq2_data_release.json"
    output_root = dataset_dir.parent / "old_artifacts_i2q"

    build_artifacts(dataset_path, output_root)


if __name__ == "__main__":
    main()
