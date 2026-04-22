from __future__ import annotations

import json
from pathlib import Path


def _speaker_names(speakers: list[dict] | None) -> list[str]:
    if not speakers:
        return []
    names: list[str] = []
    for speaker in speakers:
        name = speaker.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _summary_excerpt(summary: str, max_len: int = 240) -> str:
    text = " ".join(summary.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "..."


def build_iq2_topics(source_path: Path) -> list[dict]:
    debates = json.loads(source_path.read_text(encoding="utf-8"))
    topics: list[dict] = []

    for debate_id, debate in debates.items():
        title = (debate.get("title") or "").strip()
        if not title:
            continue

        speakers = debate.get("speakers") or {}
        transcript = debate.get("transcript") or []
        summary = (debate.get("summary") or "").strip()

        topic_entry = {
            "id": debate_id,
            "source": "iq2",
            "category": "iq2_real_debate",
            "topic": title,
            "motion": title,
            "description": _summary_excerpt(summary) if summary else f"IQ2 debate: {title}",
            "date": debate.get("date"),
            "url": debate.get("url"),
            "participants": {
                "for": _speaker_names(speakers.get("for")),
                "against": _speaker_names(speakers.get("against")),
                "moderator": (speakers.get("moderator") or {}).get("name"),
            },
            "stats": {
                "turn_count": len(transcript),
            },
            "results": debate.get("results"),
        }
        topics.append(topic_entry)

    topics.sort(key=lambda t: t["id"])
    return topics


def main() -> None:
    dataset_dir = Path(__file__).resolve().parent
    project_root = dataset_dir.parent
    source_path = dataset_dir / "iq2_data_release.json"
    output_path = project_root / "config" / "topics_iq2.json"

    topics = build_iq2_topics(source_path)
    output_path.write_text(json.dumps(topics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(topics)} IQ2 topics to {output_path}")


if __name__ == "__main__":
    main()
