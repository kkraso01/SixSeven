from __future__ import annotations

from typing import List

from ..schemas import DebateLogItem, ModeratorRecap


def render_transcript(
    topic: str,
    motion: str,
    logs: List[DebateLogItem],
    recaps: List[ModeratorRecap],
) -> str:
    lines = [f"# Debate Transcript", f"\nTopic: {topic}", f"Motion: {motion}", ""]
    round_groups = {}
    for item in logs:
        round_groups.setdefault(item.round, []).append(item)

    for recap in recaps:
        round_number = recap.round
        lines.append(f"\n## Round {round_number}")
        for item in round_groups.get(round_number, []):
            lines.append(f"\n### {item.speaker}")
            lines.append(f"Tactic: {item.tactic_used}")
            lines.append(f"Confidence: {item.confidence}")
            lines.append(f"Claim: {item.content}")
        lines.append("\n### Moderator Recap")
        lines.append(f"Agreements: {', '.join(recap.summary_agreements)}")
        lines.append(f"Disagreements: {', '.join(recap.summary_disagreements)}")
        lines.append(f"Civility: {recap.civility_score}")
        lines.append(f"Epistemic Quality: {recap.epistemic_quality_score}")
        lines.append(f"Bridge Building: {recap.bridge_building_score}")
        lines.append(
            f"Confidence Updates: CA {recap.confidence_updates.get('CA_delta', 0)}, "
            f"SA {recap.confidence_updates.get('SA_delta', 0)}"
        )
    lines.append("")
    return "\n".join(lines)
