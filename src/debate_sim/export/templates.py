from __future__ import annotations

from typing import List

from ..schemas import DebateLogItem, ModeratorRecap


def render_transcript(
    topic: str,
    motion: str,
    logs: List[DebateLogItem],
    recaps: List[ModeratorRecap],
) -> str:
    # Map speaker roles to display names
    role_to_display = {
        "proponent": "Conspiracy Advocate (CA)",
        "opponent": "Scientific Advocate (SA)",
        "moderator": "Moderator"
    }
    
    lines = [f"# Debate Transcript", f"\nTopic: {topic}", f"Motion: {motion}", ""]
    
    # Include debate_id if available
    if logs:
        lines.append(f"Debate ID: {logs[0].debate_id}")
        lines.append("")
    
    round_groups = {}
    for item in logs:
        # Only include agent turns (not moderator) in main sections
        if item.speaker_role != "moderator":
            round_groups.setdefault(item.round, []).append(item)

    for recap in recaps:
        round_number = recap.round
        lines.append(f"\n## Round {round_number}")
        for item in round_groups.get(round_number, []):
            display_name = role_to_display.get(item.speaker_role, item.speaker_role)
            lines.append(f"\n### {display_name}")
            if item.tactic_used:
                lines.append(f"Tactic: {item.tactic_used}")
            if item.confidence is not None:
                lines.append(f"Confidence: {item.confidence}")
            lines.append(f"Claim: {item.utterance}")
            if item.tool_used != "none":
                lines.append(f"Tool Used: {item.tool_used}")
                if item.tool_query:
                    lines.append(f"Search Query: {item.tool_query}")
        lines.append("\n### Moderator Recap")
        lines.append(f"Agreements: {', '.join(recap.summary_agreements)}")
        lines.append(f"Disagreements: {', '.join(recap.summary_disagreements)}")
        lines.append(f"Civility: {recap.civility_score}")
        lines.append(f"Epistemic Quality: {recap.epistemic_quality_score}")
        lines.append(f"Bridge Building: {recap.bridge_building_score}")
        lines.append(
            f"Confidence Updates: CA {recap.confidence_updates.get('CA_delta', 0):+d}, "
            f"SA {recap.confidence_updates.get('SA_delta', 0):+d}"
        )
    lines.append("")
    return "\n".join(lines)
