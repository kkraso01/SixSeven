from __future__ import annotations

import re
from typing import Any


def infer_winner_from_stance_trajectory(final_report: dict[str, Any] | None) -> dict[str, Any]:
    if not final_report:
        return {
            "winner_inferred": None,
            "winner_role": None,
            "winner_source": None,
            "winner_confidence": "low",
            "winner_evidence": None,
        }

    traj = final_report.get("stance_trajectory", {}) or {}
    ca = traj.get("CA", [])
    sa = traj.get("SA", [])

    if len(ca) >= 2 and len(sa) >= 2:
        ca_change = ca[-1] - ca[0]
        sa_change = sa[-1] - sa[0]

        if sa_change > ca_change and sa[-1] > ca[-1]:
            return {
                "winner_inferred": "SA",
                "winner_role": "opponent",
                "winner_source": "stance_trajectory",
                "winner_confidence": "medium",
                "winner_evidence": {
                    "CA_start": ca[0], "CA_end": ca[-1],
                    "SA_start": sa[0], "SA_end": sa[-1],
                },
            }

        if ca_change > sa_change and ca[-1] > sa[-1]:
            return {
                "winner_inferred": "CA",
                "winner_role": "proponent",
                "winner_source": "stance_trajectory",
                "winner_confidence": "medium",
                "winner_evidence": {
                    "CA_start": ca[0], "CA_end": ca[-1],
                    "SA_start": sa[0], "SA_end": sa[-1],
                },
            }

    return {
        "winner_inferred": None,
        "winner_role": None,
        "winner_source": None,
        "winner_confidence": "low",
        "winner_evidence": None,
    }


def infer_winner_from_final_report(
    final_report: dict[str, Any] | None,
    use_explicit_winner_fields: bool = True,
    use_stance_trajectory_fallback: bool = True,
) -> dict[str, Any]:
    if not final_report:
        return {
            "winner_inferred": None,
            "winner_role": None,
            "winner_source": None,
            "winner_confidence": "low",
            "winner_evidence": None,
        }

    outcome_summary = str(final_report.get("outcome_summary", "")).strip()
    text = outcome_summary.lower()

    winner = None
    confidence = "low"
    source = None
    evidence: Any = None

    if use_explicit_winner_fields:
        explicit_winner = (
            final_report.get("winner")
            or final_report.get("winning_side")
            or final_report.get("winner_side")
        )
        if explicit_winner:
            val = str(explicit_winner).strip().lower()
            if val in {"sa", "scientific", "scientific advocate", "opponent"}:
                winner, confidence, source, evidence = "SA", "high", "explicit_winner_field", str(explicit_winner)
            elif val in {"ca", "conspiracy", "conspiracy advocate", "proponent"}:
                winner, confidence, source, evidence = "CA", "high", "explicit_winner_field", str(explicit_winner)

    if winner is None:
        patterns = [
            (r"\bsa successfully defended\b", "SA", "high"),
            (r"\bca successfully defended\b", "CA", "high"),
            (r"\bsa won\b|\bscientific advocate won\b", "SA", "high"),
            (r"\bca won\b|\bconspiracy advocate won\b", "CA", "high"),
            (r"\bstrengthening sa'?s position\b", "SA", "medium"),
            (r"\bstrengthening ca'?s position\b", "CA", "medium"),
        ]
        for pattern, label, conf in patterns:
            if re.search(pattern, text):
                winner, confidence, source, evidence = label, conf, "outcome_summary", outcome_summary
                break

    if winner is None and use_stance_trajectory_fallback:
        stance_result = infer_winner_from_stance_trajectory(final_report)
        if stance_result.get("winner_inferred") is not None:
            return stance_result

    role_map = {"CA": "proponent", "SA": "opponent"}
    return {
        "winner_inferred": winner,
        "winner_role": role_map.get(winner),
        "winner_source": source,
        "winner_confidence": confidence,
        "winner_evidence": evidence if winner else None,
    }
