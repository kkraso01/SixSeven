from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..schemas import MemoryState


@dataclass
class TranscriptRound:
    round_number: int
    claims: Dict[str, str]
    recap_lines: List[str]


@dataclass
class TranscriptData:
    rounds: Dict[int, TranscriptRound]


@dataclass
class MetricsData:
    civility: List[int]
    epistemic_quality: List[int]
    bridge_building: List[int]


@dataclass
class RunInputs:
    run_id: str
    memory: MemoryState
    transcript: TranscriptData
    metrics: MetricsData
    run_config: Optional[Dict[str, object]]


KEYWORD_PATTERN = re.compile(
    r"\b(strong point|concession|agreement|compelling|persuasive|valid point|acknowledge)\b",
    re.IGNORECASE,
)


PROFANITY_PATTERN = re.compile(
    r"\b(idiot|stupid|moron|dumb|shut up|liar|bullshit|nonsense)\b",
    re.IGNORECASE,
)


def load_memory(run_dir: Path) -> MemoryState:
    memory_path = run_dir / "memory.json"
    return MemoryState.model_validate_json(memory_path.read_text(encoding="utf-8"))


def load_run_config(run_dir: Path) -> Optional[Dict[str, object]]:
    path = run_dir / "run_config.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_transcript(run_dir: Path) -> TranscriptData:
    path = run_dir / "transcript.md"
    if not path.exists():
        return TranscriptData(rounds={})
    text = path.read_text(encoding="utf-8")
    return parse_transcript(text)


def load_metrics(run_dir: Path, transcript: TranscriptData) -> MetricsData:
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        civility: List[int] = []
        epistemic: List[int] = []
        bridge: List[int] = []
        with metrics_path.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                civility.append(int(row.get("civility", 0)))
                epistemic.append(int(row.get("epistemic_quality", 0)))
                bridge.append(int(row.get("bridge", 0)))
        return MetricsData(civility=civility, epistemic_quality=epistemic, bridge_building=bridge)
    return metrics_from_transcript(transcript)


def parse_transcript(text: str) -> TranscriptData:
    rounds: Dict[int, TranscriptRound] = {}
    current_round: Optional[int] = None
    current_speaker: Optional[str] = None
    in_recap = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## Round "):
            try:
                current_round = int(line.replace("## Round ", "").strip())
            except ValueError:
                current_round = None
            if current_round is not None:
                rounds[current_round] = TranscriptRound(
                    round_number=current_round,
                    claims={},
                    recap_lines=[],
                )
            current_speaker = None
            in_recap = False
            continue
        if line.startswith("### "):
            header = line.replace("### ", "").strip()
            if header == "Moderator Recap":
                in_recap = True
                current_speaker = None
            else:
                current_speaker = header
                in_recap = False
            continue
        if current_round is None:
            continue
        round_data = rounds[current_round]
        if line.startswith("Claim:") and current_speaker:
            round_data.claims[current_speaker] = line.replace("Claim:", "").strip()
        if in_recap and line:
            round_data.recap_lines.append(line)

    return TranscriptData(rounds=rounds)


def metrics_from_transcript(transcript: TranscriptData) -> MetricsData:
    civility: List[int] = []
    epistemic: List[int] = []
    bridge: List[int] = []
    for round_number in sorted(transcript.rounds):
        recap_lines = transcript.rounds[round_number].recap_lines
        civility.append(_extract_score(recap_lines, "Civility"))
        epistemic.append(_extract_score(recap_lines, "Epistemic Quality"))
        bridge.append(_extract_score(recap_lines, "Bridge Building"))
    return MetricsData(civility=civility, epistemic_quality=epistemic, bridge_building=bridge)


def _extract_score(lines: List[str], label: str) -> int:
    for line in lines:
        if line.startswith(f"{label}:"):
            value = line.replace(f"{label}:", "").strip()
            if value.isdigit():
                return int(value)
    return 0


def extract_recap_text(transcript: TranscriptData, round_number: int) -> str:
    round_data = transcript.rounds.get(round_number)
    if not round_data:
        return ""
    return " ".join(round_data.recap_lines)


def detect_recap_keyword(text: str) -> bool:
    return bool(KEYWORD_PATTERN.search(text))


def excerpt_for_round(
    transcript: TranscriptData,
    round_number: int,
    agent: str,
    fallback: str,
) -> str:
    round_data = transcript.rounds.get(round_number)
    if round_data and agent in round_data.claims:
        return round_data.claims[agent][:300]
    return fallback[:300]


def safety_flags_from_text(log_texts: List[Tuple[int, str, str]]) -> List[str]:
    flags = []
    for round_number, speaker, text in log_texts:
        if PROFANITY_PATTERN.search(text):
            snippet = text.strip().split("\n")[0][:120]
            flags.append(f"Round {round_number} {speaker}: {snippet}")
    return flags


def load_run_inputs(run_dir: Path) -> RunInputs:
    memory = load_memory(run_dir)
    transcript = load_transcript(run_dir)
    metrics = load_metrics(run_dir, transcript)
    run_config = load_run_config(run_dir)
    return RunInputs(
        run_id=run_dir.name,
        memory=memory,
        transcript=transcript,
        metrics=metrics,
        run_config=run_config,
    )
