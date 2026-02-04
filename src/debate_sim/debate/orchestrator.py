from __future__ import annotations

import hashlib
import json
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import Dict, List

from debate_sim.config import DebateConfig
from debate_sim.debate.evaluation import build_metrics_table, stance_shift
from debate_sim.debate.protocol import load_prompt
from debate_sim.export.writer import ExportBundle, write_artifacts
from debate_sim.llm.instructor_wrapper import StructuredLLM
from debate_sim.llm.ollama_client import OllamaClient
from debate_sim.memory.models import (
    append_log,
    initial_memory,
    update_agent_state,
    update_round,
    update_scoreboard,
)
from debate_sim.schemas import (
    AgentTurn,
    DebateLogItem,
    FinalReport,
    MemoryState,
    ModeratorRecap,
    ScientificTurn,
)


def _round_plan(round_number: int) -> str:
    return f"Round {round_number} focus: clarify positions, test evidence, maintain civility."


def _moderator_messages(topic: str, motion: str, round_number: int, word_limit: int) -> List[Dict[str, str]]:
    content = load_prompt(
        "moderator.md",
        {
            "topic": topic,
            "motion": motion,
            "round": str(round_number),
            "word_limit": str(word_limit),
        },
    )
    return [{"role": "system", "content": content}]


def _agent_messages(prompt_name: str, topic: str, motion: str, round_number: int, word_limit: int) -> List[Dict[str, str]]:
    content = load_prompt(
        prompt_name,
        {
            "topic": topic,
            "motion": motion,
            "round": str(round_number),
            "word_limit": str(word_limit),
        },
    )
    return [{"role": "system", "content": content}]


def _memory_summary(memory: MemoryState) -> str:
    return json.dumps(
        {
            "round": memory.round,
            "agent_states": {
                key: value.model_dump() for key, value in memory.agent_states.items()
            },
            "scoreboard": memory.scoreboard.model_dump(),
        },
        indent=2,
    )


def _prompt_hashes() -> Dict[str, str]:
    prompt_dir = Path(__file__).resolve().parents[1] / "prompts"
    hashes: Dict[str, str] = {}
    for prompt_path in prompt_dir.glob("*.md"):
        content = prompt_path.read_text().encode("utf-8")
        hashes[prompt_path.name] = hashlib.sha256(content).hexdigest()
    return hashes


def _run_config(config: DebateConfig, rounds: int) -> Dict[str, object]:
    return {
        "timestamp": datetime.now().isoformat(),
        "models": {
            "moderator": config.moderator_model,
            "conspiracy": config.conspiracy_model,
            "scientific": config.scientific_model,
        },
        "temperatures": {
            "moderator": config.moderator_temperature,
            "conspiracy": config.conspiracy_temperature,
            "scientific": config.scientific_temperature,
        },
        "rounds": rounds,
        "word_limit": config.word_limit,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "api_mode": config.api_mode,
        "prompt_hashes": _prompt_hashes(),
        "analysis": {
            "shift_threshold": config.analysis_shift_threshold,
            "similarity_method": config.analysis_similarity_method,
        },
    }


def _update_memory_from_turn(memory: MemoryState, turn: AgentTurn) -> MemoryState:
    memory = append_log(
        memory,
        DebateLogItem(
            round=turn.round,
            speaker=turn.speaker,
            content=turn.claim,
            tactic_used=turn.tactic_used,
            confidence=turn.confidence,
        ),
    )
    return update_agent_state(memory, turn.speaker, turn.confidence, turn.what_changes_mind)


def _apply_recap_updates(memory: MemoryState, recap: ModeratorRecap) -> MemoryState:
    ca_delta = recap.confidence_updates.get("CA_delta", 0)
    sa_delta = recap.confidence_updates.get("SA_delta", 0)
    ca_conf = max(0, min(100, memory.agent_states["CA"].confidence + ca_delta))
    sa_conf = max(0, min(100, memory.agent_states["SA"].confidence + sa_delta))
    memory = update_agent_state(memory, "CA", ca_conf, memory.agent_states["CA"].what_changes_mind)
    memory = update_agent_state(memory, "SA", sa_conf, memory.agent_states["SA"].what_changes_mind)
    return memory


def _final_report_prompt(topic: str, motion: str, rounds: int, memory: MemoryState) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the Moderator Agent. Produce the final report. "
                "Return ONLY valid JSON matching the FinalReport schema."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\nMotion: {motion}\nRounds: {rounds}\n"
                f"Memory: {memory.model_dump_json(indent=2)}"
            ),
        },
    ]


def run_debate(
    topic: str,
    motion: str,
    rounds: int,
    config: DebateConfig,
) -> ExportBundle:
    memory = initial_memory(topic, motion)
    client = OllamaClient(config)
    llm = StructuredLLM(client)
    recaps: List[ModeratorRecap] = []
    tactic_counts: Counter[str] = Counter()

    for round_number in range(1, rounds + 1):
        previous_memory = memory
        memory = update_round(memory, round_number)
        moderator_messages = _moderator_messages(topic, motion, round_number, config.word_limit)
        moderator_messages.append(
            {
                "role": "user",
                "content": f"Round plan: {_round_plan(round_number)}\nMemory: {_memory_summary(memory)}",
            }
        )
        ca_messages = _agent_messages("conspiracy.md", topic, motion, round_number, config.word_limit)
        ca_messages.append(
            {"role": "user", "content": f"Memory: {_memory_summary(memory)}"}
        )
        ca_turn = llm.call(
            AgentTurn,
            ca_messages,
            model=config.conspiracy_model,
            temperature=config.conspiracy_temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
        memory = _update_memory_from_turn(memory, ca_turn)

        sa_messages = _agent_messages("scientific.md", topic, motion, round_number, config.word_limit)
        sa_messages.append(
            {
                "role": "user",
                "content": (
                    f"Memory: {_memory_summary(memory)}\n"
                    f"CA output: {ca_turn.model_dump_json(indent=2)}"
                ),
            }
        )
        sa_turn = llm.call(
            ScientificTurn,
            sa_messages,
            model=config.scientific_model,
            temperature=config.scientific_temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
        memory = _update_memory_from_turn(memory, sa_turn)

        recap_messages = list(moderator_messages)
        recap_messages.append(
            {
                "role": "user",
                "content": (
                    f"CA turn: {ca_turn.model_dump_json(indent=2)}\n"
                    f"SA turn: {sa_turn.model_dump_json(indent=2)}"
                ),
            }
        )
        recap = llm.call(
            ModeratorRecap,
            recap_messages,
            model=config.moderator_model,
            temperature=config.moderator_temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
        recaps.append(recap)
        memory = _apply_recap_updates(memory, recap)

        shift = stance_shift(previous_memory, memory)
        memory = update_scoreboard(
            memory,
            stance_shift=shift,
            bridge_score=recap.bridge_building_score,
            civility_score=recap.civility_score,
            epistemic_quality=recap.epistemic_quality_score,
        )

        tactic_counts.update([ca_turn.tactic_used, sa_turn.tactic_used])
        tactic_counts.update(recap.detected_fallacies_or_moves)

    final_report = llm.call(
        FinalReport,
        _final_report_prompt(topic, motion, rounds, memory),
        model=config.moderator_model,
        temperature=config.moderator_temperature,
        max_tokens=config.max_tokens,
        seed=config.seed,
    )

    metrics_table = build_metrics_table(recaps)
    bundle = write_artifacts(
        output_dir=Path(config.output_dir),
        memory=memory,
        final_report=final_report,
        recaps=recaps,
        ca_sa_log=memory.debate_log,
        metrics_table=metrics_table,
        topic=topic,
        motion=motion,
        run_config=_run_config(config, rounds),
    )
    if config.run_analysis:
        from debate_sim.analysis.analysis_runner import analyze_run

        analyze_run(str(bundle.run_dir))
    return bundle
