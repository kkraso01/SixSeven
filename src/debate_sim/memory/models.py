from __future__ import annotations

from typing import Dict, List

from debate_sim.schemas import AgentState, DebateLogItem, MemoryState, Scoreboard


def initial_memory(topic: str, motion: str) -> MemoryState:
    return MemoryState(
        topic=topic,
        motion=motion,
        round=0,
        agent_states={
            "CA": AgentState(
                confidence=55,
                values=["skepticism", "narrative cohesion"],
                preferred_tactics=["pattern-seeking"],
                rejected_frames=["authority-only"],
                what_changes_mind="Clear disconfirming evidence and reliable sources.",
            ),
            "SA": AgentState(
                confidence=55,
                values=["falsifiability", "empirical rigor"],
                preferred_tactics=["evidence"],
                rejected_frames=["anecdote-only"],
                what_changes_mind="Reproducible evidence and transparent methods.",
            ),
        },
        debate_log=[],
        scoreboard=Scoreboard(
            stance_shift={"CA_delta": 0, "SA_delta": 0},
            bridge_score=0,
            civility_score=0,
            epistemic_quality=0,
        ),
        moderator_notes=[],
        next_round_strategy={"CA": "Stay focused on narrative claims.", "SA": "Stress tests."},
    )


def append_log(memory: MemoryState, item: DebateLogItem) -> MemoryState:
    return memory.model_copy(update={"debate_log": memory.debate_log + [item]})


def update_scoreboard(
    memory: MemoryState,
    stance_shift: Dict[str, int],
    bridge_score: int,
    civility_score: int,
    epistemic_quality: int,
) -> MemoryState:
    return memory.model_copy(
        update={
            "scoreboard": Scoreboard(
                stance_shift=stance_shift,
                bridge_score=bridge_score,
                civility_score=civility_score,
                epistemic_quality=epistemic_quality,
            )
        }
    )


def update_agent_state(
    memory: MemoryState,
    speaker: str,
    confidence: int,
    what_changes_mind: str,
) -> MemoryState:
    agent_states = dict(memory.agent_states)
    agent = agent_states[speaker].model_copy(
        update={"confidence": confidence, "what_changes_mind": what_changes_mind}
    )
    agent_states[speaker] = agent
    return memory.model_copy(update={"agent_states": agent_states})


def update_round(memory: MemoryState, new_round: int) -> MemoryState:
    return memory.model_copy(update={"round": new_round})
