"""Immutable memory-state management for the debate simulator.

All mutation functions return a **new** ``MemoryState`` via
``model_copy(update=...)``, leaving the original untouched.
"""

from __future__ import annotations

from debate.core.schemas import AgentState, DebateLogItem, MemoryState, Scoreboard

#: Default initial confidence for both agents.
DEFAULT_INITIAL_CONFIDENCE: int = 55


def initial_memory(topic: str, motion: str) -> MemoryState:
    """Create a fresh :class:`MemoryState` at round 0 with default agent states."""
    return MemoryState(
        topic=topic,
        motion=motion,
        round=0,
        agent_states={
            "CA": AgentState(
                confidence=DEFAULT_INITIAL_CONFIDENCE,
                values=["skepticism", "narrative cohesion"],
                preferred_tactics=["pattern-seeking"],
                rejected_frames=["authority-only"],
                what_changes_mind="Verifiable evidence that directly contradicts the core conspiracy claim.",
            ),
            "SA": AgentState(
                confidence=DEFAULT_INITIAL_CONFIDENCE,
                values=["falsifiability", "empirical rigor"],
                preferred_tactics=["evidence"],
                rejected_frames=["anecdote-only"],
                what_changes_mind="Reproducible empirical evidence supporting the conspiracy claim.",
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
        next_round_strategy={"CA": "Establish core claim.", "SA": "Clarify and challenge."},
    )


def append_log(memory: MemoryState, item: DebateLogItem) -> MemoryState:
    """Append a debate-log entry and return a new :class:`MemoryState`."""
    return memory.model_copy(update={"debate_log": memory.debate_log + [item]})


def update_scoreboard(
    memory: MemoryState,
    stance_shift: dict[str, int],
    bridge_score: int,
    civility_score: int,
    epistemic_quality: int,
) -> MemoryState:
    """Replace the scoreboard and return a new :class:`MemoryState`."""
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
    """Update agent confidence and what_changes_mind - tracks persuasion over time."""
    agent_states = dict(memory.agent_states)
    agent = agent_states[speaker].model_copy(
        update={"confidence": confidence, "what_changes_mind": what_changes_mind}
    )
    agent_states[speaker] = agent
    return memory.model_copy(update={"agent_states": agent_states})


def update_round(memory: MemoryState, new_round: int) -> MemoryState:
    """Set the current round number and return a new :class:`MemoryState`."""
    return memory.model_copy(update={"round": new_round})
