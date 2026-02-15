from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from ..core.config import DebateConfig
from ..core.container import DebateServices, build_default_services
from ..core.protocols import PromptLoader, SearchProvider
from ..core.schemas import (
    AgentTurn,
    DebateLogItem,
    FinalReport,
    MemoryState,
    ModeratorDecision,
    ModeratorRecap,
    ScientificTurn,
    SearchPlan,
)
from ..export.writer import ExportBundle
from ..memory.models import (
    append_log,
    initial_memory,
    update_agent_state,
    update_round,
    update_scoreboard,
)
from .evaluation import build_metrics_table, stance_shift

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

#: Default number of search results returned per agent query.
SEARCH_MAX_RESULTS: int = 3
#: Maximum characters included when formatting search results for the prompt.
SEARCH_MAX_CHARS: int = 800
#: Messages-per-round estimate used when trimming history by rounds.
MESSAGES_PER_ROUND: int = 4
#: Token budget for the final report (needs more space than regular turns).
FINAL_REPORT_MAX_TOKENS: int = 2500
#: Default max retries for individual LLM calls.
LLM_MAX_RETRIES: int = 5


def _round_plan(round_number: int) -> str:
    return f"Round {round_number} focus: clarify positions, test evidence, maintain civility."


def _handle_agent_search(
    agent_turn: AgentTurn,
    agent_type: str,
    topic: str,
    search: SearchProvider,
) -> str | None:
    """
    Handle search request from an agent if present.

    Args:
        agent_turn: The agent's turn response
        agent_type: Either "CA" or "SA" to determine search strategy
        topic: The debate topic for context
        search: Injected search provider

    Returns:
        Formatted search results string if search was performed, None otherwise
    """
    if not agent_turn.search or not agent_turn.search.should_search:
        return None

    query = agent_turn.search.search_query
    if not query:
        return None

    print(f"    Searching: {query}")

    # Perform the search via injected provider
    response = search.search(query, max_results=SEARCH_MAX_RESULTS)

    if response.success and response.results:
        print(f"    Found {len(response.results)} results")
        return search.format_results(response, max_chars=SEARCH_MAX_CHARS)
    else:
        logger.warning("Search failed or returned no results for query: %s", query)
        return None


def _run_agent_turn_with_search(
    *,
    response_model: type,
    agent_messages: list[dict[str, str]],
    agent_type: str,
    topic: str,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int | None,
    llm,
    search: SearchProvider,
    max_search_rounds: int = 3,
):
    """Agentic search loop: the agent can research up to *max_search_rounds*
    times before composing its argument.

    Each iteration:
      1. Ask the LLM for a ``SearchPlan`` — should I search (more)?
      2. If yes, execute the search and accumulate results.
      3. If no (or budget exhausted), break and generate the full argument.

    All accumulated search results are injected into the prompt once before
    the final argument generation.
    """
    accumulated_results: list[str] = []
    last_query: str | None = None
    last_rationale: str | None = None

    for search_round in range(1, max_search_rounds + 1):
        # Build the planning prompt with any results gathered so far
        prior_context = ""
        if accumulated_results:
            prior_context = (
                "\n\nRESEARCH GATHERED SO FAR:\n"
                + "\n---\n".join(accumulated_results)
                + "\n\nYou may search again with a DIFFERENT query if you "
                "need additional evidence, or set should_search=false to "
                "proceed to your argument."
            )

        search_plan_messages = list(agent_messages) + [
            {
                "role": "user",
                "content": (
                    f"Research round {search_round}/{max_search_rounds}. "
                    "Before composing your argument, decide whether you need to "
                    "search the web for evidence this turn. Return ONLY a JSON "
                    "with should_search (bool), search_query (string or null), "
                    "and search_rationale (string or null)."
                    + prior_context
                ),
            }
        ]

        plan = llm.call(
            SearchPlan,
            search_plan_messages,
            model=model,
            temperature=temperature,
            max_tokens=1024,
            seed=seed,
            max_retries=LLM_MAX_RETRIES,
        )

        if not plan.should_search or not plan.search_query:
            print(f"    [{agent_type}] No more research needed (round {search_round}/{max_search_rounds})")
            break

        print(f"    [{agent_type}] Research {search_round}/{max_search_rounds}: {plan.search_query}")
        last_query = plan.search_query
        last_rationale = plan.search_rationale

        response = search.search(plan.search_query, max_results=SEARCH_MAX_RESULTS)
        if response.success and response.results:
            text = search.format_results(response, max_chars=SEARCH_MAX_CHARS)
            accumulated_results.append(f"[Query: {plan.search_query}]\n{text}")
            print(f"    [{agent_type}] Found {len(response.results)} results")
        else:
            logger.warning("Search returned no results for: %s", plan.search_query)

    # ── Generate full argument with ALL accumulated evidence ─────────────
    argument_messages = list(agent_messages)
    if accumulated_results:
        argument_messages.append(
            {
                "role": "user",
                "content": (
                    "SEARCH RESULTS (use these as evidence in your argument):\n\n"
                    + "\n---\n".join(accumulated_results)
                ),
            }
        )

    turn = llm.call(
        response_model,
        argument_messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        max_retries=LLM_MAX_RETRIES,
    )

    # Attach the search info to the turn so logging picks it up
    if accumulated_results and last_query:
        from ..core.schemas import SearchRequest

        turn.search = SearchRequest(
            should_search=True,
            search_query=last_query,
            search_rationale=last_rationale,
        )

    return turn


SpeakerRole = Literal["proponent", "opponent", "moderator"]
Stance = Literal["pro", "con", "neutral"]
ToolUsed = Literal["none", "tavily", "duckduckgo"]


def _create_debate_log_item(
    debate_id: str,
    claim: str,
    round_num: int,
    speaker: str,
    utterance: str,
    stance: Stance,
    confidence: int | None,
    tactic_used: str | None,
    tool_used: ToolUsed = "none",
    tool_query: str | None = None,
    reply_to_turn: int | None = None,
) -> DebateLogItem:
    """Create a properly formatted canonical debate log item."""
    # Map speaker codes to roles
    speaker_role_map: dict[str, SpeakerRole] = {
        "CA": "proponent",
        "SA": "opponent",
        "MA": "moderator",
    }

    return DebateLogItem(
        debate_id=debate_id,
        claim=claim,
        round=round_num,
        speaker_role=speaker_role_map.get(speaker, "moderator"),
        utterance=utterance,
        stance=stance,
        confidence=confidence,
        tactic_used=tactic_used,
        tool_used=tool_used,
        tool_query=tool_query,
        reply_to_turn=reply_to_turn,
    )


def _compact_memory_summary(memory: MemoryState) -> str:
    """Create a compact memory summary (not full JSON)."""
    ca_state = memory.agent_states.get("CA", None)
    sa_state = memory.agent_states.get("SA", None)

    return (
        f"Round {memory.round} | "
        f"CA confidence: {ca_state.confidence if ca_state else 'N/A'} | "
        f"SA confidence: {sa_state.confidence if sa_state else 'N/A'} | "
        f"Scoreboard: Bridge={memory.scoreboard.bridge_score}, "
        f"Civility={memory.scoreboard.civility_score}, "
        f"Quality={memory.scoreboard.epistemic_quality}"
    )


def _trim_history(
    history: list[dict[str, str]], config: DebateConfig, current_round: int
) -> tuple[list[dict[str, str]], bool]:
    """Trim conversation history based on config settings. Returns (trimmed_history, was_trimmed)."""
    if config.history_trim == "none":
        return history, False

    was_trimmed = False
    working_history = list(history)

    # Trim by rounds
    if config.history_trim == "rounds" and config.max_rounds_in_history:
        # Keep system messages + last N rounds worth of messages
        system_msgs = [msg for msg in working_history if msg["role"] == "system"]
        non_system = [msg for msg in working_history if msg["role"] != "system"]

        # Approximate: ~3-4 messages per round (CA, SA, moderator)
        max_msgs = config.max_rounds_in_history * MESSAGES_PER_ROUND
        if len(non_system) > max_msgs:
            non_system = non_system[-max_msgs:]
            was_trimmed = True

        working_history = system_msgs + non_system

    # Trim by message count
    if config.history_trim == "messages" and config.max_messages_in_history:
        system_msgs = [msg for msg in working_history if msg["role"] == "system"]
        non_system = [msg for msg in working_history if msg["role"] != "system"]

        if len(non_system) > config.max_messages_in_history:
            non_system = non_system[-config.max_messages_in_history :]
            was_trimmed = True

        working_history = system_msgs + non_system

    # Trim by character count
    if config.history_trim == "chars" and config.max_chars_in_history:
        total_chars = sum(len(msg.get("content", "")) for msg in working_history)
        if total_chars > config.max_chars_in_history:
            # Keep system messages, trim from oldest non-system
            system_msgs = [msg for msg in working_history if msg["role"] == "system"]
            non_system = [msg for msg in working_history if msg["role"] != "system"]

            while non_system and total_chars > config.max_chars_in_history:
                removed = non_system.pop(0)
                total_chars -= len(removed.get("content", ""))

            was_trimmed = True
            working_history = system_msgs + non_system

    return working_history, was_trimmed


def _moderator_messages(
    topic: str,
    motion: str,
    round_number: int,
    word_limit: int,
    prompts: PromptLoader,
) -> list[dict[str, str]]:
    content = prompts.load(
        "moderator.md",
        {
            "topic": topic,
            "motion": motion,
            "round": str(round_number),
            "word_limit": str(word_limit),
            "max_rounds": "TBD",  # Will be updated when called
        },
    )
    return [{"role": "system", "content": content}]


def _moderator_decision_messages(
    topic: str,
    motion: str,
    current_round: int,
    max_rounds: int,
    initial_ca_confidence: int,
    initial_sa_confidence: int,
    current_ca_confidence: int,
    current_sa_confidence: int,
    recent_recap: ModeratorRecap,
    prompts: PromptLoader,
) -> list[dict[str, str]]:
    """Build messages for moderator to decide whether to continue debate."""
    content = prompts.load(
        "moderator_decision.md",
        {
            "topic": topic,
            "motion": motion,
            "round": str(current_round),
            "max_rounds": str(max_rounds),
        },
    )

    ca_total_shift = current_ca_confidence - initial_ca_confidence
    sa_total_shift = current_sa_confidence - initial_sa_confidence
    ca_delta = recent_recap.confidence_updates.get("CA_delta", 0)
    sa_delta = recent_recap.confidence_updates.get("SA_delta", 0)

    decision_context = (
        f"PERSUASION TRACKING STATUS:\n"
        f"- Current round: {current_round}/{max_rounds}\n\n"
        f"CONFIDENCE LEVELS:\n"
        f"- CA: initial={initial_ca_confidence}, current={current_ca_confidence}, total shift={ca_total_shift:+d}\n"
        f"- SA: initial={initial_sa_confidence}, current={current_sa_confidence}, total shift={sa_total_shift:+d}\n\n"
        f"THIS ROUND'S DELTAS:\n"
        f"- CA delta: {ca_delta:+d}\n"
        f"- SA delta: {sa_delta:+d}\n\n"
        f"ROUND {current_round} QUALITY:\n"
        f"- Civility: {recent_recap.civility_score}/5\n"
        f"- Epistemic quality: {recent_recap.epistemic_quality_score}/5\n"
        f"- Bridge building: {recent_recap.bridge_building_score}/5\n\n"
        f"KEY QUESTION: Has either agent shifted 20+ points from initial position?\n"
        f"Should the debate continue to round {current_round + 1}?"
    )

    return [{"role": "system", "content": content}, {"role": "user", "content": decision_context}]


def _build_agent_messages_with_history(
    agent_role_prompt: str,
    conversation_history: list[dict[str, str]],
    memory: MemoryState,
    config: DebateConfig,
    round_number: int,
    opponent_last_message: str | None = None,
) -> list[dict[str, str]]:
    """Build complete message list for an agent with full conversation history.

    Message order:
    1. System: Agent role prompt
    2. System: Debate rules (short, fixed)
    3. History: All prior debate messages (as actual chat messages)
    4. User: Compact memory summary
    5. User: This round instruction
    6. User: Opponent's last message (highlighted) - if enabled
    """
    messages = []

    # 1. Agent role prompt
    messages.append({"role": "system", "content": agent_role_prompt})

    # 2. Debate rules (fixed, short)
    rules = (
        "DEBATE RULES:\n"
        "- Stay within word limit\n"
        "- Respond directly to opponent's points\n"
        "- Support claims with reasoning\n"
        "- Maintain civility and epistemic humility\n"
        "- Question assertions, seek clarification"
    )
    messages.append({"role": "system", "content": rules})

    # 3. Inject full conversation history (trim if needed)
    if config.history_mode == "global_full" and conversation_history:
        trimmed_history, was_trimmed = _trim_history(conversation_history, config, round_number)

        # Add trimming notice if history was trimmed
        if was_trimmed and config.summarize_if_trimmed:
            messages.append(
                {"role": "user", "content": "[Earlier rounds truncated - showing recent history]"}
            )

        # Extend with actual chat history (preserves native format)
        messages.extend(trimmed_history)

    # 4. Compact memory summary
    if config.include_memory_summary:
        memory_compact = _compact_memory_summary(memory)
        messages.append({"role": "user", "content": f"CURRENT STATE: {memory_compact}"})

    # 5. Round instruction
    round_instr = f"Round {round_number}: {_round_plan(round_number)} Respond to the debate so far."
    messages.append({"role": "user", "content": round_instr})

    # 6. Opponent's last message (highlighted) - closest to generation = most salient
    if config.highlight_opponent_last and opponent_last_message:
        messages.append(
            {
                "role": "user",
                "content": f"OPPONENT'S LAST MESSAGE (respond directly to this):\n\n{opponent_last_message}",
            }
        )

    return messages


def _prompt_hashes() -> dict[str, str]:
    prompt_dir = Path(__file__).resolve().parents[1] / "prompts"
    hashes: dict[str, str] = {}
    for prompt_path in prompt_dir.glob("*.md"):
        content = prompt_path.read_text().encode("utf-8")
        hashes[prompt_path.name] = hashlib.sha256(content).hexdigest()
    return hashes


def _run_config(config: DebateConfig, rounds: int) -> dict[str, object]:
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


def _update_memory_from_turn(
    memory: MemoryState,
    turn: AgentTurn,
    debate_id: str,
    motion: str,
    turn_number: int,
    reply_to_turn: int | None = None,
) -> MemoryState:
    """Update memory with agent turn, creating canonical debate log entry."""
    # Determine stance based on speaker
    stance: Stance = "pro" if turn.speaker == "CA" else "con"

    # Determine tool usage
    tool_used: ToolUsed = "none"
    tool_query = None
    if turn.search and turn.search.should_search:
        tool_used = "duckduckgo"
        tool_query = turn.search.search_query

    log_item = _create_debate_log_item(
        debate_id=debate_id,
        claim=motion,
        round_num=turn.round,
        speaker=turn.speaker,
        utterance=turn.claim,
        stance=stance,
        confidence=turn.confidence,
        tactic_used=turn.tactic_used,
        tool_used=tool_used,
        tool_query=tool_query,
        reply_to_turn=reply_to_turn,
    )

    memory = append_log(memory, log_item)
    return update_agent_state(memory, turn.speaker, turn.confidence, turn.what_changes_mind)


def _apply_recap_updates(memory: MemoryState, recap: ModeratorRecap) -> MemoryState:
    """Apply moderator confidence deltas to agent states."""
    ca_delta = recap.confidence_updates.get("CA_delta", 0)
    sa_delta = recap.confidence_updates.get("SA_delta", 0)

    ca_state = memory.agent_states["CA"]
    sa_state = memory.agent_states["SA"]

    ca_new_conf = max(0, min(100, ca_state.confidence + ca_delta))
    sa_new_conf = max(0, min(100, sa_state.confidence + sa_delta))

    memory = update_agent_state(memory, "CA", ca_new_conf, ca_state.what_changes_mind)
    memory = update_agent_state(memory, "SA", sa_new_conf, sa_state.what_changes_mind)

    return memory


def _final_report_prompt(
    topic: str, motion: str, rounds: int, memory: MemoryState
) -> list[dict[str, str]]:
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
    services: DebateServices | None = None,
) -> ExportBundle:
    """Run a full debate simulation and return the exported artifact bundle.

    Args:
        topic: The debate topic (e.g. "Vaccine safety").
        motion: The specific claim agents argue about.
        rounds: Maximum number of debate rounds.
        config: Runtime configuration (models, temperatures, etc.).
        services: Optional pre-built DI services; auto-wired from *config* if ``None``.

    Returns:
        An :class:`ExportBundle` containing paths to all generated artifacts.
    """
    logger.info("=" * 80)
    logger.info("DEBATE SIMULATION STARTING")
    logger.info("Topic: %s", topic)
    logger.info("Motion: %s", motion)
    logger.info(
        "Max Rounds: %d | Model: %s | History: %s",
        rounds,
        config.conspiracy_model,
        config.history_mode,
    )

    # Generate unique debate ID
    debate_id = f"debate_{uuid.uuid4().hex[:12]}"
    logger.info("Debate ID: %s", debate_id)

    # ── Wire services (use injected or build defaults) ──
    if services is None:
        services = build_default_services(config)
    llm = services.llm
    search = services.search
    prompts = services.prompts

    memory = initial_memory(topic, motion)
    recaps: list[ModeratorRecap] = []

    # Global conversation history - tracks all debate messages
    conversation_history: list[dict[str, str]] = []

    # Track turn numbers for reply_to_turn field
    turn_counter = 0
    last_ca_turn = None
    last_sa_turn = None

    # Load agent role prompts once (via injected PromptLoader)
    ca_role_prompt = prompts.load(
        "conspiracy.md",
        {"topic": topic, "motion": motion, "round": "1", "word_limit": str(config.word_limit)},
    )
    sa_role_prompt = prompts.load(
        "scientific.md",
        {"topic": topic, "motion": motion, "round": "1", "word_limit": str(config.word_limit)},
    )

    # Moderator-controlled debate loop
    round_number = 0
    max_rounds = rounds
    debate_ended_early = False
    end_reason = ""

    # Track initial confidence for persuasion measurement
    initial_ca_confidence = memory.agent_states["CA"].confidence
    initial_sa_confidence = memory.agent_states["SA"].confidence

    while round_number < max_rounds:
        round_number += 1
        logger.info("-" * 60)
        logger.info("ROUND %d/%d", round_number, max_rounds)

        previous_memory = memory
        memory = update_round(memory, round_number)

        # Track opponent's last message for highlighting
        sa_last_message = None
        ca_last_message = None

        # CONSPIRACY ADVOCATE TURN
        logger.info("Conspiracy Advocate thinking...")

        ca_messages = _build_agent_messages_with_history(
            agent_role_prompt=ca_role_prompt,
            conversation_history=conversation_history,
            memory=memory,
            config=config,
            round_number=round_number,
            opponent_last_message=sa_last_message,
        )

        ca_turn = _run_agent_turn_with_search(
            response_model=AgentTurn,
            agent_messages=ca_messages,
            agent_type="CA",
            topic=topic,
            model=config.conspiracy_model,
            temperature=config.conspiracy_temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
            llm=llm,
            search=search,
            max_search_rounds=config.max_search_rounds,
        )

        turn_counter += 1
        ca_turn_number = turn_counter

        # Add CA's response to conversation history with tags
        # NOTE: Search queries/results are NOT included - kept private from opponent
        ca_content = (
            f"[CA][Round {round_number}] {ca_turn.claim}\n"
            f"Reasons: {'; '.join(ca_turn.reasons)}\n"
            f"Question to opponent: {ca_turn.question_to_opponent}\n"
            f"Confidence: {ca_turn.confidence}\n"
            f"Tactic: {ca_turn.tactic_used}"
        )

        conversation_history.append({"role": "assistant", "content": ca_content, "speaker": "CA"})
        ca_last_message = ca_turn.claim  # For opponent highlighting

        print("\n CONSPIRACY ADVOCATE:")
        print(f"   Claim: {ca_turn.claim}")
        print(f"   Confidence: {ca_turn.confidence}/100")
        print(f"   Tactic: {ca_turn.tactic_used}")
        if ca_turn.search and ca_turn.search.should_search:
            logger.info("  CA search: %s", ca_turn.search.search_query)

        memory = _update_memory_from_turn(
            memory, ca_turn, debate_id, motion, ca_turn_number, reply_to_turn=last_sa_turn
        )
        last_ca_turn = ca_turn_number

        # SCIENTIFIC ADVOCATE TURN
        logger.info("Scientific Advocate thinking...")

        sa_messages = _build_agent_messages_with_history(
            agent_role_prompt=sa_role_prompt,
            conversation_history=conversation_history,
            memory=memory,
            config=config,
            round_number=round_number,
            opponent_last_message=ca_last_message,
        )

        sa_turn = _run_agent_turn_with_search(
            response_model=ScientificTurn,
            agent_messages=sa_messages,
            agent_type="SA",
            topic=topic,
            model=config.scientific_model,
            temperature=config.scientific_temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
            llm=llm,
            search=search,
            max_search_rounds=config.max_search_rounds,
        )

        turn_counter += 1
        sa_turn_number = turn_counter

        # Add SA's response to conversation history with tags
        # NOTE: Search queries/results are NOT included - kept private from opponent
        sa_content = (
            f"[SA][Round {round_number}] {sa_turn.claim}\n"
            f"Clarify: {sa_turn.clarify}\n"
            f"Gaps identified: {', '.join(sa_turn.evaluate_gaps)}\n"
            f"Alternative hypotheses: {', '.join(sa_turn.alternative_hypotheses)}\n"
            f"Discriminating tests: {', '.join(sa_turn.discriminating_tests)}\n"
            f"Question to opponent: {sa_turn.question_to_opponent}\n"
            f"Confidence: {sa_turn.confidence}\n"
            f"Tactic: {sa_turn.tactic_used}"
        )

        conversation_history.append({"role": "assistant", "content": sa_content, "speaker": "SA"})
        sa_last_message = sa_turn.claim  # For opponent highlighting

        print("\n SCIENTIFIC ADVOCATE:")
        print(f"   Claim: {sa_turn.claim}")
        print(f"   Confidence: {sa_turn.confidence}/100")
        print(f"   Tactic: {sa_turn.tactic_used}")
        if sa_turn.search and sa_turn.search.should_search:
            logger.info("  SA search: %s", sa_turn.search.search_query)

        memory = _update_memory_from_turn(
            memory, sa_turn, debate_id, motion, sa_turn_number, reply_to_turn=last_ca_turn
        )
        last_sa_turn = sa_turn_number

        # MODERATOR RECAP
        logger.info("Moderator analyzing...")

        moderator_messages = _moderator_messages(
            topic, motion, round_number, config.word_limit, prompts
        )
        moderator_messages.append(
            {
                "role": "user",
                "content": (
                    f"Round {round_number} debate:\n\n"
                    f"CA turn:\n{ca_turn.model_dump_json(indent=2)}\n\n"
                    f"SA turn:\n{sa_turn.model_dump_json(indent=2)}\n\n"
                    f"Provide your analysis and recap."
                ),
            }
        )

        recap = llm.call(
            ModeratorRecap,
            moderator_messages,
            model=config.moderator_model,
            temperature=config.moderator_temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
            max_retries=LLM_MAX_RETRIES,
        )

        # Add moderator recap to conversation history with tags
        recap_content = (
            f"[MODERATOR][Round {round_number} Recap]\n"
            f"Agreements: {', '.join(recap.summary_agreements)}\n"
            f"Disagreements: {', '.join(recap.summary_disagreements)}\n"
            f"Detected moves: {', '.join(recap.detected_fallacies_or_moves)}\n"
            f"Scores: civility={recap.civility_score}/5, epistemic={recap.epistemic_quality_score}/5, bridge={recap.bridge_building_score}/5\n"
            f"Next questions: {', '.join(recap.next_round_questions)}"
        )
        conversation_history.append(
            {"role": "assistant", "content": recap_content, "speaker": "MA"}
        )

        # Log moderator recap to debate log
        turn_counter += 1
        moderator_log_item = _create_debate_log_item(
            debate_id=debate_id,
            claim=motion,
            round_num=round_number,
            speaker="MA",
            utterance=recap_content,
            stance="neutral",
            confidence=None,
            tactic_used=None,
            tool_used="none",
            tool_query=None,
            reply_to_turn=None,
        )
        memory = append_log(memory, moderator_log_item)

        recaps.append(recap)
        logger.info(
            "MODERATOR RECAP: Bridge=%d/5 Civility=%d/5 Epistemic=%d/5",
            recap.bridge_building_score,
            recap.civility_score,
            recap.epistemic_quality_score,
        )

        memory = _apply_recap_updates(memory, recap)

        shift = stance_shift(previous_memory, memory)
        memory = update_scoreboard(
            memory,
            stance_shift=shift,
            bridge_score=recap.bridge_building_score,
            civility_score=recap.civility_score,
            epistemic_quality=recap.epistemic_quality_score,
        )

        # MODERATOR DECISION - Should debate continue?
        logger.info("Moderator deciding on debate continuation...")

        # Get current confidence levels for decision
        current_ca_confidence = memory.agent_states["CA"].confidence
        current_sa_confidence = memory.agent_states["SA"].confidence

        decision_messages = _moderator_decision_messages(
            topic=topic,
            motion=motion,
            current_round=round_number,
            max_rounds=max_rounds,
            initial_ca_confidence=initial_ca_confidence,
            initial_sa_confidence=initial_sa_confidence,
            current_ca_confidence=current_ca_confidence,
            current_sa_confidence=current_sa_confidence,
            recent_recap=recap,
            prompts=prompts,
        )

        decision = llm.call(
            ModeratorDecision,
            decision_messages,
            model=config.moderator_model,
            temperature=config.moderator_temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
            max_retries=LLM_MAX_RETRIES,
        )

        logger.info(
            "MODERATOR DECISION: continue=%s reason=%s", decision.should_continue, decision.reason
        )
        if decision.detected_mind_change:
            logger.info("Mind change detected: %s", decision.detected_mind_change)

        # Check if debate should end
        if not decision.should_continue:
            debate_ended_early = True
            end_reason = decision.reason
            logger.info("DEBATE ENDED BY MODERATOR AFTER ROUND %d", round_number)
            logger.info("Reason: %s", end_reason)
            logger.info(
                "CA confidence: %d -> %d (%+d)",
                initial_ca_confidence,
                current_ca_confidence,
                current_ca_confidence - initial_ca_confidence,
            )
            logger.info(
                "SA confidence: %d -> %d (%+d)",
                initial_sa_confidence,
                current_sa_confidence,
                current_sa_confidence - initial_sa_confidence,
            )
            break

    logger.info("GENERATING FINAL REPORT")
    if debate_ended_early:
        logger.info("Debate ended early after %d rounds: %s", round_number, end_reason)
    else:
        logger.info("Debate completed all %d rounds", round_number)

    # Use higher token limit for final report (needs more space for complete JSON)
    final_report = llm.call(
        FinalReport,
        _final_report_prompt(topic, motion, round_number, memory),
        model=config.moderator_model,
        temperature=config.moderator_temperature,
        max_tokens=FINAL_REPORT_MAX_TOKENS,
        seed=config.seed,
        max_retries=LLM_MAX_RETRIES,
    )

    logger.info(
        "FINAL REPORT: topic=%s rounds=%d", final_report.topic, final_report.rounds_completed
    )
    logger.info("Outcome: %s", final_report.outcome_summary)
    for moment in final_report.key_persuasion_moments:
        logger.info(
            "  Persuasion: round %d (%s) - %s", moment.round, moment.speaker, moment.why_it_mattered
        )
    for limitation in final_report.limitations:
        logger.info("  Limitation: %s", limitation)

    metrics_table = build_metrics_table(recaps)
    bundle: ExportBundle = services.exporter.write(
        output_dir=Path(config.output_dir),
        memory=memory,
        final_report=final_report,
        recaps=recaps,
        debate_log=memory.debate_log,
        metrics_table=metrics_table,
        topic=topic,
        motion=motion,
        run_config=_run_config(config, round_number),
    )

    logger.info("Artifacts saved to: %s", bundle.run_dir)
    logger.info(
        "  Transcript: %s | Memory: %s | Report: %s",
        bundle.transcript_path.name,
        bundle.memory_path.name,
        bundle.final_report_path.name,
    )
    if services.analyzer is not None:
        services.analyzer.analyze(str(bundle.run_dir))
    return bundle
