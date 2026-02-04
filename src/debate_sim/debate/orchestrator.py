from __future__ import annotations

import hashlib
import json
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from ..config import DebateConfig
from .evaluation import build_metrics_table, stance_shift
from .protocol import load_prompt
from ..export.writer import ExportBundle, write_artifacts
from ..llm.instructor_wrapper import StructuredLLM
from ..llm.ollama_client import OllamaClient
from ..memory.models import (
    append_log,
    initial_memory,
    update_agent_state,
    update_round,
    update_scoreboard,
)
from ..schemas import (
    AgentTurn,
    DebateLogItem,
    FinalReport,
    MemoryState,
    ModeratorRecap,
    ScientificTurn,
)


def _round_plan(round_number: int) -> str:
    return f"Round {round_number} focus: clarify positions, test evidence, maintain civility."


def _format_transcript(history: List[Dict[str, str]]) -> str:
    """Format conversation history as a readable transcript."""
    lines = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant":
            speaker = msg.get("speaker", "Agent")
            lines.append(f"[{speaker}]: {content}")
    return "\n\n".join(lines)


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
    history: List[Dict[str, str]], 
    config: DebateConfig, 
    current_round: int
) -> Tuple[List[Dict[str, str]], bool]:
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
        max_msgs = config.max_rounds_in_history * 4
        if len(non_system) > max_msgs:
            non_system = non_system[-max_msgs:]
            was_trimmed = True
        
        working_history = system_msgs + non_system
    
    # Trim by message count
    if config.history_trim == "messages" and config.max_messages_in_history:
        system_msgs = [msg for msg in working_history if msg["role"] == "system"]
        non_system = [msg for msg in working_history if msg["role"] != "system"]
        
        if len(non_system) > config.max_messages_in_history:
            non_system = non_system[-config.max_messages_in_history:]
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


def _build_agent_messages_with_history(
    agent_role_prompt: str,
    conversation_history: List[Dict[str, str]],
    memory: MemoryState,
    config: DebateConfig,
    round_number: int,
    opponent_last_message: str = None,
) -> List[Dict[str, str]]:
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
            messages.append({
                "role": "user", 
                "content": "[Earlier rounds truncated - showing recent history]"
            })
        
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
        messages.append({
            "role": "user",
            "content": f"OPPONENT'S LAST MESSAGE (respond directly to this):\n\n{opponent_last_message}"
        })
    
    return messages


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
    print(f"\n{'='*80}")
    print(f"DEBATE SIMULATION STARTING")
    print(f"{'='*80}")
    print(f"Topic: {topic}")
    print(f"Motion: {motion}")
    print(f"Rounds: {rounds}")
    print(f"Model: {config.conspiracy_model}")
    print(f"History Mode: {config.history_mode}")
    print(f"{'='*80}\n")
    
    memory = initial_memory(topic, motion)
    client = OllamaClient(config)
    llm = StructuredLLM(client)
    recaps: List[ModeratorRecap] = []
    tactic_counts: Counter[str] = Counter()
    
    # Global conversation history - tracks all debate messages
    conversation_history: List[Dict[str, str]] = []
    
    # Load agent role prompts once
    ca_role_prompt = load_prompt("conspiracy.md", {
        "topic": topic, "motion": motion, 
        "round": "1", "word_limit": str(config.word_limit)
    })
    sa_role_prompt = load_prompt("scientific.md", {
        "topic": topic, "motion": motion,
        "round": "1", "word_limit": str(config.word_limit)
    })

    for round_number in range(1, rounds + 1):
        print(f"\n{'─'*80}")
        print(f"ROUND {round_number}/{rounds}")
        print(f"{'─'*80}\n")
        
        previous_memory = memory
        memory = update_round(memory, round_number)
        
        # Track opponent's last message for highlighting
        sa_last_message = None
        ca_last_message = None
        
        # CONSPIRACY ADVOCATE TURN
        print("🔵 Conspiracy Advocate thinking...")
        
        ca_messages = _build_agent_messages_with_history(
            agent_role_prompt=ca_role_prompt,
            conversation_history=conversation_history,
            memory=memory,
            config=config,
            round_number=round_number,
            opponent_last_message=sa_last_message,
        )
        
        ca_turn = llm.call(
            AgentTurn,
            ca_messages,
            model=config.conspiracy_model,
            temperature=config.conspiracy_temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
        
        # Add CA's response to conversation history with tags
        ca_content = (
            f"[CA][Round {round_number}] {ca_turn.claim}\n"
            f"Reasons: {'; '.join(ca_turn.reasons)}\n"
            f"Question to opponent: {ca_turn.question_to_opponent}\n"
            f"Confidence: {ca_turn.confidence}\n"
            f"Tactic: {ca_turn.tactic_used}"
        )
        conversation_history.append({
            "role": "assistant",
            "content": ca_content
        })
        ca_last_message = ca_turn.claim  # For opponent highlighting
        
        print(f"\n🔵 CONSPIRACY ADVOCATE:")
        print(f"   Claim: {ca_turn.claim}")
        print(f"   Confidence: {ca_turn.confidence}/100")
        print(f"   Tactic: {ca_turn.tactic_used}")
        
        memory = _update_memory_from_turn(memory, ca_turn)

        # SCIENTIFIC ADVOCATE TURN
        print("\n🟢 Scientific Advocate thinking...")
        
        sa_messages = _build_agent_messages_with_history(
            agent_role_prompt=sa_role_prompt,
            conversation_history=conversation_history,
            memory=memory,
            config=config,
            round_number=round_number,
            opponent_last_message=ca_last_message,
        )
        
        sa_turn = llm.call(
            ScientificTurn,
            sa_messages,
            model=config.scientific_model,
            temperature=config.scientific_temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
        
        # Add SA's response to conversation history with tags
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
        conversation_history.append({
            "role": "assistant",
            "content": sa_content
        })
        sa_last_message = sa_turn.claim  # For opponent highlighting
        
        print(f"\n🟢 SCIENTIFIC ADVOCATE:")
        print(f"   Claim: {sa_turn.claim}")
        print(f"   Confidence: {sa_turn.confidence}/100")
        print(f"   Tactic: {sa_turn.tactic_used}")
        
        memory = _update_memory_from_turn(memory, sa_turn)

        # MODERATOR RECAP
        print("\n⚖️  Moderator analyzing...")
        
        moderator_messages = _moderator_messages(topic, motion, round_number, config.word_limit)
        moderator_messages.append({
            "role": "user",
            "content": (
                f"Round {round_number} debate:\n\n"
                f"CA turn:\n{ca_turn.model_dump_json(indent=2)}\n\n"
                f"SA turn:\n{sa_turn.model_dump_json(indent=2)}\n\n"
                f"Provide your analysis and recap."
            )
        })
        
        recap = llm.call(
            ModeratorRecap,
            moderator_messages,
            model=config.moderator_model,
            temperature=config.moderator_temperature,
            max_tokens=config.max_tokens,
            seed=config.seed,
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
        conversation_history.append({
            "role": "assistant",
            "content": recap_content
        })
        
        recaps.append(recap)
        print(f"\n⚖️  MODERATOR RECAP:")
        print(f"   Agreements: {', '.join(recap.summary_agreements)}")
        print(f"   Disagreements: {', '.join(recap.summary_disagreements)}")
        print(f"   Bridge Building: {recap.bridge_building_score}/5")
        print(f"   Civility: {recap.civility_score}/5")
        print(f"   Epistemic Quality: {recap.epistemic_quality_score}/5")
        
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

    print(f"\n{'='*80}")
    print("GENERATING FINAL REPORT")
    print(f"{'='*80}\n")
    
    final_report = llm.call(
        FinalReport,
        _final_report_prompt(topic, motion, rounds, memory),
        model=config.moderator_model,
        temperature=config.moderator_temperature,
        max_tokens=config.max_tokens,
        seed=config.seed,
    )
    
    print(f"\n{'='*80}")
    print("FINAL REPORT")
    print(f"{'='*80}")
    print(f"Verdict: {final_report.verdict}")
    print(f"Key Takeaways: {final_report.key_takeaways}")
    print(f"{'='*80}\n")

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
    
    print(f"✅ Artifacts saved to: {bundle.run_dir}")
    print(f"   - Transcript: {bundle.transcript_path.name}")
    print(f"   - Memory: {bundle.memory_path.name}")
    print(f"   - Final Report: {bundle.final_report_path.name}\n")
    if config.run_analysis:
        from debate_sim.analysis.analysis_runner import analyze_run

        analyze_run(str(bundle.run_dir))
    return bundle
