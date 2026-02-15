"""
CrewAI Debate System – Full Topic Dataset + DuckDuckGo
======================================================
Moderator-controlled structured debate with:
  - All 20 conspiracy topics from topics.py
  - Canonical JSON logging (PDF-spec fields)
  - Remote Ollama via OpenAI-compatible /v1
  - DuckDuckGo search via ddgs (replaces Tavily)
  - num_ctx hard-capped at 16000 tokens

Usage:
    # Run ALL topics once
    python debate_system.py --config ../config/config.ini --repeats 1

    # Run one category
    python debate_system.py --config ../config/config.ini --category health --repeats 3

    # Run specific topics
    python debate_system.py --config ../config/config.ini --topic_id covid_vaccine_microchips --topic_id chemtrails --repeats 2
"""

from __future__ import annotations

import os
import json
import uuid
import time
import re
import configparser
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from ddgs import DDGS

from topics import CONSPIRACY_TOPICS, DebateTopic


# =========================================================
# Config
# =========================================================

MAX_NUM_CTX = 16000  # hard ceiling on Ollama context window


@dataclass
class DebateConfig:
    # API
    base_url: str
    api_mode: str

    # Models
    moderator_model: str
    conspiracy_model: str
    scientific_model: str

    moderator_temperature: float
    conspiracy_temperature: float
    scientific_temperature: float

    # Debate
    max_tokens: int
    rounds: int
    word_limit: int
    seed: Optional[int]
    max_search_rounds: int
    num_ctx: int  # capped to <= MAX_NUM_CTX

    # Output
    output_dir: str

    # History controls
    history_trim: str
    max_rounds_in_history: Optional[int]
    summarize_if_trimmed: bool
    highlight_opponent_last: bool


def load_config(path: str) -> DebateConfig:
    cfg = configparser.ConfigParser()
    cfg.read(path)

    def get_int(
        section: str, key: str, default: Optional[int] = None
    ) -> Optional[int]:
        raw = cfg[section].get(key, "").strip()
        if raw == "":
            return default
        return int(raw)

    def get_float(section: str, key: str, default: float) -> float:
        raw = cfg[section].get(key, "").strip()
        return float(raw) if raw != "" else default

    seed_raw = cfg["debate"].get("seed", "").strip()
    seed = int(seed_raw) if seed_raw else None

    # Force max context to MAX_NUM_CTX (16000)
    num_ctx = get_int("debate", "num_ctx", MAX_NUM_CTX) or MAX_NUM_CTX
    num_ctx = min(num_ctx, MAX_NUM_CTX)

    max_rounds_raw = cfg["history"].get("max_rounds_in_history", "").strip()
    max_rounds_in_history = int(max_rounds_raw) if max_rounds_raw else None

    return DebateConfig(
        base_url=cfg["api"]["base_url"].strip(),
        api_mode=cfg["api"].get("api_mode", "openai").strip().lower(),
        moderator_model=cfg["models"]["moderator_model"].strip(),
        conspiracy_model=cfg["models"]["conspiracy_model"].strip(),
        scientific_model=cfg["models"]["scientific_model"].strip(),
        moderator_temperature=get_float("models", "moderator_temperature", 0.2),
        conspiracy_temperature=get_float("models", "conspiracy_temperature", 0.6),
        scientific_temperature=get_float("models", "scientific_temperature", 0.2),
        max_tokens=get_int("debate", "max_tokens", 900) or 900,
        rounds=get_int("debate", "rounds", 10) or 10,
        word_limit=get_int("debate", "word_limit", 300) or 300,
        seed=seed,
        max_search_rounds=get_int("debate", "max_search_rounds", 0) or 0,
        num_ctx=num_ctx,
        output_dir=cfg["output"].get("output_dir", "artifacts").strip(),
        history_trim=cfg["history"].get("history_trim", "rounds").strip(),
        max_rounds_in_history=max_rounds_in_history,
        summarize_if_trimmed=cfg["history"]
        .get("summarize_if_trimmed", "true")
        .strip()
        .lower()
        == "true",
        highlight_opponent_last=cfg["history"]
        .get("highlight_opponent_last", "true")
        .strip()
        .lower()
        == "true",
    )


# =========================================================
# Tool logging per turn (canonical schema)
# =========================================================


class TurnToolLogger:
    def __init__(self):
        self.current_turn_key: Optional[str] = None
        self.calls: Dict[str, List[Dict[str, Any]]] = {}

    def set_turn(self, turn_key: str):
        self.current_turn_key = turn_key
        self.calls.setdefault(turn_key, [])

    def log(self, tool_name: str, query: str):
        if not self.current_turn_key:
            return
        self.calls[self.current_turn_key].append(
            {
                "tool_used": tool_name,
                "tool_query": query,
                "ts": time.time(),
            }
        )

    def get_turn_tool_summary(self, turn_key: str) -> Tuple[str, str]:
        entries = self.calls.get(turn_key, [])
        if not entries:
            return ("none", "")
        last = entries[-1]
        return (last["tool_used"], last["tool_query"])


# =========================================================
# DuckDuckGo search tool (replaces Tavily)
# =========================================================


class DDGSearchTool(BaseTool):
    """DuckDuckGo search tool powered by `ddgs`, logged into canonical schema."""

    name: str = "ddg_search"
    description: str = (
        "Search the web with DuckDuckGo. "
        "Input: a query string. Output: top results with title/url/snippet."
    )

    _logger: Any = None
    _max_results: int = 5

    def __init__(self, logger: TurnToolLogger, max_results: int = 5, **kwargs):
        super().__init__(**kwargs)
        self._logger = logger
        self._max_results = max_results

    def _run(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "No query provided."

        self._logger.log("ddg", query)

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self._max_results))
        except Exception as e:
            return f"DDG search error: {e}"

        if not results:
            return "No results."

        lines = []
        for i, r in enumerate(results, start=1):
            title = (r.get("title") or "").strip()
            href = (r.get("href") or "").strip()
            body = (r.get("body") or "").strip()
            lines.append(f"{i}. {title}\n   {href}\n   {body}")
        return "\n".join(lines)


# =========================================================
# Prompt helpers
# =========================================================


def safe_json_extract(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def stance_from_footer(text: str) -> Tuple[str, int]:
    stance = "neutral"
    conf = 50
    m1 = re.search(r"STANCE:\s*(pro|con|neutral)", text, flags=re.IGNORECASE)
    if m1:
        stance = m1.group(1).lower()
    m2 = re.search(r"CONFIDENCE:\s*(\d{1,3})", text, flags=re.IGNORECASE)
    if m2:
        conf = max(0, min(100, int(m2.group(1))))
    return stance, conf


def clamp_words_keep_footer(text: str, max_words: int) -> str:
    """Enforce word limit for body while preserving the STANCE/CONFIDENCE footer."""
    text = text.strip()

    footer_match = re.search(
        r"(STANCE:\s*(pro|con|neutral)\s*\nCONFIDENCE:\s*\d{1,3}\s*)$",
        text,
        flags=re.IGNORECASE,
    )
    footer = ""
    body = text
    if footer_match:
        footer = footer_match.group(1).strip()
        body = text[: footer_match.start()].strip()

    words = body.split()
    if len(words) > max_words:
        body = " ".join(words[:max_words]).strip() + " …"

    if footer:
        return f"{body}\n\n{footer}".strip()
    return body


def format_transcript(
    transcript: List[Dict[str, Any]], highlight_last: bool
) -> str:
    lines = []
    for e in transcript:
        lines.append(
            f"[Round {e['round']} | Turn {e['turn']} | {e['role'].upper()}]\n{e['text']}\n"
        )
    if highlight_last:
        for e in reversed(transcript):
            if e["role"] in ("proponent", "opponent"):
                lines.append(
                    f"[HIGHLIGHT: last debater message ({e['role'].upper()})]\n{e['text']}\n"
                )
                break
    return "\n".join(lines).strip()


def moderator_schema_instructions(cfg: DebateConfig) -> str:
    return f"""
You are the MODERATOR and the ONLY controller of the debate.

You must:
- control turn-taking (choose who speaks next),
- track rounds,
- decide when debate ends: either if one debater changes their mind, or after {cfg.rounds} rounds.

Online search:
- If you request research, the debater will use ddg_search (DuckDuckGo) up to {cfg.max_search_rounds} times.
- Proponent research should use fringe/biased queries (e.g., add: site:reddit.com/r/conspiracy).
- Opponent research should use mainstream/reputable sources (fact-checkers, scientific orgs, reports).

Enforce max {cfg.word_limit} words per debater response (body; footer allowed).

OUTPUT: You MUST output ONLY one JSON object, no extra text, using EXACT keys:

{{
  "round": <int>,
  "next_speaker": "proponent" | "opponent" | "moderator",
  "instruction_to_speaker": "<string>",
  "research": {{
    "use_ddg": true | false,
    "query": "<string or empty>",
    "guidance": "<string or empty>"
  }},
  "end_debate": true | false,
  "end_reason": "<string or empty>",
  "log_metadata": {{
    "stance_proponent": "pro" | "con" | "neutral",
    "stance_opponent": "pro" | "con" | "neutral",
    "confidence_proponent": <0-100>,
    "confidence_opponent": <0-100>
  }}
}}
""".strip()


def debater_instructions(role: str, cfg: DebateConfig) -> str:
    if role == "proponent":
        stance_line = "You argue FOR the conspiracy motion."
        search_hint = (
            "If research requested: use fringe/biased queries "
            "(e.g., include site:reddit.com/r/conspiracy), but do NOT fabricate sources."
        )
    else:
        stance_line = "You argue AGAINST the motion using scientific/evidence-based reasoning."
        search_hint = (
            "If research requested: prioritize reputable sources, "
            "and clearly distinguish evidence vs speculation."
        )

    return f"""
You are the {role.upper()} debater.
{stance_line}

Rules:
- If the moderator requests research, you MAY use ddg_search up to {cfg.max_search_rounds} times.
- If research is NOT requested, DO NOT use tools.
- Keep your response body to <= {cfg.word_limit} words.

At the end of every message append EXACTLY:

STANCE: pro|con|neutral
CONFIDENCE: <0-100 integer>

Search guidance:
- {search_hint}
""".strip()


# =========================================================
# Agents + LLM wiring
# =========================================================


def make_llm(cfg: DebateConfig, model: str, temperature: float) -> LLM:
    return LLM(
        model=model,
        base_url=cfg.base_url,  # must be .../v1
        api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
        temperature=temperature,
        max_tokens=cfg.max_tokens,
        seed=cfg.seed,
        num_ctx=cfg.num_ctx,  # <= 16000 enforced
    )


def create_agents(cfg: DebateConfig, tool_logger: TurnToolLogger):
    ddg_tool = (
        DDGSearchTool(logger=tool_logger, max_results=5)
        if cfg.max_search_rounds > 0
        else None
    )

    moderator = Agent(
        role="Moderator",
        goal="Control the debate (turn-taking, research requests, ending condition).",
        backstory="A neutral control agent supervising a structured debate experiment.",
        llm=make_llm(cfg, cfg.moderator_model, cfg.moderator_temperature),
        verbose=False,
    )

    proponent = Agent(
        role="Proponent",
        goal="Argue for the conspiracy motion persuasively.",
        backstory="You strongly believe the motion and attempt persuasion.",
        llm=make_llm(cfg, cfg.conspiracy_model, cfg.conspiracy_temperature),
        tools=[ddg_tool] if ddg_tool else [],
        verbose=False,
    )

    opponent = Agent(
        role="Opponent",
        goal="Argue against the motion with evidence-based reasoning.",
        backstory="You resist persuasion using scientific standards and careful reasoning.",
        llm=make_llm(cfg, cfg.scientific_model, cfg.scientific_temperature),
        tools=[ddg_tool] if ddg_tool else [],
        verbose=False,
    )

    return moderator, proponent, opponent


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =========================================================
# Single-debate runner
# =========================================================


def run_single_debate(
    cfg: DebateConfig, topic: DebateTopic, run_index: int = 0
) -> Dict[str, Any]:
    """
    Run one full debate for a given topic.
    Returns metadata dict with debate_id, topic_id, output paths.
    """
    ensure_dir(cfg.output_dir)

    debate_id = str(uuid.uuid4())
    tool_logger = TurnToolLogger()
    moderator, proponent, opponent = create_agents(cfg, tool_logger)

    mod_instructions = moderator_schema_instructions(cfg)
    pro_instructions = debater_instructions("proponent", cfg)
    opp_instructions = debater_instructions("opponent", cfg)

    # Canonical log (PDF-spec fields)
    canonical_log: List[Dict[str, Any]] = []
    transcript: List[Dict[str, Any]] = []
    turn_counter = 0
    memory_summary = ""

    stance_p, conf_p = "pro", 70
    stance_o, conf_o = "con", 70

    claim = topic.motion

    def append_canonical(
        round_i: int,
        speaker_role: str,
        utterance: str,
        stance: str,
        confidence: Optional[int],
        tool_used: str,
        tool_query: str,
        reply_to_turn: Optional[int],
    ):
        canonical_log.append(
            {
                "debate_id": debate_id,
                "claim": claim,
                "round": round_i,
                "speaker_role": speaker_role,
                "utterance": utterance,
                "stance": stance,
                "confidence": confidence,
                "tool_used": tool_used,
                "tool_query": tool_query,
                "reply_to_turn": reply_to_turn,
            }
        )

    def trim_if_needed(current_round: int):
        nonlocal transcript, memory_summary
        if cfg.history_trim != "rounds" or not cfg.max_rounds_in_history:
            return

        keep_from_round = max(1, current_round - cfg.max_rounds_in_history + 1)
        older = [t for t in transcript if t["round"] < keep_from_round]
        newer = [t for t in transcript if t["round"] >= keep_from_round]
        if not older:
            return

        if cfg.summarize_if_trimmed:
            old_text = format_transcript(older, highlight_last=False)
            summarizer_task = Task(
                description=(
                    "Summarize earlier debate history in <= 180 words: "
                    "key arguments both sides, concessions, confidence shifts.\n\n"
                    f"{old_text}"
                ),
                agent=moderator,
                expected_output="A concise summary paragraph.",
            )
            crew = Crew(
                agents=[moderator],
                tasks=[summarizer_task],
                process=Process.sequential,
                verbose=False,
            )
            memory_summary = str(crew.kickoff()).strip()

        transcript = newer

    # ---- Main debate loop ----
    for round_i in range(1, cfg.rounds + 1):
        print(f"\n{'='*60}")
        print(f"  [{topic.id}] ROUND {round_i} / {cfg.rounds}")
        print(f"{'='*60}")

        trim_if_needed(round_i)

        visible = format_transcript(transcript, cfg.highlight_opponent_last)
        moderator_prompt = f"""
{mod_instructions}

TOPIC:
- id: {topic.id}
- category: {topic.category}
- topic: {topic.topic}
- description: {topic.description}

CLAIM (motion):
{claim}

KNOWN STANCES (latest reported):
- proponent: stance={stance_p}, confidence={conf_p}
- opponent: stance={stance_o}, confidence={conf_o}

MEMORY SUMMARY (may be empty):
{memory_summary}

TRANSCRIPT SO FAR:
{visible if visible else "(none yet)"}

Decide the next step for Round {round_i}. Output ONLY the JSON object.
""".strip()

        mod_task = Task(
            description=moderator_prompt,
            agent=moderator,
            expected_output="One JSON object matching the required schema.",
        )
        mod_crew = Crew(
            agents=[moderator],
            tasks=[mod_task],
            process=Process.sequential,
            verbose=False,
        )
        mod_raw = str(mod_crew.kickoff()).strip()

        mod_obj = safe_json_extract(mod_raw)
        if not mod_obj:
            # Fail-safe: log raw and stop
            print(f"  [WARNING] Moderator JSON parse failed. Raw:\n{mod_raw}")
            turn_counter += 1
            transcript.append(
                {
                    "round": round_i,
                    "turn": turn_counter,
                    "role": "moderator",
                    "text": mod_raw,
                }
            )
            append_canonical(
                round_i, "moderator", mod_raw, "neutral", None, "none", "", None
            )
            break

        # Log moderator decision as canonical utterance
        turn_counter += 1
        mod_text = json.dumps(mod_obj, ensure_ascii=False)
        transcript.append(
            {
                "round": round_i,
                "turn": turn_counter,
                "role": "moderator",
                "text": mod_text,
            }
        )
        append_canonical(
            round_i, "moderator", mod_text, "neutral", None, "none", "", None
        )

        print(f"  Moderator: next_speaker={mod_obj.get('next_speaker')}")

        if bool(mod_obj.get("end_debate", False)):
            print(
                f"  Moderator ended debate: {mod_obj.get('end_reason', '(no reason)')}"
            )
            break

        next_speaker = (mod_obj.get("next_speaker") or "proponent").strip().lower()
        instruction = (mod_obj.get("instruction_to_speaker") or "").strip()

        research = mod_obj.get("research") or {}
        use_ddg = bool(research.get("use_ddg", False))
        query = (research.get("query") or "").strip()
        guidance = (research.get("guidance") or "").strip()

        if next_speaker == "proponent":
            agent = proponent
            inst = pro_instructions
            stance_default = "pro"
        elif next_speaker == "opponent":
            agent = opponent
            inst = opp_instructions
            stance_default = "con"
        else:
            # Moderator wants to speak again; continue to next round
            continue

        # reply_to_turn: last debater turn
        reply_to_turn = None
        for t in reversed(transcript):
            if t["role"] in ("proponent", "opponent"):
                reply_to_turn = t["turn"]
                break

        research_block = "RESEARCH REQUESTED: NO (do not use tools)"
        if use_ddg and cfg.max_search_rounds > 0:
            research_block = (
                f"RESEARCH REQUESTED: YES\n"
                f"You may use ddg_search up to {cfg.max_search_rounds} times for this turn.\n"
                f"Suggested query: {query}\n"
                f"Guidance: {guidance}\n"
            )

        debater_prompt = f"""
{inst}

CLAIM (motion):
{claim}

MODERATOR INSTRUCTION:
{instruction}

{research_block}

TRANSCRIPT (visible):
{format_transcript(transcript, cfg.highlight_opponent_last)}
""".strip()

        # Tool logging context
        turn_key = f"debate={debate_id}:round={round_i}:speaker={next_speaker}:turn={turn_counter + 1}"
        tool_logger.set_turn(turn_key)

        # Run debater
        print(f"  {next_speaker.upper()} is responding...")
        deb_task = Task(
            description=debater_prompt,
            agent=agent,
            expected_output=f"<= {cfg.word_limit} words plus required footer.",
        )
        deb_crew = Crew(
            agents=[agent],
            tasks=[deb_task],
            process=Process.sequential,
            verbose=False,
        )
        deb_text_raw = str(deb_crew.kickoff()).strip()

        # Hard enforce word limit for body, keep footer
        deb_text = clamp_words_keep_footer(deb_text_raw, cfg.word_limit)

        stance, conf = stance_from_footer(deb_text)
        if stance not in ("pro", "con", "neutral"):
            stance = stance_default

        tool_used, tool_query = tool_logger.get_turn_tool_summary(turn_key)

        # Update known stances
        if next_speaker == "proponent":
            stance_p, conf_p = stance, conf
        else:
            stance_o, conf_o = stance, conf

        # Log debater utterance
        turn_counter += 1
        transcript.append(
            {
                "round": round_i,
                "turn": turn_counter,
                "role": next_speaker,
                "text": deb_text,
            }
        )
        append_canonical(
            round_i,
            next_speaker,
            deb_text,
            stance,
            conf,
            tool_used,
            tool_query,
            reply_to_turn,
        )

        print(f"  {next_speaker.upper()} stance={stance} confidence={conf}")

    # ---- Save artifacts ----
    subdir = os.path.join(cfg.output_dir, topic.id)
    ensure_dir(subdir)

    json_path = os.path.join(
        subdir, f"{topic.id}__run{run_index}__{debate_id}.json"
    )
    jsonl_path = os.path.join(
        subdir, f"{topic.id}__run{run_index}__{debate_id}.jsonl"
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(canonical_log, f, ensure_ascii=False, indent=2)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in canonical_log:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n  Saved: {json_path}")
    print(f"  Saved: {jsonl_path}")

    return {
        "debate_id": debate_id,
        "topic_id": topic.id,
        "category": topic.category,
        "motion": topic.motion,
        "run_index": run_index,
        "json_path": json_path,
        "jsonl_path": jsonl_path,
        "total_turns": turn_counter,
        "final_stance_proponent": stance_p,
        "final_confidence_proponent": conf_p,
        "final_stance_opponent": stance_o,
        "final_confidence_opponent": conf_o,
    }


# =========================================================
# Batch experiment runner
# =========================================================


def run_experiments(
    config_path: str,
    topic_ids: Optional[List[str]] = None,
    category: Optional[str] = None,
    repeats: int = 1,
):
    """
    Run debates across multiple topics with optional filtering and repetitions.
    Saves a manifest file summarizing all runs.
    """
    cfg = load_config(config_path)
    ensure_dir(cfg.output_dir)

    # Select topics
    topics = list(CONSPIRACY_TOPICS)
    if category:
        topics = [t for t in topics if t.category == category]
    if topic_ids:
        wanted = set(topic_ids)
        topics = [t for t in topics if t.id in wanted]

    if not topics:
        raise ValueError("No topics matched your selection.")

    print(f"\n{'#'*60}")
    print(f"  DEBATE EXPERIMENT")
    print(f"  Topics: {len(topics)} | Repeats: {repeats} | Total runs: {len(topics) * repeats}")
    print(f"  Models: mod={cfg.moderator_model} pro={cfg.conspiracy_model} opp={cfg.scientific_model}")
    print(f"  num_ctx={cfg.num_ctx} (capped at {MAX_NUM_CTX}) | rounds={cfg.rounds} | word_limit={cfg.word_limit}")
    print(f"  Output: {cfg.output_dir}")
    print(f"{'#'*60}\n")

    manifest: List[Dict[str, Any]] = []

    for topic in topics:
        for r in range(repeats):
            print(f"\n{'*'*60}")
            print(f"  TOPIC: {topic.id} ({topic.category}) — run {r + 1}/{repeats}")
            print(f"  MOTION: {topic.motion}")
            print(f"{'*'*60}")

            try:
                result = run_single_debate(cfg, topic, run_index=r)
                manifest.append(result)
                print(f"\n  [OK] {topic.id} run={r} debate_id={result['debate_id']}")
            except Exception as e:
                print(f"\n  [ERROR] {topic.id} run={r}: {e}")
                manifest.append(
                    {
                        "topic_id": topic.id,
                        "run_index": r,
                        "error": str(e),
                    }
                )

    # Save manifest
    manifest_path = os.path.join(
        cfg.output_dir, f"manifest_{int(time.time())}.json"
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'#'*60}")
    print(f"  ALL DONE — {len(manifest)} debate(s) completed")
    print(f"  Manifest: {manifest_path}")
    print(f"{'#'*60}")

    return manifest_path


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run moderator-controlled CrewAI debates across conspiracy topics."
    )
    parser.add_argument(
        "--config",
        default="../config/config.ini",
        help="Path to INI config (default: ../config/config.ini)",
    )
    parser.add_argument(
        "--category",
        default="",
        help="Run only topics in this category (e.g., health, government, technology)",
    )
    parser.add_argument(
        "--topic_id",
        action="append",
        default=[],
        help="Run only this topic id (can repeat for multiple topics)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat each topic (default: 1)",
    )
    args = parser.parse_args()

    cat = args.category.strip() or None
    tids = args.topic_id if args.topic_id else None

    run_experiments(args.config, topic_ids=tids, category=cat, repeats=args.repeats)
