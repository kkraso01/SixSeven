"""
CrewAI Debate Orchestrator – DuckDuckGo Edition
================================================
Moderator-controlled structured debate with canonical JSON logging,
remote Ollama via OpenAI-compatible /v1, and DuckDuckGo search (ddgs).

Changes vs debate_crewai.py:
  - num_ctx hard-capped at 16000 tokens regardless of INI value.
  - Tavily replaced with DuckDuckGo via the `ddgs` package.

Usage:
    python debate_crewai_ddgs.py --config ../config/config.ini --claim "COVID-19 vaccines contain microchips"
"""

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
from pydantic import PrivateAttr

from ddgs import DDGS


# -----------------------------
# Config
# -----------------------------

MAX_NUM_CTX = 16000  # hard ceiling on Ollama context window


@dataclass
class DebateConfig:
    base_url: str
    api_mode: str

    moderator_model: str
    conspiracy_model: str
    scientific_model: str

    moderator_temperature: float
    conspiracy_temperature: float
    scientific_temperature: float

    max_tokens: int
    rounds: int
    word_limit: int
    seed: Optional[int]

    max_search_rounds: int
    num_ctx: int  # capped at MAX_NUM_CTX

    output_dir: str

    history_mode: str
    include_memory_summary: bool
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

    max_rounds_in_history = get_int("history", "max_rounds_in_history", None)

    # CAP num_ctx to MAX_NUM_CTX (16000)
    num_ctx = get_int("debate", "num_ctx", MAX_NUM_CTX) or MAX_NUM_CTX
    num_ctx = min(num_ctx, MAX_NUM_CTX)

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
        history_mode=cfg["history"].get("history_mode", "global_full").strip(),
        include_memory_summary=cfg["history"]
        .get("include_memory_summary", "false")
        .strip()
        .lower()
        == "true",
        history_trim=cfg["history"].get("history_trim", "none").strip(),
        max_rounds_in_history=max_rounds_in_history,
        summarize_if_trimmed=cfg["history"]
        .get("summarize_if_trimmed", "false")
        .strip()
        .lower()
        == "true",
        highlight_opponent_last=cfg["history"]
        .get("highlight_opponent_last", "false")
        .strip()
        .lower()
        == "true",
    )


# -----------------------------
# Tool logging per turn
# -----------------------------


class TurnToolLogger:
    def __init__(self):
        self.current_turn_key: Optional[str] = None
        self.turn_tool_calls: Dict[str, List[Dict[str, Any]]] = {}

    def set_turn(self, turn_key: str):
        self.current_turn_key = turn_key
        self.turn_tool_calls.setdefault(turn_key, [])

    def log(self, tool_name: str, query: str):
        if not self.current_turn_key:
            return
        self.turn_tool_calls[self.current_turn_key].append(
            {
                "tool_used": tool_name,
                "tool_query": query,
                "ts": time.time(),
            }
        )

    def get_turn_tool_summary(self, turn_key: str) -> Tuple[str, str]:
        calls = self.turn_tool_calls.get(turn_key, [])
        if not calls:
            return ("none", "")
        last = calls[-1]
        return (last["tool_used"], last["tool_query"])


# -----------------------------
# DuckDuckGo tool (ddgs)
# -----------------------------


class DDGSearchTool(BaseTool):
    name: str = "ddg_search"
    description: str = (
        "Search the web using DuckDuckGo. "
        "Input should be a plain-text query. Returns top results with title, url, snippet."
    )

    # Private attrs using Pydantic's PrivateAttr for proper initialization
    _logger: Any = PrivateAttr()
    _max_results: int = PrivateAttr(default=5)

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


# -----------------------------
# Helpers
# -----------------------------


def clamp_words(text: str, max_words: int) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + " …"


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


def format_transcript_for_prompt(
    transcript: List[Dict[str, Any]], highlight_opponent_last: bool
) -> str:
    lines = []
    for e in transcript:
        lines.append(
            f"[Round {e['round']} | Turn {e['turn']} | {e['role'].upper()}]\n{e['text']}\n"
        )
    if highlight_opponent_last:
        for e in reversed(transcript):
            if e["role"] in ("proponent", "opponent"):
                lines.append(
                    f"[HIGHLIGHT: last debater message ({e['role'].upper()})]\n{e['text']}\n"
                )
                break
    return "\n".join(lines).strip()


def build_moderator_system_instructions(cfg: DebateConfig) -> str:
    return f"""
You are the MODERATOR and the ONLY controller.

You must decide:
- who speaks next (proponent or opponent, or you can speak to clarify),
- whether the next speaker must do web research using the ddg_search tool,
- and whether to end the debate.

Enforce: max {cfg.word_limit} words per debater response.

You must output ONLY one JSON object, no extra text, with schema:

{{
  "round": <int>,
  "next_speaker": "proponent" | "opponent" | "moderator",
  "instruction_to_speaker": "<what they should do/say next>",
  "research": {{
    "use_ddg": true | false,
    "query": "<string or empty>",
    "guidance": "<how to search: fringe vs mainstream>"
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

Rules:
- You never argue the claim.
- Keep research sparse: at most {cfg.max_search_rounds} search calls per debater turn.
- End debate if a debater changes stance materially or becomes neutral with low confidence, or after {cfg.rounds} rounds.
""".strip()


def build_debater_instructions(role: str, cfg: DebateConfig) -> str:
    if role == "proponent":
        stance = "You argue FOR the conspiracy claim."
        search_style = (
            "If research is requested, you may use ddg_search with fringe-leaning "
            "queries, but do not fabricate sources."
        )
    else:
        stance = "You argue AGAINST the conspiracy claim using scientific/evidence-based reasoning."
        search_style = (
            "If research is requested, you may use ddg_search and prioritize "
            "reputable sources."
        )

    return f"""
You are the {role.upper()} debater.
{stance}

Hard constraints:
- Max {cfg.word_limit} words for your response (excluding the footer).
- Use ddg_search ONLY if the moderator requested research, and at most {cfg.max_search_rounds} times.
- If no research requested: do NOT use tools.

At the end of every message, append this exact 2-line footer:

STANCE: pro|con|neutral
CONFIDENCE: <0-100 integer>

Search guidance:
- {search_style}
""".strip()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -----------------------------
# LLM + Agent wiring
# -----------------------------


def make_llm(cfg: DebateConfig, model: str, temperature: float) -> LLM:
    return LLM(
        model=model,
        base_url=cfg.base_url,  # must be .../v1
        api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
        temperature=temperature,
        max_tokens=cfg.max_tokens,
        seed=cfg.seed,
        # num_ctx is Ollama-specific; pass via extra_body so it doesn't
        # become a top-level kwarg to Completions.create()
        extra_body={"num_ctx": cfg.num_ctx},
    )


def create_agents(cfg: DebateConfig, tool_logger: TurnToolLogger):
    ddg_tool = (
        DDGSearchTool(logger=tool_logger, max_results=5)
        if cfg.max_search_rounds > 0
        else None
    )

    moderator = Agent(
        role="Moderator",
        goal="Control the debate: decide turns, require research, and end debate appropriately.",
        backstory="Neutral controller of a structured debate experiment.",
        llm=make_llm(cfg, cfg.moderator_model, cfg.moderator_temperature),
        verbose=False,
    )

    proponent = Agent(
        role="Proponent",
        goal="Persuasively argue for the conspiracy claim.",
        backstory="You sincerely believe the claim and try to persuade.",
        llm=make_llm(cfg, cfg.conspiracy_model, cfg.conspiracy_temperature),
        tools=[ddg_tool] if ddg_tool else [],
        verbose=False,
    )

    opponent = Agent(
        role="Opponent",
        goal="Resist persuasion using evidence-based reasoning.",
        backstory="Skeptical, scientific, careful about evidence.",
        llm=make_llm(cfg, cfg.scientific_model, cfg.scientific_temperature),
        tools=[ddg_tool] if ddg_tool else [],
        verbose=False,
    )

    return moderator, proponent, opponent


# -----------------------------
# Debate loop (MODERATOR DECIDES)
# -----------------------------


def run_debate(config_path: str, claim: str) -> str:
    cfg = load_config(config_path)
    ensure_dir(cfg.output_dir)

    debate_id = str(uuid.uuid4())
    tool_logger = TurnToolLogger()
    moderator, proponent, opponent = create_agents(cfg, tool_logger)

    moderator_instructions = build_moderator_system_instructions(cfg)
    proponent_instructions = build_debater_instructions("proponent", cfg)
    opponent_instructions = build_debater_instructions("opponent", cfg)

    canonical_log: List[Dict[str, Any]] = []
    transcript: List[Dict[str, Any]] = []

    memory_summary = ""
    turn_counter = 0

    stance_p, conf_p = "pro", 70
    stance_o, conf_o = "con", 70

    def append_log(
        round_i: int,
        speaker_role: str,
        utterance: str,
        stance: str,
        confidence: Optional[int],
        reply_to_turn: Optional[int],
        tool_used: str,
        tool_query: str,
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

    def trim_history_if_needed(current_round: int):
        nonlocal transcript, memory_summary
        if cfg.history_trim != "rounds" or not cfg.max_rounds_in_history:
            return

        keep_from_round = max(1, current_round - cfg.max_rounds_in_history + 1)
        older = [t for t in transcript if t["round"] < keep_from_round]
        newer = [t for t in transcript if t["round"] >= keep_from_round]
        if not older:
            return

        if cfg.summarize_if_trimmed:
            old_text = format_transcript_for_prompt(
                older, highlight_opponent_last=False
            )
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

    for round_i in range(1, cfg.rounds + 1):
        print(f"\n{'='*60}")
        print(f"  ROUND {round_i} / {cfg.rounds}")
        print(f"{'='*60}")

        trim_history_if_needed(round_i)

        visible_transcript = format_transcript_for_prompt(
            transcript, cfg.highlight_opponent_last
        )
        moderator_prompt = f"""
{moderator_instructions}

CLAIM:
{claim}

MEMORY SUMMARY (may be empty):
{memory_summary}

CURRENT KNOWN STANCES:
- proponent: stance={stance_p}, confidence={conf_p}
- opponent: stance={stance_o}, confidence={conf_o}

TRANSCRIPT SO FAR:
{visible_transcript if visible_transcript else "(none yet)"}

Now decide the next step for Round {round_i}. Output ONLY valid JSON.
""".strip()

        # --- Moderator decision ---
        mod_task = Task(
            description=moderator_prompt,
            agent=moderator,
            expected_output="One JSON object matching schema.",
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
            print(f"[WARNING] Moderator JSON parse failed. Raw:\n{mod_raw}")
            append_log(
                round_i, "moderator", mod_raw, "neutral", None, None, "none", ""
            )
            break

        turn_counter += 1
        transcript.append(
            {
                "round": round_i,
                "turn": turn_counter,
                "role": "moderator",
                "text": json.dumps(mod_obj, ensure_ascii=False),
            }
        )
        append_log(
            round_i,
            "moderator",
            json.dumps(mod_obj, ensure_ascii=False),
            "neutral",
            None,
            None,
            "none",
            "",
        )

        print(f"  Moderator decision: next_speaker={mod_obj.get('next_speaker')}")

        if bool(mod_obj.get("end_debate")):
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
            debater_instructions = proponent_instructions
            stance_default = "pro"
        elif next_speaker == "opponent":
            agent = opponent
            debater_instructions = opponent_instructions
            stance_default = "con"
        else:
            # moderator chose to speak again; proceed to next round loop
            continue

        research_block = ""
        if use_ddg and cfg.max_search_rounds > 0:
            research_block = f"""
RESEARCH REQUESTED: YES
You may use ddg_search up to {cfg.max_search_rounds} times for this turn.
Suggested query: {query}
Guidance: {guidance}
"""
        else:
            research_block = "RESEARCH REQUESTED: NO (do not use tools)"

        last_non_moderator_turn = None
        for t in reversed(transcript):
            if t["role"] in ("proponent", "opponent"):
                last_non_moderator_turn = t["turn"]
                break

        debater_prompt = f"""
{debater_instructions}

CLAIM:
{claim}

MODERATOR INSTRUCTION:
{instruction}

{research_block}

TRANSCRIPT (visible):
{format_transcript_for_prompt(transcript, cfg.highlight_opponent_last)}
""".strip()

        turn_key = f"r{round_i}:{next_speaker}:{turn_counter + 1}"
        tool_logger.set_turn(turn_key)

        # --- Run debater ---
        print(f"  {next_speaker.upper()} is responding...")
        deb_task = Task(
            description=debater_prompt,
            agent=agent,
            expected_output=f"<= {cfg.word_limit} words plus footer.",
        )
        deb_crew = Crew(
            agents=[agent],
            tasks=[deb_task],
            process=Process.sequential,
            verbose=False,
        )
        deb_text = str(deb_crew.kickoff()).strip()

        # rough clamp; footer may push slightly
        deb_text = clamp_words(deb_text, cfg.word_limit + 40)

        stance, conf = stance_from_footer(deb_text)
        if stance not in ("pro", "con", "neutral"):
            stance = stance_default

        tool_used, tool_query = tool_logger.get_turn_tool_summary(turn_key)

        if next_speaker == "proponent":
            stance_p, conf_p = stance, conf
        else:
            stance_o, conf_o = stance, conf

        turn_counter += 1
        transcript.append(
            {
                "round": round_i,
                "turn": turn_counter,
                "role": next_speaker,
                "text": deb_text,
            }
        )
        append_log(
            round_i,
            next_speaker,
            deb_text,
            stance,
            conf,
            last_non_moderator_turn,
            tool_used,
            tool_query,
        )

        print(f"  {next_speaker.upper()} stance={stance} confidence={conf}")

    # --- Save artifacts ---
    out_json = os.path.join(cfg.output_dir, f"{debate_id}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(canonical_log, f, ensure_ascii=False, indent=2)

    out_jsonl = os.path.join(cfg.output_dir, f"{debate_id}.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in canonical_log:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved debate log:\n  - {out_json}\n  - {out_jsonl}")
    return out_json


# -----------------------------
# CLI entry
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a moderator-controlled CrewAI debate (DuckDuckGo search)."
    )
    parser.add_argument(
        "--config",
        default="../config/config.ini",
        help="Path to INI config (default: ../config/config.ini)",
    )
    parser.add_argument(
        "--claim",
        required=True,
        help='Conspiracy claim to debate, e.g. "COVID-19 vaccines contain microchips"',
    )
    args = parser.parse_args()

    run_debate(args.config, args.claim)
