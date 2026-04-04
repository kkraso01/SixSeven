"""Dataset of conspiracy theories and debate topics for systematic testing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path


@dataclass
class DebateTopic:
    """A conspiracy theory debate topic."""

    id: str
    category: str
    topic: str
    motion: str
    description: str


DEFAULT_TOPICS_PATH = Path(__file__).resolve().parents[3] / "config" / "topics.json"


def _resolve_topics_path() -> Path:
    env_path = os.getenv("DEBATE_TOPICS_FILE")
    return Path(env_path).expanduser() if env_path else DEFAULT_TOPICS_PATH


def _load_topics() -> list[DebateTopic]:
    topics_path = _resolve_topics_path()
    with topics_path.open("r", encoding="utf-8") as file:
        raw_topics = json.load(file)

    if not isinstance(raw_topics, list):
        raise ValueError(f"Topics JSON must contain a list: {topics_path}")

    try:
        return [DebateTopic(**topic) for topic in raw_topics]
    except TypeError as exc:
        raise ValueError(f"Invalid topic schema in {topics_path}: {exc}") from exc


CONSPIRACY_TOPICS: list[DebateTopic] = _load_topics()


def get_topics_by_category(category: str) -> list[DebateTopic]:
    """Get all topics in a specific category."""
    return [t for t in CONSPIRACY_TOPICS if t.category == category]


def get_topic_by_id(topic_id: str) -> DebateTopic:
    """Get a specific topic by its ID."""
    for topic in CONSPIRACY_TOPICS:
        if topic.id == topic_id:
            return topic
    raise ValueError(f"Topic ID not found: {topic_id}")


def list_categories() -> list[str]:
    """Get list of all unique categories."""
    return sorted(set(t.category for t in CONSPIRACY_TOPICS))


def get_sample_topics(count: int = 5) -> list[DebateTopic]:
    """Get a sample of topics for quick testing."""
    return CONSPIRACY_TOPICS[:count]
