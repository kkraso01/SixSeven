"""Core modules: foundational data models, configuration, and topics."""

from .config import DebateConfig
from .container import DebateServices, build_default_services
from .protocols import (
    ArtifactExporter,
    DebateAnalyzer,
    LLMClient,
    PromptLoader,
    SearchProvider,
    StructuredLLMService,
)
from .schemas import (
    AgentState,
    AgentTurn,
    DebateLogItem,
    FinalReport,
    MemoryState,
    ModeratorDecision,
    ModeratorRecap,
    ModeratorTurnControl,
    ScientificTurn,
    Scoreboard,
    SearchRequest,
)
from .topics import (
    CONSPIRACY_TOPICS,
    DebateTopic,
    get_sample_topics,
    get_topic_by_id,
    get_topics_by_category,
    list_categories,
)

__all__ = [
    "DebateConfig",
    "DebateServices",
    "build_default_services",
    # Protocols
    "LLMClient",
    "StructuredLLMService",
    "SearchProvider",
    "PromptLoader",
    "ArtifactExporter",
    "DebateAnalyzer",
    # Schemas
    "AgentState",
    "AgentTurn",
    "DebateLogItem",
    "DebateTopic",
    "FinalReport",
    "MemoryState",
    "ModeratorDecision",
    "ModeratorRecap",
    "ModeratorTurnControl",
    "Scoreboard",
    "ScientificTurn",
    "SearchRequest",
    # Topics
    "CONSPIRACY_TOPICS",
    "get_sample_topics",
    "get_topic_by_id",
    "get_topics_by_category",
    "list_categories",
]
