"""Test all imports and basic functionality."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_core_imports():
    """Test core module imports."""
    print("Testing core imports...")
    try:
        from src.debate_sim import DebateConfig, run_debate
        from src.debate_sim.config import DebateConfig as ConfigDirect
        from src.debate_sim.schemas import (
            AgentTurn,
            ScientificTurn,
            DebateLogItem,
            FinalReport,
            MemoryState,
            ModeratorDecision,
            ModeratorRecap,
            SearchRequest,
        )
        print(" Core imports successful")
        return True
    except Exception as e:
        print(f" Core imports failed: {e}")
        return False


def test_topic_imports():
    """Test topic/dataset imports."""
    print("\nTesting topic imports...")
    try:
        from src.debate_sim.topics import (
            CONSPIRACY_TOPICS,
            DebateTopic,
            get_sample_topics,
            get_topic_by_id,
        )
        print(f" Topics loaded: {len(CONSPIRACY_TOPICS)} topics available")
        return True
    except Exception as e:
        print(f" Topic imports failed: {e}")
        return False


def test_export_imports():
    """Test export functionality imports."""
    print("\nTesting export imports...")
    try:
        from src.debate_sim.export.csv_export import (
            export_all_debates_to_csv,
            export_debate_log_to_csv,
        )
        from src.debate_sim.export.writer import ExportBundle, write_artifacts
        print(" Export imports successful")
        return True
    except Exception as e:
        print(f" Export imports failed: {e}")
        return False


def test_llm_imports():
    """Test LLM client imports."""
    print("\nTesting LLM imports...")
    try:
        from src.debate_sim.llm.instructor_wrapper import StructuredLLM
        from src.debate_sim.llm.ollama_client import OllamaClient
        from src.debate_sim.llm.search_tool import search_web
        print(" LLM imports successful")
        return True
    except Exception as e:
        print(f" LLM imports failed: {e}")
        return False


def test_orchestrator_imports():
    """Test orchestrator imports."""
    print("\nTesting orchestrator imports...")
    try:
        from src.debate_sim.debate.orchestrator import run_debate
        from src.debate_sim.debate.evaluation import build_metrics_table
        from src.debate_sim.debate.protocol import load_prompt
        print(" Orchestrator imports successful")
        return True
    except Exception as e:
        print(f" Orchestrator imports failed: {e}")
        return False


def test_batch_runner_import():
    """Test batch runner import."""
    print("\nTesting batch runner...")
    try:
        import batch_runner
        print(" Batch runner imports successful")
        return True
    except Exception as e:
        print(f" Batch runner import failed: {e}")
        return False


def test_config_loading():
    """Test config.ini loading."""
    print("\nTesting config loading...")
    try:
        from src.debate_sim.config import DebateConfig
        config = DebateConfig.from_ini()
        print(f" Config loaded successfully")
        print(f"  - API mode: {config.api_mode}")
        print(f"  - Moderator model: {config.moderator_model}")
        print(f"  - Gemini API key configured: {'Yes' if config.gemini_api_key else 'No'}")
        return True
    except Exception as e:
        print(f" Config loading failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("IMPORT VALIDATION TEST")
    print("=" * 80)
    
    results = []
    results.append(("Core imports", test_core_imports()))
    results.append(("Topic imports", test_topic_imports()))
    results.append(("Export imports", test_export_imports()))
    results.append(("LLM imports", test_llm_imports()))
    results.append(("Orchestrator imports", test_orchestrator_imports()))
    results.append(("Batch runner", test_batch_runner_import()))
    results.append(("Config loading", test_config_loading()))
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = " PASS" if result else " FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n All tests passed! Project is ready to run.")
        return 0
    else:
        print("\n Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
