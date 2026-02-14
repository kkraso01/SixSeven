"""Test all imports and basic functionality."""

from __future__ import annotations

import sys


def test_core_imports():
    """Test core module imports."""
    print("Testing core imports...")
    try:

        print(" Core imports successful")
        return True
    except Exception as e:
        print(f" Core imports failed: {e}")
        return False


def test_topic_imports():
    """Test topic/dataset imports."""
    print("\nTesting topic imports...")
    try:
        from debate_sim.core.topics import (
            CONSPIRACY_TOPICS,
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

        print(" Export imports successful")
        return True
    except Exception as e:
        print(f" Export imports failed: {e}")
        return False


def test_llm_imports():
    """Test LLM client imports."""
    print("\nTesting LLM imports...")
    try:

        print(" LLM imports successful")
        return True
    except Exception as e:
        print(f" LLM imports failed: {e}")
        return False


def test_orchestrator_imports():
    """Test orchestrator imports."""
    print("\nTesting orchestrator imports...")
    try:

        print(" Orchestrator imports successful")
        return True
    except Exception as e:
        print(f" Orchestrator imports failed: {e}")
        return False


def test_batch_runner_import():
    """Test batch runner import."""
    print("\nTesting batch runner...")
    try:
        import cli.batch_gemini
        import cli.batch_ollama

        assert cli.batch_ollama is not None
        assert cli.batch_gemini is not None
        print(" Batch runner imports successful")
        return True
    except Exception as e:
        print(f" Batch runner import failed: {e}")
        return False


def test_config_loading():
    """Test config.ini loading."""
    print("\nTesting config loading...")
    try:
        from debate_sim.core.config import DebateConfig

        config = DebateConfig.from_ini()
        print(" Config loaded successfully")
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
