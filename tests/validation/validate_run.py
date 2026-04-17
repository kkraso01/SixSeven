from __future__ import annotations

import sys
from pathlib import Path

from debate.analysis.analysis_runner import analyze_run
from debate.core.schemas import MemoryState


def validate_run(run_dir: str) -> None:
    path = Path(run_dir)
    memory_path = path / "memory.json"
    if not memory_path.exists():
        raise FileNotFoundError(f"Missing memory.json in {run_dir}")
    MemoryState.model_validate_json(memory_path.read_text(encoding="utf-8"))

    report = analyze_run(run_dir)
    report_path = path / "analysis_report.json"
    markdown_path = path / "analysis_report.md"
    figures_dir = path / "figures"

    if not report_path.exists():
        raise FileNotFoundError("analysis_report.json not found")
    if not markdown_path.exists():
        raise FileNotFoundError("analysis_report.md not found")
    if not figures_dir.exists() or not any(figures_dir.iterdir()):
        raise FileNotFoundError("No figures generated")

    print(f"Validated analysis for {report.run_id}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python tests/validation/validate_run.py <run_dir>")
    validate_run(sys.argv[1])


if __name__ == "__main__":
    main()
