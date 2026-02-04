from __future__ import annotations

from dataclasses import dataclass

from debate_sim import DebateConfig, run_debate


@dataclass(frozen=True)
class RunResult:
    run_dir: str
    bundle: object


def run_experiment():
    config = DebateConfig.from_env()
    result = run_debate(
        topic="Persuasion dynamics in contested claims",
        motion="Institutional narratives reduce public trust more than they build it.",
        rounds=config.rounds,
        config=config,
    )
    return RunResult(run_dir=str(result.run_dir), bundle=result)


if __name__ == "__main__":
    run_experiment()
