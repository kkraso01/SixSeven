from __future__ import annotations

from dataclasses import dataclass

from debate import DebateConfig, run_debate
from debate.core.container import build_default_services
from debate.core.logging import setup_logging

# Configure logging once
setup_logging()


@dataclass(frozen=True)
class RunResult:
    run_dir: str
    bundle: object


def run_experiment():
    config = DebateConfig.from_env()
    services = build_default_services(config)
    result = run_debate(
        topic="The safety and efficacy of mRNA vaccine technology",
        motion="Rapid development timelines and limited long-term data make mRNA vaccines fundamentally untrustworthy compared to traditional vaccine methods.",
        rounds=config.rounds,
        config=config,
        services=services,
    )
    return RunResult(run_dir=str(result.run_dir), bundle=result)


def main():
    run_experiment()


if __name__ == "__main__":
    main()
