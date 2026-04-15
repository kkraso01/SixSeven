from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

from debate import DebateConfig, run_debate
from debate.core.container import build_default_services
from debate.core.logging import setup_logging

# Configure logging once
setup_logging()


@dataclass(frozen=True)
class RunResult:
    run_dir: str
    bundle: object


DEFAULT_TOPIC = "The safety and efficacy of mRNA vaccine technology"
DEFAULT_MOTION = (
    "Rapid development timelines and limited long-term data make mRNA vaccines "
    "fundamentally untrustworthy compared to traditional vaccine methods."
)


def run_experiment(
    *,
    topic: str = DEFAULT_TOPIC,
    motion: str = DEFAULT_MOTION,
    topic_description: str | None = None,
    config_path: str = "config/config.ini",
    output_dir: str | None = None,
    rounds: int | None = None,
    word_limit: int | None = None,
) -> RunResult:
    config = DebateConfig.from_ini(config_path)
    if output_dir:
        config = replace(config, output_dir=output_dir)
    if rounds is not None:
        config = replace(config, rounds=rounds)
    if word_limit is not None:
        config = replace(config, word_limit=word_limit)

    services = build_default_services(config)
    result = run_debate(
        topic=topic,
        motion=motion,
        rounds=config.rounds,
        config=config,
        services=services,
        topic_description=topic_description,
    )
    return RunResult(run_dir=str(result.run_dir), bundle=result)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SixSeven Debate Simulator")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC, help="Debate topic")
    parser.add_argument("--motion", type=str, default=DEFAULT_MOTION, help="Debate motion")
    parser.add_argument(
        "--topic-description",
        type=str,
        default=None,
        help="Optional background context for the topic used in initial prompts",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.ini",
        help="Path to config INI file",
    )
    parser.add_argument(
        "--out",
        "--output-dir",
        "--dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Output root for artifacts (for example: new_artifacts)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override number of rounds for this run",
    )
    parser.add_argument(
        "--word-limit",
        type=int,
        default=None,
        help="Override per-turn word limit for this run",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_result = run_experiment(
        topic=args.topic,
        motion=args.motion,
        topic_description=args.topic_description,
        config_path=args.config,
        output_dir=args.output_dir,
        rounds=args.rounds,
        word_limit=args.word_limit,
    )
    print(f"Run complete: {run_result.run_dir}")


if __name__ == "__main__":
    main()
