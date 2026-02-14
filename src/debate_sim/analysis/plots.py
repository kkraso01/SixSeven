from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_stance_trajectory(
    output_dir: Path,
    per_round_confidence: dict[str, list[int]],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for agent, values in per_round_confidence.items():
        if values:
            rounds = range(1, len(values) + 1)
            plt.plot(rounds, values, marker="o", label=agent)
    plt.xlabel("Round")
    plt.ylabel("Confidence")
    plt.title("Stance Trajectory")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "stance_trajectory.png"
    plt.savefig(path)
    plt.close()
    return str(path)


def plot_quality_scores(output_dir: Path, quality_scores: dict[str, list[int]]) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for label, values in quality_scores.items():
        if values:
            rounds = range(1, len(values) + 1)
            plt.plot(rounds, values, marker="o", label=label)
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.title("Dialogue Quality Over Rounds")
    plt.legend()
    plt.tight_layout()
    path = output_dir / "quality_trends.png"
    plt.savefig(path)
    plt.close()
    return str(path)


def plot_tactic_histogram(output_dir: Path, tactic_counts: dict[str, int]) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(tactic_counts.keys())
    values = list(tactic_counts.values())
    plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
    plt.bar(labels, values)
    plt.xlabel("Tactic")
    plt.ylabel("Count")
    plt.title("CA Tactic Usage")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = output_dir / "tactic_histogram.png"
    plt.savefig(path)
    plt.close()
    return str(path)


def plot_aggregate_histogram(output_dir: Path, net_shifts: list[float]) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(net_shifts, bins=10)
    plt.xlabel("Net CA Shift")
    plt.ylabel("Run Count")
    plt.title("Distribution of Net CA Shifts")
    plt.tight_layout()
    path = output_dir / "net_ca_shift_histogram.png"
    plt.savefig(path)
    plt.close()
    return str(path)


def plot_aggregate_scatter(
    output_dir: Path,
    civility_means: list[float],
    net_shifts: list[float],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.scatter(civility_means, net_shifts)
    plt.xlabel("Mean Civility")
    plt.ylabel("Net CA Shift")
    plt.title("Civility vs Net CA Shift")
    plt.tight_layout()
    path = output_dir / "civility_vs_shift.png"
    plt.savefig(path)
    plt.close()
    return str(path)
