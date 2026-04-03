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


def plot_sentiment_comparison(
    output_dir: Path,
    turn_indices: list[int],
    sentiment_data: dict[str, list[float]],
    metric_name: str = "Polarity",
) -> str:
    """Plot polarity or subjectivity for both agents over turns."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    colors = {"CA": "firebrick", "SA": "royalblue"}
    for agent, values in sentiment_data.items():
        if values:
            plt.plot(
                turn_indices[: len(values)],
                values,
                marker="o",
                label=agent,
                color=colors.get(agent, None),
            )
    plt.xlabel("Turn Index")
    plt.ylabel(metric_name)
    plt.title(f"Speaker Comparison - {metric_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    path = output_dir / f"comparison_{metric_name.lower()}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return str(path)


def plot_emotion_distribution(output_dir: Path, agent: str, emotion_counts: dict[str, int]) -> str:
    """Plot a bar chart of dominant emotions for a specific agent."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not emotion_counts:
        return ""

    labels = list(emotion_counts.keys())
    values = list(emotion_counts.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color="mediumpurple")
    plt.xlabel("Emotion")
    plt.ylabel("Turn Count")
    plt.title(f"{agent} - Dominant Emotion Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = output_dir / f"{agent.lower()}_emotion_dist.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return str(path)


def plot_rhetorical_markers(
    output_dir: Path,
    agent: str,
    turn_indices: list[int],
    marker_data: dict[str, list[int]],
) -> str:
    """Plot rhetorical markers (uncertainty, modality) over time."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for label, values in marker_data.items():
        if values:
            plt.plot(turn_indices[: len(values)], values, marker="s", label=label.replace("_", " "))
    plt.xlabel("Turn Index")
    plt.ylabel("Score / Count")
    plt.title(f"{agent} - Rhetorical Markers Over Time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = output_dir / f"{agent.lower()}_rhetorical_markers.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return str(path)
