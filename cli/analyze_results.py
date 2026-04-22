"""Advanced Result Analyzer - Logic promoted from legacy analysis_batch.

This CLI tool performs deep linguistic and emotional analysis on completed debate runs,
utilizing BERT-based models and TextBlob for research-grade insights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from debate.analysis.utils.features import (
    EmotionAnalyzer,
    analyze_utterance_features,
    load_run_inputs,
)
from debate.analysis.analysis_runner import run_custom_analyzers
from debate.analysis.utils.winner_inference import infer_winner_from_final_report
from debate.core.config import DebateConfig

console = Console()


def _plot_sentiment_comparison(
    output_dir: Path,
    turn_indices: list[int],
    series_by_agent: dict[str, list[float]],
    metric_name: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for agent, values in series_by_agent.items():
        if values:
            rounds = turn_indices[: len(values)]
            plt.plot(rounds, values, marker="o", label=agent)
    plt.xlabel("Turn")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Comparison")
    plt.legend()
    plt.tight_layout()
    slug = metric_name.lower().replace(" ", "_")
    path = output_dir / f"{slug}_comparison.png"
    plt.savefig(path)
    plt.close()
    return str(path)


def _plot_rhetorical_markers(
    output_dir: Path,
    agent: str,
    turn_indices: list[int],
    marker_data: dict[str, list[float]],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for label, values in marker_data.items():
        if values:
            rounds = turn_indices[: len(values)]
            plt.plot(rounds, values, marker="o", label=label)
    plt.xlabel("Turn")
    plt.ylabel("Score")
    plt.title(f"Rhetorical Markers: {agent}")
    plt.legend()
    plt.tight_layout()
    path = output_dir / f"{agent.lower()}_rhetorical_markers.png"
    plt.savefig(path)
    plt.close()
    return str(path)


def _plot_emotion_distribution(
    output_dir: Path,
    agent: str,
    counts: dict[str, int],
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(counts.keys())
    values = list(counts.values())
    plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
    plt.bar(labels, values)
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.title(f"Dominant Emotion Distribution: {agent}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = output_dir / f"{agent.lower()}_emotion_distribution.png"
    plt.savefig(path)
    plt.close()
    return str(path)


def analyze_single_run(
    run_dir: Path,
    output_root: Path,
    config: DebateConfig,
    skip_emotion: bool = False,
):
    """Perform advanced analysis on a single run directory."""
    run_id = run_dir.name
    analysis_dir = output_root / "analysis" / run_id
    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_run_inputs(run_dir)
    debate_log_path = run_dir / "debate_log.csv"
    if not debate_log_path.exists():
        console.print(f"[red]Error:[/] debate_log.csv not found in {run_id}")
        return

    df = pd.read_csv(debate_log_path)
    # Filter to actual agent turns
    df_agents = df[df["speaker"].isin(["CA", "SA"])].copy()
    if df_agents.empty:
        console.print(f"[yellow]Warning:[/] No CA/SA turns found in {run_id}")
        return

    # 1. Feature Extraction
    features_list = []
    emotions_list = []

    emotion_analyzer = (
        EmotionAnalyzer.get_instance(model_name=config.adv_analysis_emotion_model)
        if not skip_emotion
        else None
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description=f"Analyzing {run_id}...", total=None)

        for _, row in df_agents.iterrows():
            text = str(row.get("claim", ""))
            features_list.append(
                analyze_utterance_features(
                    text,
                    uncertainty_lexicon=config.uncertainty_lexicon,
                    strong_modality_lexicon=config.strong_modality_lexicon,
                    weak_modality_lexicon=config.weak_modality_lexicon,
                )
            )

            if emotion_analyzer:
                emotions_list.append(emotion_analyzer.analyze(text))

    df_feats = pd.DataFrame(features_list)
    df_enriched = pd.concat([df_agents.reset_index(drop=True), df_feats], axis=1)

    if emotions_list:
        df_emotions = pd.DataFrame(emotions_list).fillna(0.0)
        df_enriched = pd.concat([df_enriched, df_emotions], axis=1)

        # Calculate dominant emotion
        emotion_cols = [c for c in df_emotions.columns if c.startswith("emotion_")]
        df_enriched["dominant_emotion"] = df_enriched[emotion_cols].idxmax(axis=1).str.replace("emotion_", "")

    # 2. Winning Inference
    final_report = inputs.memory.final_report
    winner_data = infer_winner_from_final_report(
        final_report.model_dump() if final_report else None,
        use_explicit_winner_fields=False,
        use_stance_trajectory_fallback=False,
    )

    # 3. Generating Plots
    turn_indices = list(range(1, len(df_enriched) + 1))

    # Polarity Comparison
    sentiment_data = {
        agent: df_enriched[df_enriched["speaker"] == agent]["polarity"].tolist()
        for agent in ["CA", "SA"]
    }
    _plot_sentiment_comparison(plots_dir, turn_indices, sentiment_data, "Polarity")

    # Subjectivity Comparison
    subjectivity_data = {
        agent: df_enriched[df_enriched["speaker"] == agent]["subjectivity"].tolist()
        for agent in ["CA", "SA"]
    }
    _plot_sentiment_comparison(plots_dir, turn_indices, subjectivity_data, "Subjectivity")

    # Rhetorical Markers
    for agent in ["CA", "SA"]:
        sub = df_enriched[df_enriched["speaker"] == agent]
        marker_data = {
            col: sub[col].tolist()
            for col in ["uncertainty_score", "strong_modality_score", "weak_modality_score"]
        }
        _plot_rhetorical_markers(plots_dir, agent, turn_indices, marker_data)

    # Emotion Distribution
    if not skip_emotion and "dominant_emotion" in df_enriched.columns:
        for agent in ["CA", "SA"]:
            counts = df_enriched[df_enriched["speaker"] == agent]["dominant_emotion"].value_counts().to_dict()
            _plot_emotion_distribution(plots_dir, agent, counts)

    # 4. Save results
    df_enriched.to_csv(analysis_dir / "enriched_debate_log.csv", index=False)

    report = {
        "run_id": run_id,
        "winner_inference": winner_data,
        "metrics_summary": df_feats.mean().to_dict(),
    }
    with open(analysis_dir / "advanced_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="SixSeven Advanced Result Analyzer")
    parser.add_argument("--run", type=str, help="Specific run directory to analyze")
    parser.add_argument("--dir", type=str, help="Input directory. Defaults to old_artifacts for the custom suite and results/raw for advanced analysis")
    parser.add_argument("--out", type=str, default="results", help="Root directory for analysis output")
    parser.add_argument("--artifacts", type=str, default="old_artifacts", help="Artifacts directory for role analyzer")
    parser.add_argument("--no-emotion", action="store_true", help="Skip heavy BERT emotion analysis")
    parser.add_argument("--topic-analysis-emotion-model", type=str, default="j-hartmann/emotion-english-distilroberta-base", help="BERT emotion model for topic analysis")
    parser.add_argument("--topic-analysis-batch-size", type=int, default=16, help="Batch size for topic analysis BERT emotion inference")
    parser.add_argument("--topic-analysis-max-length", type=int, default=256, help="Max token length for topic analysis BERT emotion inference")
    parser.add_argument("--topic-analysis-uncertainty-lexicon", type=str, help="Optional uncertainty lexicon override for topic analysis")
    parser.add_argument("--topic-analysis-strong-modality-lexicon", type=str, help="Optional strong modality lexicon override for topic analysis")
    parser.add_argument("--topic-analysis-weak-modality-lexicon", type=str, help="Optional weak modality lexicon override for topic analysis")
    parser.add_argument("--topic-analysis-nrc-lexicon", type=str, help="Optional NRC lexicon override for topic analysis")
    parser.add_argument("--topic-analysis-emfd-lexicon", type=str, help="Optional eMFD lexicon override for topic analysis")
    parser.add_argument("--topic-analysis-skip-emotion", action="store_true", help="Skip BERT emotion analysis for topic analysis")
    parser.add_argument("--custom-suite", dest="custom_suite", action="store_true", help="Run custom analyzers sequentially (debate_analysis -> topic_analysis -> role_analysis -> llm_analysis)")
    parser.add_argument("--advanced-analysis", dest="custom_suite", action="store_false", help="Run the older advanced analysis flow instead of the default custom suite")
    parser.add_argument("--max-runs", type=int, help="Maximum number of runs for custom suite analyzers that support it")
    parser.add_argument("--overwrite-existing", action="store_true", help="Recompute existing outputs for custom suite analyzers")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop custom suite on first analyzer failure")
    parser.set_defaults(custom_suite=True)
    args = parser.parse_args(argv)

    if args.custom_suite:
        input_runs_dir = args.dir or "old_artifacts"
        results = run_custom_analyzers(
            results_root=args.out,
            input_runs_dir=input_runs_dir,
            artifacts_dir=args.artifacts,
            skip_role_emotion=args.no_emotion,
            skip_topic_analysis_emotion=args.topic_analysis_skip_emotion or args.no_emotion,
            overwrite_existing=args.overwrite_existing,
            max_runs=args.max_runs,
            topic_analysis_emotion_model=args.topic_analysis_emotion_model,
            topic_analysis_batch_size=args.topic_analysis_batch_size,
            topic_analysis_max_length=args.topic_analysis_max_length,
            topic_analysis_uncertainty_lexicon=args.topic_analysis_uncertainty_lexicon,
            topic_analysis_strong_modality_lexicon=args.topic_analysis_strong_modality_lexicon,
            topic_analysis_weak_modality_lexicon=args.topic_analysis_weak_modality_lexicon,
            topic_analysis_nrc_lexicon=args.topic_analysis_nrc_lexicon,
            topic_analysis_emfd_lexicon=args.topic_analysis_emfd_lexicon,
            continue_on_error=not args.stop_on_error,
        )

        table = Table(title="Custom Analyzer Suite (Sequential)")
        table.add_column("Analyzer", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="magenta")

        for result in results:
            if not result.show_in_summary:
                continue
            status = "Success" if result.success else "Failed"
            details = "-" if result.success else (result.error.splitlines()[-1] if result.error else "Unknown error")
            table.add_row(result.name, status, details)

        console.print(table)
        return

    config = DebateConfig.from_ini()
    output_root = Path(args.out or config.output_dir)

    if args.run:
        run_path = Path(args.run)
        report = analyze_single_run(run_path, output_root, config, skip_emotion=args.no_emotion)
        if report:
            console.print(Panel(f"Analysis Complete: [green]{run_path.name}[/]\nWinner Inferred: [bold]{report['winner_inference']['winner_inferred']}[/]", title="Success"))
    else:
        raw_root = Path(args.dir or "results/raw")
        run_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("run_")])

        if not run_dirs:
            console.print(f"[red]Error:[/] No run directories found in {raw_root}")
            return

        # Respect config limit if not overridden (simple implementation: use config.adv_analysis_max_runs)
        limit = config.adv_analysis_max_runs
        if limit and len(run_dirs) > limit:
            console.print(f"[yellow]Note:[/] Processing only first {limit} runs (set 'max_runs' in config to change)")
            run_dirs = run_dirs[:limit]

        table = Table(title=f"Processing Batch Analysis ({len(run_dirs)} runs)")
        table.add_column("Run ID", style="cyan")
        table.add_column("Winner", style="green")
        table.add_column("Status", style="magenta")

        for rd in run_dirs:
            try:
                # Check overwrite flag
                analysis_path = output_root / "analysis" / rd.name / "advanced_report.json"
                if analysis_path.exists() and not config.adv_analysis_overwrite:
                    table.add_row(rd.name, "-", "Skipped")
                    continue

                report = analyze_single_run(rd, output_root, config, skip_emotion=args.no_emotion)
                winner = report["winner_inference"]["winner_inferred"] if report else "N/A"
                table.add_row(rd.name, str(winner), "Success")
            except Exception as e:
                table.add_row(rd.name, "Error", f"Error: {e}")

        console.print(table)


if __name__ == "__main__":
    main()
