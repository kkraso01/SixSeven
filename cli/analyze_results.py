"""Advanced Result Analyzer - Logic promoted from legacy analysis_batch.

This CLI tool performs deep linguistic and emotional analysis on completed debate runs,
utilizing BERT-based models and TextBlob for research-grade insights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from debate.analysis.features import (
    EmotionAnalyzer,
    analyze_utterance_features,
    infer_winner_from_text,
    load_run_inputs,
)
from debate.core.config import DebateConfig
from debate.analysis.plots import (
    plot_emotion_distribution,
    plot_rhetorical_markers,
    plot_sentiment_comparison,
)

console = Console()


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
    winner_data = infer_winner_from_text(final_report.outcome_summary if final_report else "")

    # 3. Generating Plots
    turn_indices = list(range(1, len(df_enriched) + 1))
    
    # Polarity Comparison
    sentiment_data = {
        agent: df_enriched[df_enriched["speaker"] == agent]["polarity"].tolist()
        for agent in ["CA", "SA"]
    }
    plot_sentiment_comparison(plots_dir, turn_indices, sentiment_data, "Polarity")

    # Subjectivity Comparison
    subjectivity_data = {
        agent: df_enriched[df_enriched["speaker"] == agent]["subjectivity"].tolist()
        for agent in ["CA", "SA"]
    }
    plot_sentiment_comparison(plots_dir, turn_indices, subjectivity_data, "Subjectivity")

    # Rhetorical Markers
    for agent in ["CA", "SA"]:
        sub = df_enriched[df_enriched["speaker"] == agent]
        marker_data = {
            col: sub[col].tolist()
            for col in ["uncertainty_score", "strong_modality_score", "weak_modality_score"]
        }
        plot_rhetorical_markers(plots_dir, agent, turn_indices, marker_data)

    # Emotion Distribution
    if not skip_emotion and "dominant_emotion" in df_enriched.columns:
        for agent in ["CA", "SA"]:
            counts = df_enriched[df_enriched["speaker"] == agent]["dominant_emotion"].value_counts().to_dict()
            plot_emotion_distribution(plots_dir, agent, counts)

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


def main():
    parser = argparse.ArgumentParser(description="SixSeven Advanced Result Analyzer")
    parser.add_argument("--run", type=str, help="Specific run directory to analyze")
    parser.add_argument("--dir", type=str, default="results/raw", help="Directory containing multiple runs")
    parser.add_argument("--out", type=str, default="results", help="Root directory for analysis output")
    parser.add_argument("--no-emotion", action="store_true", help="Skip heavy BERT emotion analysis")
    args = parser.parse_args()

    config = DebateConfig.from_ini()
    output_root = Path(args.out or config.output_dir)
    
    if args.run:
        run_path = Path(args.run)
        report = analyze_single_run(run_path, output_root, config, skip_emotion=args.no_emotion)
        if report:
            console.print(Panel(f"Analysis Complete: [green]{run_path.name}[/]\nWinner Inferred: [bold]{report['winner_inference']['winner_inferred']}[/]", title="Success"))
    else:
        raw_root = Path(args.dir)
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
