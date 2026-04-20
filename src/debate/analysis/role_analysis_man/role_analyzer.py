import argparse
import ast
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from debate.analysis.features import EmotionAnalyzer, analyze_utterance_features
from debate.analysis.lexicons import EMOTION_LEXICON

console = Console()

def extract_nrc_emotions(text: str) -> dict[str, float]:
    """Calculate raw frequency of NRC Emotion Lexicon words in the text."""
    text_lower = str(text).lower()
    tokens = set(re.findall(r"[a-z']+", text_lower))
    scores = {}
    for emotion, words in EMOTION_LEXICON.items():
        # Keep counts
        scores[f"nrc_emotion_{emotion}"] = sum(1 for t in tokens if t in words)
    return scores

def load_and_enrich_data(input_csv: Path, output_csv: Path, skip_emotion: bool = False) -> pd.DataFrame:
    """Loads the base debate log, enriches it with modality and emotion features, and saves it."""
    if output_csv.exists():
        console.print(f"[green]Found existing enriched dataset at {output_csv}, loading...[/]")
        return pd.read_csv(output_csv)

    console.print(f"[yellow]Enriched dataset not found at {output_csv}. Starting enrichment of {input_csv}...[/]")
    df = pd.read_csv(input_csv)

    # Initialize analyzers
    features_list = []
    emotion_list = []
    
    emotion_analyzer = None
    if not skip_emotion:
        try:
            emotion_analyzer = EmotionAnalyzer.get_instance()
        except ImportError:
            console.print("[red]Transformers not installed. Skipping emotion analysis.[/]")
            skip_emotion = True

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Analyzing utterances...", total=len(df))
        
        for _, row in df.iterrows():
            text = str(row.get('utterance', ''))
            
            # Linguistic features (modality, sentiment, etc.)
            features = analyze_utterance_features(text)
            features_list.append(features)
            
            # Emotion features
            row_emotions = {}
            if not skip_emotion and emotion_analyzer:
                bert_emotions = emotion_analyzer.analyze(text[:512]) # Truncate for BERT max length
                for k, v in bert_emotions.items():
                    row_emotions[k.replace('emotion_', 'bert_emotion_')] = v
            else:
                row_emotions = {}
                
            # Add NRC emotions
            row_emotions.update(extract_nrc_emotions(text))
            
            emotion_list.append(row_emotions)
            
            progress.advance(task)

    # Merge features
    df_features = pd.DataFrame(features_list)
    df_emotions = pd.DataFrame(emotion_list)
    
    df_enriched = pd.concat([df, df_features, df_emotions], axis=1)
    
    # Save enriched dataset
    df_enriched.to_csv(output_csv, index=False)
    console.print(f"[green]Successfully saved enriched dataset to {output_csv}[/]")
    
    return df_enriched


def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """Calculates Jaccard similarity between two texts."""
    set1 = set(str(text1).lower().split())
    set2 = set(str(text2).lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(intersection) / len(union)


def analyze_moderator_dynamics(df: pd.DataFrame, output_dir: Path):
    """Moderator Dynamics: Civility Management and Bridge-Building"""
    console.print("[bold blue]Running Deliverable 1: Moderator Dynamics...[/]")
    
    # Calculate sequential turn index per debate to track timeline
    df = df.copy()
    df['turn_index'] = df.groupby('debate_id').cumcount() + 1
    
    # Group by debate
    debates = df.groupby('debate_id')
    
    # 1.1 Emotion Dampening
    # Find negative emotion column (e.g. emotion_anger + emotion_sadness + emotion_fear)
    neg_cols = [c for c in df.columns if c.endswith('_anger') or c.endswith('_fear') or c.endswith('_sadness')]
    
    if not neg_cols:
        console.print("[yellow]Skipping Metric 1.1: Negative emotion columns missing.[/]")
    else:
        df['negative_score'] = df[neg_cols].sum(axis=1)
        
        # Calculate average negative score across all debates by turn index
        turn_avg = df.groupby(['turn_index', 'speaker_role'])['negative_score'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        # Plot emotion trajectory for debaters
        ax = sns.lineplot(data=turn_avg[turn_avg['speaker_role'] != 'moderator'], 
                     x='turn_index', y='negative_score', hue='speaker_role', marker='o', linewidth=2)
        
        # Add numerical labels to data points
        for _, row in turn_avg[turn_avg['speaker_role'] != 'moderator'].iterrows():
            ax.text(row['turn_index'], row['negative_score'], f"{row['negative_score']:.2f}",
                    fontsize=8, ha='center', va='bottom', color='black')
        
        # Add vertical dashed lines for moderator interventions
        mod_turns = turn_avg[turn_avg['speaker_role'] == 'moderator']['turn_index'].unique()
        for i, mt in enumerate(mod_turns):
            plt.axvline(x=mt, color='r', linestyle='--', alpha=0.6, 
                        label='Moderator Intervention' if i == 0 else "")
            
        plt.title('Emotion Trajectory over Turn Index (Moderator Interventions)')
        plt.xlabel('Turn Index (Progress)')
        plt.ylabel('Average Negative Emotion')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / '1_1_emotion_dampening_trajectory.svg')
        plt.close()

    # 1.2 Bridge-Building (Vocabulary Overlap)
    vocab_overlaps = []
    for name, group in debates:
        group = group.sort_values('turn_index').reset_index(drop=True)
        max_round = group['round'].max()
        for i in range(2, len(group)):
            if group.loc[i, 'speaker_role'] == 'moderator':
                # Text of exactly the immediately preceding CA and SA turns
                prev_roles = set(group.loc[i-2:i-1, 'speaker_role'])
                if 'proponent' in prev_roles and 'opponent' in prev_roles:
                    prev_text = " ".join(group.loc[i-2:i-1, 'utterance'].astype(str).tolist())
                    mod_text = str(group.loc[i, 'utterance'])
                    overlap = calculate_jaccard_similarity(mod_text, prev_text)
                    
                    # Determine Debate Phase (Early, Middle, Late)
                    current_round = group.loc[i, 'round']
                    pct = current_round / max_round if max_round > 0 else 0
                    if pct <= 0.33:
                        phase = "Early"
                    elif pct <= 0.66:
                        phase = "Middle"
                    else:
                        phase = "Late"
                        
                    vocab_overlaps.append({'Phase': phase, 'Jaccard Similarity': overlap})
                
    if vocab_overlaps:
        overlaps_df = pd.DataFrame(vocab_overlaps)
        overlaps_df['Phase'] = pd.Categorical(overlaps_df['Phase'], categories=["Early", "Middle", "Late"], ordered=True)
        
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(data=overlaps_df, x='Phase', y='Jaccard Similarity', hue='Phase', errorbar='ci', palette='viridis', legend=False)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
        plt.title('Moderator Bridge-Building (Vocabulary Overlap) by Phase')
        plt.xlabel('Debate Phase')
        plt.ylabel('Average Jaccard Similarity with Immediate Preceding Turns')
        plt.tight_layout()
        plt.savefig(output_dir / '1_2_bridge_building_phase.svg')
        plt.close()


def analyze_persona_profiling(df: pd.DataFrame, output_dir: Path):
    """Persona Profiling: Emotional Fingerprint and Modality"""
    console.print("[bold blue]Running Deliverable 2: Persona Profiling...[/]")
    
    agents_df = df[df['speaker_role'].isin(['proponent', 'opponent'])]
    
    # 2.1 Emotional Fingerprint
    bert_cols = [c for c in df.columns if c.startswith('bert_emotion_')]
    nrc_cols = [c for c in df.columns if c.startswith('nrc_emotion_')]
    
    if bert_cols:
        bert_means = agents_df.groupby('speaker_role')[bert_cols].mean().reset_index()
        bert_melted = bert_means.melt(id_vars='speaker_role', var_name='Emotion', value_name='Average Score')
        bert_melted['Emotion'] = bert_melted['Emotion'].str.replace('bert_emotion_', '').str.title()
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=bert_melted, x='Emotion', y='Average Score', hue='speaker_role')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
        plt.title('Emotional Fingerprint Comparison (BERT Contextual)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / '2_1_emotional_fingerprint_bert.svg')
        plt.close()
        
    if nrc_cols:
        nrc_means = agents_df.groupby('speaker_role')[nrc_cols].mean().reset_index()
        nrc_melted = nrc_means.melt(id_vars='speaker_role', var_name='Emotion', value_name='Average Score')
        nrc_melted['Emotion'] = nrc_melted['Emotion'].str.replace('nrc_emotion_', '').str.title()
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=nrc_melted, x='Emotion', y='Average Score', hue='speaker_role')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=8)
        plt.title('Emotional Fingerprint Comparison (NRC Lexicon)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / '2_1_emotional_fingerprint_nrc.svg')
        plt.close()
    # 2.2 Modality and Certainty (Density)
    if 'strong_modality_score' in df.columns and 'weak_modality_score' in df.columns and 'word_count' in df.columns:
        # Calculate density (per 100 words) to avoid length bias
        agents_df = agents_df.copy()
        valid_words = np.where(agents_df['word_count'] > 0, agents_df['word_count'], 1)
        agents_df['strong_modality_density'] = (agents_df['strong_modality_score'] / valid_words) * 100
        agents_df['weak_modality_density'] = (agents_df['weak_modality_score'] / valid_words) * 100
        
        mod_means = agents_df.groupby('speaker_role')[['strong_modality_density', 'weak_modality_density']].mean().reset_index()
        mod_means_melted = mod_means.melt(id_vars='speaker_role', var_name='Modality Type', value_name='Density (per 100 words)')
        
        # Clean up labels for plotting
        mod_means_melted['Modality Type'] = mod_means_melted['Modality Type'].str.replace('_density', '').str.replace('_', ' ').str.title()
        
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(data=mod_means_melted, y='speaker_role', x='Density (per 100 words)', hue='Modality Type', orient='h')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
        plt.title('The Hedging Gap (Modality Density Comparison)')
        plt.tight_layout()
        plt.savefig(output_dir / '2_2_hedging_gap_density.svg')
        plt.close()


def analyze_tactic_asymmetry(df: pd.DataFrame, output_dir: Path):
    """Tactic Asymmetry: Tool Usage vs Argument Strategy"""
    console.print("[bold blue]Running Deliverable 3: Tactic Asymmetry...[/]")
    
    agents_df = df[df['speaker_role'].isin(['proponent', 'opponent'])].copy()
    
    # Clean tool usage column (handles 'none', NaN, 'duckduckgo', 'tavily', etc.)
    agents_df['used_search'] = agents_df['tool_used'].apply(
        lambda x: pd.notna(x) and str(x).lower() not in ['none', '']
    )
    
    # 3.1 Tool Usage vs Argument Strategy
    if 'tactic_used' in agents_df.columns:
        # Cross-tabulate tactic mapped with search tool utilization
        tool_tactic_crosstab = agents_df.groupby(['speaker_role', 'tactic_used'])['used_search'].mean().reset_index()
        tool_tactic_crosstab['used_search_pct'] = tool_tactic_crosstab['used_search'] * 100
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=tool_tactic_crosstab, y='tactic_used', x='used_search_pct', hue='speaker_role')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=8)
        plt.title('Search Tool Invocation Frequency by Tactic')
        plt.xlabel('% of Turns Using a Search Tool (DuckDuckGo/Tavily)')
        plt.ylabel('Tactic Used')
        plt.tight_layout()
        plt.savefig(output_dir / '3_1_tool_usage_by_tactic.svg')
        plt.close()

        # Evolution of Tool Usage over the Debate
        tool_counts = agents_df.groupby(['round', 'speaker_role', 'used_search']).size().unstack(fill_value=0)
        for role in ['proponent', 'opponent']:
            if role in tool_counts.index.get_level_values('speaker_role'):
                role_tool = tool_counts.xs(role, level='speaker_role')
                # Avoid division by zero
                row_sums = role_tool.sum(axis=1)
                row_sums = row_sums.replace(0, 1)
                role_tool_pct = role_tool.div(row_sums, axis=0) * 100
                
                # Plot 100% stacked area for tool usage
                role_tool_pct.plot(kind='area', stacked=True, figsize=(10, 6), alpha=0.8)
                plt.title(f'Tool Usage Evolution Over Rounds ({role.title()})')
                plt.xlabel('Round')
                plt.ylabel('Percentage')
                plt.legend(title='Used Search Tool', loc='center left', bbox_to_anchor=(1, 0.5))
                plt.tight_layout()
                plt.savefig(output_dir / f'3_1_tool_evolution_{role}.svg')
                plt.close()

    # 3.2 Desperation (Confidence vs Tactics)
    if 'confidence' in agents_df.columns and 'tactic_used' in agents_df.columns:
        agents_df['confidence'] = pd.to_numeric(agents_df['confidence'], errors='coerce')
        
        drops = []
        for (did, role), group in agents_df.groupby(['debate_id', 'speaker_role']):
            group = group.sort_values('round').reset_index()
            # Calculate rolling 2-turn average
            group['rolling_conf'] = group['confidence'].rolling(window=2, min_periods=1).mean()
            # Calculate delta between current turn's rolling avg and previous
            group['conf_diff'] = group['rolling_conf'].diff()
            
            for i in range(1, len(group)):
                diff = group.loc[i, 'conf_diff']
                # Identify "drops" (negative delta)
                if pd.notna(diff) and diff < 0:
                    drops.append({
                        'debate_id': did,
                        'speaker_role': role,
                        'round': group.loc[i, 'round'],
                        'confidence_drop': diff,  # Negative value
                        'tactic_used': group.loc[i, 'tactic_used'],
                        'rolling_conf': group.loc[i, 'rolling_conf']
                    })
                    
        if drops:
            drops_df = pd.DataFrame(drops)
            
            plt.figure(figsize=(10, 6))
            # Plot the magnitude of the drop vs the tactic deployed
            # We take absolute value of the drop for better visualization scaling
            drops_df['drop_magnitude'] = drops_df['confidence_drop'].abs()
            sns.scatterplot(data=drops_df, x='drop_magnitude', y='tactic_used', 
                            hue='speaker_role', alpha=0.7, s=100)
            plt.title('Desperation: Tactics Deployed Following a Confidence Drop')
            plt.xlabel('Magnitude of Confidence Drop (Rolling Avg)')
            plt.ylabel('Tactic Used in Response')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / '3_2_desperation_scatter.svg')
            plt.close()


def analyze_interrogative_doubt(df: pd.DataFrame, output_dir: Path):
    """Deliverable 5.3 Option A: Interrogative Doubt (Split Boxplot)"""
    console.print("[bold blue]Running Deliverable 5.3: Interrogative Doubt (Option A)...[/]")
    
    # Filter for active debaters
    agents_df = df[df['speaker_role'].isin(['proponent', 'opponent'])].copy()
    
    if 'question_count' in agents_df.columns and 'subjectivity' in agents_df.columns:
        # Global Config for professional rendering
        sns.set_theme(style="whitegrid", context="talk")
        
        # Create categorical column
        agents_df['utterance_type'] = np.where(
            agents_df['question_count'] > 0, 
            "Interrogative (Has Questions)", 
            "Declarative (No Questions)"
        )
        
        plt.figure(figsize=(10, 8))
        
        # Split Boxplot
        palette = {"Declarative (No Questions)": "#95a5a6", "Interrogative (Has Questions)": "#e74c3c"}
        order = ['proponent', 'opponent']
        hue_order = ["Declarative (No Questions)", "Interrogative (Has Questions)"]
        
        ax = sns.boxplot(
            data=agents_df,
            x='speaker_role',
            y='subjectivity',
            hue='utterance_type',
            palette=palette,
            order=order,
            hue_order=hue_order
        )
        
        # Add numeric markers for medians
        medians = agents_df.groupby(['speaker_role', 'utterance_type'])['subjectivity'].median()
        for i, role in enumerate(order):
            for j, h in enumerate(hue_order):
                try:
                    val = medians.loc[(role, h)]
                    if pd.notna(val):
                        offset = -0.2 if j == 0 else 0.2
                        ax.text(i + offset, val + 0.01, f'{val:.3f}', 
                                ha='center', va='bottom', fontsize=10, color='black', weight='bold',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
                except KeyError:
                    pass
        
        plt.title('Interrogative Doubt ("JAQing Off" Metric)')
        plt.xlabel('Speaker Role')
        plt.ylabel('Subjectivity Score')
        
        sns.despine()
        plt.tight_layout()
        plt.savefig(output_dir / '5_1_interrogative_doubt.svg')
        plt.close()
        
        # Reset Seaborn theme back to default so it doesn't affect other plots
        sns.reset_orig()


def analyze_epistemic_stubbornness(df: pd.DataFrame, output_dir: Path):
    """Deliverable 6: Epistemic Stubbornness & Stance Stability"""
    console.print("[bold blue]Running Deliverable 6: Epistemic Stubbornness...[/]")

    agents_df = df[df['speaker_role'].isin(['proponent', 'opponent'])].copy()

    if agents_df.empty:
        return

    # Metric 6.1: Stance Flipping
    if 'stance' in agents_df.columns:
        stance_flips = []
        for (did, role), group in agents_df.groupby(['debate_id', 'speaker_role']):
            group = group.sort_values('round')
            if len(group) == 0:
                continue
            first_stance = group.iloc[0]['stance']
            
            # Changed if stance is different and not NaN
            is_flipped = False
            for s in group['stance']:
                if pd.notna(s) and str(s).lower() != str(first_stance).lower():
                    is_flipped = True
                    break
                    
            stance_flips.append({
                'speaker_role': role.title(),
                'Maintained Stance': not is_flipped,
                'Changed Stance': is_flipped
            })

        stance_df = pd.DataFrame(stance_flips)
        if not stance_df.empty:
            summary = stance_df.groupby('speaker_role')[['Maintained Stance', 'Changed Stance']].mean() * 100

            ax = summary.plot(kind='bar', stacked=True, figsize=(8, 6), color=['#2ca02c', '#d9534f'])
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%', label_type='center', fontsize=10, color='white')
            plt.title('Stance Stability: Do Agents Ever Yield?')
            plt.xlabel('Speaker Role')
            plt.ylabel('Percentage of Debates')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=0, ha='center')
            plt.tight_layout()
            plt.savefig(output_dir / '6_1_stance_stability.svg')
            plt.close()

    # Metric 6.2: The Stubbornness Index (Confidence Resilience)
    if 'confidence' in agents_df.columns:
        agents_df['confidence'] = pd.to_numeric(agents_df['confidence'], errors='coerce')

        deltas = []
        for (did, role), group in agents_df.groupby(['debate_id', 'speaker_role']):
            group = group.dropna(subset=['confidence']).sort_values('round')
            if len(group) > 1:
                early_conf = group.iloc[0]['confidence']
                late_conf = group.iloc[-1]['confidence']
                delta = late_conf - early_conf
                deltas.append({
                    'speaker_role': role.title(),
                    'Confidence Delta': delta
                })

        deltas_df = pd.DataFrame(deltas)
        if not deltas_df.empty:
            plt.figure(figsize=(8, 6))
            sns.violinplot(data=deltas_df, x='speaker_role', y='Confidence Delta', density_norm='width')
            plt.axhline(0, color='red', linestyle='--', label='No Change (Baseline)')
            plt.title('The Stubbornness Index (Late vs Early Confidence)')
            plt.xlabel('Speaker Role')
            plt.ylabel('Confidence Delta (Late - Early)')
            legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            interp_text = (
                "Distribution Guide:\n\n"
                "• Width: Data point frequency/density.\n"
                "• Above 0: Net increase over time.\n"
                "• Below 0: Net decrease over time.\n"
                "• White dot: Median metric value.\n"
                "• Thick bar: Interquartile range (IQR)."
            )
            text_box = plt.text(1.05, 0.5, interp_text, transform=plt.gca().transAxes, 
                                fontsize=9, va='center', ha='left', linespacing=1.6,
                                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='gray'))
            
            plt.savefig(output_dir / '6_2_stubbornness_index.svg', bbox_extra_artists=(legend, text_box,), bbox_inches='tight')
            plt.close()

def generate_role_summary(df: pd.DataFrame, output_dir: Path):
    """The Final Output: role_summary.csv"""
    console.print("[bold blue]Running Deliverable 4: Final Summary...[/]")
    
    # Select numeric features for aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['round']]
    
    # Calculate means
    summary_df = df.groupby('speaker_role')[numeric_cols].mean().reset_index()
    
    out_file = output_dir / '4_1_role_summary.csv'
    summary_df.to_csv(out_file, index=False)
    console.print(f"[green]Successfully saved final summary to {out_file}[/]")


def main():
    parser = argparse.ArgumentParser(description="SixSeven Role Analysis Tool")
    parser.add_argument("--artifacts", type=str, default="old_artifacts", help="Directory containing the old artifacts")
    parser.add_argument("--skip-emotion", action="store_true", help="Skip BERT emotion extraction (saves time)")
    args = parser.parse_args()

    # Read from old_artifacts but write locally to this script's directory
    script_dir = Path(__file__).parent.resolve()
    artifacts_dir = Path(args.artifacts)
    input_csv = artifacts_dir / "all_debates_ollama.csv"
    
    # Create an output directory strictly inside role-analysis-man
    local_output_root = script_dir / "generated_outputs"
    local_output_root.mkdir(parents=True, exist_ok=True)

    enriched_csv = local_output_root / "debate_log_with_modality_emotion.csv"
    output_dir = local_output_root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        console.print(f"[red]Could not find {input_csv}. Ensure you're running from the project root.[/]")
        return

    # Load & Enrich
    df = load_and_enrich_data(input_csv, enriched_csv, skip_emotion=args.skip_emotion)

    # Handle analysis modules
    analyze_moderator_dynamics(df, output_dir)
    analyze_persona_profiling(df, output_dir)
    analyze_tactic_asymmetry(df, output_dir)
    analyze_interrogative_doubt(df, output_dir)
    analyze_epistemic_stubbornness(df, output_dir)
    generate_role_summary(df, output_dir)
    
    console.print(f"\n[bold green]Role analysis complete! Check the {output_dir} folder for results.[/]")


if __name__ == "__main__":
    main()
