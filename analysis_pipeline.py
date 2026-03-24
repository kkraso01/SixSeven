import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from transformers import pipeline

# -----------------------------
# 0. Output folder
# -----------------------------
OUTPUT_DIR = "sentiment_results_detailed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_csv("debate_log.csv")
df = df[df["speaker_role"].isin(["proponent", "opponent"])].copy()
df["turn_index"] = range(1, len(df) + 1)

# -----------------------------
# 2. Lexicons
# -----------------------------
UNCERTAINTY_WORDS = {
    "may", "might", "could", "perhaps", "possibly", "suggests",
    "likely", "potential", "concern", "concerns", "reasonable"
}

STRONG_MODALITY = {
    "must", "clearly", "definitely", "proves", "demonstrates",
    "cannot", "never", "always", "undeniable", "certainly"
}

WEAK_MODALITY = {
    "may", "might", "could", "suggest", "suggests", "appears",
    "seems", "possibly", "potential"
}

def tokenize(text):
    return re.findall(r"[a-zA-Z']+", str(text).lower())

# -----------------------------
# 3. Emotion model
# -----------------------------
# You can replace this model name with another emotion classifier if needed.
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# -----------------------------
# 4. Per-utterance analysis
# -----------------------------
def analyze_utterance(text):
    text = str(text)
    tokens = tokenize(text)
    blob = TextBlob(text)

    scores = {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
        "uncertainty_score": sum(t in UNCERTAINTY_WORDS for t in tokens),
        "strong_modality_score": sum(t in STRONG_MODALITY for t in tokens),
        "weak_modality_score": sum(t in WEAK_MODALITY for t in tokens),
        "question_count": text.count("?"),
        "exclamation_count": text.count("!"),
        "word_count": len(tokens),
    }

    emotion_results = emotion_classifier(text)[0]
    for item in emotion_results:
        label = item["label"].lower().replace(" ", "_")
        scores[f"emotion_{label}"] = item["score"]

    return pd.Series(scores)

analysis = df["utterance"].apply(analyze_utterance)
df = pd.concat([df, analysis], axis=1)

# -----------------------------
# 5. Fill missing emotion columns
# -----------------------------
emotion_cols = [c for c in df.columns if c.startswith("emotion_")]
df[emotion_cols] = df[emotion_cols].fillna(0.0)

# -----------------------------
# 6. Summaries
# -----------------------------
speaker_summary = df.groupby("speaker_role").mean(numeric_only=True)
round_summary = df.groupby(["round", "speaker_role"]).mean(numeric_only=True).reset_index()

# Save text summary
with open(os.path.join(OUTPUT_DIR, "analysis_output.txt"), "w", encoding="utf-8") as f:
    f.write("=== Speaker Summary ===\n")
    f.write(speaker_summary.to_string())
    f.write("\n\n=== Round Summary ===\n")
    f.write(round_summary.to_string(index=False))

# Save CSVs
df.to_csv(os.path.join(OUTPUT_DIR, "debate_log_with_detailed_emotions.csv"), index=False)
speaker_summary.to_csv(os.path.join(OUTPUT_DIR, "speaker_summary.csv"))
round_summary.to_csv(os.path.join(OUTPUT_DIR, "round_summary.csv"), index=False)

# -----------------------------
# 7. Plot key emotions over time
# -----------------------------
key_emotions = [c for c in emotion_cols if any(
    x in c for x in ["anger", "fear", "joy", "sadness", "disgust", "surprise", "neutral"]
)]

for speaker in ["proponent", "opponent"]:
    sub = df[df["speaker_role"] == speaker]

    for col in key_emotions:
        plt.figure(figsize=(8, 4))
        plt.plot(sub["turn_index"], sub[col], marker="o")
        plt.title(f"{speaker.capitalize()} - {col} over time")
        plt.xlabel("Turn Index")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{speaker}_{col}.png"), dpi=300, bbox_inches="tight")
        plt.close()

# -----------------------------
# 8. Plot confidence vs anger/fear/neutral if present
# -----------------------------
for speaker in ["proponent", "opponent"]:
    sub = df[df["speaker_role"] == speaker]

    plt.figure(figsize=(9, 5))
    plt.plot(sub["turn_index"], sub["confidence"], marker="o", label="confidence")

    for c in ["emotion_anger", "emotion_fear", "emotion_neutral"]:
        if c in sub.columns:
            plt.plot(sub["turn_index"], sub[c], marker="s", label=c)

    plt.title(f"{speaker.capitalize()} - Confidence vs Emotions")
    plt.xlabel("Turn Index")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{speaker}_confidence_vs_emotions.png"), dpi=300, bbox_inches="tight")
    plt.close()