from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nltk
import numpy as np
import pandas as pd
import spacy
from emfdscore.scoring import score_docs as emfd_score_docs
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
from transformers import pipeline

from debate.analysis.features import (
    emotion_lexicon_scores,
)
from debate.analysis.lexicons import load_nrc_emotion_lexicon

from .config import (
    ASSERTIVE_WORDS,
    DEFAULT_NRC_EMOTIONS,
    EVIDENCE_WORDS,
    HEDGE_WORDS,
    REBUTTAL_WORDS,
    STANCE_CON_WORDS,
    STANCE_PRO_WORDS,
    STRONG_MODAL_WORDS,
    WEAK_MODAL_WORDS,
)
from .debate_metrics import lexical_count, stance_proxy_score, compute_role_alignment
from .stance_inference import load_stance_model


CITATION_PATTERN = re.compile(r"\[(\d+|[A-Za-z]+\d*)\]|\(([^)]*\d{4}[^)]*)\)")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?\b")


@dataclass
class NLPResources:
    nlp: Any
    stop_words: set[str]
    vader: SentimentIntensityAnalyzer
    embedding_model: SentenceTransformer
    emotion_classifier: Any | None
    emotion_load_error: str | None
    nrc_emotion_lexicon: dict[str, set[str]]
    stance_model: Any | None
    stance_model_error: str | None


def ensure_nltk_resource(resource_path: str, download_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        ok = nltk.download(download_name, quiet=True)
        if not ok:
            raise RuntimeError(f"Failed to download NLTK resource: {download_name}")


def ensure_spacy_model(model_name: str) -> None:
    try:
        spacy.load(model_name)
    except OSError:
        from spacy.cli import download as spacy_download

        spacy_download(model_name)


def load_nlp_resources(
    nrc_path: Path,
    use_transformer_emotion_secondary: bool,
    transformer_emotion_model: str,
) -> NLPResources:
    ensure_nltk_resource("sentiment/vader_lexicon.zip", "vader_lexicon")
    ensure_nltk_resource("corpora/stopwords", "stopwords")
    ensure_spacy_model("en_core_web_sm")

    nlp = spacy.load("en_core_web_sm")
    stop_words = set(SPACY_STOP_WORDS).union(set(stopwords.words("english")))
    vader = SentimentIntensityAnalyzer()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    emotion_classifier = None
    emotion_load_error = None
    if use_transformer_emotion_secondary:
        try:
            emotion_classifier = pipeline(
                "text-classification",
                model=transformer_emotion_model,
                top_k=None,
            )
        except Exception as exc:
            emotion_load_error = str(exc)

    nrc_lexicon = load_nrc_emotion_lexicon(
        nrc_path,
        emotions=DEFAULT_NRC_EMOTIONS,
        nlp=nlp,
        include_lemma_variants=True,
    )
    
    stance_model = None
    stance_model_error = None
    try:
        print("[NLP] Loading stance model")
        stance_model = load_stance_model()
        print("[NLP] Stance model loaded")
    except Exception as exc:
        stance_model_error = str(exc)
        print(f"[NLP] Stance model unavailable: {exc}")
    
    return NLPResources(
        nlp=nlp,
        stop_words=stop_words,
        vader=vader,
        embedding_model=embedding_model,
        emotion_classifier=emotion_classifier,
        emotion_load_error=emotion_load_error,
        nrc_emotion_lexicon=nrc_lexicon,
        stance_model=stance_model,
        stance_model_error=stance_model_error,
    )


def get_tokens_lemmas_pos(nlp: Any, text: str) -> tuple[list[str], list[str], list[str], int]:
    text = str(text)
    try:
        doc = nlp(text)
    except Exception:
        doc = None

    if doc is not None:
        tokens: list[str] = []
        lemmas: list[str] = []
        pos_tags: list[str] = []
        try:
            sent_count = max(1, sum(1 for _ in doc.sents))
        except Exception:
            sent_count = max(1, text.count(".") + text.count("!") + text.count("?"))
        for tok in doc:
            if tok.is_space:
                continue
            if tok.is_alpha:
                token = tok.text.lower()
                lemma = tok.lemma_.lower() if tok.lemma_ else token
                tokens.append(token)
                lemmas.append(lemma)
                pos_tags.append(tok.pos_ if tok.pos_ else "X")
        return tokens, lemmas, pos_tags, sent_count

    rough_tokens = re.findall(r"[A-Za-z']+", text.lower())
    return rough_tokens, rough_tokens, ["X"] * len(rough_tokens), max(1, text.count(".") + text.count("!") + text.count("?"))


def pos_distribution(pos_tags: list[str]) -> dict[str, float]:
    if not pos_tags:
        return {"noun_rate": 0.0, "verb_rate": 0.0, "adj_rate": 0.0, "adv_rate": 0.0, "pron_rate": 0.0}
    counts = pd.Series(pos_tags).value_counts()
    total = max(1, len(pos_tags))
    return {
        "noun_rate": (counts.get("NOUN", 0) + counts.get("PROPN", 0)) / total,
        "verb_rate": (counts.get("VERB", 0) + counts.get("AUX", 0)) / total,
        "adj_rate": counts.get("ADJ", 0) / total,
        "adv_rate": counts.get("ADV", 0) / total,
        "pron_rate": counts.get("PRON", 0) / total,
    }


def vader_sentiment(vader: SentimentIntensityAnalyzer, text: str) -> tuple[float, float, float, float]:
    scores = vader.polarity_scores(str(text))
    return float(scores["compound"]), float(scores["pos"]), float(scores["neu"]), float(scores["neg"])


def embedding_vector(embedding_model: SentenceTransformer, text: str) -> np.ndarray | None:
    try:
        vec = embedding_model.encode(str(text), show_progress_bar=False)
        return np.array(vec, dtype=float)
    except Exception:
        return None


def moral_frame_counts(text: str) -> dict[str, float]:
    in_df = pd.DataFrame([str(text)])
    out_df = emfd_score_docs(in_df, "emfd", "single", "bow", "vice-virtue", 1)
    if out_df is None or out_df.empty:
        raise RuntimeError("emfdscore returned empty output while scoring text.")

    row = out_df.iloc[0]

    def vv_sum(base: str) -> float:
        return float(row.get(f"{base}.virtue", 0.0)) + float(row.get(f"{base}.vice", 0.0))

    counts = {
        "moral_care_harm_count": vv_sum("care"),
        "moral_fairness_cheating_count": vv_sum("fairness"),
        "moral_loyalty_betrayal_count": vv_sum("loyalty"),
        "moral_authority_subversion_count": vv_sum("authority"),
        "moral_sanctity_degradation_count": vv_sum("sanctity"),
        "moral_liberty_oppression_count": 0.0,
    }
    counts["moral_total_count"] = float(sum(counts.values()))
    return counts


def dominant_moral_frame(row: dict[str, Any]) -> str:
    candidates = {
        k.replace("moral_", "").replace("_count", ""): v
        for k, v in row.items()
        if k.startswith("moral_") and k.endswith("_count") and k != "moral_total_count"
    }
    if not candidates:
        return "none"
    best_key, best_val = max(candidates.items(), key=lambda kv: kv[1])
    return best_key if best_val > 0 else "none"


def _compute_text_stats(text: str, tokens: list[str], clean_lemmas: list[str], sentence_count: int) -> dict[str, float | int]:
    word_count = len(tokens)
    unique_word_count = len(set(tokens))
    lemma_count = len(clean_lemmas)
    unique_lemma_count = len(set(clean_lemmas))
    char_count = len(text)
    avg_sentence_length = 0.0 if sentence_count == 0 else word_count / sentence_count
    avg_word_length = 0.0 if word_count == 0 else sum(len(t) for t in tokens) / word_count
    type_token_ratio = 0.0 if word_count == 0 else unique_word_count / word_count
    lemma_ttr = 0.0 if lemma_count == 0 else unique_lemma_count / lemma_count
    repetition_rate = 1.0 - type_token_ratio
    return {
        "word_count": word_count,
        "unique_word_count": unique_word_count,
        "lemma_count": lemma_count,
        "unique_lemma_count": unique_lemma_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "type_token_ratio": type_token_ratio,
        "lemma_type_token_ratio": lemma_ttr,
        "repetition_rate": repetition_rate,
    }


def _compute_proxy_counts(text: str, clean_lemmas: list[str], word_count: int) -> dict[str, float | int]:
    evidence_count = lexical_count(clean_lemmas, EVIDENCE_WORDS)
    rebuttal_count = lexical_count(clean_lemmas, REBUTTAL_WORDS)
    hedge_count = lexical_count(clean_lemmas, HEDGE_WORDS)
    assertive_count = lexical_count(clean_lemmas, ASSERTIVE_WORDS)
    strong_modal_count = lexical_count(clean_lemmas, STRONG_MODAL_WORDS)
    weak_modal_count = lexical_count(clean_lemmas, WEAK_MODAL_WORDS)

    citation_count = len(CITATION_PATTERN.findall(text))
    url_count = len(URL_PATTERN.findall(text))
    numeric_ref_count = len(NUMBER_PATTERN.findall(text))
    question_count = text.count("?")
    exclamation_count = text.count("!")

    evidence_proxy_total = evidence_count + citation_count + url_count + numeric_ref_count
    denom = max(1, word_count)

    return {
        "evidence_marker_count": evidence_count,
        "citation_count": citation_count,
        "url_count": url_count,
        "numeric_ref_count": numeric_ref_count,
        "evidence_proxy_total": evidence_proxy_total,
        "rebuttal_marker_count": rebuttal_count,
        "hedge_count": hedge_count,
        "assertive_count": assertive_count,
        "strong_modal_count": strong_modal_count,
        "weak_modal_count": weak_modal_count,
        "question_count": question_count,
        "exclamation_count": exclamation_count,
        "assertiveness_balance": (assertive_count - hedge_count) / denom,
        "modality_balance": (strong_modal_count - weak_modal_count) / denom,
        "evidence_density_per_1k_words": 1000.0 * evidence_proxy_total / denom,
        "rebuttal_density_per_1k_words": 1000.0 * rebuttal_count / denom,
        "hedge_density_per_1k_words": 1000.0 * hedge_count / denom,
        "assertive_density_per_1k_words": 1000.0 * assertive_count / denom,
        "strong_modal_density_per_1k_words": 1000.0 * strong_modal_count / denom,
        "weak_modal_density_per_1k_words": 1000.0 * weak_modal_count / denom,
        "has_evidence": int(evidence_proxy_total > 0),
        "has_rebuttal": int(rebuttal_count > 0),
        "has_question": int(question_count > 0),
    }


def _compute_sentiment_features(resources: NLPResources, text: str) -> dict[str, float]:
    compound, pos_sent, neu_sent, neg_sent = vader_sentiment(resources.vader, text)
    return {
        "vader_compound": compound,
        "vader_positive": pos_sent,
        "vader_neutral": neu_sent,
        "vader_negative": neg_sent,
    }


def _compute_emotion_features(resources: NLPResources, text: str, clean_lemmas: list[str]) -> dict[str, Any]:
    emotion_scores = emotion_lexicon_scores(clean_lemmas, resources.nrc_emotion_lexicon)
    transformer_emotion_scores: dict[str, float] = {}
    dominant_emotion = "unknown"
    if emotion_scores:
        dominant_emotion = max(emotion_scores, key=emotion_scores.get).replace("emotion_", "")

    if resources.emotion_classifier is not None and text.strip():
        try:
            result = resources.emotion_classifier(text[:512])
            emotion_items = result[0] if result and isinstance(result[0], list) else result
            for item in emotion_items:
                label = str(item.get("label", "unknown")).lower().replace(" ", "_")
                transformer_emotion_scores[f"emotion_tr_{label}"] = float(item.get("score", 0.0))
            if dominant_emotion == "unknown" and transformer_emotion_scores:
                dominant_emotion = max(transformer_emotion_scores, key=transformer_emotion_scores.get).replace("emotion_tr_", "")
        except Exception:
            pass

    return {
        "dominant_emotion": dominant_emotion,
        **emotion_scores,
        **transformer_emotion_scores,
    }


def _compute_stance_features(
    resources: NLPResources,
    text: str,
    role: str,
    claim: str | None = None,
) -> dict[str, Any]:
    """
    Compute zero-shot NLI stance inference and role alignment variables.
    
    Returns dict with stance prediction and role alignment scores.
    If stance model not available or no claim provided, returns neutral defaults.
    """
    defaults = {
        "predicted_stance_label": "neutral",
        "predicted_support_score": 0.33,
        "predicted_oppose_score": 0.33,
        "predicted_neutral_score": 0.34,
        "predicted_stance_margin": 0.0,
        "role_alignment_label": "neutral",
        "role_alignment_score": 0.0,
    }
    
    if resources.stance_model is None:
        return defaults
    
    if not claim or not claim.strip():
        return defaults
    
    try:
        stance_result = resources.stance_model.predict_stance(str(text), str(claim))
        
        alignment_label, alignment_score = compute_role_alignment(
            predicted_stance_label=stance_result.predicted_stance_label,
            support_score=stance_result.predicted_support_score,
            oppose_score=stance_result.predicted_oppose_score,
            speaker_role=role,
        )
        
        result = {
            "predicted_stance_label": stance_result.predicted_stance_label,
            "predicted_support_score": stance_result.predicted_support_score,
            "predicted_oppose_score": stance_result.predicted_oppose_score,
            "predicted_neutral_score": stance_result.predicted_neutral_score,
            "predicted_stance_margin": stance_result.predicted_stance_margin,
            "role_alignment_label": alignment_label,
            "role_alignment_score": alignment_score,
        }
        return result
    except Exception as e:
        print(f"[STANCE] Prediction error: {e}")
        return defaults


def analyze_utterance(
    resources: NLPResources,
    text: str,
    role: str,
    claim: str | None = None,
) -> pd.Series:
    tokens, lemmas, pos_tags, sentence_count = get_tokens_lemmas_pos(resources.nlp, str(text))
    clean_lemmas = [lemma for lemma in lemmas if lemma.isalpha() and lemma not in resources.stop_words]

    text_stats = _compute_text_stats(str(text), tokens, clean_lemmas, sentence_count)
    sentiment = _compute_sentiment_features(resources, str(text))
    proxy_counts = _compute_proxy_counts(str(text), clean_lemmas, text_stats["word_count"])
    moral_counts = moral_frame_counts(str(text))
    emotion = _compute_emotion_features(resources, str(text), clean_lemmas)
    pos_stats = pos_distribution(pos_tags)

    keyword_stance_score = stance_proxy_score(clean_lemmas, role, STANCE_PRO_WORDS, STANCE_CON_WORDS)
    moral_density = 1000.0 * float(moral_counts["moral_total_count"]) / max(1, int(text_stats["word_count"]))

    # Zero-shot stance inference if model available and claim provided
    stance_vars = _compute_stance_features(resources, str(text), role, claim)

    return pd.Series(
        {
            **text_stats,
            **sentiment,
            **proxy_counts,
            "stance_proxy_score": keyword_stance_score,  # renamed from stance_proxy_score for clarity
            "moral_density_per_1k_words": moral_density,
            **moral_counts,
            "dominant_moral_frame": dominant_moral_frame(moral_counts),
            **emotion,
            **pos_stats,
            **stance_vars,
            "tokens_list": tokens,
            "lemmas_list": clean_lemmas,
            "pos_tags_list": pos_tags,
        }
    )
