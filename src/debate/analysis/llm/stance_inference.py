"""
Zero-shot NLI-based stance inference for utterances against debate claims.

Classifies whether an utterance supports, opposes, or is neutral about the debate claim
using a pretrained entailment model. Produces scores and labels used for role alignment
analysis.
"""

from dataclasses import dataclass
from typing import Any

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
except ImportError:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    pipeline = None


@dataclass
class StanceResult:
    """Result from zero-shot stance inference."""

    predicted_stance_label: str  # "support", "oppose", or "neutral"
    predicted_support_score: float
    predicted_oppose_score: float
    predicted_neutral_score: float
    predicted_stance_margin: float  # max_score - second_max_score


class StanceInferenceModel:
    """Wrapper for zero-shot NLI stance prediction using pretrained model."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        """
        Load pretrained zero-shot NLI model.

        Args:
            model_name: HuggingFace model ID for NLI/entailment model (default: cross-encoder/nli-deberta-v3-small)
        """
        if pipeline is None:
            raise RuntimeError(
                "transformers library not installed. Install with: pip install transformers torch"
            )

        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the NLI model."""
        try:
            print(f"[STANCE] Loading model: {self.model_name}")
            self.model = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if self._gpu_available() else -1,
            )
            print("[STANCE] Model loaded")
        except Exception as e:
            print(f"[STANCE] Model load error: {e}")
            raise RuntimeError(f"Failed to load stance model {self.model_name}: {e}")

    @staticmethod
    def _gpu_available() -> bool:
        """Check if GPU is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def predict_stance(self, utterance: str, claim: str) -> StanceResult:
        """
        Classify stance of utterance toward claim using zero-shot NLI.

        Args:
            utterance: The debate utterance to classify
            claim: The debate claim/motion to evaluate against

        Returns:
            StanceResult with predicted_stance_label and scores
        """
        if not utterance or not utterance.strip():
            return StanceResult(
                predicted_stance_label="neutral",
                predicted_support_score=0.33,
                predicted_oppose_score=0.33,
                predicted_neutral_score=0.34,
                predicted_stance_margin=0.0,
            )

        if not claim or not claim.strip():
            return StanceResult(
                predicted_stance_label="neutral",
                predicted_support_score=0.33,
                predicted_oppose_score=0.33,
                predicted_neutral_score=0.34,
                predicted_stance_margin=0.0,
            )

        # Create hypotheses for zero-shot classification
        hypothesis_support = f"This utterance supports the claim that {claim}"
        hypothesis_oppose = f"This utterance opposes the claim that {claim}"
        hypothesis_neutral = f"This utterance is neutral or mixed about the claim that {claim}"

        hypotheses = [hypothesis_support, hypothesis_oppose, hypothesis_neutral]

        try:
            result = self.model(utterance, hypotheses, multi_class=True)
        except Exception:
            # Fallback to neutral on error
            return StanceResult(
                predicted_stance_label="neutral",
                predicted_support_score=0.33,
                predicted_oppose_score=0.33,
                predicted_neutral_score=0.34,
                predicted_stance_margin=0.0,
            )

        # Extract scores in order: support, oppose, neutral
        labels = result["labels"]
        scores = result["scores"]

        # Create score dict mapping hypothesis to score
        score_dict = dict(zip(labels, scores))

        # Map back to stance labels
        support_score = score_dict.get(hypothesis_support, 0.0)
        oppose_score = score_dict.get(hypothesis_oppose, 0.0)
        neutral_score = score_dict.get(hypothesis_neutral, 0.0)

        # Normalize if needed
        total = support_score + oppose_score + neutral_score
        if total > 0:
            support_score /= total
            oppose_score /= total
            neutral_score /= total

        # Determine predicted label
        scores_dict = {
            "support": support_score,
            "oppose": oppose_score,
            "neutral": neutral_score,
        }
        predicted_label = max(scores_dict, key=scores_dict.get)

        # Compute stance margin (confidence)
        sorted_scores = sorted([support_score, oppose_score, neutral_score], reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        return StanceResult(
            predicted_stance_label=predicted_label,
            predicted_support_score=float(support_score),
            predicted_oppose_score=float(oppose_score),
            predicted_neutral_score=float(neutral_score),
            predicted_stance_margin=float(margin),
        )


# Lazy-load singleton instance to avoid multiple model loads
_stance_model_instance: StanceInferenceModel | None = None


def load_stance_model(
    model_name: str = "cross-encoder/nli-deberta-v3-small",
) -> StanceInferenceModel:
    """
    Load or get cached stance model instance.

    Args:
        model_name: HuggingFace model ID (default is cross-encoder/nli-deberta-v3-small, fine-tuned for NLI)

    Returns:
        StanceInferenceModel instance
    """
    global _stance_model_instance

    if _stance_model_instance is None:
        _stance_model_instance = StanceInferenceModel(model_name=model_name)

    return _stance_model_instance


def predict_stance_with_pretrained_model(
    utterance: str, claim: str, model_name: str = "cross-encoder/nli-deberta-v3-small"
) -> dict[str, Any]:
    """
    Convenience function: predict stance without explicit model management.

    Args:
        utterance: Debate utterance to classify
        claim: Debate claim/motion to evaluate against
        model_name: HuggingFace model ID

    Returns:
        Dictionary with stance prediction and scores
    """
    model = load_stance_model(model_name=model_name)
    result = model.predict_stance(utterance, claim)

    return {
        "predicted_stance_label": result.predicted_stance_label,
        "predicted_support_score": result.predicted_support_score,
        "predicted_oppose_score": result.predicted_oppose_score,
        "predicted_neutral_score": result.predicted_neutral_score,
        "predicted_stance_margin": result.predicted_stance_margin,
    }
