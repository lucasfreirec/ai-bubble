"""Utilities for emotion detection on cleaned Bluesky posts.

This module uses the Hugging Face emotion classifier
``j-hartmann/emotion-english-distilroberta-base`` before the broader BART
zero-shot filtering pass. Low-confidence predictions are kept as raw audit
columns, while the accepted emotion columns are blanked below the confidence
threshold.
"""

from typing import Dict, Iterable, List

import pandas as pd


MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_MIN_CONFIDENCE = 0.70


def load_emotion_classifier(model_name: str = MODEL_NAME, device: int = -1):
    """Load a Hugging Face text-classification pipeline for emotion detection.

    Parameters
    ----------
    model_name:
        Hugging Face model checkpoint for emotion classification.
    device:
        Device index passed to ``transformers.pipeline``. Use ``-1`` for CPU or
        ``0`` for the first GPU.
    """
    from transformers import pipeline

    print(f"Loading emotion classifier: {model_name}")
    classifier = pipeline("text-classification", model=model_name, device=device)
    print("Emotion classifier loaded.")
    return classifier


def _normalize_prediction(prediction) -> Dict[str, object]:
    """Return the top label and score from a pipeline prediction."""
    if isinstance(prediction, list):
        if not prediction:
            return {"emotion_raw_label": None, "emotion_raw_score": None}
        prediction = max(prediction, key=lambda item: item.get("score", 0))

    return {
        "emotion_raw_label": prediction.get("label"),
        "emotion_raw_score": float(prediction.get("score", 0)),
    }


def classify_emotions(
    texts: Iterable[str],
    classifier,
    batch_size: int = 32,
    progress_every: int = 1000,
) -> List[Dict[str, object]]:
    """Classify many texts and return one emotion result dictionary per text."""
    text_list = [str(text) for text in texts]
    results = []
    next_progress = progress_every

    for start in range(0, len(text_list), batch_size):
        batch = text_list[start:start + batch_size]
        predictions = classifier(batch, truncation=True, batch_size=batch_size)
        results.extend(_normalize_prediction(prediction) for prediction in predictions)

        processed = min(start + len(batch), len(text_list))
        if progress_every and processed >= next_progress:
            print(f"Detected emotions for {processed:,} posts...")
            next_progress += progress_every

    print(f"Finished detecting emotions for {len(results):,} posts.")
    return results


def add_emotion_columns(
    df: pd.DataFrame,
    classifier,
    text_column: str = "Clean_Comment",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    batch_size: int = 32,
    progress_every: int = 1000,
) -> pd.DataFrame:
    """Add raw and confidence-filtered emotion columns to a DataFrame."""
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame.")

    classified = classify_emotions(
        texts=df[text_column].fillna(""),
        classifier=classifier,
        batch_size=batch_size,
        progress_every=progress_every,
    )

    output = df.copy()
    result_df = pd.DataFrame(
        classified,
        columns=["emotion_raw_label", "emotion_raw_score"],
        index=output.index,
    )
    output = pd.concat([output, result_df], axis=1)
    output["emotion_confident"] = output["emotion_raw_score"] >= min_confidence
    output["emotion_label"] = output["emotion_raw_label"].where(output["emotion_confident"])
    output["emotion_score"] = output["emotion_raw_score"].where(output["emotion_confident"])
    return output


def summarize_emotions(df: pd.DataFrame, emotion_column: str = "emotion_label") -> pd.DataFrame:
    """Summarize confident emotion labels."""
    if emotion_column not in df.columns:
        raise ValueError(f"Emotion column '{emotion_column}' not found in DataFrame.")

    summary = (
        df[emotion_column]
        .fillna("discarded_low_confidence")
        .value_counts(dropna=False)
        .rename_axis("emotion")
        .reset_index(name="rows")
    )
    return summary
