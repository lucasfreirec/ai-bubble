"""Utilities for FinBERT sentiment scoring on cleaned Bluesky posts."""

import os
from typing import Dict, Iterable, List

import pandas as pd


MODEL_NAME = "ProsusAI/finbert"


def load_finbert_classifier(model_name: str = MODEL_NAME, device: int = -1):
    """Load the FinBERT Hugging Face text-classification pipeline.

    Parameters
    ----------
    model_name:
        Hugging Face model checkpoint for financial sentiment classification.
    device:
        Device index passed to ``transformers.pipeline``. Use ``-1`` for CPU or
        ``0`` for the first GPU.
    """
    from transformers import pipeline

    print(f"Loading FinBERT classifier: {model_name}")
    classifier = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        device=device,
    )
    print("FinBERT classifier loaded.")
    return classifier


def _normalize_prediction(prediction) -> Dict[str, object]:
    """Return the top FinBERT label and score from a pipeline prediction."""
    if isinstance(prediction, list):
        if not prediction:
            return {"finbert_label": None, "finbert_score": None}
        prediction = max(prediction, key=lambda item: item.get("score", 0))

    return {
        "finbert_label": prediction.get("label"),
        "finbert_score": float(prediction.get("score", 0)),
    }


def classify_finbert_sentiment(
    texts: Iterable[str],
    classifier,
    batch_size: int = 32,
    progress_every: int = 1000,
) -> List[Dict[str, object]]:
    """Classify many texts and return one FinBERT result dictionary per text."""
    text_list = [str(text) for text in texts]
    results = []
    next_progress = progress_every

    for start in range(0, len(text_list), batch_size):
        batch = text_list[start:start + batch_size]
        predictions = classifier(batch, truncation=True, batch_size=batch_size)
        results.extend(_normalize_prediction(prediction) for prediction in predictions)

        processed = min(start + len(batch), len(text_list))
        if progress_every and processed >= next_progress:
            print(f"Scored FinBERT sentiment for {processed:,} posts...")
            next_progress += progress_every

    print(f"Finished FinBERT scoring for {len(results):,} posts.")
    return results


def add_finbert_columns(
    df: pd.DataFrame,
    classifier,
    text_column: str = "Clean_Comment",
    batch_size: int = 32,
    progress_every: int = 1000,
) -> pd.DataFrame:
    """Add FinBERT sentiment label and score columns to a DataFrame."""
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame.")

    classified = classify_finbert_sentiment(
        texts=df[text_column].fillna(""),
        classifier=classifier,
        batch_size=batch_size,
        progress_every=progress_every,
    )

    output = df.copy()
    result_df = pd.DataFrame(
        classified,
        columns=["finbert_label", "finbert_score"],
        index=output.index,
    )
    return pd.concat([output, result_df], axis=1)


def summarize_finbert(df: pd.DataFrame, label_column: str = "finbert_label") -> pd.DataFrame:
    """Summarize FinBERT sentiment labels."""
    if label_column not in df.columns:
        raise ValueError(f"FinBERT label column '{label_column}' not found in DataFrame.")

    return (
        df[label_column]
        .value_counts(dropna=False)
        .rename_axis("finbert_sentiment")
        .reset_index(name="rows")
    )


def save_finbert_results(df: pd.DataFrame, output_path: str) -> None:
    """Save FinBERT results to CSV, creating the output directory if needed."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows to {output_path}")
