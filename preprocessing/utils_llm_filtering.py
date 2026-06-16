"""Utilities for zero-shot LLM filtering of Bluesky AI comments.

The functions in this module use an NLI zero-shot classifier such as
``facebook/bart-large-mnli`` to assign each comment to a descriptive label.
The default labels are intentionally broad and inclusive: the first pass should
identify comments that are probably related to AI while leaving borderline rows
available for manual review.
"""

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


MODEL_NAME = "facebook/bart-large-mnli"


LABEL_MAP = {
    # Financial speculation and markets
    "stock market investing and trading related to AI companies": "Financial Speculation",
    "economic technology bubble and AI overvaluation": "Financial Speculation",
    "venture capital funding startups and business strategy in artificial intelligence": "Financial Speculation",
    "earnings revenue and corporate performance of AI technology companies": "Financial Speculation",

    # Practical software development and productivity
    "software engineering coding architecture and programming with AI": "Software Dev",
    "using AI tools for daily productivity and automation": "Software Dev",
    "prompt engineering workflows and practical use of chatbots": "Software Dev",
    "developer tools APIs and machine learning infrastructure": "Software Dev",

    # Societal impact, labor, and fear
    "AI taking human jobs automation and unemployment": "Societal Impact & Fear",
    "the future societal impact of artificial intelligence": "Societal Impact & Fear",
    "AI safety risks regulation ethics and governance": "Societal Impact & Fear",
    "misinformation surveillance privacy and harmful uses of AI": "Societal Impact & Fear",

    # Creative industries and generated media
    "AI generated art images music video and creative media": "Creative AI",
    "artists writers actors and creators reacting to generative AI": "Creative AI",
    "copyright training data plagiarism and consent in AI art": "Creative AI",

    # AI models, products, and companies
    "ChatGPT Gemini Claude Grok Copilot and other AI chatbot products": "AI Products & Models",
    "large language models generative AI and foundation models": "AI Products & Models",
    "Google OpenAI Microsoft Meta Anthropic Nvidia and AI companies": "AI Products & Models",
    "product launches announcements benchmarks and model capabilities in AI": "AI Products & Models",

    # Research, education, and technical AI
    "machine learning deep learning data science and artificial intelligence research": "AI Research & Education",
    "AI in education teaching learning and academic work": "AI Research & Education",
    "AI in healthcare science robotics and technical applications": "AI Research & Education",

    # General AI discussion
    "general discussion opinions news or jokes about artificial intelligence": "General AI Discussion",
    "personal reactions experiences and debates about AI": "General AI Discussion",

    # Noise and off-topic content
    "feedback and appreciation for a youtube video": "Noise / Video Feedback",
    "the content creator video editing channel updates and social media promotion": "Noise / Video Feedback",
    "horoscope astrology zodiac signs and the Gemini star sign": "Noise / Astrology",
    "unrelated word games memes spam giveaways or generic link promotion": "Noise / Other",
    "comments where AI means something other than artificial intelligence": "Noise / Other",
}


AI_RELATED_GROUPS = {
    "Financial Speculation",
    "Software Dev",
    "Societal Impact & Fear",
    "Creative AI",
    "AI Products & Models",
    "AI Research & Education",
    "General AI Discussion",
}


NOISE_GROUPS = {
    "Noise / Video Feedback",
    "Noise / Astrology",
    "Noise / Other",
}


def get_candidate_labels(label_map: Optional[Dict[str, str]] = None) -> List[str]:
    """Return candidate zero-shot labels in the order used by the classifier."""
    labels = label_map or LABEL_MAP
    return list(labels.keys())


def load_zero_shot_classifier(model_name: str = MODEL_NAME, device: int = -1):
    """Load a Hugging Face zero-shot classification pipeline.

    Parameters
    ----------
    model_name:
        Hugging Face model checkpoint. ``facebook/bart-large-mnli`` is the
        default used in this project.
    device:
        Device index passed to ``transformers.pipeline``. Use ``-1`` for CPU or
        ``0`` for the first GPU.
    """
    from transformers import pipeline

    print(f"Loading zero-shot classifier: {model_name}")
    classifier = pipeline("zero-shot-classification", model=model_name, device=device)
    print("Zero-shot classifier loaded.")
    return classifier


def classify_comment(
    text: str,
    classifier,
    candidate_labels: Sequence[str],
    hypothesis_template: str = "This Bluesky post is about {}.",
    multi_label: bool = False,
) -> Dict[str, object]:
    """Classify one comment and return the best label, score, and runner-up.

    ``multi_label=False`` makes labels compete with one another, which is useful
    for filtering. The score margin between the best and second-best labels is
    kept so borderline comments can be reviewed later.
    """
    result = classifier(
        str(text),
        candidate_labels=list(candidate_labels),
        hypothesis_template=hypothesis_template,
        multi_label=multi_label,
    )

    labels = result["labels"]
    scores = result["scores"]
    second_label = labels[1] if len(labels) > 1 else None
    second_score = float(scores[1]) if len(scores) > 1 else None

    return {
        "llm_label": labels[0],
        "llm_score": float(scores[0]),
        "llm_second_label": second_label,
        "llm_second_score": second_score,
        "llm_score_margin": float(scores[0] - scores[1]) if len(scores) > 1 else None,
    }


def classify_comments(
    texts: Iterable[str],
    classifier,
    candidate_labels: Sequence[str],
    hypothesis_template: str = "This Bluesky post is about {}.",
    multi_label: bool = False,
    progress_every: int = 1000,
) -> List[Dict[str, object]]:
    """Classify many comments and return one result dictionary per comment."""
    results = []

    for index, text in enumerate(texts, start=1):
        results.append(
            classify_comment(
                text=text,
                classifier=classifier,
                candidate_labels=candidate_labels,
                hypothesis_template=hypothesis_template,
                multi_label=multi_label,
            )
        )

        if progress_every and index % progress_every == 0:
            print(f"Classified {index:,} comments...")

    print(f"Finished classifying {len(results):,} comments.")
    return results


def add_llm_filter_columns(
    df: pd.DataFrame,
    classifier,
    text_column: str = "Clean_Comment",
    label_map: Optional[Dict[str, str]] = None,
    hypothesis_template: str = "This Bluesky post is about {}.",
    multi_label: bool = False,
    progress_every: int = 1000,
) -> pd.DataFrame:
    """Add zero-shot label, category, score, and review columns to a DataFrame."""
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame.")

    labels = label_map or LABEL_MAP
    candidate_labels = get_candidate_labels(labels)
    classified = classify_comments(
        texts=df[text_column].fillna(""),
        classifier=classifier,
        candidate_labels=candidate_labels,
        hypothesis_template=hypothesis_template,
        multi_label=multi_label,
        progress_every=progress_every,
    )

    output = df.copy()
    result_df = pd.DataFrame(classified, index=output.index)
    output = pd.concat([output, result_df], axis=1)
    output["llm_category"] = output["llm_label"].map(labels)
    output["llm_second_category"] = output["llm_second_label"].map(labels)
    output["llm_is_ai_related"] = output["llm_category"].isin(AI_RELATED_GROUPS)
    output["llm_is_noise"] = output["llm_category"].isin(NOISE_GROUPS)
    return output


def filter_ai_related_comments(
    df: pd.DataFrame,
    min_score: float = 0.7,
    min_margin: float = 0.05,
    keep_uncertain_ai: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split classified comments into AI-related rows and review/rejected rows.

    Parameters
    ----------
    min_score:
        Minimum score required for a confident AI-related classification.
    min_margin:
        Minimum gap between the top label and runner-up label.
    keep_uncertain_ai:
        When ``True``, comments whose top category is AI-related are retained
        even if they are below the score or margin threshold. This is safer for
        an exploratory dataset because it avoids cutting potentially relevant
        comments too aggressively.
    """
    required = {"llm_category", "llm_is_ai_related", "llm_score", "llm_score_margin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing LLM classification columns: {sorted(missing)}")

    confident_ai = (
        df["llm_is_ai_related"]
        & (df["llm_score"] >= min_score)
        & (df["llm_score_margin"].fillna(0) >= min_margin)
    )

    if keep_uncertain_ai:
        keep_mask = df["llm_is_ai_related"]
    else:
        keep_mask = confident_ai

    kept = df[keep_mask].copy().reset_index(drop=True)
    review = df[~keep_mask].copy().reset_index(drop=True)
    review["llm_review_reason"] = "top label is noise or off-topic"

    borderline = df[df["llm_is_ai_related"] & ~confident_ai].copy()
    if not borderline.empty:
        borderline["llm_review_reason"] = "AI-related label but low confidence or low margin"
        review = pd.concat([review, borderline], ignore_index=True)

    print(f"Kept {len(kept):,} AI-related comments.")
    print(f"Flagged {len(review):,} comments for rejection or manual review.")
    return kept, review


def summarize_llm_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize classified rows by LLM category and label."""
    required = {"llm_category", "llm_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing LLM classification columns: {sorted(missing)}")

    summary = (
        df.groupby(["llm_category", "llm_label"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["llm_category", "rows"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return summary


def print_label_map(label_map: Optional[Dict[str, str]] = None) -> None:
    """Print candidate labels grouped by their broader filtering category."""
    labels = label_map or LABEL_MAP
    grouped = {}
    for label, category in labels.items():
        grouped.setdefault(category, []).append(label)

    for category, category_labels in grouped.items():
        print(f"\n{category}")
        for label in category_labels:
            print(f"  - {label}")


def count_categories(labels: Iterable[str], label_map: Optional[Dict[str, str]] = None) -> Counter:
    """Count broad categories for a sequence of candidate-label strings."""
    mapping = label_map or LABEL_MAP
    return Counter(mapping.get(label, "Unknown") for label in labels)
