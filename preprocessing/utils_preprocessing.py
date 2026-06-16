import os
import re
import time
import pandas as pd
from atproto import Client
from dateutil.relativedelta import relativedelta
from langdetect import LangDetectException, detect


def connect_bsky(handle, password):
    """Authenticate with Bluesky and return a logged-in client."""
    if not handle or not password:
        raise ValueError("Set BSKY_HANDLE and BSKY_PASSWORD in your .env file before fetching posts.")

    client = Client()
    client.login(handle, password)
    print("Connected to Bluesky successfully!")
    return client


def clean_text(text):
    """Normalize comment text by removing URLs, mentions, and extra whitespace."""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text


def detect_language(text):
    """Return the ISO language code detected for text, or None when detection fails."""
    try:
        return detect(str(text))
    except LangDetectException:
        return None


def truncate_tokens(text, max_tokens=500):
    """Truncate text to at most max_tokens whitespace-separated tokens."""
    tokens = str(text).split()
    return " ".join(tokens[:max_tokens])


def token_count(text):
    """Count whitespace-separated tokens in text."""
    return len(str(text).split())


def save_dataset(df, output_path):
    """Save a DataFrame to CSV, creating the output directory if needed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows to {output_path}")

def fetch_raw_bsky_posts(
    client,
    search_queries,
    start_date,
    end_date,
    posts_per_month=100,
    sleep_seconds=0.5,
):
    """
    Fetch raw Bluesky posts month by month.

    This step intentionally does not deduplicate, clean text, detect language,
    or truncate text. Keeping it raw makes the later cleaning decisions explicit
    and reproducible.
    """
    all_rows = []
    current_date = start_date

    print(f"Starting raw Bluesky fetch from {start_date.date()} to {end_date.date()}...")

    while current_date < end_date:
        next_month = current_date + relativedelta(months=1)
        month_label = current_date.strftime("%Y-%m")
        print(f"Processing Bluesky posts for {month_label}...")

        for query in search_queries:
            date_query = (
                f"{query} "
                f"since:{current_date.strftime('%Y-%m-%dT%H:%M:%SZ')} "
                f"until:{next_month.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            )

            try:
                response = client.app.bsky.feed.search_posts({
                    "q": date_query,
                    "limit": posts_per_month,
                    "sort": "top",
                })

                for post in response.posts:
                    text = getattr(post.record, "text", "")
                    likes = getattr(post, "like_count", 0) or 0
                    reposts = getattr(post, "repost_count", 0) or 0
                    replies = getattr(post, "reply_count", 0) or 0

                    all_rows.append({
                        "Month": month_label,
                        "Date": pd.to_datetime(post.record.created_at),
                        "Query": query,
                        "Post_ID": post.uri,
                        "Text": text,
                        "Likes": likes,
                        "Reposts": reposts,
                        "Replies": replies,
                        "Weight": 1 + likes + reposts + replies,
                    })
            except Exception as exc:
                print(f"  -> Error for {month_label}, query '{query}': {exc}")

        print(f"Finished {month_label}. Raw rows collected so far: {len(all_rows):,}")
        current_date = next_month
        time.sleep(sleep_seconds)

    raw_df = pd.DataFrame(all_rows)
    if not raw_df.empty:
        raw_df["Month"] = pd.to_datetime(raw_df["Month"])

    print(f"Done! Retrieved {len(raw_df):,} raw Bluesky rows.")
    return raw_df

def drop_empty_rows_and_duplicates(df, text_column="Text", duplicate_subset=None):
    """
    Drop rows with missing/empty text and remove duplicates.

    By default, duplicates are identified by Post_ID when available; otherwise,
    the text column is used.
    """
    if duplicate_subset is None:
        duplicate_subset = ["Post_ID", text_column] if "Post_ID" in df.columns else [text_column]

    cleaned = df.copy()
    start_rows = len(cleaned)

    cleaned[text_column] = cleaned[text_column].astype(str)
    cleaned = cleaned[cleaned[text_column].str.strip().str.len() > 0]
    after_empty_drop = len(cleaned)

    for col in duplicate_subset:
        if col not in cleaned.columns:
            raise ValueError(f"Duplicate subset column '{col}' not found in DataFrame.")
        cleaned = cleaned.drop_duplicates(subset=[col]).reset_index(drop=True)
    after_dedup = len(cleaned)

    print(f"DUPLICATE SUBSET: {duplicate_subset}")
    print(f"Dropped {start_rows - after_empty_drop:,} empty text rows.")
    print(f"Dropped {after_empty_drop - after_dedup:,} duplicate rows using {duplicate_subset}.")
    return cleaned


def normalize_comment_text(df, source_column="Text", output_column="Clean_Comment"):
    """Create a normalized text column and remove rows that become empty after cleaning."""
    cleaned = df.copy()
    cleaned[output_column] = cleaned[source_column].apply(clean_text)

    start_rows = len(cleaned)
    cleaned = cleaned[cleaned[output_column].str.len() > 0].reset_index(drop=True)
    print(f"Dropped {start_rows - len(cleaned):,} rows that were empty after text normalization.")
    return cleaned


def keep_english_comments(df, text_column="Clean_Comment", language_column="Language"):
    """Detect language and keep only English comments."""
    cleaned = df.copy()
    cleaned[language_column] = cleaned[text_column].apply(detect_language)

    start_rows = len(cleaned)
    cleaned = cleaned[cleaned[language_column] == "en"].reset_index(drop=True)
    print(f"Dropped {start_rows - len(cleaned):,} non-English or undetected rows.")
    return cleaned


def truncate_comment_tokens(df, text_column="Clean_Comment", max_tokens=500):
    """Truncate the clean text column and add token-count audit columns."""
    cleaned = df.copy()
    cleaned["Original_Token_Count"] = cleaned[text_column].apply(token_count)
    cleaned[text_column] = cleaned[text_column].apply(lambda text: truncate_tokens(text, max_tokens=max_tokens))
    cleaned["Final_Token_Count"] = cleaned[text_column].apply(token_count)
    return cleaned


