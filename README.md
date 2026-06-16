# ai-bubble

Students: Lucas Freire Costa and André Santos

## Dataset

The dataset is available on Google Drive:

[Link](https://drive.google.com/drive/folders/1h_EEmXZFAMAv0F9gzUbpNnR11BBmtzE-?usp=share_link)

## youtube_api.ipynb

1. **Fetches** YouTube comments (Science & Technology videos, keyword `AI`) month by month via the YouTube Data API v3, with automatic checkpoint/resume.
2. **Cleans** raw comments — removes duplicates, strips links and mentions, filters to English only.
3. **Scores emotions** with `j-hartmann/emotion-english-distilroberta-base` (anger, disgust, fear, joy, sadness, surprise).
4. **Filters** by confidence threshold and target keywords (AI, ChatGPT, Gemini, …).
5. **Runs FinBERT** (`ProsusAI/finbert`) to label financial sentiment (positive / negative / neutral).
6. **More filters**  keeping only non-neutral, high-confidence FinBERT rows for downstream analysis.

Output files are saved incrementally to `youtube_data/`.

Create a file .env with your API key in order to execute the code.