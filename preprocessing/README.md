# Ai-bubble

This project studies public discussion around artificial intelligence and the idea of an "AI bubble" by combining social-media comments, emotion classification, FinBERT sentiment analysis, AI-topic filtering, and financial market indicators.

The analysis compares two datasets:

- Bluesky posts collected around AI-related queries.
- YouTube comments from AI-related videos.

The final outputs are cleaned CSV files and plots that summarize emotion distributions, FinBERT sentiment, AI-related topic categories, and the relationship between fear-related discussion and AI-linked asset prices.

## Main Observations

- We observe a spike on comments approximately in april 2023, which is close to GPT - 4 [lanching date (14/03/2023)](https://ipspecialist.net/the-history-of-chatgpt-from-gpt-1-to-gpt-4/). Also, it was the first [public anounce of Gemini](https://mobisoftinfotech.com/resources/blog/google-io-2023).

- In september 2024, we obverse a decay, that happened simultaneously with a stock price decay. Also, there was a lot of speculation over the cut of interest rates in the USA, which leads investors to migrate from large cap companies to the small cap ones. [Source](https://fortune.com/2024/08/02/ai-bubble-tech-stocks-nvidia-amazon-meta-microsoft-amd-intel/)

- Most posts from Bluesky are classified as negative by FinBERT, but the correlation between the fear ratio and AI-related asset prices is still positive. This is interesting because it suggests that negative or fearful public discussion does not necessarily move opposite to market attention. In this dataset, fear can rise together with financial interest in AI, which may point to hype, uncertainty, and speculation growing at the same time.

## Repository Organization

```text
.
├── preprocessing/
│   ├── get_comments_bsky.ipynb
│   ├── bart_filter.ipynb
│   ├── emotion_detection.ipynb
│   ├── finbert.ipynb
│   ├── utils_preprocessing.py
│   ├── utils_llm_filtering.py
│   ├── utils_emotions.py
│   └── utils_finbert.py
├── analysis/
│   └── analyze_plot.ipynb
├── data/
│   ├── raw_posts_bsky.csv
│   ├── clean_posts_bsky.csv
│   ├── bsky_final.csv
│   ├── youtube_final.csv
│   └── final_plots/
├── requirements.txt
├── .gitignore
└── README.md
```

## Code Workflow

The project is organized as a pipeline.

1. Data collection and cleaning happens in `preprocessing/`.
2. Emotion detection labels comments with emotions such as fear, joy, anger, disgust, sadness, and surprise.
3. LLM/zero-shot filtering identifies whether comments are AI-related and assigns broad AI topic groups.
4. FinBERT sentiment analysis labels comments as positive, negative, or neutral.
5. Final merged datasets are saved in `data/`.
6. Analysis notebooks generate the final plots in `data/final_plots/`.

## Preprocessing Folder

`preprocessing/` contains the notebooks and helper modules used to build the final datasets.

- `get_comments_bsky.ipynb`: collects or prepares Bluesky posts.
- `bart_filter.ipynb`: applies zero-shot classification to filter AI-related content.
- `emotion_detection.ipynb`: assigns emotion labels to posts/comments.
- `finbert.ipynb`: applies FinBERT sentiment classification.
- `utils_preprocessing.py`: shared cleaning and preprocessing utilities.
- `utils_llm_filtering.py`: zero-shot label definitions and filtering helpers.
- `utils_emotions.py`: emotion-classification helpers.
- `utils_finbert.py`: FinBERT helper functions.

The AI-related topic labels are defined in `preprocessing/utils_llm_filtering.py`. 

## Analysis Folder

`analysis/` contains the notebooks used to inspect the processed datasets and generate figures.

The main plotting workflow creates combined Bluesky/YouTube figures where each dataset is shown as a subplot. These plots compare:

- emotion distributions
- emotion evolution through time
- FinBERT sentiment distributions
- FinBERT sentiment by emotion
- AI-related topic categories for top emotions
- fear ratio versus AI-related asset prices
- fear ratio correlations with AI asset prices

## Data Folder

`data/` contains intermediate and final CSV files. It is ignored by git because the files can be large and are generated artifacts.

Important files include:

- `raw_posts_bsky.csv`: raw Bluesky post data.
- `clean_posts_bsky.csv`: cleaned Bluesky data.
- `bsky_final.csv`: final Bluesky dataset with emotion, FinBERT sentiment, and AI-related group labels.
- `youtube_final.csv`: final YouTube dataset with emotion, FinBERT sentiment, and AI-related group labels.
- `final_plots/`: exported plots used in the final analysis.

## Notes

- The final CSVs are the main source for the analysis plots.
- The project uses generated intermediate data, so rerunning notebooks in a different order can overwrite CSVs and plots.
- `data/`, virtual environments, notebook checkpoints, and local agent/tooling folders are ignored by git.
