import pandas as pd
import re
from pathlib import Path

INPUT = "data/raw/amazon_vfl_reviews.csv"
OUTPUT = "data/processed/sentiment_reviews.csv"

df = pd.read_csv(INPUT)

print("===== DATASET INFORMATION =====")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nColumn names:", df.columns.tolist())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print("Duplicate rows:", df.duplicated().sum())

print("\n===== RATING DISTRIBUTION =====")
print(df["rating"].value_counts().sort_index())

print("\n===== EMPTY REVIEWS =====")
print("Empty reviews:", df["review"].isna().sum())

# Remove reviews with missing text
df = df.dropna(subset=["review"])

# Remove exact duplicate records
df = df.drop_duplicates()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_review"] = df["review"].apply(clean_text)

def get_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    return "Positive"

df["sentiment"] = df["rating"].apply(get_sentiment)

print("\n===== SENTIMENT DISTRIBUTION =====")
print(df["sentiment"].value_counts())

Path("data/processed").mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\n===== PROCESSED DATASET SAVED =====")
print("Saved to:", OUTPUT)
print("Final rows:", len(df))
