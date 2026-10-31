import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

DATA = "data/processed/sentiment_reviews.csv"

df = pd.read_csv(DATA)
df["clean_review"] = df["clean_review"].fillna("")

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(df["clean_review"])
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("===== TF-IDF FEATURES =====")
print("Number of reviews:", X.shape[0])
print("Number of features:", X.shape[1])
print("Training reviews:", X_train.shape[0])
print("Testing reviews:", X_test.shape[0])
