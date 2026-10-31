import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA = "data/processed/sentiment_reviews.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df["clean_review"] = df["clean_review"].fillna("")

X_text = df["clean_review"]
y = df["sentiment"]

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)
X = vectorizer.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("===== LOGISTIC REGRESSION RESULTS =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred, zero_division=0))

joblib.dump(model, MODEL_DIR / "logistic_regression.pkl")
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")

print("\nModel saved in models/")
