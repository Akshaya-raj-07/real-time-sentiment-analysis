import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Load processed dataset
df = pd.read_csv("data/processed/sentiment_reviews.csv")

# Input and output
X_text = df["clean_review"].fillna("")
y = df["sentiment"]

# Convert text into numerical features
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(X_text)

# Create SVM model
model = LinearSVC()

# Train the model
model.fit(X, y)

# Create models folder if needed
Path("models").mkdir(exist_ok=True)

# Save model and vectorizer
joblib.dump(model, "models/svm_model.pkl")
joblib.dump(vectorizer, "models/svm_vectorizer.pkl")

print("===== BEST MODEL SAVED =====")
print("Model: SVM")
print("Accuracy from comparison: 83.09%")
print("Model saved: models/svm_model.pkl")
print("Vectorizer saved: models/svm_vectorizer.pkl")