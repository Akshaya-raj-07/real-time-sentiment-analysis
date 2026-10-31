import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

df = pd.read_csv("data/processed/sentiment_reviews.csv")
df["clean_review"] = df["clean_review"].fillna("")

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(df["clean_review"])
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC(),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="weighted", zero_division=0
    )
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": p,
        "Recall": r,
        "F1": f1
    })

result_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print("\n===== MODEL COMPARISON =====")
print(result_df.to_string(index=False))
result_df.to_csv("data/processed/model_comparison.csv", index=False)
