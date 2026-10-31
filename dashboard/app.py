import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from pain_points import detect_pain_points

st.set_page_config(page_title="Real-Time Sentiment Analysis", page_icon="📊", layout="wide")

st.title("📊 Real-Time Sentiment Analysis")
st.caption("NLP + Machine Learning based review analysis")

data_path = ROOT / "data" / "processed" / "sentiment_reviews.csv"
model_path = ROOT / "models" / "svm_model.pkl"
vectorizer_path = ROOT / "models" / "svm_vectorizer.pkl"

if not data_path.exists():
    st.warning("Processed dataset not found. Run: python src/preprocessing.py")
    st.stop()

df = pd.read_csv(data_path)

st.subheader("Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Reviews", len(df))
c2.metric("Products", df["name"].nunique())
c3.metric("Positive Reviews", int((df["sentiment"] == "Positive").sum()))

st.subheader("Product Selection")

products = ["All Products"] + sorted(df["name"].dropna().unique().tolist())

selected_product = st.selectbox(
    "Choose a product:",
    products
)

if selected_product == "All Products":
    filtered_df = df
else:
    filtered_df = df[df["name"] == selected_product]

st.write("Reviews available:", len(filtered_df))


st.subheader("Sentiment Distribution")

sentiment_counts = (
    filtered_df["sentiment"]
    .value_counts()
    .reindex(["Negative", "Neutral", "Positive"], fill_value=0)
)

st.bar_chart(sentiment_counts)

st.subheader("Analyze a New Review")
review = st.text_area("Enter a product review:")

if st.button("Analyze Review"):
    if not model_path.exists() or not vectorizer_path.exists():
        st.error("Train the model first using: python src/train.py")
    elif not review.strip():
        st.warning("Please enter a review.")
    else:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        prediction = model.predict(vectorizer.transform([review]))[0]

        st.success(f"Predicted Sentiment: {prediction}")

        pain_points = detect_pain_points(review)
        if pain_points:
            st.info("Detected pain points: " + ", ".join(pain_points))
        else:
            st.info("No major predefined pain point detected.")



st.dataframe(
    filtered_df[["name", "date", "rating", "review", "sentiment"]].head(20),
    use_container_width=True
)
