import joblib

# Load the trained SVM model
model = joblib.load("models/svm_model.pkl")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("models/svm_vectorizer.pkl")


def predict_sentiment(review):
    # Convert the review into TF-IDF numbers
    review_vector = vectorizer.transform([review])

    # Predict sentiment
    prediction = model.predict(review_vector)

    return prediction[0]


# Take a review from the user
review = input("Enter a product review: ")

# Predict sentiment
result = predict_sentiment(review)

print("\n===== SENTIMENT RESULT =====")
print("Review:", review)
print("Predicted Sentiment:", result)