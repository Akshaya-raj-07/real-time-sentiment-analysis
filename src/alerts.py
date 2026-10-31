from predict import predict_sentiment

def negative_alert(review):
    sentiment = predict_sentiment(review)
    if sentiment == "Negative":
        return "ALERT: Negative sentiment detected."
    return "No negative sentiment alert."

if __name__ == "__main__":
    review = input("Enter a review: ")
    print(negative_alert(review))
