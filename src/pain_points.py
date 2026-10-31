import re

PAIN_POINTS = {
    "quality": ["quality", "poor quality", "damaged", "defective", "broken"],
    "delivery": ["delivery", "late", "delay", "arrived late", "shipping"],
    "price": ["expensive", "price", "costly", "overpriced"],
    "packaging": ["packaging", "package", "box", "leak", "leaked"],
    "performance": ["not working", "doesn't work", "performance", "ineffective"],
    "service": ["customer service", "support", "refund", "replacement"]
}

def detect_pain_points(text):
    text = text.lower()
    found = []
    for category, keywords in PAIN_POINTS.items():
        if any(re.search(r"\b" + re.escape(k) + r"\b", text) for k in keywords):
            found.append(category)
    return found

if __name__ == "__main__":
    text = input("Enter a review: ")
    print("Pain points:", detect_pain_points(text))
