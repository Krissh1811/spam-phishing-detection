from model import model
from preprocess import vectorizer, clean_text
from fuzzy import get_risk_level

while True:
    text = input("\nEnter message (type 'exit' to stop): ")

    if text.lower() == "exit":
        break

    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized.toarray())[0][0]

    label = "SPAM" if prediction > 0.5 else "NOT SPAM"
    risk = get_risk_level(prediction)

    print("\n--- RESULT ---")
    print("Prediction:", label)
    print("Confidence:", round(prediction, 2))
    print("Risk Level:", risk)
    print("--------------")