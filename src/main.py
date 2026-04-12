import pandas as pd
import os
from model import model
from preprocess import vectorizer, clean_text
from fuzzy import get_risk_level

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load dataset
with open(os.path.join(BASE_DIR, "data", "demo_dataset.txt"), "r") as f:
    messages = [line.strip() for line in f if line.strip()]

results = []

for msg in messages:
    cleaned = clean_text(msg)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized.toarray())[0][0]
    # soften prediction
    prediction = 0.7 * prediction + 0.15

    label = "SPAM" if prediction > 0.5 else "NOT SPAM"
    risk = get_risk_level(prediction)

    results.append({
        "Message": msg,
        "Prediction": label,
        "Confidence": round(prediction, 2),
        "Risk": risk
    })

# Convert to table
df = pd.DataFrame(results)

print("\n===== RESULTS TABLE =====\n")
print(df)

# Optional: save for report
df.to_csv("results.csv", index=False)

# Accuracy (optional but impressive)
from preprocess import y_test, X_test

test_preds = model.predict(X_test.toarray())
test_labels = (test_preds > 0.5).astype(int)

accuracy = (test_labels.flatten() == y_test.values).mean()

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nRisk Distribution:\n", df['Risk'].value_counts())