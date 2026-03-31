from flask import Flask, request, jsonify
import re
import pickle
from tensorflow.keras.models import load_model
import os
from fuzzy import get_risk_level

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..","artifacts", "model.h5")
vectorizer_path = os.path.join(BASE_DIR, "..","artifacts", "vectorizer.pkl")

model = load_model(model_path)
vectorizer = pickle.load(open(vectorizer_path, "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']

    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized.toarray())[0][0]
    label = "Spam" if prediction > 0.5 else "Not Spam"
    risk = get_risk_level(prediction)

    return jsonify({
        "prediction": label,
        "confidence": float(prediction),
        "risk": risk
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)