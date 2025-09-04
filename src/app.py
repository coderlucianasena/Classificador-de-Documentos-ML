
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Carregar o vetorizador e o modelo
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/multinomial_nb_model.pkl")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify_document_api():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Nenhum texto fornecido para classificação."}), 400

    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    category = prediction[0]

    return jsonify({"text": text, "category": category})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


