
import joblib
import pandas as pd

def classify_document(text):
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    model = joblib.load("models/multinomial_nb_model.pkl")

    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]

if __name__ == "__main__":
    new_documents = [
        "Este é um documento sobre as últimas notícias do mercado financeiro.",
        "Receita de um delicioso bolo de cenoura com cobertura de chocolate.",
        "Artigo sobre os avanços da inteligência artificial e robótica.",
        "Regulamentação para a construção de novas moradias."
    ]

    for doc in new_documents:
        category = classify_document(doc)
        print(f"Documento: \"{doc}\" -> Categoria: {category}")


