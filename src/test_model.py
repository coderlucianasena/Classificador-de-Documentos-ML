
import unittest
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict import classify_document

class TestDocumentClassification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Carregar os dados de treino e teste reais gerados por data_preparation.py
        train_df = pd.read_csv("data/train.csv")

        # Treinar o vetorizador e o modelo com os dados reais
        cls.vectorizer = TfidfVectorizer(stop_words=None, max_features=1000)
        X_train = cls.vectorizer.fit_transform(train_df["text"])
        y_train = train_df["category"]

        cls.model = MultinomialNB()
        cls.model.fit(X_train, y_train)

        # Salvar temporariamente para que classify_document possa carregá-los
        joblib.dump(cls.vectorizer, "models/tfidf_vectorizer.pkl")
        joblib.dump(cls.model, "models/multinomial_nb_model.pkl")

    def test_classify_document(self):
        test_cases = [
            ("Este é um documento sobre as últimas notícias do mercado financeiro.", "Finanças"),
            ("Receita de um delicioso bolo de cenoura com cobertura de chocolate.", "Culinária"),
            ("Artigo sobre os avanços da inteligência artificial e robótica.", "Tecnologia"),
            ("Notícias sobre a nova política econômica do governo.", "Política"),
            ("Discussão sobre os avanços da medicina e biotecnologia.", "Saúde"),
            ("Guia completo para iniciantes em programação Python.", "Tecnologia"),
            ("Relatório anual da empresa sobre lucros e perdas.", "Finanças"),
            ("Notícias sobre esportes e resultados de jogos.", "Esportes"),
            ("Manual de instruções para montagem de móveis.", "Casa"),
            ("Estudo sobre a psicologia do consumidor.", "Educação"),
            ("Lei sobre proteção de dados pessoais e privacidade.", "Legal"),
            ("Contrato de prestação de serviços jurídicos.", "Legal"),
            ("Parecer jurídico sobre um caso de propriedade intelectual.", "Legal"),
            ("Novas tecnologias para carros autônomos.", "Tecnologia"),
            ("Desenvolvimento de software para automação industrial.", "Legal"),
        ]

        for text, expected_category in test_cases:
            predicted_category = classify_document(text)
            self.assertEqual(predicted_category, expected_category, f"Falha para o texto: {text}. Esperado: {expected_category}, Obtido: {predicted_category}")

if __name__ == '__main__':
    unittest.main()


