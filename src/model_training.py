
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

if __name__ == '__main__':
    train_df = pd.read_csv('data/train.csv')

    # Treinar o TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words=None, max_features=1000)
    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['category']

    # Treinar o modelo Naive Bayes
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Salvar o vetorizador e o modelo
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    joblib.dump(model, 'models/multinomial_nb_model.pkl')

    print('Modelo e vetorizador treinados e salvos em models/tfidf_vectorizer.pkl e models/multinomial_nb_model.pkl')


