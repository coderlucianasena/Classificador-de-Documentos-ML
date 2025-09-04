
import pandas as pd
import joblib
from sklearn.metrics import classification_report

if __name__ == '__main__':
    test_df = pd.read_csv('data/test.csv')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    model = joblib.load('models/multinomial_nb_model.pkl')

    X_test = vectorizer.transform(test_df['text'])
    y_test = test_df['category']

    y_pred = model.predict(X_test)

    print('Relatório de Classificação:')
    print(classification_report(y_test, y_pred))


