from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class NaiveBayesModel:
    def __init__(self):
        self.model = make_pipeline(CountVectorizer(), MultinomialNB())

    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        return data['text'], data['label']

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        print(classification_report(y_test, predictions))
        print(confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    # Load data
    nb_model = NaiveBayesModel()
    X, y = nb_model.load_data('data/raw/semisupervised-bert-xlnet.csv')

    # Split data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    nb_model.train(X_train, y_train)

    # Evaluate the model
    nb_model.evaluate(X_test, y_test)