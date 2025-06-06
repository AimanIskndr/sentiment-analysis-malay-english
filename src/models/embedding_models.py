# embedding_models.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences

class EmbeddingModels:
    def __init__(self, max_words=5000, max_len=100):
        self.max_words = max_words
        self.max_len = max_len
        self.vectorizer = TfidfVectorizer(max_features=self.max_words)
        self.model = None

    def preprocess_data(self, data):
        # Encode labels
        label_encoder = LabelEncoder()
        data['label'] = label_encoder.fit_transform(data['label'])
        return data

    def create_embedding_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.max_words, 128, input_length=self.max_len))
        self.model.add(SpatialDropout1D(0.2))
        self.model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(3, activation='softmax'))  # Assuming 3 classes: Positive, Neutral, Negative
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, X, y, epochs=5, batch_size=64):
        X_pad = pad_sequences(X, maxlen=self.max_len)
        self.model.fit(X_pad, y, epochs=epochs, batch_size=batch_size, verbose=2)

    def predict(self, X):
        X_pad = pad_sequences(X, maxlen=self.max_len)
        return np.argmax(self.model.predict(X_pad), axis=-1)

    def evaluate(self, X, y):
        X_pad = pad_sequences(X, maxlen=self.max_len)
        return self.model.evaluate(X_pad, y)