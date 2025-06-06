# text_preprocessor.py

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english')).union(set(stopwords.words('malay')))
    
    def clean_text(self, text):
        text = self.remove_links(text)
        text = self.remove_mentions_and_hashtags(text)
        text = self.normalize_text(text)
        return text
    
    def remove_links(self, text):
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    def remove_mentions_and_hashtags(self, text):
        text = re.sub(r'@\w+|#\w+', '', text)
        return text.strip()
    
    def normalize_text(self, text):
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)