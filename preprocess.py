# preprocess.py

import re
import nltk
import pickle
import os
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# 🧹 Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)     # Remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# 🔍 Load trained model from model directory
def load_model():
    model_path = os.path.join('model', 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# 🔍 Load TF-IDF vectorizer from model directory
def load_vectorizer():
    vectorizer_path = os.path.join('model', 'vectorizer.pkl')
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer
