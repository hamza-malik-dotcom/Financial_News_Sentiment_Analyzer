import streamlit as st
st.set_page_config(page_title="Financial News Sentiment Analyzer", layout="centered")
import joblib
import numpy as np
import tensorflow as tf
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load all models and vectorizers
@st.cache_resource
def load_all_models():
    models = {}

    # Logistic Regression
    models['log_model'] = joblib.load("logistic_model.pkl")
    models['tfidf'] = joblib.load("tfidf_vectorizer.pkl")

    # Enhanced LSTM
    models['lstm'] = load_model("enhanced_lstm_model.h5")
    models['lstm_tokenizer'] = joblib.load("lstm_tokenizer.pkl")

    # Word2Vec + LSTM
    models['word2vec_lstm'] = load_model("word2vec_lstm_model.h5")
    models['word2vec_tokenizer'] = joblib.load("word2vec_tokenizer.pkl")
    models['word2vec_model'] = Word2Vec.load("word2vec.model")

    # FinBERT
    models['finbert_model'] = BertForSequenceClassification.from_pretrained("finbert_model")
    models['finbert_tokenizer'] = BertTokenizer.from_pretrained("finbert_model")

    return models

models = load_all_models()

# Preprocessing function (same as training)
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def predict_logistic(text):
    text_clean = preprocess(text)
    vector = models['tfidf'].transform([text_clean])
    pred = models['log_model'].predict(vector)[0]
    return pred

def predict_lstm(text):
    text_clean = preprocess(text)
    seq = models['lstm_tokenizer'].texts_to_sequences([text_clean])
    pad = pad_sequences(seq, maxlen=120, padding='post')
    pred = np.argmax(models['lstm'].predict(pad), axis=-1)[0]
    return pred

def predict_word2vec_lstm(text):
    text_clean = preprocess(text)
    seq = models['word2vec_tokenizer'].texts_to_sequences([text_clean])
    pad = pad_sequences(seq, maxlen=120, padding='post')
    pred = np.argmax(models['word2vec_lstm'].predict(pad), axis=-1)[0]
    return pred

def predict_finbert(text):
    inputs = models['finbert_tokenizer'](text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = models['finbert_model'](**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs).item()
    return label

# Label Mapping
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ================= Streamlit UI ====================

st.title("📰 Financial News Sentiment Analyzer")
st.markdown("Predict the sentiment of financial news headlines using **4 trained models**.")

text = st.text_area("Enter a financial news headline:")

model_choice = st.selectbox(
    "Select the model:",
    ("Logistic Regression (TF-IDF)", "Enhanced LSTM", "Word2Vec + LSTM", "FinBERT")
)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter a headline.")
    else:
        with st.spinner("Analyzing..."):
            if model_choice == "Logistic Regression (TF-IDF)":
                pred = predict_logistic(text)
            elif model_choice == "Enhanced LSTM":
                pred = predict_lstm(text)
            elif model_choice == "Word2Vec + LSTM":
                pred = predict_word2vec_lstm(text)
            elif model_choice == "FinBERT":
                pred = predict_finbert(text)

            st.success(f"**Sentiment:** {label_map[pred]}")

