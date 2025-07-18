import streamlit as st
import pickle
from src.preprocessing import clean_text

model = pickle.load(open('models/sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))


def predict_sentiment(tweet):
    processed = clean_text(tweet)
    vector = vectorizer.transform([processed])
    pred = model.predict(vector)[0]
    labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return labels.get(pred, 'Unknown')


st.title("Twitter Sentiment Analyzer")

tweet = st.text_area("Enter a Tweet")

if st.button("Analyze"):
    sentiment = predict_sentiment(tweet)
    st.write(f"Sentiment: {sentiment}")
