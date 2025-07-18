import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from src.preprocessing import clean_text



df = pd.read_csv('data/tweets.csv', encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ['sentiment', 'tweet']
df['sentiment'] = df['sentiment'].map({0: 0, 2: 1, 4: 2})
df['clean_tweet'] = df['tweet'].apply(clean_text)

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_tweet'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)

pickle.dump(model, open('models/sentiment_model.pkl', 'wb'))
pickle.dump(tfidf, open('models/tfidf_vectorizer.pkl', 'wb'))