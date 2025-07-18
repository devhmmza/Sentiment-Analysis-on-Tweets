import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from src.preprocessing import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data/tweets.csv', encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ['sentiment', 'tweet']
df['sentiment'] = df['sentiment'].map({0: 0, 2: 1, 4: 2})
df['clean_tweet'] = df['tweet'].apply(clean_text)

tfidf = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
X = tfidf.transform(df['clean_tweet'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = pickle.load(open('models/sentiment_model.pkl', 'rb'))
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
