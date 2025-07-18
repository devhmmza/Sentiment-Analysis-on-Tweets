# 🚀 Sentiment Analysis on Tweets

> A powerful NLP project that **classifies tweet sentiments** (Positive, Neutral, Negative) using classic ML & custom preprocessing.  
> Built with Python, scikit-learn, and clean, battle-tested text pipelines.

---

## 🔥 Why This Project Rocks

- **Real-world Twitter data** cleaning & preprocessing (no fluff, just raw text magic)  
- Advanced text vectorization using **TF-IDF**  
- Solid machine learning model: **Logistic Regression / Naive Bayes** — pick your fighter  
- Easy-to-use CLI / app interface for instant sentiment prediction  
- Modular, clean code structure for rapid upgrades & customization

---

## 🗂️ Project Structure

sentiment-analysis-tweets/
├── data/
│ └── tweets.csv # Raw tweet data with sentiment labels
├── models/
│ ├── sentiment_model.pkl # Trained ML model saved with pickle
│ └── tfidf_vectorizer.pkl # TF-IDF vectorizer saved for prediction
├── src/
│ ├── preprocessing.py # Text cleaning & normalization functions
│ └── train.py # Model training & saving pipeline
├── app.py # Command-line / Streamlit app for prediction
├── requirements.txt # Python dependencies
└── README.md # This badass file



---

## 🛠️ Tech Stack & Libraries

- Python 3.x  
- [pandas](https://pandas.pydata.org/) — data wrangling  
- [scikit-learn](https://scikit-learn.org/) — ML modeling & vectorization  
- [nltk](https://www.nltk.org/) — natural language preprocessing  
- [pickle](https://docs.python.org/3/library/pickle.html) — model serialization  
- [Streamlit](https://streamlit.io/) (optional) — slick UI for quick demos

---

## ⚙️ How to Run

### 1. Clone the repo

```bash
git clone https://github.com/devhmmza/Sentiment-Analysis-on-Tweets.git
cd Sentiment-Analysis-on-Tweets
