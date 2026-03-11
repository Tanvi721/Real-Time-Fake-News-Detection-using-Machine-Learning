import streamlit as st
import pandas as pd
import re
import requests

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------
# PAGE CONFIG
# -----------------------

st.set_page_config(page_title="AI Fake News Detector", page_icon="🧠", layout="wide")

st.title("🧠 Real-Time Fake News Detector")
st.write("Machine Learning + AI Similarity + Live News Search")


# -----------------------
# LOAD DATA (ONLINE)
# -----------------------

@st.cache_data
def load_data():

    fake_url = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake.csv"
    true_url = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/true.csv"

    fake = pd.read_csv(fake_url)
    true = pd.read_csv(true_url)

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true])

    df = df.sample(frac=1).reset_index(drop=True)

    df = df.sample(15000)

    return df


# -----------------------
# CLEAN TEXT
# -----------------------

def clean_text(text):

    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text


df["content"] = df["text"]
df["content"] = df["content"].apply(clean_text)


# -----------------------
# MODEL TRAINING
# -----------------------

@st.cache_resource
def train_model():

    X = df["content"]
    y = df["label"]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        max_df=0.7,
        ngram_range=(1,2)
    )

    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return model, vectorizer, accuracy


model, vectorizer, model_accuracy = train_model()


# -----------------------
# LOAD SIMILARITY MODEL
# -----------------------

@st.cache_resource
def load_similarity_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


similarity_model = load_similarity_model()


# -----------------------
# NEWS API
# -----------------------

API_KEY = st.secrets["NEWS_API_KEY"]


def get_latest_news(query):

    keywords = " ".join(query.split()[:4])

    url = f"https://newsapi.org/v2/everything?q={keywords}&apiKey={API_KEY}"

    response = requests.get(url).json()

    articles = []

    if "articles" in response:

        for a in response["articles"][:5]:
            articles.append(a["title"])

    return articles


# -----------------------
# SIMILARITY CHECK
# -----------------------

def check_similarity(user_news, articles):

    if len(articles) == 0:
        return 0

    user_embedding = similarity_model.encode([user_news])

    scores = []

    for article in articles:

        article_embedding = similarity_model.encode([article])

        score = cosine_similarity(user_embedding, article_embedding)[0][0]

        scores.append(score)

    return max(scores)


# -----------------------
# DASHBOARD
# -----------------------

st.subheader("📊 Model Dashboard")

col1, col2 = st.columns(2)

col1.metric("Model Accuracy", f"{round(model_accuracy*100,2)} %")
col2.metric("Dataset Size", len(df))


# -----------------------
# INPUT AREA
# -----------------------

st.subheader("💬 Check News")

news = st.text_area("Enter News Text")


if st.button("Analyze News"):

    with st.spinner("Analyzing news with AI..."):

        cleaned = clean_text(news)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)[0]

        confidence = max(probability)

        st.write("### 🧠 ML Prediction")

        if prediction == 0:
            st.error("🚨 Fake News")
        else:
            st.success("✅ Real News")

        st.write(f"Prediction Confidence: **{round(confidence*100,2)} %**")

        st.write("### 🌐 Searching Latest News")

        articles = get_latest_news(news)

        if len(articles) == 0:
            st.warning("No related news articles found")

        else:

            st.write("Top related headlines:")

            for a in articles:
                st.write("-", a)

        st.write("### 🔎 AI Similarity Check")

        similarity_score = check_similarity(news, articles)

        st.write("Similarity Score:", round(similarity_score, 2))

        if similarity_score > 0.6:
            st.success("Similar news found online → Likely Real")
        else:
            st.warning("Low similarity with online news → Possibly Fake")


