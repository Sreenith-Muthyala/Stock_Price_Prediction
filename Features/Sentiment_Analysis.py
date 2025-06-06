import streamlit as st
import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


API_KEY = "4c80436e00c64e288c86e6df8a416ff0"

def get_news(company):
    url = f"https://newsapi.org/v2/everything?q={company}&language=en&sortBy=publishedAt&pageSize=20&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    return data.get("articles", [])

def analyze_sentiment(news_list):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []
    headlines = []

    for article in news_list:
        title = article.get("title", "")
        score = analyzer.polarity_scores(title)
        sentiment_scores.append(score)
        headlines.append(title)

    df = pd.DataFrame(sentiment_scores)
    df["Headline"] = headlines
    df["Sentiment"] = df["compound"].apply(lambda x: "Positive" if x > 0.2 else ("Negative" if x < -0.2 else "Neutral"))
    return df

def run():
    st.header("📰 Sentiment Analysis on News Headlines")

    company = st.text_input("Enter Company Name or Stock Ticker (e.g., Apple or AAPL)", value="Apple")

    if st.button("Analyze Sentiment"):
        with st.spinner("Fetching news and analyzing sentiment..."):
            articles = get_news(company)
            if not articles:
                st.warning("No articles found. Try another company.")
                return

            df = analyze_sentiment(articles)

            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = df["Sentiment"].value_counts()
            st.bar_chart(sentiment_counts)

            st.subheader("🧠 Average Sentiment Score")
            st.write(df[["pos", "neu", "neg", "compound"]].mean())

            st.subheader("📰 Recent Headlines and Sentiment")
            st.dataframe(df[["Headline", "Sentiment", "compound"]])

            st.caption("Powered by NewsAPI and VADER Sentiment")

