import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from Data.source import get_stock_data

def add_features(df):
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=10).std()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['EMA_10'] = df['Close'].ewm(span=10).mean()
    df.dropna(inplace=True)
    return df

def label_risk(df):
    threshold = df['Volatility'].mean()
    df['Risk_Label'] = df['Volatility'].apply(lambda x: 1 if x > threshold else 0)
    return df

def run():
    st.header("📉 Risk Prediction ")

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("today"))

    if st.button("Predict Risk"):
        df = get_stock_data(ticker, start=start_date, end=end_date)
        df = add_features(df)
        df = label_risk(df)

        features = ['Return', 'SMA_10', 'SMA_30', 'EMA_10']
        X = df[features]
        y = df['Risk_Label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        latest_data = X.tail(1)
        prediction = model.predict(latest_data)[0]

        risk = "🔴 High Risk" if prediction == 1 else "🟢 Low Risk"
        st.subheader(f"Predicted Risk: {risk}")

        st.line_chart(df[['Volatility']])
        with st.expander("Show data"):
            st.dataframe(df.tail(50))
