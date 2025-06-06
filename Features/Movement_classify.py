import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Data.source import get_stock_data

def add_features(df):
    df['Return'] = df['Close'].pct_change()
    df['Direction'] = df['Return'].apply(lambda x: 1 if x > 0 else 0)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df.dropna(inplace=True)
    return df

def run():
    st.header("📈 Classify Price Movements")

    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("today"))

    if st.button("Classify"):
        df = get_stock_data(ticker, start=start_date, end=end_date)
        df = add_features(df)

        features = ['SMA_10', 'SMA_30']
        X = df[features]
        y = df['Direction']

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        latest_data = X.tail(1)
        next_move = model.predict(latest_data)[0]
        label = "🔼 Price will go UP" if next_move == 1 else "🔻 Price will go DOWN"

        st.subheader(f"Predicted Movement: {label}")
        st.success(f"Accuracy on Test Set: {acc*100:.2f}%")

        st.line_chart(df[['Close']])
        with st.expander("Show data"):
            st.dataframe(df.tail(50))
