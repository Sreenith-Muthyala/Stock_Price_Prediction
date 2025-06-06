import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from math import sqrt

#importing source from data folder
from Data.source import get_stock_data 

def run():
    st.subheader("📈 Price Prediction ")

    ticker = st.text_input("Enter Stock Ticker Symbol", value="AAPL")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    if st.button("Run Prediction"):
        # Step 1: Fetch data
        df = get_stock_data(ticker, start_date, end_date)

        if df.empty:
            st.error("No data found for the selected ticker/date range.")
            return
        else:st.write(df)

        # Step 2: Create target variable
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        # Step 3: Prepare features and target
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = df[features]
        y = df['Target']

        # Step 4: Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        
        # Step 5: Train XGBoost model
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X_train, y_train)

        # Step 6: Predict
        predictions = model.predict(X_test)

        # Step 7: Evaluation
        rmse = sqrt(mean_squared_error(y_test, predictions))

        # step 8: Predict tomorrow's price
        latest_data = df[features].iloc[-1:]  # today's last known values
        tomorrow_pred = model.predict(latest_data)[0]

        st.info(f"📅 Predicted Closing Price for Tomorrow: **${tomorrow_pred:.2f}**")

        #step 9: Plot actual vs predicted
        st.subheader("Actual Price vs Predicted Price")
        result_df = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': predictions
        }, index=y_test.index)

        st.line_chart(result_df)

        # Display chart of historical Close prices
        st.subheader("📊 Historical Closing Prices")
        st.line_chart(df[['Close']])
