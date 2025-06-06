import streamlit as st
from Features.Price_Prediction import run as price_predict_run
from Features.Risk_Prediction import run as risk_predict_run
from Features.Movement_classify import run as movement_run
from Features.Sentiment_Analysis import run as sentiment_run

st.set_page_config(page_title="Our Profit Predictions from Stocks", layout="wide")

option = st.sidebar.selectbox(
    "Select Feature",
    ("Price Prediction", "Risk Prediction", "Movement Classification", "Sentiment Analysis")
)
if option == "Price Prediction": price_predict_run()
elif option == "Risk Prediction": risk_predict_run()
elif option == "Movement Classification": movement_run()
elif option == "Sentiment Analysis": sentiment_run()