import streamlit as st
import requests
import os

st.set_page_config(page_title="StocKast", layout="wide")
API_BASE_URL = os.environ.get("INTERNAL_API_URL", "http://localhost:8000/api")

st.title("StocKast")
st.write("Streamlit interface for StocKast")

st.text_input("Enter stock ticker symbol (e.g. AAPL)")
if st.button("Predict"):
    st.info("Prediction logic will connect to FastAPI...")
