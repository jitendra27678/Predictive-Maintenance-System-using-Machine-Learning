import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("model/model.pkl")
preprocess = joblib.load("model/preprocess.pkl")
columns = joblib.load("model/columns.pkl")

st.title("Credit Risk Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data")
    st.dataframe(df.head())

    df = df[columns]

    processed = preprocess.transform(df)
    predictions = model.predict(processed)

    df["Prediction"] = predictions

    st.write("Predictions")
    st.dataframe(df.head())

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Predictions",
        csv,
        "predictions.csv",
        "text/csv",
    )
