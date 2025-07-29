import streamlit as st
import joblib


model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

st.title("Sentiment Analysis App")
user_input = st.text_area("Enter your comment")

if st.button("Predict Sentiment"):
    vectorized_input = tfidf.transform([user_input])
    pred_encoded = model.predict(vectorized_input)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    st.success(f"Predicted Sentiment: {pred_label}")
