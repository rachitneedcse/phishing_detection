import streamlit as st

st.markdown(
    """
    <style>
    /* Set the page background to white */
    body, [data-testid="stAppViewContainer"] {
        background-color: white;
    }
    /* Make titles (h1, h2, h3) black */
    h1, h2, h3 {
        color: black !important;
    }
    p, span, div {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)




import joblib
import re
import numpy as np
from lime.lime_text import LimeTextExplainer
from streamlit.components.v1 import html


# Load saved ML model and TF-IDF vectorizer
ml_model = joblib.load('model/email_spam_nb.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

st.title("Phishing Email Detection")
st.markdown("Enter the email text below to classify it as 0:**Phishing** or 1:**Safe**")

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_proba(texts):
    features = vectorizer.transform(texts)
    return ml_model.predict_proba(features)

email_text = st.text_area("Email Text")

if st.button("Predict"):
    if email_text.strip():
        cleaned_text = preprocess_text(email_text)
        ml_features = vectorizer.transform([cleaned_text])
        ml_pred = ml_model.predict(ml_features)

        st.write(f"Machine Learning Model Prediction: **{ml_pred[0]}**")

        

        explainer = LimeTextExplainer(class_names=ml_model.classes_)
        exp = explainer.explain_instance(cleaned_text, predict_proba, num_features=10)

        
        html_exp = exp.as_html()
        html(html_exp, height=800)
    else:
        st.error("Please enter the email text to classify.")
