import streamlit as st
import joblib
import pandas as pd
import os
import re

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    text = ' '.join(word for word in text.split() if word not in manual_stopwords)  # Remove stopwords
    return text

# Define manual stopwords
manual_stopwords = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    # Truncated for brevity...
}

# Paths to resources
dataset_path = r'twitter_training.csv'
data_directory = os.path.dirname("twitter_traning.csv")
vectorizer_path = os.path.join(data_directory, 'tfidf_vectorizer.pkl')
model_path = os.path.join(data_directory, 'sentiment_rf_model_optimized.pkl')

# Load the TF-IDF vectorizer and the trained model
try:
    assert os.path.exists(vectorizer_path), f"Vectorizer file not found: {vectorizer_path}"
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Streamlit app layout
st.set_page_config(page_title="Sentiment Analysis", page_icon=":sparkles:", layout="wide")

# Title and description
st.markdown("<h1 style='text-align: center;'>🗣️ Sentiment Analysis Tool</h1>", unsafe_allow_html=True)

# Text input
text_input = st.text_area("Enter text below for analysis:", "Type something...", height=150)

# Button to analyze sentiment
if st.button("Analyze Sentiment"):
    if text_input.strip() == "":
        st.error("Please enter some text before analyzing.")
    else:
        # Clean and vectorize the input text
        cleaned_text = clean_text(text_input)
        try:
            vectorized_text = vectorizer.transform([cleaned_text]).toarray()
            prediction = model.predict(vectorized_text)[0]
            sentiment = "Positive 😊" if prediction == 1 else "Negative 😔"
            st.success(f"Sentiment: {sentiment}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
