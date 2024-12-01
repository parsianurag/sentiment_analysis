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

# Load the TF-IDF vectorizer and the trained model
dataset_path = r'D:/Guvi/Own_Projects/Sentiment_Analysis_on_Social_Media/twitter_training.csv'
data_directory = os.path.dirname(dataset_path)

vectorizer_path = os.path.join(data_directory, 'tfidf_vectorizer.pkl')
model_path = os.path.join(data_directory, 'sentiment_rf_model_optimized.pkl')

vectorizer = joblib.load(vectorizer_path)
model = joblib.load(model_path)

# Define manual stopwords
manual_stopwords = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
    "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

# Streamlit app layout
st.set_page_config(page_title="Sentiment Analysis", page_icon=":sparkles:", layout="wide")

# CSS for styling
st.markdown("""
    <style>
        .main {
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .input-box {
            background-color: #ffffff;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 20px;
        }
        .text-output {
            padding: 15px;
            border-radius: 8px;
            color: #ffffff;
        }
        .positive {
            background-color: #4CAF50;
        }
        .negative {
            background-color: #F44336;
        }
        .stButton > button {
            background-color: #007BFF;
            color: white;
            border-radius: 8px;
            padding: 10px;
            border: none;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
        .title {
            color: #333333; /* Dark color for better visibility */
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<div class='main'><h1 class='title' style='text-align:center;'>🗣️ Sentiment Analysis Tool</h1></div>", unsafe_allow_html=True)
st.markdown("""
    <div class='main'>
    <p style='text-align:center;'>Welcome to the Sentiment Analysis Tool! 🎉</p>
    <p style='text-align:center;'>Enter your text below to get sentiment analysis results.</p>
    </div>
""", unsafe_allow_html=True)


# Text input
st.markdown("<div class='input-box'><h3>Input Text:</h3></div>", unsafe_allow_html=True)
text_input = st.text_area("", "Type something...", height=150)

# Button to analyze sentiment
if st.button("Analyze Sentiment"):
    if text_input.strip() == "":
        st.error("Please enter some text before analyzing.")
    else:
        # Clean and vectorize the input text
        cleaned_text = clean_text(text_input)
        vectorized_text = vectorizer.transform([cleaned_text]).toarray()

        # Make a prediction
        prediction = model.predict(vectorized_text)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😔"
        sentiment_class = "positive" if prediction == 1 else "negative"

        # Display the result
        st.markdown(f"""
            <div class='text-output {sentiment_class}'>
                <h2 style='text-align:center;'>{sentiment}</h2>
            </div>
            <div class='main'>
                <h3>Your Text:</h3>
                <p>{text_input}</p>
            </div>
        """, unsafe_allow_html=True)



