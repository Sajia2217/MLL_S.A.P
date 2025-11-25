import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

    
# Load model and tokenizer
model = tf.keras.models.load_model("sentiment_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = text.strip()
    return text

# Prediction Function
def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100)  # same maxlen used during training
    pred = model.predict(padded)[0][0]

    if pred >= 0.5:
        return "Positive 😄", float(pred)
    else:
        return "Negative 😠", float(pred)

# Streamlit UI
st.title("Sentiment Analysis Web App (LSTM/GRU Model)")
st.write("Enter text below to detect sentiment.")

user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, confidence = predict_sentiment(user_input)
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: {confidence:.4f}")
