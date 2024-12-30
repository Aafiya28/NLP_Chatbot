import streamlit as st
import random
import json
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

# Load model and data
model = load_model("chatbot_model.h5")
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
lemmatizer = WordNetLemmatizer()

# Function to preprocess user input
def preprocess_input(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

# Function to predict intent
def predict_class(sentence):
    bag = preprocess_input(sentence)
    res = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Function to get response
def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    for i in intents_json["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])

# Load intents
with open("intents.json", "r") as file:
    intents = json.load(file)

# Streamlit UI
st.title("AI Chatbot")
st.write("Talk to me!")

user_input = st.text_input("You:")
if st.button("Send"):
    if user_input:
        intents_list = predict_class(user_input)
        response = get_response(intents_list, intents)
        st.write("Bot:", response)
