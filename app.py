import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st
import json
import random
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%D - %H:%M \n")

def load_files():
    model = keras.models.load_model('chatbot_model.h5')
    intents = json.loads(open('intents.json').read())
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    lencoder = pickle.load(open('lencoder.pkl', 'rb'))

    return model, intents, vectorizer, lencoder

model, intents, vectorizer, lencoder = load_files()


def predict_class(sentence, vectorizer, lencoder, model):
    yhat = model.predict(vectorizer.transform([sentence]).todense())
    tag = lencoder.inverse_transform([np.argmax(yhat)])[0]
    pred = {'tag': tag, 'prob': np.round(np.max(yhat), 2)}
    return pred

def chatbot_response(sentence, vectorizer, lencoder, model, intents):
    pred = predict_class(sentence, vectorizer, lencoder, model)
    for i in intents['intents']:
        if i['tag'] == pred['tag']:
            result = random.choice(i['responses'])
            break
    return result

st.header('Welcome to Food Delivery Service')
nav = st.sidebar.radio("MENU",["Pizza-100/-","Burger-150/-","Sandwich-100/-","Cutlet-150/-","Idli-80/-","Dosa-100/-"])



if st.checkbox('Order Food Here'):
    msg = st.text_input('You: ')
    st.text(current_time)
    if st.button('Send'):
        response = chatbot_response(msg, vectorizer, lencoder, model, intents)
        st.text_input('Chatbot: ', value=response)
        st.text(current_time)
st.image('image.jpg')
