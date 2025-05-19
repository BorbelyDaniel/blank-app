import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

@st.cache_resource
def load_models():
    # loads your serialized artifacts
    le = joblib.load("labelencoder")
    embed_model = joblib.load("embedmodel")
    keras_model = tf.keras.models.load_model("model.keras")
    return le, embed_model, keras_model

le, embed_model, keras_model = load_models()

st.title("Text Classification with Keras & Embeddings")

user_input = st.text_area("Enter text to classify:")

if st.button("Predict"):
    preds = keras_model.predict(embed_model.predict(np.array([le.transform(user_input.split(" ")[:20])])).reshape(1, -1))[0] 
    st.success(f"Prediction: Black:{preds[0]} Draw:{preds[1]} White:{preds[2]}")
