import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

@st.cache_resource
def load_models():
    # loads your serialized artifacts
    le = joblib.load("models/labelencoder")
    embed_model = joblib.load("models/embedmodel")
    keras_model = tf.keras.models.load_model("models/model.keras")
    return le, embed_model, keras_model

le, embed_model, keras_model = load_models()

st.title("Text Classification with Keras & Embeddings")

user_input = st.text_area("Enter text to classify:")

if st.button("Predict"):
    # 1. Embed your input
    #    adjust this call if your embedmodel uses .encode() instead of .transform()
    emb = embed_model.transform([user_input])
    emb_array = np.array(emb)

    # 2. Run the Keras model
    preds = keras_model.predict(emb_array)
    class_idx = np.argmax(preds, axis=1)[0]

    # 3. Decode the label
    label = le.inverse_transform([class_idx])[0]

    st.success(f"**Prediction:** {label}")
