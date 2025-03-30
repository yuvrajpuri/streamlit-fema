# Final Output Page

import streamlit as st
import tensorflow as tf
import io
from contextlib import redirect_stdout

st.title("Final Output Page")

st.write("Not entirely sure what to put here yet. Maybe a combination of everything all working together? Maybe show a cleaner version of the individual parts?")

st.write("If there are any suggestions for what to include, please let me know.")

st.header("Test run for tensorflow models.")

# Sample model making method for the cache-ing test

@st.cache_resource
def load_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
  ])
  return model

model = load_model()

# note that the time it took for the page to load in was increased due to the actual import + creation of the model


# We have newly cache'd the actual model being made. Now the only computation time that should occur is showing the model.

# New method to redirect output from console so we can present it on the screen

# Capture model.summary() output
def get_model_summary(model):
    buffer = io.StringIO()
    with redirect_stdout(buffer):  # Redirect console output to buffer
        model.summary()
    return buffer.getvalue()

ms_str = get_model_summary(model)

st.title("Model Summary")
st.text(ms_str)
# Using st.text had a bit of an odd output where the symbols were out of line with the text. Attempting st.write.

# st.write(ms_str)
# Using st.write was even worse when it came to alignment. For now, these types of console outputs will have to be excluded.
