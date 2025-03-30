# Final Output Page

import streamlit as st
import tensorflow as tf

st.title("Final Output Page")

st.write("Not entirely sure what to put here yet. Maybe a combination of everything all working together? Maybe show a cleaner version of the individual parts?")

st.write("If there are any suggestions for what to include, please let me know.")

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
st.write(model.summary())
print((model.summary()))

# The same problems I had with Jupyter that stopped me from running anything with tensorflow are rearing their head here.
