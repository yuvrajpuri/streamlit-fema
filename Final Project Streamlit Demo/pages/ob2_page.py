import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
import numpy as np
import os

# Absolute path to the model in Colab - assuming its been pre-uploaded. Need to adjust for a better solution.
MODEL_PATH = "/content/yolo11x.pt"

# Load model once
@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, device

model, DEVICE = load_model()

# User interface - the title and the guiding text.
st.title("YOLOv11 Object Detection (Sample)")
st.write("Upload an image and detect objects using your trained YOLOv11 model.")

# Upload image
insert_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if insert_file is not None:
    pic = Image.open(insert_file).convert("RGB")
    st.image(pic, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Inference"):
        with st.spinner("Detecting..."):
            results = model(pic, device=DEVICE)

            # Get image with boxes (BGR -> RGB)
            result_array = results[0].plot()
            result_image = Image.fromarray(result_array[..., ::-1])  # Convert BGR to RGB

            st.image(result_image, caption="Detection Results", use_column_width=True)
