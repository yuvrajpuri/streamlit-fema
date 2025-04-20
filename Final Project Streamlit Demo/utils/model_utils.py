import torch
from ultralytics import YOLO
import streamlit as st

# Absolute path to the model in Colab - assuming its been pre-uploaded. Need to adjust for a better solution.
MODEL_PATH = "/content/best.pt"


# Load model once
@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, device
