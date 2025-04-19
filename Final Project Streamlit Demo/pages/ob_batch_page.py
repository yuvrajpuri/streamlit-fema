# Same original start
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import torch
import numpy as np
import os
import json
from io import BytesIO

# Needed for COCO annotation generation
import uuid
from datetime import datetime

# After loading image via PIL
from PIL.ExifTags import TAGS

# Absolute path to the model in Colab - assuming its been pre-uploaded. Need to adjust for a better solution.
MODEL_PATH = "/content/best.pt"



# Load model once
@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, device

model, DEVICE = load_model()

# helper function to observe EXIF metadata for the date the image was captured. defaults to today if none
def get_date_captured(pil_image):
    try:
        exif = pil_image._getexif()
        if exif is not None:
            for tag, value in exif.items():
                if TAGS.get(tag) == "DateTimeOriginal":
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S").isoformat()
    except Exception:
        pass
    return datetime.now().isoformat()

# when making an annotation, clean it of the filetype (e.g. not img1.jpg_annotations but img1_annotations)
def clean_annotation(filename):
    return os.path.splitext(filename)[0]

# ----------------------------

# Display and Work

st.title("Batch Object Detection Tool")

batch_files = st.file_uploader(
    "Upload a folder of images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if batch_files:
  process = st.button("Run Batch Inference")
  if process:
    st.write("Joke")
