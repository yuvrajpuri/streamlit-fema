import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
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
        with st.spinner("Running inference..."):
            results = model(pic, device=DEVICE)
            result = results[0]

            # Display image with detections
            plotted_img = result.plot()
            result_image = Image.fromarray(plotted_img[..., ::-1])  # BGR to RGB
            st.image(result_image, caption="Detection Results", use_container_width=True)

            # Show detection results in a table
            boxes = result.boxes
            if boxes and boxes.xyxy is not None and len(boxes) > 0:
                # Extract data
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                names = result.names

                # Build DataFrame
                detections = []
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    detections.append({
                        "Class": names[int(cls[i])],
                        "Confidence": round(float(conf[i]), 3),
                        "X1": int(x1),
                        "Y1": int(y1),
                        "X2": int(x2),
                        "Y2": int(y2)
                    })

                df = pd.DataFrame(detections)
                st.subheader("Detected Objects")
                st.dataframe(df, use_container_width=True)
            else:
                st.subheader("Detected Objects")
                st.write("🔍 None found.")
