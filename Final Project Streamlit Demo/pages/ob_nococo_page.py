import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import torch
import numpy as np
import os

# Needed for COCO annotation generation
import uuid

# Absolute path to the model in Colab - assuming its been pre-uploaded. Need to adjust for a better solution.
MODEL_PATH = "/content/best.pt"

assert os.path.exists(MODEL_PATH), "Model file not found!"


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
st.write("Upload an image and detect disaster damage using a trained YOLOv11 model.")

# Upload image
insert_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Map category ids to the type of building damage
CATEGORY_MAP = {
    "Affected_Building": 1,
    "Major_Damage": 2
}

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
                # Extract data - most important for annotations
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                names = result.names

                # Bonus for display
                conf = boxes.conf.cpu().numpy()

                # Build DataFrame and usage for making annotations
                
                # 1. Display "detections" 
                detections = []

                # 2. Annotations list "detections"
                ann_det = []
                sess_det = []
                
                for i in range(len(xyxy)):
                    # Coordinates for bounding boxes - used for display and annotations
                    x1, y1, x2, y2 = xyxy[i]

                    # Getting category names/category ids for the annotations
                    category_name = names[int(cls[i])]

                    # Display information on streamlit
                    detections.append({
                        "Class": names[int(cls[i])],
                        "Confidence": round(float(conf[i]), 3),
                        "X1": int(x1),
                        "Y1": int(y1),
                        "X2": int(x2),
                        "Y2": int(y2)
                    })

                    width_box = x2 - x1
                    height_box = y2 - y1
                    category_id = CATEGORY_MAP[category_name]  # Same map as before

                    session_detections.append({
                        "bbox": [int(x1), int(y1), int(width_box), int(height_box)],
                        "category_id": category_id
                    })
                  
                    # Annotations information
                    ann_det.append({
                        "filename": insert_file.name,
                        "xmin": x1,
                        "ymin": y1,
                        "xmax": x2,
                        "ymax": y2,
                        "class": category_name,
                    })

                df = pd.DataFrame(detections)
               

                st.session_state["last_uploaded_image"] = pic
                st.session_state["last_detections"] = session_detections

                # Show the dataframe that has the displayable detections
                st.subheader("Detected Objects")
                st.dataframe(df, use_container_width=True)
            else:
                st.subheader("Detected Objects")
                st.write("🔍 None found.")
