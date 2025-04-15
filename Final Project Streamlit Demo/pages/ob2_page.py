import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import torch
import numpy as np
import os

# Needed for COCO annotation generation
from pylabel import LabelDataset, importer
import uuid
from datetime import datetime

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
                ann_df = pd.DataFrame(ann_det)

                # Add bbox column in COCO format: [x, y, width, height]
                ann_df["bbox"] = ann_df.apply(
                    lambda row: [row["xmin"], row["ymin"], row["xmax"] - row["xmin"], row["ymax"] - row["ymin"]],
                    axis=1
                )

                # dataset = importer.ImportYOLOv5(path=None, df=ann_df, path_to_images=None)
                
                
                # Manually set category map 
                dataset.df["cat_id"] = dataset.df["class"].map(CATEGORY_MAP)
                
                # Export annotations dataset to COCO
                coco_json_path = f"{insert_file.name}_coco.json"
                dataset.export.ExportToCoco(coco_json_path)

                # Create a path to re-utilize the same COCO annotations and Image we uploaded for cropping
                with open(coco_json_path, "r") as f:
                    coco_json_str = f.read()

                st.success("COCO annotation created!")

                # Downloadable COCO annotations (use for debugging)
                st.download_button(
                    label="Download COCO JSON",
                    data=coco_json_str,
                    file_name=coco_json_path,
                    mime="application/json"
                )

                # Save COCO annotations that we generated to the session_state. This way we don't have to search for an image and annotations file
                st.session_state["last_uploaded_image"] = pic
                st.session_state["last_annotations"] = json.loads(coco_json_str)

                # Show the dataframe that has the displayable detections
                st.subheader("Detected Objects")
                st.dataframe(df, use_container_width=True)
            else:
                st.subheader("Detected Objects")
                st.write("🔍 None found.")
