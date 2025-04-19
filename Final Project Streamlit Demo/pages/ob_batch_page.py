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
    with st.spinner("Running batch object detection..."):

        # Lists and counters to keep track of where we're at
        all_annotations = []
        all_images = []
        skipped = []
        img_id_counter = 1
        anno_id_counter = 1

        for uploaded in batch_files:
            try:
                image = Image.open(uploaded).convert("RGB")
            except Exception:
                skipped.append(uploaded.name + " (invalid image)")
                continue

            w, h = image.size
            img_name = uploaded.name
            conception = get_date_captured(image)

            results = model(image, device=DEVICE)
            result = results[0]
            boxes = result.boxes

            if boxes and boxes.xyxy is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                names = result.names

                # Image section of the JSON (Image, Annotations, Categories)
                img_dict = {
                    "id": img_id_counter,
                    "file_name": img_name,
                    "width": w,
                    "height": h,
                    "date_captured": conception
                }
                all_images.append(img_dict)

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    category_name = names[int(cls[i])]

                    category_id = CATEGORY_MAP.get(category_name)
                    if not category_id:
                        continue

                    width = x2-x1
                    height = y2-y1

                    # Annotations for the given image
                    annotations = {
                        "id": anno_id_counter,
                        "image_id": img_id_counter,
                        "category_id": category_id,
                        "bbox": [int(x1), int(y1), int(w), int(h)],
                        "area": int(width * height),
                        "iscrowd": 0
                    }
                    all_annotations.append(annotations)
                    anno_id_counter +=1

                img_id_counter +=1

            # No bounding boxes then skip it
            else:  
                skipped.append(img_name)

        # Make the JSON with all the images and annotations
        big_coco_json = {
            "images": all_images,
            "annotations": all_annotations,
            "categories": [
                {"id": 1, "name": "Affected building"},
                {"id": 2, "name": "Major damage"}
            ]
        }

        # Big JSON , made downloadable
        if all_annotations:
            
            json_str = json.dumps(big_coco_json, indent=2)
            st.download_button(
                label="Download COCO Annotations",
                data = json_str,
                file_name="batch_annotations.json"
            )
        else:
            st.info("No objects were detected. No annotations were generated - nothing to download. Submit different images.")

        if len(skipped) == len(batch_files):
            st.warning("All uploaded images had no detectable objects.")
        elif skipped:
            st.subheader("Images Skipped (no detections present)")
            for name in skipped:
                st.markdown(f"- ❌ **{name}**")
        else:
            st.success("All images processed successfully!")
        
