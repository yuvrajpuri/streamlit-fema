# Same original start
import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
import pandas as pd
import torch
import numpy as np
import os
import json
from io import BytesIO
import zipfile

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

CATEGORY_MAP = {
    "Affected_Building": 1,
    "Major_Damage": 2
}

CATEGORY_LABELS = {
    1: "Affected building",
    2: "Major damage"
}


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

def crop_bbox(image, bbox):
    x, y, w, h = bbox
    return image.crop((x, y, x + w, y + h))

def draw_bounding_boxes(image, annotations):
    draw = ImageDraw.Draw(image)
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        color = "purple" if ann["category_id"] == 1 else "yellow"
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
    return image

# ----------------------------

# Display and Work

st.title("Batch Object Detection Tool")

batch_files = st.file_uploader(
    "Upload a folder of images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if batch_files:

    # Clear ZIP batch when new batch is uploaded
    uploaded_names = [f.name for f in batch_files]
    if uploaded_names != st.session_state.get("last_uploaded_batch",[]):
        st.session_state["batch_zip_ready"] = False
        st.session_state["batch_zip"] = None
        st.session_state["last_uploaded_batch"] = uploaded_names

    # Run the batch inference process
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
                    conf = boxes.conf.cpu().numpy()
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
                            "bbox": [int(x1), int(y1), int(width), int(height)],
                            "area": int(width * height),
                            "score": float(conf[i]),
                            "segmentation": [],
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
                if "batch_zip_ready" not in st.session_state:
                    st.session_state["batch_zip_ready"] = False
            
                json_str = json.dumps(big_coco_json, indent=2)
                st.download_button(
                    label="Download COCO Annotations",
                    data = json_str,
                    file_name="batch_annotations.json"
                )

                # Second Part: ZIP of crops and images
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zipf:
                    img_id_map = {img["id"]: img["file_name"] for img in big_coco_json["images"]}
                    image_lookup = {f.name: Image.open(f).convert("RGB") for f in batch_files}

                    # Annotations grouped by image_id
                    group_annotations = {}
                    for ann in big_coco_json["annotations"]:
                        group_annotations.setdefault(ann["image_id"], []).append(ann)

                    for image_info in big_coco_json["images"]:
                        image_id = image_info["id"]
                        image_name = image_info["file_name"]

                        if image_name not in image_lookup:
                            continue

                        picture = image_lookup[image_name]

                        # Save a picture with bounding boxes
                        pic_boxes = draw_bounding_boxes(picture.copy(), group_annotations.get(image_id, []))
                        pic_bytes = BytesIO()
                        pic_boxes.save(pic_bytes, format="JPEG")

                        clean = clean_annotation(image_name)
                        
                        # Make the folder via prefix and put bounding box image + crops in it
                        zipf.writestr(f"{clean}/{clean}_bboxes.jpg", pic_bytes.getvalue())

                        # Save the crops
                        for i, anno in enumerate(group_annotations.get(image_id, [])):
                            crop = crop_bbox(picture, anno["bbox"])
                            crop_buffer = BytesIO()
                            crop.save(crop_buffer, format="JPEG")
                            crop_buffer.seek(0)
                            zipf.writestr(f"{clean}/{clean}_crop_{i+1}.jpg", crop_buffer.read())

                        # Make the COCO JSON in the batch folder
                    zipf.writestr("batch_annotations.json", json.dumps(big_coco_json, indent=2))

                # Save the zip file to the session_state since we are effectively rerunning the script
                zip_buffer.seek(0)
                st.session_state["batch_zip"] = zip_buffer.getvalue()
                st.session_state["batch_zip_ready"] = True
                    
                if st.session_state.get("batch_zip_ready", False):
                    st.download_button(
                    label="Download Crops & Annotations (ZIP)",
                    data=st.session_state["batch_zip"],
                    file_name="batch_crops_bundle.zip",
                    mime="application/zip"
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
                
            
