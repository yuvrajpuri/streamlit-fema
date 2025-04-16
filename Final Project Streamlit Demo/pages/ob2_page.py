import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import torch
import numpy as np
import os

# Needed for COCO annotation generation
import uuid
from datetime import datetime

# After loading image via PIL
from datetime import datetime
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

# helper function to build the COCO that we obtain. it's barebones, focused on 1 image at a time
def build_coco_json(image_name, width, height, detections, picture):
    """
    detections: list of dicts, each with:
        {
            "bbox": [x, y, width, height],
            "category_id": int
        }
    """
    
    # default image id for a singular image uploaded - i.e. the image that we get the crops from
    image_id=1

    # for the annotations portion of the COCO JSON
    annotations=[]

    # x & y are starting initial coordinate for the bounding box
    # w and h are "width" and "height" of the bounding box
    for idx, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        annotations.append({
            "id": idx,
            "image_id": image_id,
            "category_id": det["category_id"],
            "bbox": [x, y, w, h],
            "area": w * h,
            "segmentation":[],
            "iscrowd": 0
        })

    # for loop making the annotations complete, lets make the COCO
    date_data = get_date_captured(picture)
    coco = {
        "images": [{
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height,
            "date_captured": date_data
        }],
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "Affected building"},
            {"id": 2, "name": "Major damage"}
        ]
    }

    return coco


    
    

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
                
                # 1. Display "detections" + showing on the cropping page
                detections = []
                session_detections = []
                
                # 2. Annotations list "detections" for JSON
                ann_det = []

                # For build_coco_json helper function
                width, height = pic.size
                pic_name = insert_file.name
                
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
                     # Calculate width and height
                    width_box = x2 - x1
                    height_box = y2 - y1
                    category_id = CATEGORY_MAP[category_name]  # Same map as before

                    session_detections.append({
                        "bbox": [int(x1), int(y1), int(width_box), int(height_box)],
                        "category_id": category_id
                    })

                    # Annotations information
                    ann_det.append({
                        "category_id": category_id,
                        "bbox": [int(x1), int(y1), int(width_box), int(height_box)],
                        "area": int(width_box * height_box)
                    })

                df = pd.DataFrame(detections)

                coco_json = build_coco_json(pic_name, width, height, session_detections, pic)
                coco_json_str = json.dumps(coco_json, indent=2)

                # Show the dataframe that has the displayable detections
                st.subheader("Detected Objects")
                st.dataframe(df, use_container_width=True)

                st.subheader("COCO JSON Preview (Annotations Only)")
                st.dataframe(pd.DataFrame(ann_det), use_container_width=True)
                
                #st.success("COCO annotation created!")

                # Downloadable COCO annotations (use for debugging)
                #st.download_button(
                    #label="Download COCO JSON",
                    #data=coco_json_str,
                    #file_name=coco_json_path,
                    #mime="application/json"
                #)

                # Save COCO annotations that we generated to the session_state. This way we don't have to search for an image and annotations file
                st.session_state["last_uploaded_image"] = pic
                #st.session_state["last_annotations"] = json.loads(coco_json_str)
                st.session_state["last_detections"] = session_detections


            else:
                st.subheader("Detected Objects")
                st.write("🔍 None found.")
