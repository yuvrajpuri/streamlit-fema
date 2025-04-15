import streamlit as st
from PIL import Image, ImageDraw
import os
import uuid

# Helper function: Crop and return image
def crop_bbox(image, bbox):
    left, top, width, height = bbox
    right = left + width
    bottom = top + height
    return image.crop((left, top, right, bottom))

# Helper function: Draw bounding boxes
def draw_bounding_boxes(image, annotations):
    draw = ImageDraw.Draw(image)
    for ann in annotations:
        bbox = ann["bbox"]
        category_id = ann["category_id"]

        left, top, width, height = bbox
        right = left + width
        bottom = top + height

        color = "red"
        if category_id == 1:
            color = "purple"
        elif category_id == 2:
            color = "yellow"

        draw.rectangle([left, top, right, bottom], outline=color, width=3)
    return image

# Main section
st.title("Crop & Visualize Detected Regions")

CATEGORY_LABELS = {
    1: "Affected building",
    2: "Major damage"
}

# Ensure we have the necessary session state
if "last_uploaded_image" in st.session_state and "last_annotations" in st.session_state:
    image = st.session_state["last_uploaded_image"]
    annotation_data = st.session_state["last_annotations"]

    coco_annotations = annotation_data["annotations"]
    image_meta = annotation_data["images"][0]  # only one image per run

    # Draw bounding boxes on image
    boxed_img = draw_bounding_boxes(image.copy(), coco_annotations)
    st.image(boxed_img, caption="Image with Bounding Boxes", use_container_width=True)

    st.subheader("Cropped Objects")
    
    # Crop and display all detected areas
    for i, ann in enumerate(coco_annotations):
        bbox = ann["bbox"]  # [x, y, width, height]
        category_id = ann["category_id"]
        label = CATEGORY_LABELS.get(category_id, "Unknown")


        cropped = crop_bbox(image, bbox)
        st.image(cropped, caption=f"Crop {i+1} : {label}", width=200)
else:
    st.warning("No image or annotation data found in session. Please run detection first.")
