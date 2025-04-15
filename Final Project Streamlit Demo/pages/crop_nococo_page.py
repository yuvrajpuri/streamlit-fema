import streamlit as st
from PIL import Image, ImageDraw

CATEGORY_LABELS = {
    1: "Affected building",
    2: "Major damage"
}

def crop_bbox(image, bbox):
    x, y, w, h = bbox
    return image.crop((x, y, x + w, y + h))

def draw_bounding_boxes(image, detections):
    draw = ImageDraw.Draw(image)
    for det in detections:
        x, y, w, h = det["bbox"]
        category_id = det["category_id"]
        color = "purple" if category_id == 1 else "yellow"
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
    return image

# Start of page
st.title("Cropping Tool")

if "last_uploaded_image" in st.session_state and "last_detections" in st.session_state:
    image = st.session_state["last_uploaded_image"]
    detections = st.session_state["last_detections"]

    # Show full image with bounding boxes
    image_with_boxes = draw_bounding_boxes(image.copy(), detections)
    st.image(image_with_boxes, caption="Detected Objects", use_container_width=True)

    st.subheader("Cropped Regions")
    for i, det in enumerate(detections):
        cropped = crop_bbox(image, det["bbox"])
        label = CATEGORY_LABELS.get(det["category_id"], "Unknown")
        st.image(cropped, caption=f"Crop {i+1}: {label}", width=200)
else:
    st.warning("No image or detections found. Please run object detection first.")
