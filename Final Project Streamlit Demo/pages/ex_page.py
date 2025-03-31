import cv2
import numpy as np
import requests
import torch
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

# Testing if this works

# Load YOLO model once
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")  # Load the model once and reuse it

# Function to run YOLO and return the processed image
def detect_objects(image, draw_boxes=True):
    model = load_model()

    # Convert to OpenCV format if it's a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Run YOLO inference
    results = model(image)

    # If bounding boxes are OFF, return the original image
    if not draw_boxes:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw bounding boxes
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        names = [result.names[i] for i in class_ids]

        for (box, class_name, conf) in zip(boxes, names, confidences):
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_name} ({conf:.2f})"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)  # Label text

    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB

# Streamlit App
st.title("YOLO Object Detection in Streamlit 🚀")

# Upload Image or Use URL
option = st.radio("Choose Image Input:", ["Upload", "URL"])

if option == "Upload":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif option == "URL":
    image_url = st.text_input("Enter image URL:")
    if image_url:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

# If image is available, process it
if "image" in locals():
    st.image(image, caption="Original Image", use_container_width=True)

    # Toggle for bounding boxes
    draw_boxes = st.checkbox("Show Bounding Boxes", value=True)

    # Run YOLO detection with the toggle setting
    detected_image = detect_objects(image, draw_boxes=draw_boxes)

    # Display the processed image
    st.image(detected_image, caption="Processed Image", use_container_width=True)
