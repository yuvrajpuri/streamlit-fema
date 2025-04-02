# Cropping Example page
import os
import pandas as pd
import cv2
import json
import streamlit as st
from PIL import Image

st.header("Cropping Tool Example")
st.write("Here is where we will showcase the cropping tool developed by another teammate.")

import streamlit as st
import cv2
import json
import numpy as np
from PIL import Image

# File paths for the pre-existing image and JSON
IMAGE_PATH = "CSCI_E-599a-Bounding_Box_Cropping/source data/image_00191.png"
JSON_PATH = "CSCI_E-599a-Bounding_Box_Cropping/source data/image_00191.json"

def crop_bounding_boxes(image, annotations):
    """
    Crop bounding boxes from the given image based on JSON annotation data.
    
    :param image: Original image as a NumPy array.
    :param annotations: JSON data containing bounding boxes.
    :return: List of cropped images.
    """
    img_height, img_width, _ = image.shape
    cropped_images = []

    for i, bbox in enumerate(annotations["boxes"], start=1):
        # Convert coordinates to integers
        x, y, w, h = map(lambda v: round(float(v)), [bbox["x"], bbox["y"], bbox["width"], bbox["height"]])

        # Ensure bounding box is within image bounds
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = max(1, min(w, img_width - x))
        h = max(1, min(h, img_height - y))

        # Crop the bounding box
        cropped_img = image[y:y+h, x:x+w]
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB
        cropped_images.append(Image.fromarray(cropped_img))

    return cropped_images

# Streamlit UI
st.title("Bounding Box Cropping Tool")

# Load image
image = Image.open(IMAGE_PATH)
image_np = np.array(image)  # Convert to NumPy array for OpenCV

# Load JSON
with open(JSON_PATH, 'r') as f:
    annotations = json.load(f)

# Display original image
st.image(image, caption="Original Image", use_column_width=True)

# Crop images
cropped_images = crop_bounding_boxes(image_np, annotations)

# Display cropped images
st.subheader("Cropped Objects")
for idx, cropped_img in enumerate(cropped_images):
    st.image(cropped_img, caption=f"Cropped Object {idx+1}", use_container_width=False)
