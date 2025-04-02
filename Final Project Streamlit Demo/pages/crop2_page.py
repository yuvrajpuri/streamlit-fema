# Cropping Example page
import os
import pandas as pd
import cv2
import json
import sys
import streamlit as st
from PIL import Image

st.header("Cropping Tool Example")
st.write("Here is where we will showcase the cropping tool developed by another teammate.")


# Add the "CSCI_E-599a-Bounding_Box_Cropping" folder to sys.path
os.chdir(os.path.join(os.getcwd(), "CSCI_E-599a-Bounding_Box_Cropping", "source data"))

# Import existing script
# FIXED: The fix is to rename the file for the test. 
import test_600x600  

# File paths (Assume pre-existing files)
# Current error: file not found. Updating the path.
# Error found: the file self calls the image paths within the local directory - i.e. "imagexxxx.png". Need to change the local directory first.
# Use the sys path append.
# UPDATE: move the test_600x600.py that was renamed to the source data directory.

print(os.getcwd())
IMAGE_PATH = "CSCI_E-599a-Bounding_Box_Cropping/source data/image_00191.png"
JSON_PATH = "CSCI_E-599a-Bounding_Box_Cropping/source data/image_00191.json"
OUTPUT_DIR = "cropped_objects"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Run the existing script automatically
st.title("Bounding Box Cropping Results")
st.subheader("Original Image")
st.image(IMAGE_PATH, caption="Original Image", use_column_width=True)

st.subheader("Running the Cropping Program...")

sys.path.append(os.path.join(os.getcwd(), "CSCI_E-599a-Bounding_Box_Cropping", "source data"))
test_600x600.crop_bounding_boxes(IMAGE_PATH, JSON_PATH, OUTPUT_DIR)

# Display cropped images
st.subheader("Cropped Objects")
cropped_files = sorted([os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")])

for file in cropped_files:
    st.image(file, caption=os.path.basename(file), use_container_width=False)

st.success("Processing Complete! All images displayed.")
