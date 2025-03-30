# Object detection page
import streamlit as st
import os
from PIL import Image
import pandas as pd


st.title("Object Detection Demo")

st.write("""
         Here is where we show an example of the Object detection - e.g. an original image and the bounding boxes detected in it.
         
         In this example, I will merely show an image from the previous group and the bounding box label from the sample images inference folder.
         
         """)

BASE_DIR = os.path.abspath(os.getcwd())
img_folder = os.path.join(BASE_DIR, "sample_images_for_inference")

if not os.path.exists(img_folder):
    st.error(f"Error: The folder '{img_folder}' does not exist.")
    st.stop()

img_files = [f for f in os.listdir(img_folder) if f.endswith(".jpg")]

if not img_files:
    st.error("No images found in the folder.")
    st.stop()

# Dropdown selection to choose an image
select = st.selectbox("Choose an image:", img_files)

# Define paths
img_path = os.path.join(img_folder, select)
bbox_path = os.path.join(img_folder, select.replace(".jpg", ".txt"))

# Display the selected image
st.image(Image.open(img_path), caption="Chosen Image", use_container_width=True)

# Function to read bounding box data from a text file
def load_bounding_boxes(txt_file):
    try:
        # Load data, skip the first column, and rename the remaining columns
        df = pd.read_csv(txt_file, sep=" ", header=None, usecols=[1, 2, 3, 4], names=["val_1", "val_2", "val_3", "val_4"])
        return df
    except Exception as e:
        st.error(f"Error reading {txt_file}: {e}")
        return pd.DataFrame(columns=["x_min", "y_min", "width", "height"])


# Check if bounding box data file exists
if os.path.exists(bbox_path):
    bboxes = load_bounding_boxes(bbox_path)
    st.write("Bounding Box Data:")
    st.dataframe(bboxes)
else:
    st.warning("No bounding box data found for this image.")
