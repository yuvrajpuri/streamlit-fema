# Captioning Page
import streamlit as st
from PIL import Image
import os
from striprtf.striprtf import rtf_to_text


st.title("Captioning Demo")

st.write("""
         Here is where we show examples of how the captioning occurs.
         
         For now, all I'll show is an example of the caption provided by Tim.
         
         """)

# Get the current working directory dynamically
BASE_DIR = os.path.abspath(os.getcwd())  # This sets the base directory dynamically

# Set folder for cropped images and caption text files
folder = os.path.join(BASE_DIR, "caption_dataset_example")

# Ensure folder exists
if not os.path.exists(folder):
    st.error(f"Error: The folder '{folder}' does not exist.")
    st.stop()

img = [f for f in os.listdir(folder) if f.endswith(".png")]

if not img:
    st.error("No images found in the folder.")
    st.stop()

select = st.selectbox("Choose a cropped image with a caption:", img)

img_path = os.path.join(folder, select)
cap_path = os.path.join(folder, select.replace(".png", ".txt"))

# Display the selected image
st.image(Image.open(img_path), caption="Cropped Image", use_container_width=True)

# Function to clean and extract caption from the RTF file
def extract_caption(rtf_file):
    try:
        with open(rtf_file, "r", encoding="utf-8") as file:
            rtf_content = file.read()
            # Convert RTF to plain text
            caption = rtf_to_text(rtf_content)
            return caption.strip()
    except Exception as e:
        st.error(f"Error reading caption from {rtf_file}: {e}")
        return "No caption found."

caption = extract_caption(cap_path)

# Display the caption
st.caption(caption)

