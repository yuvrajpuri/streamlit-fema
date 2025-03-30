# Cropping Page
import streamlit as st
from PIL import Image
import os

st.title("Cropping Page")

st.write("""
         Not sure what to put here yet. 
         My first thought was to display an image (with or without bounding boxes?) and then next to / beneath it, show all the cropped bounding boxes sourced from it.
         For now, I'll just display a cropped image.
         """)



BASE_DIR = os.path.abspath(os.getcwd())
folder = os.path.join(BASE_DIR, "caption_dataset_example")

if not os.path.exists(folder):
    st.error(f"Error: The folder '{folder}' does not exist.")
    st.stop()

img = [f for f in os.listdir(folder) if f.endswith(".png")]

if not img:
    st.error("No images found in the folder.")
    st.stop()



select = st.selectbox("Choose a cropped image:", img)
img_path = os.path.join(folder, select)

# Display the cropped image
st.image(Image.open(img_path), caption="Cropped Image", use_container_width=True)


