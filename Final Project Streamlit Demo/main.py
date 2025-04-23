import streamlit as st
import pandas as pd
import numpy as np

# First action
st.set_page_config(page_title="FEMA Disaster Image Damage Detection", layout="centered", initial_sidebar_state="expanded")


st.title("Demo (of Full Demo)")
st.sidebar.success("Select a page to view the demonstration.")

st.write("This app demonstrates object detection, image cropping, captioning, and geolocation mapping on disaster images. And a final output.")
st.write("Still under development is a page to show or run our model and the details about how it compares or works without taking too much time / computation.")

st.markdown("# Main page")
st.sidebar.markdown("# Main page")
st.sidebar.write("*Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.*")

st.write(
    """
    Below this, you will see a number of hyperlinks to direct you to the previous bodies of research or groups that provided data that guided this project.
    
    **Example Research and Sources (final product may not reflect this):**
    - [Original LADIv2 Paper Done By MIT on arXiv](https://arxiv.org/abs/2406.02780)
    - [LADIv2 Repository on HuggingFace](https://huggingface.co/datasets/MITLL/LADI-v2-dataset)
    - [Civil Air Patrol website](https://www.gocivilairpatrol.com/)
    - [FEMA website](https://www.fema.gov/)
    
    """)
