import streamlit as st

main_page = st.Page("main.py", title="Main Page")
ob_page = st.Page("pages/ob_page.py", title="Object Detection")
cap_page = st.Page("pages/cap_page.py", title = "Captioning")
crop_page = st.Page("pages/crop_page.py", title= "Cropping")
geo_page = st.Page("geo_page.py", title = "Georeferencing")
fin_page = st.Page("pages/fin_page.py", title = "Final Output")




# Edit : adding a test example page for trying to use Cropping tool
# crop2_page = st.Page("pages/crop2_page.py", title = "Cropping Tool Examples")

crop3_page = st.Page("pages/crop3_page.py", title = "Cropping using Uploaded Images")
crop_nococo_page = st.Page("pages/crop_nococo_page.py", title = "Cropping without JSON")

# Note: does not have the implementation yet to account for cropping an image selected yet. 

# Edit : adding a test object detection page for utilizing the YOLO model we have developed
ob2_page = st.Page("pages/ob2_page.py", title = "Object Detection (example)")
# ob_nococo_page = st.Page("pages/ob_nococo_page.py", title = "Object Detection No COCO")

# cap2_page = st.Page("pages/cap2_page.py", title = "Captioning in Streamlit (PaliGemma)")

ob_batch = st.Page("pages/ob_batch_page.py", title= "Batch Object Detection")

pg = st.navigation([main_page, ob_page, cap_page, crop_page, geo_page, fin_page, ob2_page, crop_nococo_page, ob_batch])

# Run the selected page
pg.run()
