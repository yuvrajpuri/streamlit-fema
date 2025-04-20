import streamlit as st

main_page = st.Page("main.py", title="Main Page")

ob2_page = st.Page("pages/ob2_page.py", title = "Object Detection (example)")

crop_nococo_page = st.Page("pages/crop_nococo_page.py", title = "Cropping without JSON")

ob_batch = st.Page("pages/ob_batch_page.py", title= "Batch Object Detection")

cap_page = st.Page("pages/cap_page.py", title = "Captioning (Video Demonstration)")

pg = st.navigation([main_page, ob2_page, crop_nococo_page, ob_batch, cap_page])

# Run the selected page
pg.run()


#ob_page = st.Page("pages/ob_page.py", title="Object Detection")
#crop_page = st.Page("pages/crop_page.py", title= "Cropping")
#crop2_page = st.Page("pages/crop2_page.py", title = "Cropping Tool Examples")
#crop3_page = st.Page("pages/crop3_page.py", title = "Cropping using Uploaded Images")



