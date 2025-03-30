import streamlit as st

main_page = st.Page("main.py", title="Main Page")
ob_page = st.Page("pages/ob_page.py", title="Object Detection")
cap_page = st.Page("pages/cap_page.py", title = "Captioning")
crop_page = st.Page("pages/crop_page.py", title= "Cropping")
geo_page = st.Page("geo_page.py", title = "Georeferencing")
fin_page = st.Page("pages/fin_page.py", title = "Final Output")

pg = st.navigation([main_page, ob_page, cap_page, crop_page, geo_page, fin_page])

# Run the selected page
pg.run()
