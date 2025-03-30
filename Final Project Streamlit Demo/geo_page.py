# Georeferencing Page
import streamlit as st
from exif import Image
import os
import pandas as pd

st.title("Georeferencing Demo")

st.write("Here's an example of using the EXIF metadata to geolocate where the images were taken. Ideally, I could potentially include a temporal component and maybe even assign them to the disasters to filter them by disaster on the map.")
st.write("For now, that's a work in progress or will need more time to complete.")

# Convert EXIF coordinates to map coordinates (degrees, minutes, seconds to just degrees)
# South and West are negative; North and East are positive.
def dms_to_decimal(dms, ref):
    if dms is None:
        return None
    degrees, minutes, seconds = dms
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:  
        decimal = -decimal
    return decimal


# Function to extract EXIF GPS coordinates from an image
def get_gps_coordinates(image_path):
    with open(image_path, "rb") as img_file:
        image = Image(img_file)

        if image.has_exif and image.gps_latitude and image.gps_longitude:
            lat = dms_to_decimal(image.gps_latitude, image.gps_latitude_ref)
            lon = dms_to_decimal(image.gps_longitude, image.gps_longitude_ref)
            return lat, lon
    return None, None


image_folder = "sample_images_for_inference"

# Find all .jpg images in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

# Add the image data to a list then a DataFrame
data = []
for img in image_files:
    img_path = os.path.join(image_folder, img)
    lat, lon = get_gps_coordinates(img_path)
    if lat is not None and lon is not None:  # Only keep images with GPS data
        data.append({"Image": img, "LATITUDE": lat, "LONGITUDE": lon})
        
gps = pd.DataFrame(data)

st.write("Extracted GPS Coordinates:")
st.dataframe(gps)

# Plot the data on a streamlit map
if not gps.empty:
    st.map(gps[['LATITUDE', 'LONGITUDE']])
else:
    st.write("No images with GPS metadata found.")

