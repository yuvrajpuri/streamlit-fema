# Cropping Example page
import os
import pandas as pd
import cv2
import json
import sys
import streamlit as st
import re
from PIL import Image, ImageDraw

st.header("Cropping Tool Example")
st.write("Here is where we will showcase the cropping tool developed by another teammate.")

# Part 1: Initial variables and Methods

# Variables

# Base directory
source_base_dir = "/content/streamlit-fema/Final Project Streamlit Demo/pages/source"  # Base directory for source data

# JSON file name
json_file_name = "_annotations.coco.json"

# Function to process a single directory
def process_directory(source_dir):
    json_path = os.path.join(source_dir, json_file_name)
    output_json_path = json_path  # Save the updated JSON in the same directory

    # Read the original COCO JSON file
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    # Get list of images in the source directory (long names)
    image_files = {img for img in os.listdir(source_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))}

    # Create a mapping dictionary long names -> short names
    name_mapping = {}

    for img_name in image_files:
        # Extract the main identifier from the short name (example: "image_00003")
        base_name_match = re.match(r"(image_\d+)", img_name)
        if base_name_match:
            base_name = base_name_match.group(1) + ".jpg"  # Desired short format
            name_mapping[img_name] = base_name

    # Rename files in the source directory
    for old_name, new_name in name_mapping.items():
        old_path = os.path.join(source_dir, old_name)
        new_path = os.path.join(source_dir, new_name)
        os.rename(old_path, new_path)

    # Filter images and update names in the JSON
    filtered_images = []
    for img in coco_data["images"]:
        if img["file_name"] in name_mapping:
            img["file_name"] = name_mapping[img["file_name"]]  # Replace with the short name
            filtered_images.append(img)

    # Get the IDs of the selected images
    selected_image_ids = {img["id"] for img in filtered_images}

    # Filter annotations related to the selected images
    filtered_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in selected_image_ids]

    # Create the new JSON with corrected names
    filtered_coco = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": coco_data["categories"]  # Keep original categories
    }

    # Save the new JSON in the same directory
    with open(output_json_path, "w") as f:
        json.dump(filtered_coco, f, indent=4)

    print(f"✅ Files renamed and new JSON saved at: {output_json_path}")


# Processing images
# Function to crop and save images
def crop_and_save(image_path, bbox, output_path):
    image = Image.open(image_path)
    left, upper, width, height = bbox
    right = left + width
    lower = upper + height
    cropped_image = image.crop((left, upper, right, lower))
    cropped_image.save(output_path)

# Function to draw bounding boxes on the image and save it
def draw_bounding_boxes(image_path, bboxes, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        left, upper, width, height = bbox
        right = left + width
        lower = upper + height
        draw.rectangle([left, upper, right, lower], outline="red", width=2)
    image.save(output_path)

# Function to process a single directory
def process_directory2(source_dir):
    json_path = os.path.join(source_dir, json_file_name)
    processed_dir = os.path.join(source_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Load annotations from the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create a dictionary to map image_id to file_name
    image_id_to_filename = {image['id']: image['file_name'] for image in data['images']}

    # Dictionary to group bounding boxes by image_id
    image_bboxes = {}

    # Iterate over annotations and group bounding boxes by image_id
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        if image_id not in image_bboxes:
            image_bboxes[image_id] = []
        image_bboxes[image_id].append(bbox)

    # Make a list to contain the whole processed images
    processed_images = []
    
    # Iterate over images and process them
    for image_id, bboxes in image_bboxes.items():
        image_filename = image_id_to_filename[image_id]
        image_path = os.path.join(source_dir, image_filename)
        
        # Save the image with bounding boxes drawn
        output_path_with_boxes = os.path.join(processed_dir, f"{os.path.splitext(image_filename)[0]}_with_boxes.jpg")
        draw_bounding_boxes(image_path, bboxes, output_path_with_boxes)

        # Make a list to hold the cropped images
        cropped_pics = []
        
        # Crop and save each bounding box individually
        for i, bbox in enumerate(bboxes):
            output_path = os.path.join(processed_dir, f"{os.path.splitext(image_filename)[0]}_crop_{i}.jpg")
            crop_and_save(image_path, bbox, output_path)
            cropped_pics.append(output_path)

        # Make the final returnable list
        processed_images.append((image_filename, output_path_with_boxes, cropped_pics))

        #print(f"Processed {image_filename} with {len(bboxes)} bounding boxes")

    return processed_images

# Part 2: Processing and Cropping the Images
# Target only the 'test' subdirectory
test_dir = os.path.join(source_base_dir, "test")

# Check if 'test' directory exists before processing
if os.path.isdir(test_dir):
    process_directory(test_dir)
else:
    print(f"Test directory not found: {test_dir}")

# Crop
test_processed_pics = process_directory2(test_dir)

if test_processed_pics:
    image_options = [img[0] for img in test_processed_pics]
    selected_image = st.selectbox("Select an image", image_options)

    for img_name, bbox_img, crops in test_processed_pics:
        if img_name == selected_image:
            st.image(bbox_img, caption="Original Image with Bounding Boxes", use_container_width=True)
            st.subheader("Cropped Objects:")
            cols = st.columns(min(len(crops), 4))  # Display up to 4 per row

            for i, crop in enumerate(crops):
                with cols[i % 4]:
                    st.image(crop, caption=f"Crop {i+1}", use_container_width=True)
else:
    st.warning("No processed images found.")


