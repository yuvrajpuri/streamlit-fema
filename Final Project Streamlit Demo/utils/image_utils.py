from PIL import ImageDraw
from io import BytesIO

def crop_bbox(image, bbox):
    x, y, w, h = bbox
    return image.crop((x, y, x + w, y + h))

def draw_bounding_boxes(image, detections):
    draw = ImageDraw.Draw(image)
    for det in detections:
        x, y, w, h = det["bbox"]
        color = "purple" if det["category_id"] == 1 else "yellow"
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
    return image

# Needs ds_bbox?
