from datetime import datetime
from PIL.ExifTags import TAGS

CATEGORY_MAP = {
    "Affected_Building": 1,
    "Major_Damage": 2
}

#def get_date_captured(image):
#    try:
#        exif = image._getexif()
#        if exif:
#            for tag, val in exif.items():
#                if tag == 36867:
#                    return datetime.strptime(val, "%Y:%m:%d %H:%M:%S").isoformat()
#    except:
#        pass
#    return datetime.now().isoformat()

def get_date_captured(image, verbose=True):
    """
    Attempts to extract the original capture date from image EXIF metadata.
    Falls back to current datetime if unavailable or invalid.
    
    Args:
        image (PIL.Image): The image to extract EXIF metadata from.
        verbose (bool): If True, print EXIF debug information.
    
    Returns:
        str: ISO 8601 formatted datetime string.
    """
    try:
        # Support both newer and older Pillow EXIF access
        exif = getattr(image, "getexif", image._getexif)()
        if exif:
            preferred_tags = ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]
            for tag, val in exif.items():
                tag_name = TAGS.get(tag, str(tag))
                if verbose:
                    print(f"EXIF → {tag_name}: {val}")
                if tag_name in preferred_tags:
                    try:
                        return datetime.strptime(val, "%Y:%m:%d %H:%M:%S").isoformat()
                    except Exception as e:
                        if verbose:
                            print(f"⛔ Failed parsing {tag_name}: {e}")
    except Exception as e:
        if verbose:
            print(f"⚠️ EXIF read error: {e}")

    # Fallback if no valid EXIF date is found
    if verbose:
        print("📆 No valid EXIF date found, using current timestamp.")
    return datetime.now().isoformat()

# def build_coco_json?
