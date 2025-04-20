from datetime import datetime
from PIL.ExifTags import TAGS

CATEGORY_MAP = {
    "Affected_Building": 1,
    "Major_Damage": 2
}

# if we remove the debugging, use this version
#def get_date_captured(image):
#    """
#    Extracts the most accurate capture datetime from EXIF metadata.
#    Falls back to current timestamp if unavailable.
#    """
#    try:
#        # Prefer modern getexif(), fallback to older _getexif()
#        exif = image.getexif() if hasattr(image, "getexif") else image._getexif()
#        if exif:
#            # Priority order for EXIF date fields
#            for tag, val in exif.items():
#                tag_name = TAGS.get(tag, str(tag))
#                if tag_name in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
#                    try:
#                        return datetime.strptime(val, "%Y:%m:%d %H:%M:%S").isoformat()
#                    except Exception:
#                        pass  # Ignore parse errors
#    except Exception:
#        pass

    # Fallback to current datetime
#    return datetime.now().isoformat()

def get_date_captured(image, verbose=False):
    """
    Extracts date from EXIF if available. Tries .getexif() first, then _getexif().
    """
    try:
        # Try the modern EXIF method first
        if hasattr(image, "getexif"):
            exif = image.getexif()
        else:
            exif = image._getexif()

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

    if verbose:
        print("📆 No valid EXIF date found, using current timestamp.")
    return datetime.now().isoformat()
# def build_coco_json?
