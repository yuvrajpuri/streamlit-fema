from datetime import datetime
from PIL.ExifTags import TAGS

CATEGORY_MAP = {
    "Affected_Building": 1,
    "Major_Damage": 2
}

def get_date_captured(image):
    try:
        exif = image._getexif()
        if exif:
            for tag, val in exif.items():
                if tag == 36867:
                    return datetime.strptime(val, "%Y:%m:%d %H:%M:%S").isoformat()
    except:
        pass
    return datetime.now().isoformat()
