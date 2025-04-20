import zipfile
import json
import io
from io import BytesIO
import os

# when making an annotation, clean it of the filetype (e.g. not img1.jpg_annotations but img1_annotations)
def clean_annotation(filename):
    return os.path.splitext(filename)[0]

# Something for making the actual zip file?
