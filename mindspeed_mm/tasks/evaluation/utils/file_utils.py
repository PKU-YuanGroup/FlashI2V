import base64
import csv
import io
import mimetypes
import os
import pickle
from multiprocessing import Lock
from PIL import Image

lock = Lock()


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pkl(data, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def is_valid_image(image_path):
    if not os.path.exists(image_path):
        return False
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        if width <= 0 or height <= 0:
            raise ValueError('Image dimensions are invalid')
        return True
    except Exception as e:
        print(f"Error reading image '{image_path}': {e}")
        return False


def parse_file(s):
    if os.path.exists(s) and s != '.':
        if not os.path.isfile(s):
            raise ValueError(f'{s} is not a file')

        suffix = os.path.splitext(s)[1].lower()
        mime = mimetypes.types_map.get(suffix, 'unknown')
        return (mime, s)
    else:
        return (None, s)


def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    with lock:
        image = decode_base64_to_image(base64_string, target_size=target_size)
        image.save(image_path)


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    with Image.open(io.BytesIO(image_data)) as image:
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        if target_size > 0:
            image.thumbnail((target_size, target_size))
        return image.copy()


def save_csv(data, f, quoting=csv.QUOTE_ALL):
    data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)
