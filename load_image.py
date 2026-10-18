"""
Helper function to load image and raise error if it can't be read
"""
import cv2 as cv

def load_image(img_path, flags=cv.IMREAD_COLOR):
    img = cv.imread(img_path, flags)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    return img