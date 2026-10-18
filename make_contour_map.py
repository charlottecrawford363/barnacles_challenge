"""
Create a contour map of an image

Source: https://towardsdatascience.com/image-contouring-with-opencv-5b18f17a8386/
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from load_image import load_image

SMOOTHING_KERNEL = (15, 15)     # kernel size for Gaussian blur
NUM_CONTOURS = 4                # number of intensity levels to contour
ALPHA = 0.6                     # opacity of image under contours

def make_contour_map(img_path):
    """
    Plot intensity contour lines over a blurred grayscale of the image

    Args:
        img_path (str): path to original image
    """
    IMG_GRAY = load_image(img_path, cv.IMREAD_GRAYSCALE)

    # Smooth the image using Gaussian blur
    smoothed_img = cv.GaussianBlur(IMG_GRAY, SMOOTHING_KERNEL, sigmaX=0)

    fig, ax = plt.subplots()

    # Set the contour levels
    contour_levels = np.linspace(0, 255, NUM_CONTOURS)

    # Create the contour plot
    contour_plot = ax.contour(smoothed_img, levels=contour_levels, colors='black')

    # Add the smoothed image
    ax.imshow(smoothed_img, 
            cmap='gray', 
            alpha=ALPHA)

    ax.set_title('Smoothed Image with Contours')
    ax.set_axis_off()

    plt.show()

