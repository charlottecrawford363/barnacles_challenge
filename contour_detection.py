"""
Barnacle detection using thresholding and contour finding.

Experimenting w/ different image thresholding: https://docs.opencv.org/3.4.20/d7/d4d/tutorial_py_thresholding.html
Starting w/ using this to remove background: https://www.geeksforgeeks.org/python/removing-black-background-and-make-transparent-using-python-opencv/
Using single color channel: https://learnopencv.com/contour-detection-using-opencv-python-c/#steps-for-detecting-contours 
"""

import cv2 as cv
import matplotlib.pyplot as plt
from draw_contours import draw_contours
from load_image import load_image

# set plot title font size
plt.rcParams['axes.titlesize'] = 8


def contour_detection_v1(img_path, shape="circle"):
    """
    Threshold the grayscale image at a fixed value and draw contours.
  
    Args:
        img_path (str): Path to the original image.
        shape (str): Shape of contour: "circle", "ellipse", or "hull".
    """

    # read image and convert it to grayscale and RGB 
    image = load_image(img_path)
    gray_image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    rgb_image=cv.cvtColor(image,cv.COLOR_BGR2RGB)

    # create binary image 
    _, binary_image= cv.threshold(gray_image, 100, 255, cv.THRESH_TOZERO)
        
    # find contours
    contours, _ = cv.findContours(binary_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # draw contours onto image
    contour_count, avg_radius = draw_contours(rgb_image, contours, shape)    

    f, axarr = plt.subplots(1, 3, constrained_layout=True)
    axarr[0].imshow(gray_image)
    axarr[0].set_title("Grayscale Image")
    axarr[1].imshow(binary_image)
    axarr[1].set_title("Binary Image")
    axarr[2].imshow(rgb_image)
    axarr[2].set_title("Color Image w/ Contour")

    plt.show()


def contour_detection_channel_v1(img_path, shape="circle"):
    """
    Add an Otsu mask for threshold of grayscale image, adaptive-threshold the result,
    and draw contours.
 
    Args:
        img_path (str): Path to the original image.
        shape (str): Shape of contour: "circle", "ellipse", or "hull".
    """
    
    # read image and convert it to grayscale and RGB 
    image = load_image(img_path)
    gray_image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    rgb_image=cv.cvtColor(image,cv.COLOR_BGR2RGB)

    # create binary image
    _, binary_image= cv.threshold(gray_image, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # remove background       
    b, g, r = cv.split(image)
    rgba = [b, g, r, binary_image]
    dst = cv.merge(rgba, 4)        

    gray = cv.cvtColor(dst, cv.COLOR_BGRA2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)    

    contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    contour_count, avg_radius = draw_contours(rgb_image, contours, shape)           

    f, axarr = plt.subplots(1, 4, constrained_layout=True)
    axarr[0].imshow(gray_image)
    axarr[0].set_title("Grayscale Image")
    axarr[1].imshow(dst)
    axarr[1].set_title("Binary w/out bg")
    axarr[2].imshow(binary)
    axarr[2].set_title("Binary Grayscale w/ Channel")
    axarr[3].imshow(rgb_image)
    axarr[3].set_title("Color Image w/ Contour")
    
    plt.show()

def contour_detection_channel(img_path, channel, shape="circle"):
    """
    Adaptive-threshold a single color channel, mask out the background, and draw contours.
 
    Args:
        img_path (str): Path to the original image.
        channel (int): Color channel to use: 0 = blue, 1 = green, 2 = red.
        shape (str): Shape of contour: "circle", "ellipse", or "hull".
    """
    
    # read image and convert it to grayscale and RGB 
    image = load_image(img_path)
    gray_image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    rgb_image=cv.cvtColor(image,cv.COLOR_BGR2RGB)

    # create binary image from Gaussian blur
    blur = cv.GaussianBlur(gray_image, (5,5), 0)
    _, binary_image= cv.threshold(blur, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # color channel
    if channel not in (0, 1, 2):
        raise ValueError(f"channel must be 0 (blue), 1 (green), or 2 (red), got {channel!r}")
    channel_image = image[:, :, channel]    # 0,1,2 : B,G,R    
    binary = cv.adaptiveThreshold(channel_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 3)
    binary = cv.bitwise_and(binary, binary_image)

    contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    contour_count, avg_radius = draw_contours(rgb_image, contours, shape)
    
    f, axarr = plt.subplots(1, 3, constrained_layout=True)
    axarr[0].imshow(gray_image)
    axarr[0].set_title("Grayscale Image")
    axarr[1].imshow(binary)
    axarr[1].set_title("Binary Grayscale w/ Channel")
    axarr[2].imshow(rgb_image)
    axarr[2].set_title("Color Image w/ Contour")

    plt.show()
