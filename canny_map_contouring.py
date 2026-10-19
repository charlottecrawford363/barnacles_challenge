"""
Detecting barnacles with a Canny edge map with morphological
closing and contours or Hough circle detection

- I used this source to get a basic understanding of how to use canny edge detection: 
https://docs.opencv.org/4.13.0/da/d22/tutorial_py_canny.html
- I referenced some code in here to determine how to use the morphing technique:
https://www.scaler.com/topics/contour-analysis-opencv/ 
- The code for Hough circles is based on: https://docs.opencv.org/3.4.20/da/d53/tutorial_py_houghcircles.html
with changes to the parameters and layout.
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from draw_contours import draw_contours
from load_image import load_image

def canny_map_contouring(img_path, method):
    """
    Detect barnacles from a Canny edge map and plot results.
 
    Args:
        img_path (str): Path to the original image.
        method (str): "morphology" to close edges and draw circles around
            contours, or "hough_circles" to detect circles with Hough Circles.
    """

    img = load_image(img_path)
    if method not in ("morphology", "hough_circles"):
        raise ValueError(f'method must be "morphology" or "hough_circles", got {method!r}')
    gray_image=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # bilateral filtering to blur image to get rid of some of the background noise
    bilateral = cv.bilateralFilter(img, 15, 160, 200)  
    bilateral_gray = cv.cvtColor(bilateral, cv.COLOR_BGR2GRAY) 

    # Canny edge map  
    canny_map = cv.Canny(image=bilateral_gray, threshold1=120, threshold2=180, L2gradient=True)

    if (method == "morphology"):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
        canny_map = cv.morphologyEx(canny_map, cv.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv.findContours(canny_map, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contour_count, avg_radius = draw_contours(img, contours, "circle")
            
    
    if (method == "hough_circles"):
        circles = cv.HoughCircles(
            bilateral_gray,
            cv.HOUGH_GRADIENT,
            dp=1,
            minDist=30,      
            param1=100,         
            param2=30,       
            minRadius=10,       
            maxRadius=45
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))      
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]          
                
                cv.circle(img, center, radius, (255, 0, 0), 2)
                cv.circle(img, center, 2, (0, 0, 255), 3)

    f, axarr = plt.subplots(1, 3, constrained_layout=True)
    axarr[0].imshow(bilateral_gray)
    axarr[0].set_title("Bilateral Grayscale Image")
    axarr[1].imshow(canny_map)
    axarr[1].set_title("Canny Map")
    axarr[2].imshow(img)
    axarr[2].set_title("Image w/ Contours")

    plt.show()


def canny_map_hough_circles(img_path, side_mm=None):
    """
    Detect barnacles from a Canny edge map using Hough's circles and plot results
    as well as # of barnacles and average radius of the barnacles (contours).
 
    Args:
        img_path (str): Path to the original image.
        side_mm (float): side length of image in millimeters (assumes image is square)
    """

    img = load_image(img_path)
    gray_image=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # bilateral filtering to blur image to get rid of some of the background noise
    bilateral = cv.bilateralFilter(img, 15, 160, 200)  
    bilateral_gray = cv.cvtColor(bilateral, cv.COLOR_BGR2GRAY) 

    # Canny edge map  
    canny_map = cv.Canny(image=bilateral_gray, threshold1=120, threshold2=180, L2gradient=True)
    
    circles = cv.HoughCircles(
        bilateral_gray,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=30,      
        param1=100,         
        param2=30,       
        minRadius=10,       
        maxRadius=45
    )

    contour_count = 0
    avg_radius = 0
    if circles is not None:
        # Convert the coordinates and radii to integers
        circles = np.uint16(np.around(circles))
        
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]          
            
            # Draw the outer circle outline (in green)
            cv.circle(img, center, radius, (255, 0, 0), 2)
            # Draw the center point of the shape (in red)
            cv.circle(img, center, 2, (0, 0, 255), 3)

            avg_radius += radius
            contour_count += 1

        avg_radius = avg_radius / contour_count if contour_count else 0

    # convert avg_radius from pixels to mm
    if side_mm is not None:
        h, w = img.shape[:2]
        mm_per_px = side_mm / ((w + h) / 2)   # average of both sides, in case the crop isn't perfectly square
        avg_radius_mm = avg_radius * mm_per_px
        stats = f"{contour_count} barnacles, avg radius {avg_radius_mm:.2f} mm"
    else:
        stats = f"{contour_count} barnacles, avg radius {avg_radius:.1f} px"

    f, ax = plt.subplots(1, 1, constrained_layout=True)
    ax.imshow(img)
    ax.set_title(f"Image w/ Contours\n{stats}")

    plt.show()