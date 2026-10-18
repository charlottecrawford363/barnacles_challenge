"""
Helper function for drawing the contour shapes

Source for the drawing of contours: https://medium.com/@siromermer/contour-detection-using-opencv-detecting-and-counting-coins-2d5192597e3c 
"""
import cv2 as cv

# min and max area used to remove invalid contours
MIN_AREA = 40
MAX_AREA = 400

def draw_contours(img, contours, shape):
    """
    Draws shapes around the contours 

    Args:
        img (np.ndarray): image to draw contours on
        contours (list): a list of contours from cv.findContours
        shape (str): "circle", "ellipsse", or "hull"

    Returns:
        tuple: (contour_count, avg_radius), the number of valid contours drawn
            and their average radius in pixels (only computed for shape type 
            "circle", else 0)
    """

    if shape not in ("circle", "ellipse", "hull"):
        raise ValueError(f'shape must be "circle", "ellipse", or "hull", got {shape!r}')
    
    contour_count = 0
    avg_radius = 0     
    for contour in contours:
        if (shape == "ellipse"):
            if len(contour) >= 5:
                ellipse = cv.fitEllipse(contour)
                if cv.contourArea(contour) > MIN_AREA and cv.contourArea(contour) < MAX_AREA:
                    cv.ellipse(img, ellipse, (255, 0, 0), 2)
                    contour_count += 1

        if cv.contourArea(contour) > MIN_AREA and cv.contourArea(contour) < MAX_AREA:
            if (shape == "circle"):
                (x,y),radius = cv.minEnclosingCircle(contour) 
                center = (int(x),int(y))
                radius = int(radius)
                cv.circle(img, center, radius,(255,0,0),2)
                contour_count += 1
                avg_radius += radius

            if (shape == "hull"):
                hull = cv.convexHull(contour)
                cv.drawContours(img, [hull], 0, (255, 0, 0), 2) 
                contour_count += 1  

    avg_radius = avg_radius / contour_count if contour_count else 0
    return contour_count, avg_radius