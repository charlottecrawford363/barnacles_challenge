# Barnacles Challenge

Detecting, counting and measuring barnacles in photos using classical image processing and computer vision with OpenCV.

## Approach

I compared several contouring methods, primarily on `unseen_img2.png` and `img2_cropped.png`:

1. **Contour mapping.** Intensity contours over a blurred image. Clustered barnacles blended together, so individual barnacles were very difficult to identify.
2. **Threshold-based contour detection.** Thresholding followed by `findContours`. This worked reasonably on `unseen_img2.png` but produced a lot of false positives on `img2_cropped.png`, which has a larger background area. Background removal and single color channels (using a Gaussian blur) gave little to no improvement.
3. **Canny edge maps.** A bilateral filter was used to reduce background noise while keeping edges, then contours were found either after morphological closing or with the Hough Circle Transform. Hough's circles method particularly gave far fewer false positives than the threshold methods for both test images.

Hough circles worked best overall, so `canny_map_hough_circles` was created to count barnacles and compute their average radius, converted to millimeters when the real side length of the image is known. The full process, results, conclusions and thoughts on learning are detailed in the notebook.

## Files

| File | Description |
|---|---|
| `barnacles_challenge.ipynb` | Main notebook: walks through each method with plots, conclusions and learning reflections. |
| `make_contour_map.py` | Plots intensity contour lines over a blurred grayscale image. |
| `contour_detection.py` | Threshold-based methods: fixed threshold, background removal, and adaptive threshold on a single color channel. |
| `canny_map_contouring.py` | Canny edge map methods: morphological closing + contours, Hough circles, and `canny_map_hough_circles` for counting and average radius. |
| `draw_contours.py` | Draws circles, ellipses or convex hulls around contours in a valid size range and returns the count and average radius. |
| `load_image.py` | Loads an image and raises a clear error if the file can't be read. |
| `barnacles/` | Image folder for testing. |

## How to run

Requires Python 3 with OpenCV, NumPy, Matplotlib and Jupyter:

```
pip install opencv-python numpy matplotlib jupyter
```

To reproduce the results, open the notebook `barnacles_challenge.ipynb` from the repository root and run all cells.

## Summary of results

Hough circles produced the fewest false positives, but still missed many barnacles and gave some false positives in tight clusters. Parameters were tuned by hand for these images, and performance dropped on `unseen_img1_cropped.png` in darker, shadowed areas. Other possible methods and details are in the notebook.