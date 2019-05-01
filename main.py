from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

colored = cv.imread('sheets/TP5_AltoTest_600.png', cv.IMREAD_COLOR)
img = cv.cvtColor(colored, cv.COLOR_BGR2GRAY)
gray = cv.bitwise_not(img)
bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
horizontal = np.copy(bw)
vertical = np.copy(bw)

def show_wait_destroy(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

# Specify size on vertical axis
rows = vertical.shape[0]
verticalsize = rows/30
# Create structure element for extracting vertical lines through morphology operations
verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, int(verticalsize)))
# Apply morphology operations
vertical = cv.erode(vertical, verticalStructure)
vertical = cv.dilate(vertical, verticalStructure)
# Show extracted vertical lines
show_wait_destroy("vertical", vertical)

# Inverse vertical image
vertical = cv.bitwise_not(vertical)
show_wait_destroy("vertical_bit", vertical)
'''
Extract edges and smooth image according to the logic
1. extract edges
2. dilate(edges)
3. src.copyTo(smooth)
4. blur smooth img
5. smooth.copyTo(src, edges)
'''
# Step 1
edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                            cv.THRESH_BINARY, 3, -2)
show_wait_destroy("edges", edges)
# Step 2
kernel = np.ones((2, 2), np.uint8)
edges = cv.dilate(edges, kernel)
show_wait_destroy("dilate", edges)
# Step 3
smooth = np.copy(vertical)
# Step 4
smooth = cv.blur(smooth, (2, 2))
# Step 5
(rows, cols) = np.where(edges != 0)
vertical[rows, cols] = smooth[rows, cols]
# Show final result
show_wait_destroy("smooth - final", vertical)


