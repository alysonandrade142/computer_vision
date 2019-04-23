from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

colored = cv.imread('sheets/TP23_SonatineI-2_600.png', cv.IMREAD_COLOR)
img = cv.cvtColor(colored, cv.COLOR_BGR2GRAY)

akaze = cv.AKAZE_create(threshold=0.001, nOctaves=4)
kp = akaze.detect(img)

print ("Total Keypoints: ", len(kp))
 
#Plotting the image with the more relevants keypoints with direction
image = cv.drawKeypoints(img, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('img', image)
cv.waitKey(0)
