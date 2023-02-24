 #hello

import cv2
import numpy as np

img = cv2.imread("images/ocr_a_reference.png")
print("img shape: ", img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst = cv2.cornerHarris(gray, 2, 3, 0.04)
print("dst shape: ", dst.shape)

img[dst > 0.1 * dst.max()] = [0, 0, 255]
cv2.imshow("result", img)
cv2.waitKey(0)