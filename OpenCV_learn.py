
# https://www.bilibili.com/video/BV1eR4y1R7PP/
# 【唐博士带你学AI】唐宇迪 OpenCV

import cv2
import matplotlib.pyplot as plt
import numpy as np

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)

def img_color(img, color = 'o'):
    if color == 'o':
        return img
    cur_img = img.copy()
    color_item = ['B', 'G', 'R']
    for i in range(3):
        if color != color_item[i]:
            cur_img[:, :, i]= 0
    return cur_img

img = cv2.imread("cat.jpg")

input()
#4 filling
top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)

replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType = cv2.BORDER_REPLICATE)

plt.subplots(231), plt.imshow(img, 'gray'), plt.title('original')
plt.subplots(232), plt.imshow(replicate, 'gray'), plt.title('replicate')
#plt.subplots(1), plt.imshow(img, 'gray'), plt.title('original')
#plt.subplots(1), plt.imshow(img, 'gray'), plt.title('original')
#plt.subplots(1), plt.imshow(img, 'gray'), plt.title('original')
#plt.subplots(1), plt.imshow(img, 'gray'), plt.title('original')
plt.show()

#3 ROI
b, g, r = cv2.split(img)
print(b.shape)

img = img[0:200, 0:200]
#('img_200', img)

cv_show('img_red', img_color(img, 'G'))

#2 video
vc = cv2.VideoCapture('1.mp4')

if vc.isOpened():
    open_vc, frame = vc.read()
else:
    open_vc = False

while open_vc:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('result', gray)
        if cv2.waitKey(2) & 0xFF == 27:
            break

vc.release()
cv2.destroyAllWindows()

#1 img
print(img.shape)

img = cv2.imread("1_1.jpg", cv2.IMREAD_GRAYSCALE)

cv_show('1_1.jpg', img)
cv2.imwrite('1_gray.png', img)



