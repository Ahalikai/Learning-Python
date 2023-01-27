
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
img_1 = cv2.imread('1_1.jpg')

#input()

#5 image compute
img_1 = cv2.resize(img_1, (img.shape[0], img.shape[1]))
#cv_show('1_1', img_1)
img_u = cv2.addWeighted(img, 0.6, img_1, 0.4, 0)
#cv_show('union', img_u)
img_fxn = cv2.resize(img, (0, 0), fx = 2, fy = 1)
#cv_show('fxn', img_fxn)

img = img[:5, :5, 0]
print(img)
print(img + 10)

#4 filling
top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)

# REPLICATE： 复制最边缘上的一个点，所有的维度都使用当前的点
#REPLICATE = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REPLICATE)
REFLECT = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT)
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType = cv2.BORDER_REPLICATE)
REFLECT_101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
WRAP = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
CONST = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value = 255)

plt.subplot(231), plt.imshow(img), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(REFLECT), plt.title('REFLECT')
plt.subplot(234), plt.imshow(REFLECT_101), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(WRAP), plt.title('WRAP')
plt.subplot(236), plt.imshow(CONST), plt.title('CONST')
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



