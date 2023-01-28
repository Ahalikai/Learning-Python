
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

img_1 = cv2.imread("cat.jpg")
img = cv2.imread('1_1.jpg')

#img = cv2.resize(img, (0, 0), fx = 0.3, fy = 0.3)

### Section 7

#7-5 & 7-7 Contours
draw_img = img.copy()
draw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(draw_img, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


cnt = contours[120]

epsilon = 0.1 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

draw_img = img.copy()
res = cv2.drawContours(draw_img, [approx], -1, (0, 255, 0), 2)

cv_show('Contours', res)
input()


#7-6 template
template = img[200:350, 50:200, :]
w, h, _ = template.shape
cv_show('small', template)
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img2 = img.copy()
    method = eval(meth)
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img2, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()

#7-4
#拉普拉斯金字塔
img_l = cv2.pyrDown(img)
img_l = cv2.pyrUp(img_l)
img_l = img - img_l
cv_show('lap', img_l)

#高斯金字塔
img_up = cv2.pyrUp(img)
cv_show('up', img_up)
img_down = cv2.pyrDown(img)
cv_show('down', img_down)

### Section 6
'''
S1 除噪
S2 算子求梯度
S3 NMS
S4 Double-Threshold
S5 抑制孤立的弱边缘
'''
v1 = cv2.Canny(img, 80, 150)
v2 = cv2.Canny(img, 50, 100)
result = np.vstack((v1, v2))
cv_show('result', result)

### Section 5 sobel scharr laplacian
#sobel
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 3)
sobelx = cv2.convertScaleAbs(sobelx)

sobely = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 3)
sobely = cv2.convertScaleAbs(sobely)

sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

#scharr
scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharrx = cv2.convertScaleAbs(scharrx)

scharry = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharry = cv2.convertScaleAbs(scharry)

scharr = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

#laplacian
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

result = np.vstack((sobel, scharr, laplacian))
cv_show('result', result)

### Section 4
kernal = np.ones((10,10), np.uint8)

dige_erosion = cv2.erode(img, kernal, iterations = 1)
dige_dilate = cv2.dilate(dige_erosion, kernal, iterations = 1)

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernal)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernal)

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernal)

tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernal)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernal)

res1 = np.hstack((dige_erosion, dige_dilate, opening, closing))
res2 = np.hstack((gradient, tophat, blackhat, img))
res = np.vstack((res1, res2))
cv_show('total', res)

### Section 3

#3-1 & 3-2 slide  blur guess mid
cv_show('Org', img)

blur = cv2.blur(img, (3, 3))
aussian = cv2.GaussianBlur(img, (5, 5), 1)
mid = cv2.medianBlur(img, 5)

res = np.hstack((blur, aussian, mid))
cv_show('total', res)

#cv_show('blur', blur)

#3-0 limit

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv_show('img_gray', img_gray)
ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Org', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img_gray, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(3, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

### Section 2

#2-5 image compute
img_1 = cv2.resize(img_1, (img.shape[0], img.shape[1]))
#cv_show('1_1', img_1)
img_u = cv2.addWeighted(img, 0.6, img_1, 0.4, 0)
#cv_show('union', img_u)
img_fxn = cv2.resize(img, (0, 0), fx = 2, fy = 1)
#cv_show('fxn', img_fxn)

img = img[:5, :5, 0]
print(img)
print(img + 10)

#2-4 filling
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

#2-3 ROI
b, g, r = cv2.split(img)
print(b.shape)

img = img[0:200, 0:200]
#('img_200', img)

cv_show('img_red', img_color(img, 'G'))

#2-2 video
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

#2-1 img
print(img.shape)

img = cv2.imread("1_1.jpg", cv2.IMREAD_GRAYSCALE)

cv_show('1_1.jpg', img)
cv2.imwrite('1_gray.png', img)

