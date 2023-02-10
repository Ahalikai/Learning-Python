# import
from PIL import Image
import numpy as np
import argparse
import cv2
import os
import pytesseract

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="images/page.jpg", required=False, help="input a image")
args = vars(ap.parse_args())


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    # 0123 -> 左上， 右上， 右下， 左下
    # 左上 右下
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 右上 左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

# 透视变换-函数
def four_point_transform(image, pts):
    rect = order_points(pts)
    (A, B, C, D) = rect

    # 两点距离 求长度
    width_up = np.sqrt(((A[0] - B[0]) ** 2) + ((A[1] - B[1]) ** 2))
    width_low = np.sqrt( ( (C[0] - D[0]) ** 2 ) + ( (C[1] - D[1]) ** 2 ) )
    maxWidth = max(width_up, width_low)
    hight_left = np.sqrt(((A[0] - D[0]) ** 2) + ((A[1] - D[1]) ** 2))
    hight_right = np.sqrt(((B[0] - C[0]) ** 2) + ((B[1] - C[1]) ** 2))
    maxHight = max(hight_left, hight_right)

    # 变化后的坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHight - 1],
        [0, maxHight - 1]
    ], dtype="float32")

    # 计算 变化矩阵M
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (int(maxWidth), int(maxHight)))

    return warped

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation= inter)
    return resized

# 输入图像
image = cv2.imread(args["image"])
# 坐标变化
ratio = image.shape[0] / 500.0
orig = image.copy()

image = resize(orig, height = 500)

# 预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

print("Step1: 边缘检测")
cv2.imshow("Image", image)
cv2.imshow("Canny", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

print("Step2: 轮廓检测")
cv2.drawContours(image, [screenCnt], -1, (0, 0, 255), 2)
cv2.imshow("OutLine", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 透视变换
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# 再处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
print("Step3: 透视变换")
cv2.imshow("Result", resize(ref, height=650))

# text
print("Step4: 文本检测")

text = pytesseract.image_to_string(ref)
print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()