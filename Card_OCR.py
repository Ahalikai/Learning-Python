#import
from imutils import contours
import numpy as np
import argparse
import cv2
import Ahalk_utils


# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', default='images/credit_card_02.png', help='path to input image')
ap.add_argument('-t', '--template', default='images/ocr_a_reference.png', help='path to template')
args = vars(ap.parse_args())

# 卡的类别
FRIST_NUMBER = {
    '3' : 'Amercian Express',
    '4' : 'Visa',
    '5' : 'MasterCard',
    '6' : 'Discover Card'
}

# 显示图像
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#模板读取
img = cv2.imread(args['template'])
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray_inv = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)[1]

#模板轮廓
refCnts, hierarchy = cv2.findContours(img_gray_inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
cv_show('img', img)

refCnts = Ahalk_utils.sort_contours(refCnts)[0]

digits = {}

# 遍历每一个box
for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = img_gray_inv[y : y + h, x : x + w]
    roi = cv2.resize(roi, (57, 88))

    digits[i] = roi

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

#读入图像
image = cv2.imread(args['image'])
image = Ahalk_utils.resize(image, width=300)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#礼帽
tophat = cv2.morphologyEx(image_gray, cv2.MORPH_TOPHAT, rectKernel)

gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1) #ksize=-1 相当于3*3
gradX = np.absolute(gradX)

(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ( (gradX - minVal) / (maxVal - minVal) ))






















