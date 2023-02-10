#import
from imutils import contours
import numpy as np
import argparse
import cv2
import Ahalk_utils


# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', default='images/credit_card_01.png', help='path to input image')
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
gradX = gradX.astype('uint8')

print(np.array(gradX).shape)

# 闭操作
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh1 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

# 轮廓计算
threshCnts, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img, cnts, -1, (0,0,255), 3)
cv_show('result', cur_img)

locs = []
# 轮廓筛选
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    if ar > 2.5 and ar < 4.0:
        if(w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x, y, w, h))
locs = sorted(locs, key=lambda x:x[0])
output = []

# 遍历轮廓中的数字
for (i, (gx, gy, gw, gh)) in enumerate(locs):
    groupOutput = []
    # 根据坐标提取每一个组
    group = image_gray[gy - 5: gy + gh + 5, gx - 5 : gx + gw + 5]
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group', group)
    # 框出每一组
    digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
    #计算每一组的每一个数值
    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y : y + h, x : x + w]
        roi = cv2.resize(roi, (57, 88))
        cv_show('roi', roi)

        scores = []

        for (i, digitROI) in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _,) = cv2.minMaxLoc(result)
            scores.append(score)

        groupOutput.append(str(np.argmax(scores)))


    # 画出结果
    cv2.rectangle(image, (gx - 5, gy - 5),
                  (gx + gw + 5, gy + gh + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gx, gy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # 得到结果
    output.extend(groupOutput)

# print result
print("Credit Card Type: {}".format(FRIST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv_show("Image", image)
