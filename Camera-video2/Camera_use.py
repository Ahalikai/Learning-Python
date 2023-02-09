

# python调用opencv相机拍照代码
import cv2
camera = cv2.VideoCapture(0)
i = 0
ret, img = camera.read()
print('输入j,下载当前图片')
print('输入q,终止程序')
while ret:

    cv2.imshow('img', img)
    ret, img = camera.read()

    if cv2.waitKey(1) & 0xFF == ord('j'):  # 按j保存一张图片
        i += 1
        firename = str('./img' + str(i) + '.jpg')
        cv2.imwrite(firename, img)
        print('写入：', firename)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.release()
cv2.destroyAllWindows()
