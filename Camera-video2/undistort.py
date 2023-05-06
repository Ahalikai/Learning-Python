
# 对视频 进行畸变矫正
import cv2 as cv
import numpy as np
import glob

def undistort1(frame):

	k=np.array( [[408.96873567 ,  0.         ,329.01126845],
 [  0.       ,  409.20308599 ,244.73617469],
 [  0.       ,    0.       ,    1.        ]])

	k = np.array([[1184.52534766571, 0, 974.429294015024],
				  [0., 1184.13524856527,475.933442967284],
				  [0., 0., 1.]])

	# d=np.array([-0.33880708 , 0.16416173 ,-0.00039069 ,-0.00056267 ,-0.056967  ])
	d = np.array([-0.293031197659429, 0.0625952472919022,0,0,0 ])
	h,w=frame.shape[:2]
	mapx,mapy=cv.initUndistortRectifyMap(k,d,None,k,(w,h),5)
	return cv.remap(frame,mapx,mapy,cv.INTER_LINEAR)

# cap=cv.VideoCapture(0)# 换成要打开的摄像头编号
# ret,frame=cap.read()
# while ret:
# 	cv.imshow('later',frame)
# 	cv.imshow('img',undistort(frame))
# 	ret,frame=cap.read()
# 	if cv.waitKey(1)&0xff==27:
# 		break
#
# cap.release()

def undistort(frame):
    # fx = 1229.4
    # cx = 634.8157
    # fy = 1229.5
    # cy = 482.9607
    # k1, k2, p1, p2, k3 = -0.4310, 0.2038, 0, 0, 0    #ipcam 12

    # 相机坐标系到像素坐标系的转换矩阵
    k = np.array([
		[1184.52534766571, 0., 974.429294015024],
		[0., 1184.13524856527, 475.933442967284],
		[0., 0., 1.]
        #ipcam 12
        # [1229.4, -0.2041, 634.8157],
        #    [0.000000, 1229.5, 482.9607],
        #     [0.000000, 0.000000, 1.000000]
            # [450.43735, 0, 321.5635],
            # [0.000000, 450.149, 179.50135],
            # [0.000000, 0.000000, 1.000000]
    ])
    # 畸变系数
    d = np.array([
        # -0.3665, 0.1389, -2.9633e-04, -3.2308e-04, 0
        # -0.4310, 0.2038, 0, 0, 0    #ipcam12
        -0.293031197659429, 0.0625952472919022, 0, 0, 0
    ])
    h, w = frame.shape[:2]
    adjusted_k, roi = cv.getOptimalNewCameraMatrix(k, d, (w, h), 0, (w, h), 1)
    mapx, mapy = cv.initUndistortRectifyMap(k, d, None, adjusted_k, (w, h), 5)
    return cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)

images = glob.glob('E:/Matrix/262camera/32/*.jpg')  #   拍摄的十几张棋盘图片所在目录
for fname in images:
	print(fname)
	img = cv.imread(fname)
	#target_img = cv.resize(img, (1280, 960))
	#cv.imshow('img', img)
	cv.imshow('img', undistort(img))
	cv.imwrite('undistort.jpg', undistort(img))
	cv.waitKey(0)
cv.destroyAllWindows()
