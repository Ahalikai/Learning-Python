
# 对视频 进行畸变矫正
import cv2 as cv
import numpy as np
import glob

def undistort(frame):

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

images = glob.glob('E:/Matrix/262camera/32/*.jpg')  #   拍摄的十几张棋盘图片所在目录
for fname in images:
	print(fname)
	img = cv.imread(fname)
	#target_img = cv.resize(img, (1280, 960))
	#cv.imshow('img', img)
	cv.imshow('img', undistort(img))
	cv.waitKey(0)
cv.destroyAllWindows()
