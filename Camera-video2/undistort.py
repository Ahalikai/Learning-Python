
# 对视频 进行畸变矫正
import cv2 as cv
import numpy as np
import glob

def undistort(frame):

	k=np.array( [[408.96873567 ,  0.         ,329.01126845],
 [  0.       ,  409.20308599 ,244.73617469],
 [  0.       ,    0.       ,    1.        ]])

	k = np.array([[1.31688101e+03, 0.00000000e+00, 9.66128276e+02],
 [0.00000000e+00,1.31720898e+03,5.34038509e+02],
 [0.00000000e+00,0.00000000e+00,1.00000000e+00]])

	# d=np.array([-0.33880708 , 0.16416173 ,-0.00039069 ,-0.00056267 ,-0.056967  ])
	d = np.array([-3.56536017e-01, 2.12249787e-01,-1.24758904e-04, 9.01910398e-04,-9.52484138e-02 ])
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

images = glob.glob('1448camera/*.jpg')  #   拍摄的十几张棋盘图片所在目录
for fname in images:
	print(fname)
	img = cv.imread(fname)
	target_img = cv.resize(img, (1920, 1080))
	#cv.imshow('img', img)
	cv.imshow('img', undistort(target_img))
	cv.waitKey(0)
cv.destroyAllWindows()
