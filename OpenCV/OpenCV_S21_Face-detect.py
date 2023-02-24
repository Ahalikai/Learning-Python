
from collections import OrderedDict
import numpy as np
import argparse
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape", required=False,
                default="other/shape_predictor_68_face_landmarks.dat", help="path to landmark predictor")
ap.add_argument("-i", "--image", required=False,
                default="images/liudehua2.jpg", help="path to input image")
args = vars(ap.parse_args())

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

FACIAL_LANDMARKS_5_IDXS = OrderedDict([
	("right_eye", (2, 3)),
	("left_eye", (0, 1)),
	("nose", (4))
])

def resize_image(image, width = None, high = None):
	(h, w) = image.shape[:2]
	dim = (h, w)
	if(width is None and high is not None):
		r = high / float(h)
		dim = (int(w * r), high)
	elif(width is not None and high is None):
		r = width / float(w)
		dim = (width, int(h * r))
	return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def visualize_facial_landmarks(image, shape, colors = None, alpha = 0.75):
	overlay = image.copy()
	output = image.copy()
	# set colors
	if colors is None:
		colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220)]

	# for part
	for(i, name) in enumerate(FACIAL_LANDMARKS_68_IDXS.keys()):
		# get every point
		(j, k) = FACIAL_LANDMARKS_68_IDXS[name]
		pts = shape[j:k]
		#check station
		if name == "jaw":
			# links
			for l in range(1, len(pts)):
				ptA = tuple(pts[l - 1])
				ptB = tuple(pts[l])
				cv2.line(overlay, ptA, ptB, colors[i], 2)
		else:
			# computer convex
			hull = cv2.convexHull(pts)
			cv2.drawContours(overlay, [hull], -1, colors[i], -1)

	# put information in orgin image
	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
	return  output


# 关键点定位
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape"])

image = cv2.imread(args["image"])
image = resize_image(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Face detect
rects = detector(gray, 1)

# for rects
for (i, rect) in enumerate(rects):
	shape = predictor(gray, rect)
	shape = shape_to_np(shape)

	# for every part
	for (name, (i, j)) in FACIAL_LANDMARKS_68_IDXS.items():
		clone = image.copy()
		cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
					0.7, (0, 0, 255), 2)

		# draw ponit
		for (x, y) in shape[i: j]:
			cv2.circle(clone, (x, y), 3, (0, 0, 255), -1)

		# ROI
		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
		roi = image[y: y + h, x : x + w]
		roi = resize_image(roi, width=250)

		# show every part
		cv2.imshow("roi", roi)
		cv2.imshow("image", image)
		cv2.waitKey(0)


	output = visualize_facial_landmarks(image, shape)
	cv2.imshow("image", output)
	cv2.waitKey(0)








