# import
import cv2

def sort_contours(cnts, method='L2R'):
    reverse = False
    i = 0

    if method == 'R2L' or method == 'B2T':
        reverse = True
    if method == 'B2T' or method == 'T2B':
        i = 1
    boundingBoxs = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxs) = zip(*sorted(zip(cnts, boundingBoxs),
                                       key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxs
F
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
        dim = (width, int(h*r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

