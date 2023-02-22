
from skimage.exposure import rescale_intensity
import argparse
import cv2
import numpy as np


def convolve(image, kernel):

    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    pad = (kW - 1) // 2

    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):

            roi = image[y - pad : y + pad + 1, x - pad : x + pad + 1]
            output[y - pad, x - pad] = (roi * kernel).sum()

    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output


