import cv2
import math
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt


font = cv2.FONT_HERSHEY_SIMPLEX


def orientated_non_max_suppression(mag, ang):
    ang_quant = np.round(ang / (np.pi/4)) % 4
    winE = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    winSE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    winS = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    winSW = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    magE = non_max_suppression(mag, winE)
    magSE = non_max_suppression(mag, winSE)
    magS = non_max_suppression(mag, winS)
    magSW = non_max_suppression(mag, winSW)

    mag[ang_quant == 0] = magE[ang_quant == 0]
    mag[ang_quant == 1] = magSE[ang_quant == 1]
    mag[ang_quant == 2] = magS[ang_quant == 2]
    mag[ang_quant == 3] = magSW[ang_quant == 3]
    return mag


def non_max_suppression(data, win):
    data_max = scipy.ndimage.filters.maximum_filter(
        data, footprint=win, mode='constant')
    data_max[data != data_max] = 0
    return data_max


# start calulcation
image_path = "000001396.jpg"
image_path_1=cv2.resize(cv2.imread(image_path),(450,450),interpolation=cv2.INTER_AREA)
gray_image = cv2.imread(image_path, 0)
img = cv2.imread(image_path)
with_nmsup = True  # apply non-maximal suppression
fudgefactor = 1.5  # with this threshold you can play a little bit
sigma = 50  # for Gaussian Kernel
kernel = 2*math.ceil(2*sigma)+1  # Kernel size

gray_image = gray_image/255.0
blur = cv2.GaussianBlur(gray_image, (kernel, kernel), sigma)
gray_image = cv2.subtract(gray_image, blur)

# compute sobel response
sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

mag = np.hypot(sobelx, sobely)
ang = np.arctan2(sobely, sobelx)

# threshold
threshold = 4 * fudgefactor * np.mean(mag)
mag[mag < threshold] = 0

# either get edges directly
if with_nmsup is False:
    mag = cv2.normalize(mag, 0, 255, cv2.NORM_MINMAX)
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
    num_white = np.sum(result == 255)
    num_black = np.sum(result == 0)
    print(num_white)
    print(num_black)
    print((num_white/num_black)*100)
    # print(result)
    cv2.imshow('im', result)
    cv2.waitKey()

# or apply a non-maximal suppression
else:

    # non-maximal suppression
    mag = orientated_non_max_suppression(mag, ang)
    # create mask
    mag[mag > 0] = 255
    mag = mag.astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
    num_white = np.sum(result == 255)
    num_black = np.sum(result == 0)
    print(num_white)
    print(num_black)
    ratio = (num_white/num_black)*100
    print(ratio)
    if ratio > 1:
        print("Cracked")
        cv2.putText(img, 'Cracked', (0, 30),
                    font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    else:
        cv2.putText(img, 'Not Cracked', (0, 30),
                    font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        print("Not Cracked")
    # plt.hist(result.ravel(), 256, [0, 256])

    # print(result)
    cv2.imshow('im', result)
    cv2.imshow('gray_image', img)
    # plt.show()
    cv2.waitKey()
# 4614 crack
# 2266
