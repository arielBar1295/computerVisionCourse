"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import matplotlib.pyplot as plt
import numpy as np
# from PIL import Image, ImageDraw
import cv2
from math import sqrt, pi, cos, sin
from matplotlib import cm

from collections import defaultdict


def conv1D(inSignal:np.ndarray,kernel1:np.ndarray)->np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param inSignal: 1-D array
    :param kernel1: 1-D array as a kernel
    :return: The convolved array
    """
    #flip the kernel for the conv
    kernel1=np.flip(kernel1)
    output=np.zeros(len(inSignal)+len(kernel1)-1)
    #padding with 0
    inSignal=np.pad(inSignal, (len(kernel1)-1, len(kernel1)-1), 'constant')
    index=0
    #iterate the array and multiply by the kernel.
    for i in range(len(inSignal)-(len(kernel1)-1)):
         output[index]=((kernel1*inSignal[i:i+len(kernel1)]).sum())
         index+=1

    return output

def conv2D(inImage:np.ndarray,kernel2:np.ndarray)->np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    :return: The convolved image
    """
    k_height, k_width = kernel2.shape
    img_height, img_width = inImage.shape
    paddedmat = np.pad(inImage, ((k_height, k_height), (k_width, k_width)), 'edge')

    convolved_mat = np.zeros((img_height, img_width), dtype="float32")
    for i in range(img_height):
        for j in range(img_width):
            x_head = j + 1 + k_width
            y_head = i + 1 + k_height
            convolved_mat[i, j] = (paddedmat[y_head:y_head + k_height, x_head:x_head + k_width] * kernel2).sum()

    return convolved_mat

def convDerivative(inImage:np.ndarray) -> (np.ndarray,np.ndarray,np.ndarray,np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
    """

    kernel = np.array([[1, 0, -1]])
    x_direction = conv2D(inImage, kernel)
    y_direction = conv2D(inImage,kernel.T)
    direction = np.arctan2(y_direction, x_direction)
    magnitude = np.sqrt((np.power(x_direction, 2) + np.power(y_direction, 2)))

    return direction, magnitude, x_direction, y_direction

def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    new_image_x = conv2D(img, filter)
    new_image_y = conv2D(img, np.flip(filter.T))
    sobel = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    # if the intensity is bigger than thresh ,set the pixel to 1 ,else set to 0
    sobel/=255
    sobel[(sobel>=thresh)] =1   #x
    sobel[(sobel < thresh)]=0
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
    #soble by opencv
    cv_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    cv_sobel *= 255.0 / cv_sobel.max()
    return cv_sobel,sobel

def edgeDetectionZeroCrossingSimple(image:np.ndarray)->(np.ndarray) :
    """
    Detecting edges using the "ZeroCrossing" method
    :param I: Input image
    :return: Edge matrix
    Hence, a zero-cross is defined in the case that a pixel with negative value has a positive value around, or a pixel with a positive value has a negative value around.
    """
    laplacian = np.array([[0, 1, 0], [1, -4, 1],[0, 1, 0]])
    img=conv2D(image,laplacian)
    min = cv2.morphologyEx(img, cv2.MORPH_ERODE, np.ones((3, 3)))
    max = cv2.morphologyEx(img, cv2.MORPH_DILATE, np.ones((3, 3)))
    ZeroCrossingSimple = np.logical_or(np.logical_and(min < 0, img > 0), np.logical_and(max > 0, img < 0))
    return ZeroCrossingSimple

def edgeDetectionZeroCrossingLOG(img:np.ndarray)->(np.ndarray):
    """
    Detecting edges using the "ZeroCrossingLOG" method
    :param I: Input image
    :return: :return: Edge matrix
    """
    #smooth with 2d Gaussian
    blur_img = blurImage1(img,3)
    return edgeDetectionZeroCrossingSimple(blur_img)

def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """

    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
    cv_sobel = np.hypot(sobely, sobelx)
    cv_sobel = cv_sobel / cv_sobel.max() * 255
    theta = np.arctan2(sobely, sobelx)
    nonmaximg=non_max_suppression(cv_sobel,theta)
    thresh=threshold(nonmaximg,0.05,0.15)
    cannyFinal=hysteresis((thresh))
    cannyOpencv = cv2.Canny(img, img.shape[0], img.shape[1])
    return cannyOpencv,cannyFinal

def blurImage1(in_image:np.ndarray,kernel_size:np.ndarray)->np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param inImage: Input image
    2
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    sig = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    img = conv2D(in_image, kernel)
    return img

def blurImage2(in_image:np.ndarray,kernel_size:np.ndarray)->np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    # Creates Gaussian kernel
    kernel = cv2.getGaussianKernel(kernel_size, -1)
    # Apply the above Gaussian kernel.
    img=cv2.sepFilter2D(in_image, -1, kernel, kernel)
    return img

def non_max_suppression(img, theta):
        h, w = img.shape
        output = np.zeros((h, w), dtype=np.int32)
        angle = theta * 180. / np.pi
        # normalize the angle
        angle[angle < 0] += 180
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                q = 255
                r = 255
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]
                # if the pixel is bigger than its neighbors ,keep it,else change the pixel to 0
                if (img[i, j] >= q) and (img[i, j] >= r):
                    output[i, j] = img[i, j]
                else:
                    output[i, j] = 0

        return output

def threshold(img, low, high):
    h, w = img.shape
    highT = img.max() * high;
    lowT = highT * low;
    out = np.zeros((h, w), dtype=np.int32)
    strongPixI, strongPixJ = np.where(img >= highT)
    zeros_i, zeros_j = np.where(img < lowT)
    weakPixI, weakPixJ = np.where((img <= highT) & (img >= lowT))
    strong = 255
    weak = 75
    out[strongPixI, strongPixJ] = strong
    out[weakPixI, weakPixJ] = weak
    return out

def hysteresis(img, weak=75, strong=255):
    h, w = img.shape
    # check for each pixel if one of its nneighbors is a strong intensity
    # if it is ,set to strong.
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if (img[i, j] == weak):
                if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                        or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0

    return img

def houghCircle(img:np.ndarray,min_radius:float,max_radius:float)->list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    :param I: Input image
    :param minRadius: Minimum circle radius
    :param maxRadius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """

    rmin = min_radius
    rmax = max_radius
    steps = 100
    threshold = 0.4

    points = []
    for r in range(rmin, rmax + 1):
        for t in range(steps):
            points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

    acc = defaultdict(int)
    #take the edges using edgeDetectionCanny and take from the output all the index which are edges (means intensitiy 255)
    cvAns, myAns = edgeDetectionCanny(img, 1, 1)
    myAns = np.argwhere(myAns == 255)
    for x, y in myAns:
        for r, dx, dy in points:
            a = x - dx
            b = y - dy
            acc[(a, b, r)] += 1

    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            circles.append((x, y, r))

    return circles