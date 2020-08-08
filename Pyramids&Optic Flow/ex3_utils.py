
from typing import List
import numpy as np
import cv2
import matplotlib.pyplot as plt

def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    im1/=255
    im2/=255
    Fx = cv2.Sobel(im1, cv2.CV_64F, 1, 0, ksize=5)
    Fy = cv2.Sobel(im1, cv2.CV_64F, 0, 1, ksize=5)
    Ft = im2 -im1
    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)
    x_y = []
    u_v = []
    # building the window,check the pixels according to the step size.
    for i in range(win_size, im1.shape[0] - win_size,step_size):
        for j in range(win_size, im1.shape[1] - win_size,step_size):
            Ix = Fx[i - win_size:i + win_size + 1, j - win_size:j + win_size + 1].flatten()
            Iy = Fy[i - win_size:i + win_size + 1, j - win_size:j + win_size + 1].flatten()
            It = Ft[i - win_size:i + win_size + 1, j - win_size:j + win_size + 1].flatten()
            B = np.reshape(It, (It.shape[0], 1))
            A = np.vstack((Ix, Iy)).T
            lk1, lk2 = np.linalg.eigvals(np.matmul(A.T, A))
            l1 = max(lk1,lk2)
            l2 = min(lk1,lk2)
            #the constraints
            if l1 >= l2 and l2 >1 and l1/l2 < 100 :
                nu = np.matmul(np.linalg.pinv(A), -B)
                u_v.append([nu[0],nu[1]])
                x_y.append([j,i])
    return (np.array(x_y), np.array(u_v))

def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    output = []
    kernel = cv2.getGaussianKernel(5, -1)
    gaussPyr=gaussianPyr(img, levels)
    for i in range(len(gaussPyr) - 1):
        expandedImg = (gaussExpand(gaussPyr[i + 1],kernel))
        # Crop
        if (len(expandedImg) != len(gaussPyr[i])):
            expandedImg = expandedImg[0:len(gaussPyr[i]), :]
        if (len(expandedImg[0]) != len(gaussPyr[i][0])):
            expandedImg = expandedImg[:, 0:len(gaussPyr[i][0])]
        output.append((gaussPyr[i] - expandedImg))

    output.append((gaussPyr[len(gaussPyr) - 1]))
    return output

def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    kernel = cv2.getGaussianKernel(5, -1)
    output = lap_pyr[len(lap_pyr) - 1]
    for i in range(len(lap_pyr) - 1, 0, -1):
        # expand to next level
        expandedImg = gaussExpand(output,kernel)
        # Crop expanded image if wrong target dimensions
        if (len(expandedImg) != len(lap_pyr[i - 1])):
            expandedImg = expandedImg[0:len(lap_pyr[i - 1]), :]
        if (len(expandedImg[0]) != len(lap_pyr[i - 1][0])):
            expandedImg = expandedImg[:, 0:len(lap_pyr[i - 1][0])]

        output = expandedImg + lap_pyr[i - 1]
    return output

def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """

    list=[]
    list.append(img)
    kernel = cv2.getGaussianKernel(5, -1)
    for k in range(1, levels):
       img = cv2.sepFilter2D(list[k-1], -1, kernel, kernel)
       img = img[::2, ::2]
       list.append(img)
    return list

def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    output = img
    #check the img type and build output accordingly
    if len(img.shape)>2:
        output=np.zeros((2 * len(img), 2 * len(img[0]),3))
        output[::2,::2] = img
    else:
        output = np.zeros((2 * len(img), 2 * len(img[0])))
        output[::2,::2]=img
    output[output<0]=0
    output=(cv2.sepFilter2D(output, -1, gs_k,gs_k)*4)
    return output


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """
    gaussPyrMask=gaussianPyr(mask)
    laplPyrWhite=laplaceianReduce(img_1)
    laplPyrBlack=laplaceianReduce(img_2)
    blended_pyr = []
    for i in range(len(gaussPyrMask)):
        blended_pyr.append(gaussPyrMask[i]*laplPyrWhite[i] + (1 -  gaussPyrMask[i])*laplPyrBlack[i])
    pyrBlending=laplaceianExpand(blended_pyr)
    naive=naiveBlending(img_1,img_2,mask)

    return (naive, pyrBlending)
def naiveBlending(img1,img2,mask):
    """
    Blends two images using naive method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :return: Blended Image
    """
    output = img1.copy()
    output[mask == 0] = img2[mask == 0]
    return output