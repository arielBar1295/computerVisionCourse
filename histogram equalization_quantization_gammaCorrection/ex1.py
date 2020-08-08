

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
import cv2
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return

def isRGB (image: np.ndarray) -> bool:
    if len(image.shape) == 3:
        return True
    return False
def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns and returns in converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    if representation == 1:
        # channel 0 means grascale
        image = cv2.imread(filename, 0)
        return image/255

    elif representation == 2:
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image/255
    else:
        raise ValueError('Only RGB or GRAY_SCALE ')
    pass


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    if representation != 1 and representation != 2:
        raise ValueError('The image can not be displayed')
    image = imReadAndConvert(filename, representation)
    if representation == 1:
        plt.imshow(image, cmap=plt.get_cmap('gray'))
    else:
        plt.imshow(image)
    plt.show()
    pass


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    toYIQ = np.array([[0.299,0.587,0.114],[0.5959,-0.2746,-0.3213],[0.2115,-0.5227,0.3112]])
    ans = np.dot(imgRGB,toYIQ.T)
    return ans
    pass


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    toRGB = np.array([[1,0.956,0.619],[1,-0.272,-0.647],[1,-1.106,1.703]])
    ans = np.dot(imgYIQ,toRGB.T)
    return ans
    pass


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    image = imgOrig
    if isRGB(imgOrig):
        image = cv2.normalize(image.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        image = transformRGB2YIQ(image)
        imgEq = image[:, :, 0]
        imgEq = cv2.normalize(imgEq, None, 0, 255, cv2.NORM_MINMAX)
        imgEq = imgEq.astype('uint8')
    else:
        imgEq = image * 255
        imgEq = imgEq.astype('uint8')

    lut = np.zeros(256, dtype=imgEq.dtype)  # Create an empty lookup table

    histOrg, bins = np.histogram(imgEq.flatten(), 256, [0, 256])
    cdf = histOrg.cumsum()  # Calculate cumulative histogram
    cdf_normalized = cdf * histOrg.max() / cdf.max()
    cdf_normalized = cdf * histOrg.max() / cdf.max()
    plt.title('origin image histogram')
    plt.plot(cdf_normalized, color='b')
    plt.hist(imgEq.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.show()
    cdf_m = np.ma.masked_equal(cdf, 0)  # Remove the 0 value in the histogram
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (
            cdf_m.max() - cdf_m.min())  # equal to the lut[i] = int(255.0 *p[i]) formula described earlier

    cdf = np.ma.filled(cdf_m, 0).astype('uint8')  # Add the masked elements to 0
    linearImg = imgEq
    linearImg = cdf[linearImg]
    histEq, bins = np.histogram(linearImg.flatten(), 256, [0, 255])
    cdflin = histEq.cumsum()
    cdf_normalized_lin = cdflin * histEq.max() / cdflin.max()
    plt.title('linear CDF ')
    plt.plot(cdf_normalized_lin, color='b')
    plt.hist(linearImg.flatten(), 256, [0, 255], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()
    if isRGB(imgOrig):
        imgEq = cv2.normalize(imgEq.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        image[:, :, 0] = imgEq
        imgEq = transformYIQ2RGB(image)
        imgEq = imgEq * 255
        imgEq = imgEq.astype('uint8')

    imgEq = cv2.LUT(imgEq, cdf)

    return imgEq, histOrg, histEq
    pass

def rePaint(img: np.ndarray,boundaries:np.array)->np.ndarray:
    newImg = np.zeros(img.shape)
    for j in range(len(boundaries) - 1):
        newImg[(img < boundaries[j + 1]) & (img >= boundaries[j])] = boundaries[j]
    return newImg
def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    imOrig*=255
    # means RGB image ,take only the y channel
    if isRGB(imOrig):
        yiq = transformRGB2YIQ(imOrig)
        img = yiq[:, :, 0].copy()
    else:
        img = imOrig.copy()
    # first insert the boundaries(equal division at first),place at position 0,0 and in the last position 255
    boundaries = np.array([0, 255])
    for i in range(1, nQuant):
        boundaries = np.insert(boundaries, i, int((256 / nQuant) * i))
    histOrig, binsOrig = np.histogram(img.flatten(), bins=255, range=(0, 255))
    error_i = []
    qImage_i = []
    colorArray = np.zeros(nQuant)
    for i in range(nIter):
        # take from the origin hist and bins the values according to the boundaries,for each segment calculate the weighted mean
        for j in range(len(boundaries) - 1):
            bins = binsOrig[boundaries[j].astype(int):boundaries[j + 1].astype(int)]
            hist = histOrig[boundaries[j].astype(int):boundaries[j + 1].astype(int)]
            numofPix = hist.sum()
            weightedMean = (hist.flatten() * bins.flatten()).sum() / numofPix
            if np.isnan(weightedMean):
                weightedMean = np.nan_to_num(weightedMean)
            colorArray[j] = weightedMean
        # find the new bounderies (by finding the middle of each 2 avg).
        for j in range(1, len(boundaries) - 1):
            boundaries[j] = (colorArray[j - 1] + colorArray[j]) / 2
        # setting the new colors
        newImg = rePaint(img, boundaries)
        # calculating the error
        MSE = np.sqrt(np.power(np.subtract(newImg, img), 2).sum()) / (imOrig.shape[0] * imOrig.shape[1])
        error_i.append(MSE)
        qImage_i.append(newImg.copy())
    # back to RGB
    if isRGB(imOrig):
        yiq[:, :, 0] = newImg
        newImg = transformYIQ2RGB(yiq) / 255
        plt.imshow(newImg)
        plt.show()
        plt.plot(error_i)
        plt.show()
    else:
        plt.gray()
        plt.imshow(newImg)
        plt.show()
        plt.plot(error_i)
        plt.show()

    return qImage_i,error_i
    pass

