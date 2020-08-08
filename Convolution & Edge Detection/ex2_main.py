from ex2_utils import *
import numpy as np

import matplotlib.pyplot as plt

def myID():
    return 313383259,315392852

def deriv():
    img = cv2.imread('zebra.jpeg', cv2.IMREAD_GRAYSCALE) / 255
    ori, mag, x_der,y_der= convDerivative(img)
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.imshow(ori)
    ax2.imshow(mag)
    ax3.imshow(x_der)
    ax4.imshow(y_der)
    ax1.title.set_text('tetha')
    ax2.title.set_text('magnitude')
    ax3.title.set_text('x derivative')
    ax4.title.set_text('y derivative')
    plt.show()

def canny(input,t1,t2):
    cvAns,ourImplemntation=edgeDetectionCanny(input,t1,t2)
    plt.imshow(ourImplemntation,cmap='gray')
    plt.title('canny - our implementation')
    plt.show()

def zeroCrossing(input):
    answer=edgeDetectionZeroCrossingSimple(input)
    plt.imshow(answer,cmap='gray')
    plt.title("simple zero crossing")
    plt.show()

def zeroLoGCrossing(input):
    answer=edgeDetectionZeroCrossingLOG(input)
    plt.imshow(answer,cmap='gray')
    plt.title("LoG zero crossing")
    plt.show()

def sobel(input,t):
    cv,answer=edgeDetectionSobel(input,t)
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(cv,cmap='gray')
    ax[1].imshow(answer,cmap='gray')
    ax[0].title.set_text('sobel - cv')
    ax[1].title.set_text('sobel- our implementation')
    plt.show()

def blur(img,kernel_size):
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(blurImage2(img, kernel_size),cmap='gray')
    ax[1].imshow(blurImage1(img,kernel_size),cmap='gray')
    ax[0].title.set_text('blur - cv')
    ax[1].title.set_text('blur - our implementation')
    plt.show()

def hough(img,minR,maxR):
    circles = houghCircle(img, minR, maxR)
    print("found circle:", circles)
    # draw the circles on the input images:
    for vertex in circles:
        cv2.circle(img, (vertex[1], vertex[0]), vertex[2], (0, 255, 0), 1)
        cv2.rectangle(img, (vertex[1] - 2, vertex[0] - 2), (vertex[1] - 2, vertex[0] - 2), (0, 0, 255), 3)
    plt.imshow(img, cmap='gray')
    plt.title("hough circle")
    plt.show()

def testconv1D(array,kernel):
    npAns=np.convolve(array, kernel,'full')
    myAns=conv1D(array,kernel)
    if npAns.all()==myAns.all():
        print("good conv1D")

def testconv2D(img,kernel):
    borderType=cv2.BORDER_REPLICATE
    npAns=cv2.filter2D(img,-1,kernel,borderType=borderType)
    myAns=conv2D(img,kernel)
    if npAns.all()==myAns.all():
        print("good conv2D")

def main():
     print("ID:", myID())
     beachImg=cv2.imread('beach.jpg',cv2.IMREAD_GRAYSCALE)
     imgPeople = cv2.imread('people.png', cv2.IMREAD_GRAYSCALE)
     boxImg = cv2.imread('boxman.jpeg',cv2.IMREAD_GRAYSCALE)
     monkeyImg = cv2.imread('codeMonkey.jpeg', cv2.IMREAD_GRAYSCALE)
     manImage= cv2.imread('manWithCamera.jpg', cv2.IMREAD_GRAYSCALE)
     foodImage= cv2.imread('food.jpg', cv2.IMREAD_GRAYSCALE)

     #conv1D
     kernel=np.array([1/3,1/3,1/3])
     array=np.array([1,2,3,4,5,6,7])
     testconv1D(array,kernel)

     #conv2D
     kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
     testconv2D(imgPeople,kernel)

     # convDerivative
     deriv()

     #sobel
     sobel(beachImg,0.7)
     sobel(manImage,0.9)


     #canny
     canny(imgPeople,1,1)
     canny(boxImg,1,1)
     canny(foodImage,0.7,1)

     #zero crossing
     zeroCrossing(monkeyImg)

    # zero crossing LOG
     zeroLoGCrossing(monkeyImg)

    #blur -bonus.
     blur(beachImg,20)
     blur(foodImage,40)

    # houghCircle
     img = cv2.imread('smallC.jpg', cv2.IMREAD_GRAYSCALE)
     hough(img,5,11)


if __name__ == '__main__':
    main()