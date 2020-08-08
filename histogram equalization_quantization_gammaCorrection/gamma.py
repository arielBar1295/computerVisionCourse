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
from ex1_utils import LOAD_GRAY_SCALE
import numpy as np
import cv2

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    if rep == 2:
        # reading as BGR
        image = cv2.imread(img_path)/255
    elif rep == 1:
        image=cv2.imread(img_path,0)/255
    else:
        raise ValueError('Only RGB or GRAY_SCALE ')

    # the callback function, to find the gamma -divide by 100 to get values between 0.01 to 2
    def on_trackbar(val):
        gamma = val / 100
        corrected_image = np.power(image, gamma)
        cv2.imshow('Gamma display', corrected_image)

    cv2.namedWindow('Gamma display')
    trackbar_name = 'Gamma'
    cv2.createTrackbar(trackbar_name, 'Gamma display', 1, 200, on_trackbar)
    # Show some stuff
    on_trackbar(1)
    # Wait until user press some key
    cv2.waitKey()
    pass


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
