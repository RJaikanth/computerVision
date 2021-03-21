"""
This file contains functions for convolving images with odd-sized square kernels.
It also contains code for displaying images in the frequency domain
"""


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def basicConvolve(image: np.array, kernel: np.array, padding:int = 2, stride:int = 1) -> np.array:
    """
    Basic Convolution Method

    Args:
        image (np.array): Input image
        kernel (np.array): Convolution Kernel
        padding (int, optional): Padding to be added to the image. Zero padding only. Defaults to 2.
        stride (int, optional): Kernel Stride. Defaults to 1.

    Returns:
        np.array: Convolved image
    """
    # Flip kernel
    kernel = np.flipud(np.fliplr(kernel))
    
    # Get output shape
    xOutput = int(((image.shape[0] - kernel.shape[0] - 2 * padding) / stride) + 1)
    yOutput = int(((image.shape[1] - kernel.shape[1] - 2 * padding) / stride) + 1)
    output = np.zeros((xOutput, yOutput))
    
    if padding != 0:
        imagePadded = np.zeros((image.shape[0]+padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1*padding), int(padding):int(-1*padding)] = image
    else:
        imagePadded = image
    
    for y in range(image.shape[1]):
        if (y > image.shape[1] - kernel.shape[1]):
            break
        for x in range(image.shape[0]):
            if (x > image.shape[0] - kernel.shape[0]):
                break    
            try:
                output[x, y] = (kernel * imagePadded[x: x + kernel.shape[0], y:y+kernel.shape[1]]).sum()
            except:
                break
    
    return output
    
   
def convolveFFT(image: np.array, kernel: np.array, padding:int = 2, stride:int = 1) -> np.array:
    """
    Basic Convolution Method

    Args:
        image (np.array): Input image
        kernel (np.array): Convolution Kernel
        padding (int, optional): Padding to be added to the image. Zero padding only. Defaults to 2.
        stride (int, optional): Kernel Stride. Defaults to 1.

    Returns:
        np.array: Convolved image
    """
    
    return np.real(np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(np.flipud(np.fliplr(kernel)), s=image.shape)))
    
     
   
   
if __name__ == "__main__":
    kernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    
    cap = cv.VideoCapture(0)
    count=0
    
    while (True):
        ret, frame = cap.read()
        
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        output = convolveFFT(frame, kernel)
        
        cv.imshow("Image - Python", frame)
        cv.imshow("Convolved Image", output)
        
        if cv.waitKey(1) & 0xFF==ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()



