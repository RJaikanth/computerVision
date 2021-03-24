"""
"""
import cv2 as cv
import numpy as np



def smoothing(image):
    return cv.GaussianBlur(image, (3,3), 1)


def sobel(image):
    gradx = cv.Sobel(image, cv.CV_16S, 1, 0, ksize=3)
    grady = cv.Sobel(image, cv.CV_16S, 0, 1, ksize=3)
    
    return cv.addWeighted(cv.convertScaleAbs(gradx), 0.5, cv.convertScaleAbs(grady), 0.5, 0)


def prewitt(image):
    dx = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    dy = np.flipud(dx.T)
    
    # Smooth the image
    image = smoothing(image)

    gradx = cv.filter2D(image, cv.CV_16S, dx)
    grady = cv.filter2D(image, cv.CV_16S, dy)
    
    return cv.addWeighted(cv.convertScaleAbs(gradx), 0.5, cv.convertScaleAbs(grady), 0.5, 0)


def roger(image):
    dx = np.array([
        [0, 1],
        [-1, 0]
    ])
    dy = np.flipud(dx.T)

    # Smooth the image
    image = smoothing(image)

    gradx = cv.filter2D(image, cv.CV_16S, dx)
    grady = cv.filter2D(image, cv.CV_16S, dy)
    
    return cv.addWeighted(cv.convertScaleAbs(gradx), 0.5, cv.convertScaleAbs(grady), 0.5, 0)


def rogerSobel(image):
    return cv.addWeighted(sobel(image), 0.8, roger(image), 0.2, 0)


def rogerPrewitt(image):
    return cv.addWeighted(prewitt(image), 0.8, roger(image), 0.2, 0)


def canny(image):
    return cv.Canny(image, 120, 150)



if __name__ == "__main__":
    cap = cv.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        
        # Show the frame
        cv.imshow("Image", frame)
        
        # Edge Det
        cv.imshow("Prewitt", prewitt(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)))
        cv.imshow("Sobel", sobel(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)))
        cv.imshow("Roger", roger(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)))
        cv.imshow("Roger + Sobel", rogerSobel(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)))
        cv.imshow("Roger + Prewitt", rogerPrewitt(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)))
        cv.imshow("Canny", canny(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)))
        
        if cv.waitKey(1) == ord('q'):
            break
        
    # cv.destroyAllWindows()
    # cap.release()
