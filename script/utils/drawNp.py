# Convert 0-1 Image into 0-255 Image
import numpy as np
from cv2init import init

def drawNp(img: np.array) -> np.array:
    blank = init()
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img[i][j] == 1 :
                blank[i][j] = 0

    return blank