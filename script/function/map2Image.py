import numpy as np
import os
import pandas as pd

def map2Image(minX,minY,maxX,maxY,csv_file):

  inputImage = np.zeros([512,512], dtype=np.uint8)

  for i in range(0,csv_file.shape[0]):
    x = csv_file.loc[i][0]
    y = csv_file.loc[i][1]

    mapX = int(round(np.interp(x,[minX,maxX],[0,500])))
    mapY = int(round(np.interp(y,[minY,maxY],[0,500])))
    inputImage[mapX][mapY] = 1

  outputImage = drawNp(inputImage)

  rotImage = np.rot90(outputImage)

  return rotImage
