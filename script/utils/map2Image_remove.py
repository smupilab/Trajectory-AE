import pandas as pd
import numpy as np
import random

from drawNp import drawNp

def map2Image_remove(minX: float, minY: float, maxX: float, maxY: float, csv_file: pd.DataFrame) -> np.array:
    inputImage = np.zeros([512,512], dtype=np.uint8)

    removeList = [ ]
    fileNum = csv_file.shape[0]
    for _ in range( int( fileNum * 0.5 ) ):
        idx = random.randint( 0, fileNum )
        while ( idx in removeList ):
            idx = random.randint( 0, fileNum )

        removeList.append( idx )

    for i in range(0, fileNum):
        if ( i in removeList ):
            continue

        x = csv_file.loc[i][0]
        y = csv_file.loc[i][1]

        # Print Dot
        mapX = int(round(np.interp(x,[minX,maxX],[0,500])))
        mapY = int(round(np.interp(y,[minY,maxY],[0,500])))
        inputImage[mapX][mapY] = 1

    outputImage = drawNp(inputImage)

    rotImage = np.rot90(outputImage)

    return rotImage