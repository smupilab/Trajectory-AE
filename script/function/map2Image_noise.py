import numpy as np
import pandas as pd
import random

# Convert csv File to Image with Noise
def map2Image_noise(minX: float, minY: float, maxX: float, maxY: float, csv_file: pd.DataFrame) -> np.array:
    inputImage = np.zeros([512,512], dtype=np.uint8)

    randomList = set()
    while len(randomList) < int(csv_file.shape[0] / 7):
        randomList.add(random.randint(0,csv_file.shape[0]))

    randomList=list(randomList)
    dicisionList = [1,-1]

    for i in range(0, csv_file.shape[0]):
        try:
            # Generate Noise
            randomList.index(i)

            r = random.uniform((minX - maxX) / 40,(minX - maxX) / 20)
            D = random.choice(dicisionList)

            x = csv_file.loc[i][0] - (D * r)
            y = csv_file.loc[i][1] - (D * r)

            # Paint dot
            mapX = int(round(np.interp(x,[minX,maxX],[0,500])))
            mapY = int(round(np.interp(y,[minY,maxY], [0,500])))
            inputImage[mapX][mapY] = 1

        except:
            x = csv_file.loc[i][0]
            y = csv_file.loc[i][1]

            mapX = int(round(np.interp(x,[minX,maxX],[0,500])))
            mapY = int(round(np.interp(y,[minY,maxY], [0,500])))
            inputImage[mapX][mapY] = 1


    outputImage = drawNp(inputImage)

    rotImage = np.rot90(outputImage)

    return rotImage