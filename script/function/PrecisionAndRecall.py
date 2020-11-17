def PrecisionAndRecall(decodeImage,OriginalImage,csvFile,Threshold):

  blank = np.zeros([512,512],dtype=np.uint8)
  blank.fill(255)
  blank = cv2.resize(blank,(512,512),1)

  for x in range(0,512):
    for y in range(0,512):
      if decodeImage[x][y] * 255 < Threshold :
      
        blank[x][y] = OriginalImage[x][y] * 255

  inputImage = np.zeros([512,512], dtype=np.uint8)
  blankRot = np.rot90(blank)
  blankRot = np.rot90(blankRot)
  blankRot = np.rot90(blankRot)
  minX,minY,maxX,maxY = coorMaxMin(csvFile)
  FN = TP = TN = FP = 0
  for i in range(0,csvFile.shape[0]):
    x = csv_file.loc[i][0]
    y = csv_file.loc[i][1]

    mapX = int(round(np.interp(x,[minX,maxX],[0,500])))
    mapY = int(round(np.interp(y,[minY,maxY],[0,500])))
    inputImage[mapX][mapY] = 1

    if inputImage[mapX][mapY]==1 and blankRot[mapX][mapY] == 0:
      #print(i,csvFile.loc[i][3])
      if csvFile.loc[i][3] == False:
        FN = FN + 1
      else : 
        TN = TN + 1
    else:
      #print(i,"erased : ",csv_file.loc[i][3])
      if csvFile.loc[i][3] == True:
        TP = TP + 1
      else:
        FP = FP + 1


  outputImage = drawNp(inputImage)

  rotImage = np.rot90(outputImage)
  #print(blank)
  cv2_imshow(blank)
  cv2_imshow(rotImage)

  print("FN : ",FN,"TN : ",TN,"TP : ",TP,"FP : ",FP)

  return FN,TN,TP,FP

