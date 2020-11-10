def drawNp(img):
    blank = init()
    # print(img.shape[0])
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if img[i][j] == 1 :
                blank[i][j] = 0

    return blank