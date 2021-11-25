import random, cv2
import numpy as np
import pandas as pd

class Map2ImageGenerator:
    def __init__( self, width, height, dot = 0 ):
        self.WIDTH = width
        self.HEIGHT = height
        self.dot = dot

    def ConvertImage( self, csv_file: pd.DataFrame, form: str = 'default' ) -> np.array:
        maxmin = self.coorMaxMin( csv_file )
        if ( form == 'default' ):
            return self.map2Image( maxmin, csv_file )
        elif ( form == 'noise' ):
            return self.map2Image_noise( maxmin, csv_file )
        elif ( form == 'remove' ):
            return self.map2Image_remove( maxmin, csv_file )


    # 빈 캔버스 만들기
    def emptyCanvas( self ) -> np.array: 
        blank = np.zeros([self.HEIGHT,self.WIDTH],dtype=np.uint8)
        blank.fill(0)
        blank = cv2.resize(blank,(self.HEIGHT,self.WIDTH))

        return blank

    # Convert 0-1 Images into 0-255 Image
    def drawNp( self, img: np.array ) -> np.array:
        '''
        dotForm = 0 : 1x1 dot
        dotForm = 1 : crosshead (3x3)
        dotForm = 2 : 3x3 dot
        '''
        blank = self.emptyCanvas()

        rowDiff = [ [ 0 ], [ -1, 0, 0, 0, 1 ], [ -1, -1, -1, 0, 0, 0, 1, 1, 1 ] ]
        colDiff = [ [ 0 ], [ 0, -1, 0, 1, 0 ], [ -1, 0, 1, -1, 0, 1, -1, 0, 1 ] ]

        for i in range( 0, img.shape[0] ):
            for j in range( 0, img.shape[1] ):
                if img[i][j] == 1 :
                    for rr, cc in zip( rowDiff[self.dot], colDiff[self.dot] ):
                        newRow, newCol = i + rr, j + cc
                        if ( 0 <= newRow < img.shape[0] and 0 <= newCol < img.shape[1] ):
                            blank[newRow][newCol] = 255

        return blank


    # Convert csv File to Image
    def map2Image( self, min_max: tuple, csv_file: pd.DataFrame ) -> np.array:
        inputImage = np.zeros([self.HEIGHT,self.WIDTH], dtype=np.uint8)

        minX, minY, maxX, maxY = min_max

        for i in range( 0, len( csv_file ) ):
            x = csv_file.loc[i]['lat']
            y = csv_file.loc[i]['long']

            # Print Dot
            mapX = int( round( np.interp( x, [ minX, maxX ], [ 0, int( self.WIDTH * 0.9 ) ] ) ) )
            mapY = int( round( np.interp( y, [ minY, maxY ], [ 0, int( self.HEIGHT * 0.9 ) ] ) ) )
            inputImage[mapX][mapY] = 1

        outputImage = self.drawNp(inputImage)

        rotImage = np.rot90(outputImage)

        return rotImage


    # Convert csv File to Image with Noise
    def map2Image_noise( self, min_max: tuple, csv_file: pd.DataFrame ) -> np.array:
        inputImage = np.zeros([self.HEIGHT,self.WIDTH], dtype = np.uint8 )

        minX, minY, maxX, maxY = min_max

        randomList = set()
        while len( randomList ) < int( len( csv_file ) / 7):
            randomList.add( random.randint( 0,len( csv_file ) ) )

        randomList = list( randomList )
        dicisionList = [ 1, -1 ]

        for i in range( 0, len( csv_file ) ):
            try:
                # Generate Noise
                randomList.index(i)

                r = random.uniform((minX - maxX) / 40,(minX - maxX) / 20)
                D = random.choice(dicisionList)

                x = csv_file.loc[i]['lat'] - (D * r)
                y = csv_file.loc[i]['long'] - (D * r)

                # Paint dot
                mapX = int( round( np.interp( x, [ minX, maxX ], [ 0, int( self.WIDTH * 0.9 ) ] ) ) )
                mapY = int( round( np.interp( y, [ minY, maxY ], [ 0, int( self.HEIGHT * 0.9 ) ] ) ) )
                inputImage[mapX][mapY] = 1

            except:
                x = csv_file.loc[i]['lat']
                y = csv_file.loc[i]['long']

                mapX = int( round( np.interp( x, [ minX, maxX ], [ 0, int( self.WIDTH * 0.9 ) ] ) ) )
                mapY = int( round( np.interp( y, [ minY, maxY ], [ 0, int( self.HEIGHT * 0.9 ) ] ) ) )
                inputImage[mapX][mapY] = 1


        outputImage = self.drawNp(inputImage)

        rotImage = np.rot90(outputImage)

        return rotImage


    def map2Image_remove( self, min_max: tuple, csv_file: pd.DataFrame ) -> np.array:
        inputImage = np.zeros([self.HEIGHT,self.WIDTH], dtype=np.uint8)

        minX, minY, maxX, maxY = min_max

        removeList = [ ]
        fileNum = len( csv_file )
        for _ in range( int( fileNum * 0.5 ) ):
            idx = random.randint( 0, fileNum )
            while ( idx in removeList ):
                idx = random.randint( 0, fileNum )

            removeList.append( idx )

        for i in range(0, fileNum):
            if ( i in removeList ):
                continue

            x = csv_file.loc[i]['lat']
            y = csv_file.loc[i]['long']

            # Print Dot
            mapX = int( round( np.interp( x, [ minX, maxX ], [ 0, int( self.WIDTH * 0.9 ) ] ) ) )
            mapY = int( round( np.interp( y, [ minY, maxY ], [ 0, int( self.HEIGHT * 0.9 ) ] ) ) )
            inputImage[mapX][mapY] = 1

        outputImage = self.drawNp(inputImage)

        rotImage = np.rot90(outputImage)

        return rotImage


    # Return Max and Min X,Y Coordinate Value of file
    def coorMaxMin( self, file: pd.DataFrame ) -> (float, float, float, float):
        minX, minY = ( file.loc[0]['lat'], file.loc[0]['long'] )
        maxX, maxY = ( file.loc[0]['lat'], file.loc[0]['long'] )
        for i in range( 1, len( file ) ):
            x = file.loc[i]['lat']
            y = file.loc[i]['long']
            if x > maxX :
                maxX = x
            if x < minX :
                minX = x
            if y > maxY :
                maxY = y
            if y < minY :
                minY = y
        return minX, minY, maxX, maxY
