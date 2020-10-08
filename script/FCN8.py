# FCN_8S.py
import sys
stdout = sys.stdout

file_name = 'FCN_8S'
output_stream = open( 'log({}).txt'.format( file_name ), 'wt' )
error_stream = open( 'errors({}).txt'.format( file_name ), 'wt' )
sys.stdout = output_stream
sys.stderr = error_stream

SIZE = 512

# Image Load #
stdout.write( 'Start Load Images... \n' )

import os, cv2, glob
import numpy as np

currDir = '/taejin/Taejin/TrajectoryAugmentation/'
saveDir = currDir + 'Trajectory-Augumentation/'
dataDir = currDir + 'Trajectory_Data/432-Image/'

## Load Train Data ##
def GetImage( path ):
	img = cv2.imread( path, 0 )
	resized = cv2.resize( img, ( SIZE, SIZE ) )

	return resized

X_trainDir = dataDir + 'Input-50/'
Y_trainDir = dataDir + 'Val/'

X_train, Y_train = [ ], [ ]

os.chdir( X_trainDir )
X_trainFiles = glob.glob( '*png' )
for f in X_trainFiles:
	X_train.append( GetImage( f ) )

os.chdir( Y_trainDir )
Y_trainFiles = glob.glob( '*png' )
for f in Y_trainFiles:
	Y_train.append( GetImage( f ) )

## Load Test Data ##
testDir = dataDir + 'Test_Image/'

X_test = [ ]

os.chdir( testDir )
testFiles = glob.glob( '*png' )
for f in testFiles:
	X_test.append( GetImage( f ) )

## Resize Images for CNN ##
X_train, Y_train = np.array( X_train ), np.array( Y_train )
X_test = np.array( X_test )

X_train = X_train.astype( 'float32' ) / 255.
Y_train = Y_train.astype( 'float32' ) / 255.
X_test = X_test.astype( 'float32' ) / 255.

print( X_train.shape, Y_train.shape, X_test.shape )

X_train = np.reshape( X_train, ( len( X_train ), SIZE, SIZE, 1 ) )
Y_train = np.reshape( Y_train, ( len( Y_train ), SIZE, SIZE, 1 ) )
X_test = np.reshape( X_test, ( len( X_test ), SIZE, SIZE, 1 ) )

print( 'train shape (X, Y): ({},{})'.format( X_train.shape, Y_train.shape ) )
print( 'test shape (X): ({})'.format( X_test.shape ) )

stdout.write( 'Finish Load Images! \n' )

# Construct Model #
stdout.write( 'Start Make Model... \n' )

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16

## Hyper Parameter ##
kernel = 4
pooling = ( 2, 2 )
acti, pad = 'relu', 'same'
encoding_channels = [ 64, 32, 16 ]
decoding_channels = reversed( encoding_channels )

## Input Image ##
input_img = layers.Input( shape = ( SIZE, SIZE, 1 ) )

vgg16 = VGG16( include_top = False, weights = 'imagenet', input_tensor = input_img )

f3 = vgg16.get_layer('block3_pool').output
f4 = vgg16.get_layer('block4_pool').output
f5 = vgg16.get_layer('block5_pool').output

f5_conv1 = layers.Conv2D( 4086, 7, padding = pad, activation = acti )(f5)
f5_drop1 = layers.Dropout( 0.5 )(f5_conv1)
f5_conv2 = layers.Conv2D( 4086, 1, padding = pad, activation = acti )(f5_drop1)
f5_drpo2 = layers.Dropout( 0.5 )(f5_conv2)
f5_conv3 = layers.Conv2D( 1, 1, padding = pad, activation = None )(f5_drop2)

f5_conv3_x2 = layers.Conv2DTranspose( 1, 4, 2, use_bias = False, padding = pad, activation = acti )(f5_conv3)
f4_conv1 = layers.Conv2D( 1, 1, padding = pad, activation = None )(f4)

merge1 = layers.add( [ f4_conv1, f5_conv3_x2 ] )
merge1_x2 = layers.Conv2DTranspose( 1, 4, 2, use_bias = False, padding = pad, activation = acti )(merge1)

f3_conv1 = layers.Conv2D( 1, 1, padding = pad, activation = None )(f3)
merge2 = layers.add( [ f3_conv1, merge1_x2 ] )

output = layers.Conv2DTranspose( 1, 16, 8, padding = pad, activation = None )(merge2)

fcn_8s = keras.Model(input_img, output)
fcn_8s.summary()

fcn_8s.compile(optimizer = 'adam', loss = 'binary_crossentropy')

stdout.write( 'Finish Making Model! \n' )

# Train Model #
stdout.write( 'Start Training Model... \n' )

## Hyper Parameter ##
EPOCH = 50
BATCH = 20
SHUFFLE = True

history = fcn_8s.fit( X_train, Y_train, epochs = EPOCH, batch_size = BATCH, shuffle = SHUFFLE )

stdout.write( 'Finish Train Model! \n' )

# Test Model #
stdout.write( 'Start Testing Model... \n' )
decoded_img = fcn_8s.predict( X_test )

import matplotlib.pyplot as plt

os.chdir( saveDir )
n = 10
plt.figure( figsize = ( 20, 4 ) )
for i in range( n ):
	ax = plt.subplot( 2, n, i + 1 )
	plt.imshow( X_test[i].reshape( SIZE, SIZE ) )
	plt.gray()

	ax.get_xaxis().set_visible( False )
	ax.get_yaxis().set_visible( False )

	ax = plt.subplot( 2, n, n + i + 1 )
	plt.imshow( decoded_img[i].reshape( SIZE, SIZE ) )
	plt.gray()

	ax.get_xaxis().set_visible( False )
	ax.get_yaxis().set_visible( False )

plt.savefig( 'Result.png', dpi = 300 )
plt.show()

stdout.write( 'Finish Testing Model! \n' )

stdout.write( 'Good job\n' )

output_stream.close()
error_stream.close()