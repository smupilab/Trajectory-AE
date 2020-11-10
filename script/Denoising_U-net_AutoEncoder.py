# Denoising_U-net_AutoEncoder.py

# Denoising Trajectory Data using U-net Auto Encoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys
stdout = sys.stdout

file_name = 'Denoising_U-net_AutoEncoder'
output_stream = open( 'log({}).txt'.format( file_name ), 'wt' )
error_stream = open( 'errors({}).txt'.format( file_name ), 'wt' )
sys.stdout = output_stream
sys.stderr = error_stream

SIZE = 512

# Image Load #
stdout.write( 'Start Load Images... ' )

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

X_train = np.reshape( X_train, ( len( X_train ), SIZE, SIZE, 1 ) )
Y_train = np.reshape( Y_train, ( len( Y_train ), SIZE, SIZE, 1 ) )
X_test = np.reshape( X_test, ( len( X_test ), SIZE, SIZE, 1 ) )

print( 'train shape (X, Y): ({},{})'.format( X_train.shape, Y_train.shape ) )
print( 'test shape (X): ({})'.format( X_test.shape ) )

stdout.write( 'Finish Load Images! ' )

# Construct Model #
stdout.write( 'Start Make Model... ' )

## Hyper Parameter ##
kernel = ( 3, 3 )
pooling = ( 2, 2 )
acti, pad = 'relu', 'same'
encoding_channels = [ 64, 32, 16 ]
decoding_channels = list(reversed( encoding_channels ))

## Input Image ##
input_img = layers.Input( shape = ( SIZE, SIZE, 1 ) )

## Encoding ##
conv1 = layers.Conv2D( encoding_channels[0], kernel, activation = acti, padding = pad )( input_img )
pool1 = layers.MaxPooling2D( pooling, padding = pad )( conv1 )

conv2 = layers.Conv2D( encoding_channels[1], kernel, activation = acti, padding = pad )( pool1 )
pool2 = layers.MaxPooling2D( pooling, padding = pad )( conv2 )

conv3 = layers.Conv2D( encoding_channels[2], kernel, activation = acti, padding = pad )( pool2 )
pool3 = layers.MaxPooling2D( pooling, padding= pad )( conv3 )

## Decoding ##
conv4 = layers.Conv2D( decoding_channels[0], kernel, activation = acti, padding = pad )( pool3 )
pool4 = layers.UpSampling2D( pooling )( conv4 )

conv5 = layers.Conv2D( decoding_channels[1], kernel, activation = acti, padding = pad )( pool4 )
pool5 = layers.UpSampling2D( pooling )( conv5 )

conv6 = layers.Conv2D( decoding_channels[2], kernel, activation = acti, padding = pad )( pool5 )
pool6 = layers.UpSampling2D( pooling )( conv6 )

merge1 = layers.concatenate( [ conv1, pool6 ] )

output = layers.Conv2D( 1, kernel, activation = 'sigmoid', padding = 'same' )( merge1 )

## Compile Model ##
autoencoder = keras.models.Model( input_img, output )
autoencoder.compile( optimizer = 'adadelta', loss = 'binary_crossentropy' )

autoencoder.summary()

stdout.write( 'Finish Making Model! ' )

# Train Model #
stdout.write( 'Start Training Model... ' )

## Hyper Parameter ##
EPOCH = 50
BATCH = 20
SHUFFLE = True

history = autoencoder.fit( X_train, Y_train, epochs = EPOCH, batch_size = BATCH, shuffle = SHUFFLE )

stdout.write( 'Finish Train Model! ' )

# Test Model #
stdout.write( 'Start Testing Model... ' )
decoded_img = autoencoder.predict( X_test )

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

stdout.write( 'Finish Testing Model! ' )

stdout.write( 'Good job' )

output_stream.close()
error_stream.close()