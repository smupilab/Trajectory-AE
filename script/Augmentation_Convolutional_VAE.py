# Augmentation_Convolutional_VAE.py

'''
Trajectory Data Augmentation using VAE
'''

import sys
stdout = sys.stdout

file_name = 'Trajectory-Augmentation_VAE'
output_stream = open( 'log({}).txt'.format( file_name ), 'wt' )
error_stream = open( 'errors({}).txt'.format( file_name ), 'wt' )
sys.stdout = output_stream
sys.stderr = error_stream

SIZE = 512

# Loading file and converting to Image #
stdout.write( 'Start Load files...' )

import os, glob
import numpy as np
from utils import convertImage

trajectoryAugmentataionDir = '/home/taejin/Trajectory-Augmentation/'
trajectoryDataDir = '/home/taejin/Trajectory_Data/'

currDir = trajectoryAugmentataionDir + '/script/'
dataDir = trajectoryDataDir + '/VirtualData/'

os.chdir( dataDir )
files = glob.glob( '*csv' )

trainSize = int( len( files ) * 0.8 )
trainFiles, testFiles = files[ : trainSize], files[trainSize : ]

X_train, Y_train = [ ], [ ]
for file in trainFiles:
	csv_file = pd.read_csv( file, names = [ 'lat', 'long', 'num' ], header = None )
	maxmin = convertImage.coorMaxMin( csv_file )
	X_train.append( convertImage.map2Image_remove( *maxmin, csv_file ) )
	Y_train.append( convertImage.map2Image( *maxmin, csv_file ) )

X_test = [ ]
for file in testFiles:
	csv_file = pd.read_csv( file, names = [ 'lat', 'long', 'num' ], header = None )
	maxmin = convertImage.coorMaxMin( csv_file )
	X_test.append( convertImage.map2Image_remove( *maxmin, csv_file ) )

X_train, Y_train = np.array( X_train ), np.array( Y_train )
X_test = np.array( X_test )

X_train = X_train / 255.
Y_train = Y_train / 255.
X_test = X_test / 255.

X_train = np.reshape( X_train, ( len( X_train ), SIZE, SIZE, 1 ) )
Y_train = np.reshape( Y_train, ( len( Y_train ), SIZE, SIZE, 1 ) )
X_test = np.reshape( X_test, ( len( X_test ), SIZE, SIZE, 1 ) )

stdout.write( 'Finish Load files! ' )

# Constructing Model #
import tensorflow as tf; tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from tensorflow.kears import layers
import tensorflow.keras.backend as K

def ComputeLatent( x ):
	mu, sigma = x
	eps = K.random_normal( shape = ( K.shape( mu )[0], K.int_shape( mu )[1] ) )

	return mu + K.exp( sigma / 2 ) * eps

latent = 2

## Constructing Encoder ##
stdout.write( 'Start Constructing Model... ' )
encoder_input = layers.Input( shape = ( SIZE, SIZE, 1 ) )
encoder_conv = layers.Conv2D( 128, 3, 2, 'same', activation = 'relu' )( encoder_input )
encoder_conv = layers.Conv2D( 256, 3, 2, 'same', activation = 'relu' )( encoder_conv )

encoder = layers.Flatten()( encoder_conv )

mu = layers.Dense( latent )( encoder )
sigma = layers.Dense( latent )( encoder )

latent_space = layers.Lambda( ComputeLatent, output_shape = ( latent_dim, ) )( [ mu, sigma ] )

conv_shape = K.int_shape( encoder_conv )

## Construction Decoder ##
decoder_input = layers.Input( shape = ( latent, ) )

units = conv_shape[1] * conv_shape[2] * conv_shape[3]
decoder = layers.Dense( units, 'relu' )( decoder_input )
decoder = layers.Reshape( ( conv_shape[1], conv_shape[2], conv_shape[3] ) )( decoder )

decoder_conv = layers.Conv2DTranspose( 256, 3, 2, 'same', activation = 'relu' )( decoder )
decoder_conv = layers.Conv2DTranspose( 128, 3, 2, 'same', activation = 'relu' )( decoder_conv )
decoder_output = layers.Conv2DTranspose( 1, 3, padding = 'same', activation = 'sigmoid' )( decoder_conv )

encoder = keras.models.Model( encoder_input, latent_space, name = 'Encoder' )
decoder = keras.models.Model( decoder_input, decoder_output, name = 'Decoder' )

vae = keras.models.Model( encoder_input, decoder( encoder( encoder_input ) ), name = 'VAE' )

encoder.summary()
decoder.summary()
vae.summary()

stdout.write( 'Finish Constructing Model! ' )

# Compiling model and Training #
stdout.write( 'Start Training Mode...' )

def KL_Reconstruction_Loss( true, pred ):
	reconstruction_loss = keras.losses.binary_crossentropy( K.flatten( true ), K.flatten( pred ) ) * SIZE * SIZE

	kl_loss = 1 + sigma - K.square( mu ) - K.exp( sigma )
	kl_loss = K.sum( kl_loss, axis = -1 )
	kl_loss *= -0.5

	return K.mean( reconstruction_loss + kl_loss )

vae.compile( 'adam', KL_Reconstruction_Loss )

BATCH_SIZE = 10
EPOCHS = 30

history = vae.fit( X_train, Y_train, BATCH_SIZE, EPOCHS )

stdout.write( 'Finish Training Model! ' )

# Test model #
stdout.write( 'Start Testing Model... ' )
decoded_img = vae.predict( X_test )

import matplotlib.pyplot as plt

os.chdir( currDir )

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

output_stream.close()
error_stream.close()
