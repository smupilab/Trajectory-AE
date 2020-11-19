# VariationalAutoEncoder.py

##########
# Import #
##########
import tensorflow as tf; tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(25)
tf.executing_eagerly()

####################
# Define Functions #
####################
def ComputeLatent( x ):
	mu, sigma = x

	batch = K.shape(mu)[0]
	dim = k.int_shape(mu)[1]
	eps = K.random_normal( shape = ( batch, dim ) )

	return mu + K.exp( sigma / 2 ) * eps

def DisplayImageSequence( x_start, y_start, x_end, y_end, no_of_imgs ):
	x_axis = np.linspace( x_start, x_end, no_of_imgs )
	y_axis = np.linspace( y_start, y_end, no_of_imgs )

	x_axis = x_axis[:, np.newaxis]
	y_axis = y_axis[:, np.newaxis]

	new_points = np.hstack( ( x_axis, y_axis ) )
	new_images = decoder.predict( new_points )
	new_images = new_images.reshape( new_images.shape[0], new_images.shape[1], new_images.shape[2] )

	fig, axes = plt.subplots( ncols = no_of_imgs, sharex = False, sharey = True, figsize = ( 20, 7 ) )

	counter = 0
	for i in range( no_of_imgs ):
		axes[counter].imshow( new_images[i], cmap = 'gray' )
		axes[counter].get_xaxis().set_visible( False )
		axes[counter].get_yaxis().set_visible( False )

		counter += 1
	plt.show()


