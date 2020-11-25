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
from function import coorMaxMin, map2Image, map2Image_remove

currDir = os.environ['TrajetoryAugmentation'] + '/script/'
dataDir = os.environ['TrajectoryData'] + '/직접찍은데이터/'

files = glob.glob( '*csv' )

X_train, X_test = files[ :]
