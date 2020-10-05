# NoiseGenerator.py
import os, glob, random
import pandas as pd
import numpy as np

dataDir = '/taejin/Taejin/TrajectoryAugmentation/Trajectory_Data/'
csvDir = dataDir + '직접찍은데이터/'
workDir = dataDir + 'NoiseData/'

# Load csv Files
os.chdir( csvDir )
files = glob.glob( '*csv' )

# Add Noise to csv Files
os.chdir( workDir )

for i, file in enumerate(files):
	csv_file = pd.read_csv( file, name = [ 'lat', 'long', 'num' ], header = None )

	lat_range = ( min( csv_file['lat'] ), max( csv_file['lat'] ) )
	long_range = ( min( csv_file['long'] ), max( csv_file['long'] ) )

	last_num = csv_file['num'][-1]

	noise_size = int(csv_file.shape[0] / 0.3)
	noises = np.random.normal( ( noise_size, 2 ) )
	for i in range( noise_size ):
		random_value = random.randint( 0, last_num );

		noise_data = {
			'lat' : [ csv_file['lat'][random_value] + noises[i][0] ],
			'long' : [ csv_file['long'][random_value] + noises[i][1] ],
			'num' : [ last_num + ( i + 1 ) ]
		}
		noise_data = pd.dataFrame(noise_data)
		csv_file.append( noise_data, ignore_index = True )

	csv_file.to_csv( 'noise_' + i + '.csv' )
