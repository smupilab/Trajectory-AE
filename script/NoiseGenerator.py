# NoiseGenerator.py
import os, glob, random
import pandas as pd
import numpy as np

dataDir = '/taejin/Taejin/TrajectoryAugmentation/Trajectory_Data/'
csvDir = dataDir + '직접찍은데이터/'

workDir = dataDir + 'Noises/'
noiseDir = workDir + 'NoiseDatas/'
originDir = workDir + 'OriginDatas/'

# Load csv Files
os.chdir( csvDir )
files = glob.glob( '*csv' )

# Add Noise to csv Files
for idx, file in enumerate(files):
	print( 'Add Noise to', file )
	original_file = pd.read_csv( filese, names = [ 'lat', 'long', 'num' ], header = None )
	copy_file = pd.DataFrame.copy( original_file )

	num = str(idx) if ( idx > 9 ) else '0' + str(idx)
	original_file.to_csv( originDir + 'origin_' + num + '.csv', header = False, index = False )

	last_num = copy_file.shape[0] - 1

	number_noise = int(copy_file.shape[0] * 0.3)
	noises = np.random.normal( size = ( number_noise, 2 ) )

	noise_data = { 'lat' : [ ], 'long' : [ ], 'num' : [ ] }
	for i in range( number_noise ):
		random_value = random.randint( 0, last_num );
		noise_data['lat'].append( copy_file['lat'][random_value] + noises[i][0] )
		noise_data['long'].append( copy_file['long'][random_value] + noises[i][1] )
		noise_data['num'].append( last_num + ( i + 1 ) )

	noise_data = pd.DataFrame(noise_data)
	copy_file.append( noise_data, ignore_index = True )

	copy_file.to_csv( noiseDir + 'noise_' + num + '.csv', header = False, index = False )

# Plot 
import matplotlib.pyplot as plt

os.chdir( originDir )
original_files = glob.glob( '*csv' )

os.chdir( noiseDir )
noise_files = glob.glob( '*csv' )

n = 10
for i in range( n ):
	ax = plt.subplot( 2, n, i + 1 )

	file_name = originDir + original_files[i]
	original_csv = pd.read_csv( file_name, names = ['lat', 'long', 'num'], header = None )

	datas = ( original_csv['lat'], original_csv['long'] )
	plt.scatter( datas[0], datas[1], c = 'black', s = 1 )
	plt.axis( 'off' )

	ax.get_xaxis().set_visible( False )
	ax.get_yaxis().set_visible( False )

	ax = plt.subplot( 2, n, n + i + 1 )

	file_name = noiseDir + noise_files[i]
	noise_csv = pd.read_csv( file_name, names = ['lat', 'long', 'num'], header = None )

	datas = ( noise_csv['lat'], noise_csv['long'] )
	plt.scatter( datas[0], datas[1], c = 'black', s = 1 )
	plt.axis( 'off' )

	ax.get_xaxis().set_visible( False )
	ax.get_yaxis().set_visible( False )

plt.savefig( 'Result.png', dpi = 300 )
