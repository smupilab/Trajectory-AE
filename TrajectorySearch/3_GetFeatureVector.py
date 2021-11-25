# 3_GetFeatureVector.py
import os
import numpy as np
import pickle as pkl

from tensorflow import keras

WIDTH, HEIGHT = 256, 256
CROP_WIDTH, CROP_HEIGHT = 32, 32
CROP_IMAGE_NUMBER = (WIDTH // CROP_WIDTH) * (HEIGHT // CROP_HEIGHT)

ROOT_DIR = os.getcwd()
os.chdir(ROOT_DIR)

if ('Results' not in os.listdir()):
	os.mkdir('Results')
RESULT_DIR = os.path.join(ROOT_DIR, 'Results')

EXPERIMENT_DATA = {
    'name' : 'GetFeatureVector',
    'number' : '3',
    'description' : 'Get Feature Vecotor\n'
    }

os.chdir(RESULT_DIR)
curr_result_name = f"{EXPERIMENT_DATA['number']}_{EXPERIMENT_DATA['name']}_Results"
if (curr_result_name not in os.listdir()):
    os.mkdir(curr_result_name)

CURR_RESULT_DIR = os.path.join(RESULT_DIR, curr_result_name)
os.chdir(CURR_RESULT_DIR)

with open('Description.txt', 'w') as f:
    f.write(EXPERIMENT_DATA['description'])

os.chdir(os.path.join(RESULT_DIR, "1_DataPreprocessing_Results"))
X_train = pkl.load('Train_Data')

os.chdir(os.path.join(RESULT_DIR, "2_TrainModel_Results"))
encoder = keras.models.load_model('encoder.h5')
auto_encoder = keras.models.load_model('auto_encoder.h5')

low_dimension_data, prevIdx = [ ], 0
for i in range(0, len(X_train), 10_000):
    low_dimension_data.extend(encoder(X_train[prevIdx:i]))
    prevIdx = i
low_dimension_data.extend(encoder(X_train[prevIdx:]))

low_dimension_data = np.array(low_dimension_data)
print(X_train.shape)
print(low_dimension_data.shape)

os.chdir(CURR_RESULT_DIR)
if ('Low_Dimension' not in os.listdir()):
    os.mkdir('Low_Dimension')
os.chdir('Low_Dimension')

for img_idx in range(0, len(low_dimension_data), CROP_IMAGE_NUMBER):
    os.chdir(os.path.join(CURR_RESULT_DIR, 'Low_Dimension'))
    os.mkdir(f'{img_idx // CROP_IMAGE_NUMBER}th data')
    os.chdir(f'{img_idx}th data')

    for i in range(CROP_IMAGE_NUMBER):
        with open(f'{i}th data', 'wb') as f:
            pkl.dump(low_dimension_data[img_idx + i], f)