# 1_DataPreprocessing.py
import os, cv2, glob
import numpy as np
import pandas as pd
import pickle as pkl
import convertImage as utils

WIDTH, HEIGHT = 256, 256
CROP_WIDTH, CROP_HEIGHT = 32, 32
CROP_IMAGE_NUMBER = (WIDTH // CROP_WIDTH) * (HEIGHT // CROP_HEIGHT)

ROOT_DIR = os.getcwd()
os.chdir(ROOT_DIR)

if ('Results' not in os.listdir()):
	os.mkdir('Results')
RESULT_DIR = os.path.join(ROOT_DIR, 'Results')

GEOLIFE_DIR = os.path.join(ROOT_DIR, 'Geolife')
DATA_DIR = os.path.join(GEOLIFE_DIR, 'Data')

EXPERIMENT_DATA = {
    'name' : 'DataPreprocessing',
    'number' : '1',
    'description' : 'GPS Data Preprocessing\n'
    }

os.chdir(RESULT_DIR)
curr_result_name = f"{EXPERIMENT_DATA['number']}_{EXPERIMENT_DATA['name']}_Results"
if (curr_result_name not in os.listdir()):
    os.mkdir(curr_result_name)

CURR_RESULT_DIR = os.path.join(RESULT_DIR, curr_result_name)
os.chdir(CURR_RESULT_DIR)

with open('Description.txt', 'w') as f:
    f.write(EXPERIMENT_DATA['description'])

print(f'ROOT_DIR:        {ROOT_DIR}')
print(f'DATA_DIR:        {DATA_DIR}')
print(f'RESULT_DIR:      {RESULT_DIR}')
print(f'CURR_RESULT_DIR: {CURR_RESULT_DIR}')

original_images = [ ]
generator = utils.Map2ImageGenerator(WIDTH, HEIGHT, 0)

print("========== Image Preprocessing 1 ==========")
cnt = 0
for directory in sorted(os.listdir(DATA_DIR)):
    os.chdir(os.path.join(DATA_DIR,directory,'Trajectory'))
    files = glob.glob('*plt')

    for i, file in enumerate(files):
        csv_file = pd.read_csv(file, names = [ 'lat', 'long', 'zero', 'alti', 'date_number', 'date_string', 'time'  ])[6:]
        csv_file.index = range(0, len(csv_file))
        original_images.append(generator.ConvertImage(csv_file))
        cnt += 1
        if (cnt % 3000 == 0):
            print(f'count:{cnt}')

os.chdir(CURR_RESULT_DIR)
if ('Image_Files' not in os.listdir()):
    os.mkdir('Image_Files')
os.chdir('Image_Files')

cnt = 1
for img in original_images:
    file_name = f'Geolife_trajectory{cnt}.png'
    cv2.imwrite(file_name, img)
    
    if (cnt % 3000 == 0):
        print(f'Save {file_name}')
    cnt += 1

print("========== Image Preprocessing 2 ==========")
cnt = 0
cropped_images_train = [ ]
for image in original_images:
    for i in range( 0, HEIGHT, CROP_HEIGHT ):
        for j in range( 0, WIDTH, CROP_WIDTH ):
            curr_image = [ ]
            for ii in range( i, i + CROP_HEIGHT ):
                curr_image.append( image[ii][j : j + CROP_WIDTH] )
            cropped_images_train.append( curr_image )
    cnt += 1
    if (cnt % 3000 == 0):
        print(f'count: {cnt}')

X_train = np.array(cropped_images_train).astype('float32') / 255.
X_train = np.reshape(X_train, (-1, CROP_HEIGHT, CROP_WIDTH, 1))

with open(f'Train_Data', 'wb') as f:
	pkl.dump(X_train, f)