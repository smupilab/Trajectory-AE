# 4_TrajectorySearch.py
import os, cv2, glob
import numpy as np
import pandas as pd
import pickle as pkl

import matplotlib.pyplot as plt

WIDTH, HEIGHT = 256, 256
CROP_WIDTH, CROP_HEIGHT = 32, 32
CROP_IMAGE_NUMBER = (WIDTH // CROP_WIDTH) * (HEIGHT // CROP_HEIGHT)

ROOT_DIR = os.getcwd()
os.chdir(ROOT_DIR)

GEOLIFE_DIR = os.path.join(ROOT_DIR, 'Geolife')

if ('Results' not in os.listdir()):
	os.mkdir('Results')
RESULT_DIR = os.path.join(ROOT_DIR, 'Results')

IMAGE_DIR = os.path.join(RESULT_DIR, "1_DataPreprocessing_Results", "Image_Files")
LOW_DIMENSION_DIR = os.path.join(RESULT_DIR, "3_GetFeatureVector_Results", "Low_Dimension")

EXPERIMENT_DATA = {
    'name' : 'TrajectorySearch',
    'number' : '4',
    'description' : 'Search Top 10 Similar Trajectories\n'
    }

os.chdir(RESULT_DIR)
curr_result_name = f"{EXPERIMENT_DATA['number']}_{EXPERIMENT_DATA['name']}_Results"
if (curr_result_name not in os.listdir()):
    os.mkdir(curr_result_name)

CURR_RESULT_DIR = os.path.join(RESULT_DIR, curr_result_name)
os.chdir(CURR_RESULT_DIR)

with open('Description.txt', 'w') as f:
    f.write(EXPERIMENT_DATA['description'])

os.chdir(IMAGE_DIR)
file_names = sorted(glob.glob('*png'))

trajectory_images = [ ]
for file in file_names:
    trajectory_images.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE))

def load_low_dimension_data(dir_name: str) -> list:
    low_dimension = []

    os.chdir(os.path.join(LOW_DIMENSION_DIR, dir_name))
    for data in sorted(glob.glob('*data')):
        with open(data, 'rb') as f:
             low_dimension.append(pkl.load(f))
                
    return np.array(low_dimension)

map_table = pd.read_csv(GEOLIFE_DIR + '\Map_Table.csv', header = 0)
def match_image_to_lowDim(img_name: str) -> int:
    front = int(img_name.split('.')[0][len('Geolife_trajectory'):])
    
    return str(map_table[map_table['ImageNumber'] == front].iloc[0][' LowDimNumber'])

from scipy.spatial import distance

def get_MSE(list1: np.array, list2: np.array) -> float:
    similarities = [ ]
    for low_dim1, low_dim2 in zip(list1, list2):
        similarities.append(distance.cosine(low_dim1, low_dim2))

    return sum(np.square(similarities)) / len(similarities)

def search_similar_trajectories(target_img: str, top = 10) -> pd.DataFrame:
    print("Get Similar image with", target_img, ".....")
    base_low_dimension = load_low_dimension_data(match_image_to_lowDim(target_img) + 'th data')

    similar_images = []    
    for idx, curr_file_name in enumerate(file_names):
        if (idx % 3_000 == 0):
            print(f'{idx}/{len(file_names)}')
        lowDim_name = match_image_to_lowDim(curr_file_name)
        curr_low_dimension = load_low_dimension_data(lowDim_name + 'th data')

        similar_images.append([get_MSE(base_low_dimension, curr_low_dimension), curr_file_name])

    df = pd.DataFrame(similar_images, columns = ['MSE', 'Image File Name'])
    df = df.sort_values(by = 'MSE', axis = 0)
    
    os.chdir(CURR_RESULT_DIR)
    df.to_csv(f'Similar_trajectories_with({target_img}).csv')
    
    print("Done", target_img, ".....")
    
    return df.head(top)

targets = []
with open("TargetFiles.txt", "wt") as f:
	target_names = f.readlines()
	for target_name in target_names:
		targets.append(target_name)

similar_data_frames = []
for target in targets:
    similar_data_frames.append(search_similar_trajectories(target))

find_image = lambda name: trajectory_images[file_names.index(name)]

groups = []
for i, target in enumerate(targets):
    fig, ax = plt.subplots(1, 11, sharex = True, sharey = True, figsize = (110, 10))
    ax[0].imshow(find_image(target))
    
    df = similar_data_frames[i]
    names = []
    for j in range(1, 11):
        names.append(df.iloc[j - 1, 1])
        ax[j].set_title(f'{df.iloc[j - 1, 0]:.5f}', fontsize = 20)
        ax[j].imshow(find_image(names[-1]))
    
    groups.append(names)
    
    os.chdir(CURR_RESULT_DIR)
    plt.gray()
    plt.savefig(f'Similar_trajectories_with({target}).png', dpi = 300)
    plt.show()

def plot_overlap(images, overlap, file_name):
    group_images = [ ]
    os.chdir(IMAGE_DIR)
    for name in images:
        group_images.append(cv2.imread(name, cv2.IMREAD_GRAYSCALE))

    result = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for img in group_images:
        for row in range(HEIGHT):
            for col in range(WIDTH):
                if (img[row][col]):
                    result[row][col] += overlap
                    result[row][col] = min(result[row][col], 255)
                        
        
    os.chdir(CURR_RESULT_DIR)

    plt.gray()
    plt.imshow(result)
    plt.savefig(file_name, dpi = 100)
    plt.show()

for idx, group in enumerate(groups):
    plot_overlap(group, 100, f'Result_100_{idx}.png')