# 2_ModelTrain.py
import os
import pickle as pkl

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

WIDTH, HEIGHT = 256, 256
CROP_WIDTH, CROP_HEIGHT = 32, 32

ROOT_DIR = os.getcwd()
os.chdir(ROOT_DIR)

if ('Results' not in os.listdir()):
	os.mkdir('Results')
RESULT_DIR = os.path.join(ROOT_DIR, 'Results')

EXPERIMENT_DATA = {
    'name' : 'TrainModel',
    'number' : '2',
    'description' : 'Train CAE model\n'
    }

os.chdir(RESULT_DIR)
curr_result_name = f"{EXPERIMENT_DATA['number']}_{EXPERIMENT_DATA['name']}_Results"
if (curr_result_name not in os.listdir()):
    os.mkdir(curr_result_name)

CURR_RESULT_DIR = os.path.join(RESULT_DIR, curr_result_name)
os.chdir(CURR_RESULT_DIR)

with open('Description.txt', 'w') as f:
    f.write(EXPERIMENT_DATA['description'])

af, pd = 'relu', 'same' # activation function and padding value

########## Construct Encoder ##########
encode_input = layers.Input((CROP_HEIGHT, CROP_WIDTH, 1))
x = layers.Conv2D(32, (3, 3), activation = af, padding = pd)(encode_input)
x = layers.MaxPooling2D((2, 2), padding = pd)(x)

x = layers.Conv2D(32, (3, 3), activation = af, padding = pd)(x)
x = layers.MaxPooling2D((2, 2), padding = pd)(x)

x = layers.Conv2D(16, (3, 3), activation = af, padding = pd)(x)
x = layers.MaxPooling2D((2, 2), padding = pd)(x)

x = layers.Conv2D(16, (3, 3), activation = af, padding = pd)(x)
x = layers.MaxPooling2D((2, 2), padding = pd)(x)

feature_map_shape = x.shape
flatten_size = feature_map_shape[1] * feature_map_shape[2] * feature_map_shape[3]

x = layers.Flatten()(x)
encode_output = layers.Dense(flatten_size , activation = af)(x)

encoder = keras.Model(encode_input, encode_output, name = 'Encoder')
encoder.summary()

########## Construct Decoder ##########
decode_input = layers.Input((flatten_size))

x = layers.Dense(flatten_size, activation = af)(decode_input)
x = layers.Reshape(feature_map_shape[1:])(x)

x = layers.Conv2DTranspose(16, (3, 3), activation = af, padding = pd)(x)
x = layers.UpSampling2D((2, 2))(x)

x = layers.Conv2DTranspose(16, (3, 3), activation = af, padding = pd)(x)
x = layers.UpSampling2D((2, 2))(x)

x = layers.Conv2DTranspose(32, (3, 3), activation = af, padding = pd)(x)
x = layers.UpSampling2D((2, 2))(x)

x = layers.Conv2DTranspose(32, (3, 3), activation = af, padding = pd)(x)
x = layers.UpSampling2D((2, 2))(x)

decode_output = layers.Conv2DTranspose(1, (3, 3), activation = af, padding = pd)(x)

decoder = keras.Model(decode_input, decode_output, name = 'Decoder')
decoder.summary()

########## Construct Autoencoder ##########
auto_encoder = keras.Model(encode_input, decoder(encoder(encode_input)), name = 'Auto_Encoder')
auto_encoder.summary()

auto_encoder.compile('adam', loss = 'mse')

EPOCH = 300
BATCH = 256

os.chdir(os.path.join(RESULT_DIR, "1_DataPreprocessing_Results"))
X_train = pkl.load('Train_Data')

history = auto_encoder.fit(X_train, X_train, epochs = EPOCH, batch_size = BATCH)

os.chdir(CURR_RESULT_DIR)
encoder.save('encoder.h5')
decoder.save('decoder.h5')
auto_encoder.save('auto_encoder.h5')