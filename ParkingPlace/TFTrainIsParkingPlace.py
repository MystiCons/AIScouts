import numpy as np
import os
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data,  dropout,  fully_connected
from tflearn.layers.estimator import regression

from tqdm import tqdm
from random import shuffle
from PIL import Image

IS_PARK_DIR = './parking/'
IS_NOT_PARK_DIR = './notparking/'
TEST_DIR = './test_data/'
MODEL_NAME = 'test'
IMG_SIZE = 128
LR = 1e-3

is_parking_place_imgs = []
is_not_parking_place_imgs = []

def load_training_data():
    training_data = []
    for img in tqdm(os.listdir(IS_PARK_DIR)):
        label = [1, 0]
        path = os.path.join(IS_PARK_DIR,  img)
        img = Image.open(path)
        img.thumbnail(IMG_SIZE,  Image.ANTIALIAS)
        img = img.convert('L')
        training_data.append([np.array(img),  np.array(label)])
        
    for img in tqdm(os.listdir(IS_NOT_PARK_DIR)):
        label = [0,1]
        path = os.path.join(IS_NOT_PARK_DIR,  img)
        img = Image.open(path)
        img.thumbnail(IMG_SIZE,  Image.ANTIALIAS)
        img = img.convert('L')
        training_data.append([np.array(img),  np.array(label)])
        
    shuffle(training_data)
    np.save('train_data.npy',  training_data)
    return training_data
    
def train_network(train_data):
    convnet = input_data(shape=[None,  IMG_SIZE,  1],  name='input')
    
    convnet = conv_2d(convnet,  32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = conv_2d(convnet,  64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = fully_connected(convnet,  1024,  activation='relu')
    convnet = dropout(convnet,  0.8)
    
    convnet = fully_connected(convnet,  10,  activation='softmax')
    convnet = regression(convnet,  optimizer='adam',  learning_rate=LR,  loss='categorical_crossentropy',  name='targets')
    
    model = tflearn.DNN(convnet,  tensorboard_dir='log')
    
    if(os.path.exists('{}.meta'.format(MODEL_NAME))):
        model.load(MODEL_NAME)
        print('model loaded!')
    
    train = train_data[:-500]
    test = train_data[-500:]
        
    X = np.array([i[0] for i in train]).reshape(-1,  IMG_SIZE,  IMG_SIZE,  1)
    Y = [i[0] for i in train]
    
    test_X = np.array([i[0] for i in test]).reshape(-1,  IMG_SIZE,  IMG_SIZE,  1)
    test_Y = [i[0] for i in test]
    
    model.fit({'input': X},  {'targets': Y},  10,  ({'input': test_X}, {'targets': test_Y}),  500,  True,  MODEL_NAME)
    
    model.save(MODEL_NAME)
    
def main():
    if(os.path.isfile('train_data.npy')):
        train_data = np.load('train_data.npy')
    else:
        train_data = load_training_data()
    train_network(train_data)
    
    
if __name__ == '__main__':
    main()
    
