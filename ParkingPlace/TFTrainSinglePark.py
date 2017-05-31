import numpy as np
import os
import tflearn
import cv2
import matplotlib.pyplot as plt
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data,  dropout,  fully_connected
from tflearn.layers.estimator import regression

from tqdm import tqdm
from random import shuffle

IS_PARK_DIR = './train_data/true/'
IS_NOT_PARK_DIR = './train_data/false/'
TEST_DIR = './test_data/'
MODEL_NAME = 'test'
IMG_SIZE = 128
LR = 1e-4

is_parking_place_imgs = []
is_not_parking_place_imgs = []


def load_training_data():
    training_data = []
    count = 0
    for img in tqdm(os.listdir(IS_PARK_DIR)):
        count += 1
        label = [1, 0]
        path = os.path.join(IS_PARK_DIR,  img)
        img = cv2.resize(cv2.imread(path,  cv2.IMREAD_GRAYSCALE),  (IMG_SIZE,  IMG_SIZE))
        training_data.append([np.array(img),  np.array(label)])
        
    for img in tqdm(os.listdir(IS_NOT_PARK_DIR)):
        label = [0, 1]
        path = os.path.join(IS_NOT_PARK_DIR,  img)
        img = cv2.resize(cv2.imread(path,  cv2.IMREAD_GRAYSCALE),  (IMG_SIZE,  IMG_SIZE))
        training_data.append([np.array(img),  np.array(label)])
        
    shuffle(training_data)
    np.save('train_data.npy',  training_data)
    return training_data


def train_network(train_data):
    convnet = input_data(shape=[None, IMG_SIZE,  IMG_SIZE,  1],  name='input')
    
    convnet = conv_2d(convnet,  32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = conv_2d(convnet,  64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = conv_2d(convnet,  32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = conv_2d(convnet,  64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = conv_2d(convnet,  32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = conv_2d(convnet,  64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = fully_connected(convnet,  1024,  activation='relu')
    convnet = dropout(convnet,  0.8)
    
    convnet = fully_connected(convnet,  2,  activation='softmax')
    convnet = regression(convnet,  optimizer='adam',  learning_rate=LR,  loss='categorical_crossentropy',  name='targets')
    
    model = tflearn.DNN(convnet,  tensorboard_dir='log')
    
    if(os.path.exists(MODEL_NAME + '.meta')):
        model.load(MODEL_NAME)
        print('model loaded!')
    else:
        
        test_ratio = int(0.8*len(train_data))
        train_ratio = int(0.2*len(train_data))
        train = train_data[:train_ratio]
        test = train_data[-test_ratio:]
            
        X = np.array([i[0] for i in train]).reshape(-1,  IMG_SIZE,  IMG_SIZE,  1)
        Y = [i[1] for i in train]
        
        test_X = np.array([i[0] for i in test]).reshape(-1,  IMG_SIZE,  IMG_SIZE,  1)
        test_Y = [i[1] for i in test]
        
        model.fit({'input': X},  {'targets': Y}, 10,  ({'input': test_X}, {'targets': test_Y}),  100,  True,  MODEL_NAME)
        
        model.save(MODEL_NAME)
    
    return model
    
def  load_testing_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,  img)
        img_label = img.split('.')[0]
        img = cv2.resize(cv2.imread(path,  cv2.IMREAD_GRAYSCALE),  (IMG_SIZE,  IMG_SIZE))
        testing_data.append([np.array(img),  img_label])
    np.save('test_data.npy',  testing_data)
    return testing_data
    


def main():
    if(os.path.isfile('train_data.npy')):
        train_data = np.load('train_data.npy')
    else:
        train_data = load_training_data()
    if(os.path.isfile('test_data.npy')):
        test_data = np.load('test_data.npy')
    else:
        test_data = load_testing_data()
    
    model = train_network(train_data)
    
    fig = plt.figure()
    
    for label,  data in enumerate(test_data[:12]):
        img_label = data[1]
        img_data = data[0]
        y = fig.add_subplot(3, 4, label+1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE,  IMG_SIZE,  1)
        model_out = model.predict([data])[0]
        if np.argmax(model_out) == 1: str_label='No'
        else: str_label = 'Yes'
        y.imshow(orig,  cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()
    
    
if __name__ == '__main__':
    main()
