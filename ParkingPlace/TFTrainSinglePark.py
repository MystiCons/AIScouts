import numpy as np
import os, sys
import tflearn
import cv2
import matplotlib.pyplot as plt
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data,  dropout,  fully_connected
from tflearn.layers.estimator import regression
import getopt

from tqdm import tqdm
from random import shuffle

IS_PARK_DIR = './train_data/true/'
IS_NOT_PARK_DIR = './train_data/false/'
TEST_DIR = './test_data/'
MODEL_NAME = ''
IMG_SIZE = 128
LR = 1e-4
layers = 3
epochs = 10

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
    global layers
    convnet = input_data(shape=[None, IMG_SIZE,  IMG_SIZE,  1],  name='input')

    for _ in range(layers):
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
        train_ratio = int(0.8*len(train_data))
        test_ratio = int(0.2*len(train_data))
        train = train_data[:train_ratio]
        test = train_data[-test_ratio:]
            
        X = np.array([i[0] for i in train]).reshape(-1,  IMG_SIZE,  IMG_SIZE,  1)
        Y = [i[1] for i in train]
        
        test_X = np.array([i[0] for i in test]).reshape(-1,  IMG_SIZE,  IMG_SIZE,  1)
        test_Y = [i[1] for i in test]
        
        model.fit({'input': X},  {'targets': Y}, epochs,  ({'input': test_X}, {'targets': test_Y}),  100,  True,  MODEL_NAME)

        if not os.path.exists('models'):
            os.makedirs('models')
        os.chdir('models')

        model.save(MODEL_NAME)

        os.chdir('..')

    return model


def load_testing_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,  img)
        img_label = img.split('.')[0]
        img = cv2.resize(cv2.imread(path,  cv2.IMREAD_GRAYSCALE),  (IMG_SIZE,  IMG_SIZE))
        testing_data.append([np.array(img),  img_label])
    np.save('test_data.npy',  testing_data)
    return testing_data
    

def main(argv):
    global LR
    global layers
    global MODEL_NAME
    global epochs
    try:
        opts, args = getopt.getopt(argv, "hl:r:m:e:", ["layers", "rate", "model", "epochs"])
    except getopt.GetoptError:
        print('Usage: -l <layers count> -r <learning rate> -m <output model name> -e <epochs>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: -l <layers count> -r <learning rate> -m <output model name> -e <epochs>')
        elif opt in ('-l', '--layers'):
            layers = int(arg)
        elif opt in ('-r', '--rate'):
            LR = float(arg)
        elif opt in ('-m', '--model'):
            MODEL_NAME = arg
        elif opt in ('-e', '--epochs'):
            epochs = int(arg)

    if(os.path.isfile('train_data.npy')):
        train_data = np.load('train_data.npy')
    else:
        train_data = load_training_data()

    model = train_network(train_data)
    ''' 
    if(os.path.isfile('test_data.npy')):
        test_data = np.load('test_data.npy')
    else:
        test_data = load_testing_data()

   
    
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
    
 '''
if __name__ == '__main__':
    main(sys.argv[1:])
