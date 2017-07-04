import numpy as np
import os
import sys
import tflearn
import cv2
import matplotlib.pyplot as plt
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data,  dropout,  fully_connected
from tflearn.layers.estimator import regression
import getopt

from tqdm import tqdm
from random import shuffle

DATA_FOLDER = '/media/cf2017/levy/backup/tensorflow/images/'
IS_PARK_DIR = 'train_data/true/'
IS_NOT_PARK_DIR = 'train_data/false/'
TEST_DIR = 'test_data/'
MODELS_FOLDER = 'models/'
MODEL_NAME = ''
IMG_SIZE = 128
LR = 1e-4
layers = 3
epochs = 10
visualize = False


is_parking_place_imgs = []
is_not_parking_place_imgs = []


def load_training_data():
    training_data = []
    count = 0
    dir = DATA_FOLDER + IS_PARK_DIR
    for img in tqdm(os.listdir(dir)):
        count += 1
        label = [1, 0]
        path = os.path.join(dir,  img)
        img = cv2.resize(cv2.imread(path,  cv2.IMREAD_GRAYSCALE),  (IMG_SIZE,  IMG_SIZE))
        training_data.append([np.array(img),  np.array(label)])

    dir = DATA_FOLDER + IS_NOT_PARK_DIR
    for img in tqdm(os.listdir(dir)):
        label = [0, 1]
        path = os.path.join(dir,  img)
        img = cv2.resize(cv2.imread(path,  cv2.IMREAD_GRAYSCALE),  (IMG_SIZE,  IMG_SIZE))
        training_data.append([np.array(img),  np.array(label)])
        
    shuffle(training_data)
    np.save(DATA_FOLDER + 'train_data.npy',  training_data)
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

    model = tflearn.DNN(convnet,  tensorboard_dir='log', checkpoint_path='/home/cf2017/PycharmProjects/AIScouts/AIScouts/IPCameraDetection/checkpoints/')

    if not os.path.exists(DATA_FOLDER + 'models'):
        os.makedirs(DATA_FOLDER + 'models')
    os.chdir(DATA_FOLDER + 'models')

    train_ratio = int(0.8*len(train_data))
    test_ratio = int(0.2*len(train_data))
    train = train_data[:train_ratio]
    test = train_data[-test_ratio:]

    X = np.array([i[0] for i in train]).reshape(-1,  IMG_SIZE,  IMG_SIZE,  1)
    Y = [i[1] for i in train]

    test_X = np.array([i[0] for i in test]).reshape(-1,  IMG_SIZE,  IMG_SIZE,  1)
    test_Y = [i[1] for i in test]

    model.fit({'input': X},  {'targets': Y}, epochs,  ({'input': test_X}, {'targets': test_Y}), show_metric=True, shuffle=False,  snapshot_epoch=True,  run_id=MODEL_NAME)

    model.save(MODEL_NAME)

    os.chdir('..')

    return model


def load_testing_data():
    testing_data = []
    for img in tqdm(os.listdir(DATA_FOLDER + TEST_DIR)):
        path = os.path.join(DATA_FOLDER + TEST_DIR,  img)
        img_label = img.split('.')[0]
        img = cv2.resize(cv2.imread(path,  cv2.IMREAD_GRAYSCALE),  (IMG_SIZE,  IMG_SIZE))
        testing_data.append([np.array(img),  img_label])
    np.save(DATA_FOLDER + 'test_data.npy',  testing_data)
    return testing_data


def test_and_visualize(model):
    if (os.path.isfile(DATA_FOLDER + 'test_data.npy')):
        test_data = np.load(DATA_FOLDER + 'test_data.npy')
    else:
        test_data = load_testing_data()

    fig = plt.figure()

    for label, data in enumerate(test_data[:12]):
        img_label = data[1]
        img_data = data[0]
        y = fig.add_subplot(3, 4, label + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]
        print(model_out)
        if np.argmax(model_out) == 1:
            str_label = 'No'
        else:
            str_label = 'Yes'
        y.imshow(orig, cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()


def main(argv):
    global LR
    global layers
    global MODEL_NAME
    global epochs
    global visualize
    try:
        opts, args = getopt.getopt(argv, "hl:r:m:e:v:", ["layers", "rate", "model", "epochs", "visualize"])
    except getopt.GetoptError:
        print('Usage: -l <layers count> -r <learning rate> -m <output model name> -e <epochs> -v <visualize(True/False)>')
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
        elif opt in ('-v', '--visualize'):
            if arg.lower() == 'true':
                visualize = True
            else:
                visualize = False

    if(os.path.isfile(DATA_FOLDER + 'train_data.npy')):
        train_data = np.load(DATA_FOLDER + 'train_data.npy')
    else:
        train_data = load_training_data()

    model = train_network(train_data)

    if visualize:
        test_and_visualize(model)


if __name__ == '__main__':
    main(sys.argv[1:])
