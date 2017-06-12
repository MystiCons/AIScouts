import tflearn
import numpy as np
import os
import cv2
import pickle
import matplotlib as plt
import heapq
import config

from tqdm import tqdm
from random import shuffle
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data,  dropout,  fully_connected
from tflearn.layers.estimator import regression


class Model:

    learning_rate = 1e-3
    img_size = 128
    layers = 3
    epochs = 10
    model = None
    debug = False
    model_name = 'default'
    data_folder = '/media/cf2017/levy/tensorflow/images/'
    test_path = ''
    label_folders = {}
    labels = {}

    def __init__(self, label_folders, data_folder='.',
                   learning_rate=1e-3, img_size=128, layers=3,
                   epochs=10, model_name=''):
        self.label_folders = label_folders
        self.learning_rate = learning_rate
        self.img_size = img_size
        self.layers = layers
        self.epochs = epochs
        self.debug = False
        self.model_name = model_name
        self.data_folder = data_folder

        convnet = input_data(shape=[None, self.img_size, self.img_size, 1], name='input')

        for _ in range(self.layers):
            convnet = conv_2d(convnet, 32, 2, activation='relu')
            convnet = max_pool_2d(convnet, 2)

            convnet = conv_2d(convnet, 64, 2, activation='relu')
            convnet = max_pool_2d(convnet, 2)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=self.learning_rate,
                             loss='categorical_crossentropy',
                             name='targets')

        self.model = tflearn.DNN(convnet, tensorboard_dir='log',
                            checkpoint_path=self.data_folder + 'checkpoints/')


    # Loads a saved instance of class Model
    @classmethod
    def load_model(cls, path):
        f = open(path, 'rb')
        tmp_dict = dill.load(f)

        f.close()
        print(tmp_dict)

        #self.__dict__.update(tmp_dict)
        #return pickle.load(path)

    def save_model(self):
        f = open(self.model_name + '.settings', 'wb')
        dill.dump(self.__dict__, f, 2)
        f.close()

    def load_testing_data(self, test_path):
        testing_data = []
        for img in tqdm(os.listdir(self.data_folder + test_path)):
            path = os.path.join(self.data_folder + test_path, img)
            img_name = img.split('.')[0]
            img_label = ''.join([i for i in img_name if not i.isdigit()])
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),
                             (self.img_size, self.img_size))
            testing_data.append([np.array(img), img_label])
        np.save(self.data_folder + self.model_name + '_test_data' + '.npy', testing_data)
        return testing_data


    # Loads training data from paths array, each path should be a folder which contains
    # images for each possible answer
    # Example: path[0] = 'images/cat/'
    #          path[1] = 'images/dog/'
    #          path[2] = 'images/cow/'
    # paths will be appended to initialized 'data_folder'
    # Each folder name will become a label
    def load_training_data(self):
        training_data = []
        for i in range(len(self.label_folders)):
            path = self.data_folder + self.label_folders[i]
            label = np.zeros(len(self.label_folders), np.int32)
            tmp_dict = {self.label_folders[i]: label}
            self.labels.update(tmp_dict)
            label[i] = 1
            for img in tqdm(os.listdir(path)):
                path2 = os.path.join(path, img)
                img = cv2.resize(cv2.imread(path2, cv2.IMREAD_GRAYSCALE),
                                 (self.img_size, self.img_size))
                training_data.append([np.array(img), np.array(label)])

        shuffle(training_data)
        np.save(self.data_folder + self.model_name + '_train_data' + '.npy', training_data)
        print(self.labels)
        return training_data

    def train_model(self):

        if os.path.isfile(self.data_folder + self.model_name + '_train_data' + '.npy'):
            train_data = np.load(self.data_folder + self.model_name + '_train_data' + '.npy')
        else:
            train_data = self.load_training_data()

        if not os.path.exists(self.data_folder + 'models'):
            os.makedirs(self.data_folder + 'models')
        os.chdir(self.data_folder + 'models')

        #Use 80% of the data for training and 20% for validation
        train_ratio = int(0.8 * len(train_data))
        test_ratio = int(0.2 * len(train_data))
        train = train_data[:train_ratio]
        test = train_data[-test_ratio:]

        X = np.array([i[0] for i in train]).reshape(-1, self.img_size, self.img_size, 1)
        Y = [i[1] for i in train]

        test_X = np.array([i[0] for i in test]).reshape(-1, self.img_size, self.img_size, 1)
        test_Y = [i[1] for i in test]

        self.model.fit({'input': X}, {'targets': Y}, self.epochs,
                  ({'input': test_X}, {'targets': test_Y}), show_metric=True,
                  shuffle=False, snapshot_epoch=True, run_id=self.model_name)

        self.model.save(self.model_name)
        os.chdir('..')


    def predict(self, img, predictions=1):
        data = img.reshape(self.img_size, self.img_size, 1)
        return self.model.predict([data])[0]
        #highest = heapq.nlargest(predictions, out)
        #ret = []



    def test_model(self, path='test_data/'):
        if os.path.isfile(self.data_folder + self.model_name + '_test_data' + '.npy'):
            test_data = np.load(self.data_folder + self.model_name + '_test_data' + '.npy')
        else:
            test_data = self.load_testing_data(path)

        fig = plt.figure()

        for label, data in enumerate(test_data[:12]):
            img_label = data[1]
            img_data = data[0]
            y = fig.add_subplot(3, 4, label + 1)
            orig = img_data
            data = img_data.reshape(self.img_size, self.img_size, 1)
            model_out = self.model.predict([data])[0]
            if np.argmax(model_out) == 1:
                str_label = 'No'
            else:
                str_label = 'Yes'
            y.imshow(orig, cmap='gray')
            plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)
        plt.show()


