import tflearn
import numpy as np
import os
import pickle

from PIL import Image
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


# Represents a convolutional neural network
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
    shuffle = False

    def __init__(self, label_folders, data_folder='./',
                 learning_rate=1e-3, img_size=128, layers=4,
                 epochs=10, model_name=''):
        if model_name == '':
            model_name = 'L' + str(layers) + 'R' + str(learning_rate) + 'E' + str(epochs)
        self.label_folders = label_folders
        self.learning_rate = learning_rate
        self.img_size = img_size
        self.layers = layers
        self.epochs = epochs
        self.debug = False
        self.model_name = model_name
        self.data_folder = data_folder

        if not self.data_folder[-1] == '/':
            self.data_folder += '/'

        for i in range(len(self.label_folders)):
            if not self.label_folders[i][-1] == '/':
                self.label_folders[i] += '/'
        self.conv_nn()

    def conv_nn(self):
        convnet = input_data(shape=[None, self.img_size, self.img_size, 1], name='input')

        for _ in range(self.layers):
            convnet = conv_2d(convnet, 32, 2, activation='relu')
            convnet = max_pool_2d(convnet, 2)

            convnet = conv_2d(convnet, 64, 2, activation='relu')
            convnet = max_pool_2d(convnet, 2)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, len(self.label_folders), activation='softmax')
        convnet = regression(convnet, optimizer='Adam', shuffle_batches=False, learning_rate=self.learning_rate,
                             loss='categorical_crossentropy',
                             name='targets')

        self.model = tflearn.DNN(convnet, tensorboard_dir='log',
                                 checkpoint_path=self.data_folder + 'checkpoints/' + self.model_name + '/')

    # Loads a saved instance of class Model
    @classmethod
    def load_model(cls, path):
        if not os.path.isfile(path + '.settings'):
            print('.settings file for file ' + path + ' not found!')
            exit(2)
        f = open(path + '.settings', 'rb')
        dict = pickle.load(f)
        mod = cls(dict['label_folders'], data_folder=dict['data_folder'],
                  learning_rate=dict['learning_rate'],
                  img_size=dict['img_size'], layers=dict['layers'],
                  epochs=dict['epochs'], model_name=dict['model_name'])
        if os.path.exists(path + '.meta'):
            mod.model.load(path)
            print('model loaded!')
        else:
            print('Configuration loaded, but no trained model found, call train_model method after this.')
        f.close()
        # Relable the model
        mod.relable()
        return mod



    def relable(self):
        tmp_dict = {}
        for i in range(len(self.label_folders)):
            tmp_dict.update({i: self.label_folders[i].split('/')[-2]})
        self.labels = tmp_dict

    def predict_with_path(self, path, predictions=1):
        try:
            img = Image.open(path)
            img = img.convert('L')
            img = img.thumbnail((self.img_size, self.img_size), Image.ANTIALIAS)
            img = np.asarray(img)
        except Exception:
            print(path + ' not found!')
            exit(2)
        data = img.reshape(self.img_size, self.img_size, 1)
        out = self.model.predict([data])[0]
        index = np.argmax(out)
        return self.labels[index], round(out[index], 3)

    def predict(self, img, predictions=1):
        img = img.thumbnail((self.img_size, self.img_size), Image.ANTIALIAS)
        img = np.asarray(img)
        data = img.reshape(self.img_size, self.img_size, 1)
        out = self.model.predict([data])[0]
        index = np.argmax(out)
        return self.labels[index], round(out[index], 3)



