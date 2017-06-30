import tflearn
import numpy as np
import os
import cv2
import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm
from random import shuffle
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data,  dropout,  fully_connected
from tflearn.layers.estimator import regression
from tflearn.optimizers import Adam


# Represents a convolutional neural network
class Model:

    learning_rate = 1e-3
    img_size = 128
    layers = 3
    epochs = 10
    image_channels = 1
    model = None
    debug = False
    model_name = 'default'
    data_folder = '/media/cf2017/levy/tensorflow/images/'
    test_path = ''
    label_folders = {}
    labels = {}
    shuffle = True


    def __init__(self, label_folders, data_folder='./',
                   learning_rate=1e-3, img_size=128, layers=4,
                   epochs=10, image_channels=1, model_name=''):
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
        self.image_channels = image_channels

        if not self.data_folder[-1] == '/':
            self.data_folder += '/'

        for i in range(len(self.label_folders)):
            if not self.label_folders[i][-1] == '/':
                self.label_folders[i] += '/'
        self.conv_nn()

    def conv_nn(self):
        convnet = input_data(shape=[None, self.img_size, self.img_size, self.image_channels], name='input')

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

        if not os.path.isdir(self.data_folder + 'checkpoints/' + self.model_name + '/'):
            os.mkdir(self.data_folder + 'checkpoints/' + self.model_name + '/')
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
                   img_size=dict['img_size'],layers=dict['layers'],
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

    def save_settings(self):
        dict = {key: value for key, value in self.__dict__.items() if
                not key.startswith('__') and not callable(value) and not key == 'model'}
        f = open(self.data_folder + 'models/' + self.model_name + '.settings', 'wb')
        pickle.dump(dict, f, 2)
        f.close()

    def save_model(self):
        dict = {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(value) and not key == 'model'}
        self.model.save(self.data_folder + 'models/' + self.model_name)
        f = open(self.data_folder + 'models/' + self.model_name + '.settings', 'wb')
        pickle.dump(dict, f, 2)
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


    def load_training_data(self):
        training_data = np.array([])
        data = []
        count = 0
        iteration = 0

        if not os.path.isdir(self.data_folder + self.model_name):
            os.mkdir(self.data_folder + self.model_name)
        save_path = self.data_folder + self.model_name + '/'

        for i in range(len(self.label_folders)):
            path = self.data_folder + self.label_folders[i]
            label = np.zeros(len(self.label_folders), np.int32)
            label[i] = 1
            tmp_dict = {count: self.label_folders[i].split('/')[-2]}
            self.labels.update(tmp_dict)
            for img in tqdm(os.listdir(path)):
                path2 = os.path.join(path, img)
                try:
                    img = cv2.resize(cv2.imread(path2, cv2.IMREAD_GRAYSCALE),
                                     (self.img_size, self.img_size))
                    data.append([np.array(img), np.array(label)])
                except Exception:
                    print('failed to load image: ' + path2)
                iteration += 1
                if iteration > 40000:
                    if training_data.size == 0:
                        training_data = data.copy()
                    else:
                        training_data = np.concatenate((training_data, data))
                    self.save_data_set_partition(data, save_path)
                    data.clear()
                    iteration = 0
            count += 1
        # Save excess data
        if not iteration == 0:
            if training_data.size == 0:
                training_data = data
            else:
                training_data = np.concatenate((training_data, data))
            self.save_data_set_partition(data, save_path)
            data.clear()
        np.random.shuffle(training_data)
        return training_data

    def save_data_set_partition(self, data, save_path):
        np.save(save_path + self.model_name + '_train_data' + str(len(os.listdir(save_path))) + '.npy', data)

    def load_saved_data_set(self, path):
        data = np.array([])
        for file in os.listdir(path):
            load = np.load(path + file)
            if data.size == 0:
                data = load
            else:
                data = np.concatenate((data, load))
        if data.size == 0:
            print('Data set loading failed.')
            exit(2)
        np.random.shuffle(data)

        return data



    def train_model(self, saved_train_data_path=None):

        if saved_train_data_path is None:
            train_data = self.load_training_data()
        else:
            if os.path.isdir(saved_train_data_path):
                train_data = self.load_saved_data_set(saved_train_data_path)
                self.relable()
            else:
                print('Given data file not found, loading data.')
                train_data = self.load_training_data()

        if not os.path.exists(self.data_folder + 'models'):
            os.makedirs(self.data_folder + 'models')
        if os.path.isfile(self.data_folder + 'models/' + self.model_name + '.meta'):
            i = input('Model with the name: ' + self.model_name + ' already exists, do you want to continue training or load old one? ([c]ontinue/[l]oad)')
            if not i.lower() == 'continue' or 'c':
                self.model = Model.load_model(self.data_folder + 'models/' + self.model_name)
                return
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
                  shuffle=shuffle, snapshot_epoch=True, run_id=self.model_name)

        self.save_model()
        
    def relable(self):
        tmp_dict = {}
        for i in range(len(self.label_folders)):
            tmp_dict.update({i: self.label_folders[i].split('/')[-2]})
        self.labels = tmp_dict

    def predict_with_path(self, path, predictions=1):
        try:
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),
                             (self.img_size, self.img_size))
        except Exception:
            print(path + ' not found!')
            exit(2)
        data = img.reshape(self.img_size, self.img_size, 1)
        out = self.model.predict([data])[0]
        index = np.argmax(out)
        return self.labels[index], round(out[index], 3)


    def predict(self, img, predictions=1):
        img = cv2.resize(img, (self.img_size, self.img_size))
        data = img.reshape(self.img_size, self.img_size, 1)
        out = self.model.predict([data])[0]
        index = np.argmax(out)
        return self.labels[index], round(out[index], 3)

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
            str_label = self.predict(data)

            y.imshow(orig, cmap='gray')
            plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)
        plt.show()


class DataManipulation:

    data_folder = None

    def __init__(self, data_folder):
        self.data_folder = data_folder

    def try_cluster_training_data(self, path, clusters, img_size=128):
        dir = os.listdir(path)
        data = []
        data_orig = []
        for i in tqdm(range(len(dir))):
            try:
                img_orig = cv2.imread(path + str(i) + '.bmp', cv2.IMREAD_COLOR)
                img = cv2.resize(img_orig, (img_size, img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                data.append(np.array(img, dtype=np.float32))
                data_orig(np.array(img_orig))
            except Exception:
                pass

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        data = np.array(data)
        compactness, labels, centers = cv2.kmeans(data, clusters, None, criteria, 10, flags)
        A = ord('A')
        if not os.path.isdir(self.data_folder + 'clustered_images/'):
            os.mkdir(self.data_folder + 'clustered_images')
        for i in range(clusters):
            images = data[labels.ravel() == i]
            print(labels.ravel())
            if not os.path.isdir(self.data_folder + 'clustered_images/' + chr(A + i)):
                os.mkdir(self.data_folder + 'clustered_images/' + chr(A + i))
            count = 0
            for image in tqdm(images):
                cv2.imwrite(self.data_folder + 'clustered_images/' + chr(A + i) + '/' + str(count) + '.bmp',
                            image)
                count += 1

    def color_quantization(self, imgs_path, colors, img_size, save_images_path=None):
        dir = os.listdir(imgs_path)
        images = []
        for i in tqdm(range(len(dir))):
            img = cv2.resize(cv2.imread(imgs_path + str(i) + '.bmp', cv2.IMREAD_GRAYSCALE),
                             (img_size, img_size))
            images.append(np.array(img))

        new_images = []
        for i in range(len(images)):
            data = images[i].reshape((-1, 1))
            # convert to np.float32
            data = np.float32(data)
            # define criteria, number of clusters(K) and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret, label, center = cv2.kmeans(data, colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            # Now convert back into uint8, and make original image
            center = np.uint8(center)
            res = center[label.flatten()]
            res2 = res.reshape(images[i].shape)
            if save_images_path is None:
                new_images.append(res2)
            else:
                dirlen = len(os.listdir(save_images_path))
                cv2.imwrite(save_images_path + str(dirlen) + '.bmp',
                            res2)

        if save_images_path is not None:
            return new_images


    def flip_images(self, src, dest, img_size):
        dir = os.listdir(src)
        images = []
        for i in tqdm(range(len(dir))):
            img = cv2.imread(src + str(i) + '.bmp', cv2.IMREAD_GRAYSCALE)
            images.append(np.array(img))


        for i in range(len(images)):
            try:
                img = images[i]
                cv2.flip(images[i], 1, img)
                cv2.imwrite(dest + str(i) + '.bmp',
                            img)
            except Exception:
                pass


if __name__ == '__main__':
    inp = "/media/cf2017/levy/tensorflow/parking_place/new_training_data/"
    out = "/media/cf2017/levy/tensorflow/parking_place/new_training_data/"
    #mod = Model.load_model("/media/cf2017/levy/tensorflow/images/" + "models/testi1")
    #mod.try_cluster_training_data("/media/cf2017/levy/tensorflow/images/new_training_data/", 2)
    #mod.color_quantization(mod.data_folder + 'clustered_images/A/', mod.data_folder + 'clustered_images/A/')
    manipulator = DataManipulation("/media/cf2017/levy/tensorflow/parking_place/")
    #manipulator.try_cluster_training_data("/media/cf2017/levy/tensorflow/parking_place/new_training_data/", 3)
    #manipulator.color_quantization(inp + "Parks/", 24, 128, save_images_path=inp + "Parks/")
    #manipulator.color_quantization(inp + "Parks/", 16, 128, save_images_path=inp + "Parks/")
    #manipulator.color_quantization(inp + "Parks/", 12, 128, save_images_path=inp + "Parks/")
    #manipulator.color_quantization(inp + "Cars/", 16, 128, save_images_path=inp + "Cars/")
    #manipulator.color_quantization(inp + "Cars/", 12, 128, save_images_path=inp + "Cars/")
    manipulator.color_quantization(inp + "Cars/", 6, 128, save_images_path=inp + "Cars/")


    #manipulator.flip_images(inp + "Parks/", inp + "temp_parks/", 128)
    manipulator.flip_images(inp + "Cars/", inp + "temp_cars/", 128)



