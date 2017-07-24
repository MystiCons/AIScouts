import tflearn
import numpy as np
import os
import cv2
import pickle
import matplotlib.pyplot as plt
from scipy.misc import imresize

from tqdm import tqdm
from random import shuffle
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d
from tflearn.layers.core import input_data,  dropout,  fully_connected
from tflearn.layers.estimator import regression
from tflearn.datasets import oxflower17, mnist
import tflearn
from tflearn.optimizers import Adam
import tensorflow as tf
import math

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
    shuffle = False
    separate_validation_data = False

    # DCGAN
    z_dim = 2000
    gen_net = None
    is_dcgan = False
    gen_model = None


    def __init__(self, label_folders, data_folder='./',
                   learning_rate=1e-3, img_size=128, layers=4,
                   epochs=10, image_channels=1, model_name='', DCGAN=False):
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
        self.is_dcgan = DCGAN

        if not self.data_folder[-1] == '/':
            self.data_folder += '/'

        for i in range(len(self.label_folders)):
            if not self.label_folders[i][-1] == '/':
                self.label_folders[i] += '/'
        if DCGAN:
            self.build_DCGAN()
        else:
            self.conv_nn()

    def conv_nn(self):
        convnet = input_data(shape=[None, self.img_size, self.img_size, self.image_channels], name='input')
        # TODO given layers are currently x2 the given layer amount
        for _ in range(self.layers):
            convnet = conv_2d(convnet, 32, 2, activation='relu')
            convnet = max_pool_2d(convnet, 2)

            convnet = conv_2d(convnet, 64, 2, activation='relu')
            convnet = max_pool_2d(convnet, 2)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.5)

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
                    epochs=dict['epochs'], model_name=dict['model_name'], DCGAN=dict['is_dcgan'])
        if os.path.exists(path + '.meta'):
            mod.model.load(path)
            print('model loaded!')
        else:
            print('Configuration loaded, but no trained model found, call train_model or DCGAN method after this.')
        f.close()
        # Relable the model
        mod.relable()
        if mod.is_dcgan:
            mod.gen_model = tflearn.DNN(mod.gen_net, session=mod.model.session)
        return mod

    def save_settings(self):
        dict = {key: value for key, value in self.__dict__.items() if
                not key.startswith('__') and not callable(value) and not key == 'model'}
        f = open(self.data_folder + 'models/' + self.model_name + '.settings', 'wb')
        pickle.dump(dict, f, 2)
        f.close()

    def save_model(self):
        dict = {key: value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(value)
                and not key == 'model' and not key == 'gen_net' and not key == 'gen_model'}
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
                    if len(training_data) == 0:
                        training_data = data.copy()
                    else:
                        training_data = np.concatenate((training_data, data))
                    self.save_data_set_partition(data, save_path)
                    data.clear()
                    iteration = 0
            count += 1
        # Save excess data
        if len(training_data) == 0:
            training_data = data.copy()
        else:
            training_data = np.concatenate((training_data, data))
        self.save_data_set_partition(data, save_path)
        data.clear()
        np.random.shuffle(training_data)
        return training_data

    def load_validation_data(self, save_folder=None):
        validation_data = np.array([])
        data = []
        count = 0
        iteration = 0

        if not os.path.isdir(self.data_folder + self.model_name):
            os.mkdir(self.data_folder + self.model_name)
        if not save_folder:
            save_path = self.data_folder + self.model_name + '/'
        else:
            save_path = save_folder

        for i in range(len(self.label_folders)):
            path = self.data_folder + self.label_folders[i].split('/')[1] + '_validation'
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
                    if len(validation_data) == 0:
                        validation_data = data.copy()
                    else:
                        validation_data = np.concatenate((validation_data, data.copy()))
                    self.save_validation_set_partition(data, save_path)
                    data.clear()
                    iteration = 0
            count += 1
        # Save excess data
        if len(validation_data) == 0:
            validation_data = data.copy()
        else:
            validation_data = np.concatenate((validation_data, data.copy()))
        self.save_validation_set_partition(data, save_path)
        data.clear()
        if len(validation_data) == 0:
            self.separate_validation_data = False
        else:
            np.random.shuffle(validation_data)
            self.separate_validation_data = True
        return validation_data

    def save_data_set_partition(self, data, save_path):
        np.save(save_path + self.model_name + '_train_data' + str(len(os.listdir(save_path))) + '.npy', data)

    def save_validation_set_partition(self, data, save_path):
        np.save(save_path + self.model_name + '_validation_data' + str(len(os.listdir(save_path))) + '.npy', data)

    def load_saved_data_set(self, path):
        data = np.array([])
        validation_data = np.array([])
        for file in os.listdir(path):
            if '_train_data' in file:
                load = np.load(path + file)
                if data.size == 0:
                    data = load
                else:
                    data = np.concatenate((data, load))
            elif '_validation_data' in file and self.separate_validation_data:
                load = np.load(path + file)
                if validation_data.size == 0:
                    validation_data = load
                else:
                    validation_data = np.concatenate((validation_data, load))

        if data.size == 0:
            print('Data set loading failed.')
            exit(2)
        if validation_data.size == 0:
            self.separate_validation_data = False
        np.random.shuffle(data)
        if self.separate_validation_data:
            np.random.shuffle(validation_data)

        return data, validation_data

    def train_model(self, saved_train_data_path=None, separate_validation_data=False):
        # If there is no separate validation data just load training data and use 20% of it as validation
        if separate_validation_data is False:
            self.separate_validation_data = False
        elif saved_train_data_path is None:
            validation_data = self.load_validation_data()

        if saved_train_data_path is None:
            train_data = self.load_training_data()
        else:
            if os.path.isdir(saved_train_data_path):
                train_data, validation_data = self.load_saved_data_set(saved_train_data_path)
                # If we have saved train_data, but validation_data wasn't found and separate_validation_data is true,
                # load validation data
                if not self.separate_validation_data and separate_validation_data:
                    validation_data = self.load_validation_data(save_folder=saved_train_data_path)
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

        if separate_validation_data and not self.separate_validation_data:
            print('Seperate validation data not found! Using 20% of the training data')
        if self.separate_validation_data:
            test = validation_data
            train = train_data
        else:
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

    def predict_with_path(self, path):
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


    def predict(self, img):
        img = cv2.resize(img, (self.img_size, self.img_size))
        data = img.reshape(self.img_size, self.img_size, 1)
        out = self.model.predict([data])[0]
        index = np.argmax(out)
        return self.labels[index], round(out[index], 3)

    def predict_DCGAN(self, noise):
        out = np.array(self.gen_model.predict({'input_gen_noise': noise}))
        return out

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

    def DCGAN(self, mnist_data=False, oxford_flower=False):
        # Preprocess
        if oxford_flower:
            X, Y = oxflower17.load_data(one_hot=True, resize_pics=(self.img_size, self.img_size))
        else:
            if not mnist_data:
                train = self.load_training_data()
                # i[0] is the image, i[1] the label
                X = np.array([i[0] for i in train]).reshape(-1, self.img_size, self.img_size, 1)
            else:
                X, Y, test_X, test_Y = mnist.load_data()
                X = X.reshape(-1, 28, 28, 1)
        preprocessed = []
        for image in tqdm(X):
            dim = (self.img_size, self.img_size)
            img = cv2.resize(image, dim)
            if oxford_flower:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #cv2.imshow('main', img)
            #cv2.waitKey()
            preprocessed.append(img)

        if oxford_flower:
            X = np.array(preprocessed.copy())
            for i in range(4):
                X = np.concatenate((X, np.array(preprocessed.copy())))

        else:
            X = np.array(preprocessed.copy())
        np.random.shuffle(X)
        X = X.reshape(-1, self.img_size, self.img_size, 1)
        total_samples = len(X)
        m = np.amax(X)
        print(m)

        disc_noise = np.random.normal(0., 1., size=[total_samples, self.z_dim])
        # Prepare target data to feed to the discriminator (0: fake image, 1: real image)
        y_disc_fake = np.zeros(shape=[total_samples])
        y_disc_real = np.ones(shape=[total_samples])
        # Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.
        y_disc_fake = tflearn.data_utils.to_categorical(y_disc_fake, 2)
        y_disc_real = tflearn.data_utils.to_categorical(y_disc_real, 2)

        # Prepare input data to feed to the stacked generator/discriminator
        gen_noise = np.random.normal(0., 1., size=[total_samples, self.z_dim])
        # Prepare target data to feed to the discriminator
        # Generator tries to fool the discriminator, thus target is 1 (e.g. real images)
        y_gen = np.ones(shape=[total_samples])
        y_gen = tflearn.data_utils.to_categorical(y_gen, 2)

        visual_callback = Visual_CallBack(self.gen_net)

        #logger_callback = tflearn.callbacks.TermLogger()

        # Start training, feed both noise and real images.
        self.model.fit(X_inputs={'input_gen_noise': gen_noise,
                          'input_disc_noise': disc_noise,
                          'input_disc_real': X},
                            Y_targets={'target_gen': y_gen,
                           'target_disc_fake': y_disc_fake,
                           'target_disc_real': y_disc_real},
                            n_epoch=self.epochs, show_metric=True,
                            shuffle=True, run_id=self.model_name,
                            callbacks=visual_callback) #logger_callback])

        self.save_model()
        self.gen_model = tflearn.DNN(self.gen_net, session=self.model.session)
        # Create another model from the generator graph to generate some samples
        # for testing (re-using same session to re-use the weights learnt).



    def build_DCGAN(self):
        gen_input = input_data(shape=[None, self.z_dim], name='input_gen_noise')
        input_disc_noise = input_data(shape=[None, self.z_dim], name='input_disc_noise')

        input_disc_real = input_data(shape=[None, self.img_size, self.img_size, 1], name='input_disc_real')

        disc_fake = self.discriminator(self.generator(input_disc_noise))
        disc_real = self.discriminator(input_disc_real, reuse=True)
        disc_net = tf.concat([disc_fake, disc_real], axis=0)

        gen_net = self.generator(gen_input, reuse=True)
        stacked_gan_net = self.discriminator(gen_net, reuse=True)

        disc_vars = tflearn.get_layer_variables_by_scope('Discriminator')

        disc_target = tflearn.multi_target_data(['target_disc_fake', 'target_disc_real'],
                                                shape=[None, 2])

        adam = Adam(learning_rate=self.learning_rate, beta1=0.9)
        disc_model = regression(disc_net, optimizer=adam,
                                placeholder=disc_target,
                                loss='categorical_crossentropy',
                                trainable_vars=disc_vars,
                                name='target_disc', batch_size=64,
                                op_name='DISC')

        gen_vars = tflearn.get_layer_variables_by_scope('Generator')
        gan_model = regression(stacked_gan_net, optimizer=adam,
                               loss='categorical_crossentropy',
                               trainable_vars=gen_vars,
                               name='target_gen', batch_size=64,
                               op_name='GEN')

        self.model = tflearn.DNN(gan_model, tensorboard_dir='log',
                          checkpoint_path=self.data_folder + 'checkpoints/' + self.model_name + '/')
        self.gen_net = gen_net

    def noise_layer(self, inp, std):
        noise = tf.random_normal(shape=tf.shape(inp), mean=0.0, stddev=std, dtype=tf.float32)
        return inp + noise
    # Generator
    def generator(self, x, reuse=False):
        s = self.img_size
        s2 = self.divide(s, 2)
        s4 = self.divide(s2, 2)
        s8 = self.divide(s4, 2)
        s16 = self.divide(s8, 2)

        with tf.variable_scope('Generator', reuse=reuse):
            x = tflearn.fully_connected(x, s16 * s16 * 256)
            x = tf.reshape(x, shape=[-1, s16, s16, 256])
            x = tflearn.dropout(x, 0.8)
            x = tflearn.batch_normalization(x)
            x = tflearn.conv_2d_transpose(x, 128, 5, [s8, s8], strides=[2, 2], activation='relu')
            self.noise_layer(x, 0.2)
            x = tflearn.batch_normalization(x)
            x = tflearn.conv_2d_transpose(x, 64, 5, [s4, s4], strides=[2, 2], activation='relu')
            self.noise_layer(x, 0.2)
            x = tflearn.batch_normalization(x)
            x = tflearn.conv_2d_transpose(x, 32, 5, [s2, s2], strides=[2, 2], activation='relu')
            self.noise_layer(x, 0.2)
            x = tflearn.batch_normalization(x)
            x = tflearn.conv_2d_transpose(x, 1, 2, [s, s], strides=[2, 2], activation='relu')
            return tf.nn.tanh(x)

    # Discriminator
    def discriminator(self, x, reuse=False):
        with tf.variable_scope('Discriminator', reuse=reuse):
            #self.noise_layer(x, 0.2)
            x = tflearn.conv_2d(x, 64, 2, activation='relu')
            x = tflearn.avg_pool_2d(x, 2)
            # x = tflearn.batch_normalization(x)
            x = tflearn.conv_2d(x, 128, 2, activation='relu')
            x = tflearn.avg_pool_2d(x, 2)
            # x = tflearn.batch_normalization(x)
            x = tflearn.conv_2d(x, 64, 2, activation='relu')
            x = tflearn.avg_pool_2d(x, 2)
            # x = tflearn.batch_normalization(x)
            x = tflearn.conv_2d(x, 128, 2, activation='relu')
            x = tflearn.avg_pool_2d(x, 2)
            # x = tflearn.batch_normalization(x)
            x = tflearn.fully_connected(x, 1024, activation='relu')
            x = tflearn.dropout(x, 0.8)
            x = tflearn.fully_connected(x, 2, activation='softmax')
            return x

    def divide(self, size, stride):
        return math.ceil(float(size) / float(stride))


class Visual_CallBack(tflearn.callbacks.Callback):
    gen = None
    gen_net = None
    z = None
    f = None
    a = None
    image_count = 0
    images_row = 0
    img_size = 0
    def __init__(self, gen_net, img_count=10, img_size=64):
        self.gen_net = gen_net
        # Noise input.
        self.z = np.random.normal(0., 1., size=[int(img_count / 2), 2000])
        self.image_count = img_count
        self.images_row = int(math.floor(math.sqrt(img_count)))
        self.img_size = img_size
        self.gen = tflearn.DNN(self.gen_net, session=tf.get_default_session())

    def on_batch_end(self, training_state, snapshot):
        _z = np.random.normal(0., 1., size=[int(self.image_count / 2), 2000])
        images = np.array(self.gen.predict({'input_gen_noise': self.z}))
        images2 = np.array(self.gen.predict({'input_gen_noise': _z}))
        images = np.concatenate((images, images2))
        images = denormalize_image(images)
        new_im = np.hstack([images[i] for i in range(self.image_count)])
        cv2.imshow("main", new_im)
        cv2.waitKey(1)
        dirlen = len(os.listdir('/media/cf2017/levy/tensorflow/DCGAN/time_lapse5/'))
        cv2.imwrite('/media/cf2017/levy/tensorflow/DCGAN/time_lapse5/' + str(dirlen) + '.bmp', new_im)

    def on_epoch_end(self, training_state):

        images = np.array(self.gen.predict({'input_gen_noise': self.z}))
        images = denormalize_image(images)
        new_im = np.hstack([images[i] for i in range(int(self.image_count / 2))])
        dirlen = len(os.listdir('/media/cf2017/levy/tensorflow/DCGAN/time_lapse4/'))
        cv2.imwrite('/media/cf2017/levy/tensorflow/DCGAN/time_lapse4/' + str(dirlen) + '.bmp', new_im)



def denormalize_image(images):
    return np.array([(x) * 255 for x in images], dtype='uint8')

