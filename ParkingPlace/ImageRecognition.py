
from fractions import Fraction
import numpy as np
import os
import tflearn
import cv2

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data,  dropout,  fully_connected
from tflearn.layers.estimator import regression


crop_size_width = 128
crop_size_height = 128
image_aspect = [16, 9]
image_width = 0
image_height = 0
image = None
move_ratio = 16
auto_config_search_accuracy = 0.01
auto_config_search_start_accuracy = 0.05
inverted_aspect = 'n'
model_name = ''
model = None
count2 = 0

TF_Image_size = 128


def load_tfmodel():
    global model
    global model_name
    convnet = input_data(shape=[None, TF_Image_size, TF_Image_size, 1], name='input')

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists(model_name + '.meta'):
        model.load(model_name)
        print('model loaded!')
    else:
        print('model not found!')


def search_image():
    curr_position = [0, 0]
    while True:
        while True:
            crop = image[curr_position[1]:curr_position[1]+crop_size_height,  curr_position[0]: curr_position[0]+crop_size_width]
            if not(predict(crop)):
                curr_position[0] = int(curr_position[0] + move_ratio)
            else:
                curr_position[0] = curr_position[0] + crop_size_width
            if curr_position[0] + crop_size_width >= image_width:
                break
        curr_position[1] = curr_position[1] + crop_size_height
        curr_position[0] = 0
        if curr_position[1] + crop_size_height >= image_height:
            break


def predict(crop):
    img = cv2.resize(crop, (TF_Image_size, TF_Image_size))
    data = img.reshape(TF_Image_size, TF_Image_size, 1)
    cv2.imshow('main2', crop)
    cv2.waitKey(1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 0:
        return True
    return False


def auto_configure():
    global crop_size_width
    global crop_size_height
    global auto_config_search_accuracy
    global auto_config_search_start_accuracy
    global inverted_aspect
    global image
    if inverted_aspect.lower() == 'n' or inverted_aspect.lower() == 'no':
        crop_size_width = int(image_width * auto_config_search_start_accuracy)
        crop_size_height = int(image_width * auto_config_search_start_accuracy)
        #crop_size_height = int((image_aspect[1] * crop_size_width) / image_aspect[0])
    else:
        crop_size_height = int(image_width * auto_config_search_start_accuracy)
        crop_size_width = int((image_aspect[1] * crop_size_height) / image_aspect[0])
    curr_position = [0, 0]
    move_ratio_width = int(crop_size_width * 0.1)
    move_ratio_height = int(crop_size_height * 0.1)
    found = False
    count = 1
    image2 = image.copy()
    cv2.imshow('main', image)
    cv2.waitKey(1)
    for _ in range(2):
        while True:
            if curr_position[1] + crop_size_height >= image_height:
                count += 1
                if inverted_aspect.lower() == 'n' or inverted_aspect.lower() == 'no':
                    crop_size_width = int(image_width * (auto_config_search_start_accuracy + auto_config_search_accuracy * count))
                    crop_size_height = int(
                        image_width * (auto_config_search_start_accuracy + auto_config_search_accuracy * count))
                    # crop_size_height = int((image_aspect[1] * crop_size_width) / image_aspect[0])
                else:
                    crop_size_height = int(image_width * (auto_config_search_start_accuracy + auto_config_search_accuracy * count))
                    crop_size_width = int((image_aspect[1] * crop_size_height) / image_aspect[0])
                print('iteration: ' + str(count) + ' crop size: ' + str([crop_size_width, crop_size_height]))
                curr_position = [0, 0]
                if crop_size_width >= image_width or crop_size_height >= image_height:
                    break
            else:
                curr_position[1] = curr_position[1] + move_ratio_height
                curr_position[0] = 0
            while True:
                crop = image2[curr_position[1]:curr_position[1] + crop_size_height, curr_position[0]:curr_position[0] + crop_size_width]
                if predict(crop):
                    cv2.rectangle(image, (curr_position[0], curr_position[1]),
                                  (curr_position[0] + crop_size_width, curr_position[1] + crop_size_height),
                                  (255, 0, 0), 2)
                    cv2.imshow('main', image)
                    cv2.waitKey(1)
                    curr_position[0] = int(curr_position[0] + crop_size_width / 2)
                else:
                    curr_position[0] = int(curr_position[0] + move_ratio_width)
                if curr_position[0] + crop_size_width >= image_width:
                    break
            if found:
                break

        if found:
            break
        else:
            if inverted_aspect == 'n':
                inverted_aspect = 'y'
                print('Changing to inverted aspect ratio')
            else:
                inverted_aspect = 'n'
                print('Changing to non inverted aspect ratio')
            curr_position = [0, 0]
            count = 0


    if found:
        print('Right crop size found: ' + str([crop_size_width, crop_size_height]))
    else:
        print('Auto configuration failed: Could not find any taught objects in the given image')
    cv2.imshow(image)
    cv2.waitKey()


def init():
    global image_width
    global image_height
    global crop_size_width
    global crop_size_height
    global image
    global inverted_aspect
    global model_name
    img_file = input('Image file location: ')
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    image_height, image_width = image.shape
    fraction = Fraction(image_width,  image_height)
    image_aspect[0] = fraction.numerator
    image_aspect[1] = fraction.denominator
    auto_config = ''
    while not auto_config.lower() == 'y' or auto_config.lower() == 'n':
        auto_config = input('Do you want to try auto configure the crop size? (y/n) ')
    inverted_aspect = input('Use inverted aspect ratio for crop? (Default: no) (y/n) ')
    model_name = input('Model path: ')
    load_tfmodel()
    if auto_config == 'n':
        crop_size_width = int(input('Crop width size (Must be divisible by given images aspect width [' + str(image_aspect[0]) + ']): '))
        while not(crop_size_width % image_aspect[0] == 0):
            print("Not divisible by aspect ratio")
            crop_size_width = int(input('Crop width size (Must be divisible by given images aspect width [' + str(image_aspect[0]) + ']): '))
        crop_size_height = int((image_aspect[1] * crop_size_width) / image_aspect[0])
    else:
        auto_configure()


def main():
    cv2.namedWindow("main", cv2.WINDOW_AUTOSIZE)
    init()
    search_image()


if __name__ == '__main__':
    main()
