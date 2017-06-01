
from fractions import Fraction
import numpy as np
import os, sys, getopt
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
image_edited = None
move_ratio = 16
auto_config_search_accuracy = 0.01
auto_config_search_start_accuracy = 0.14
inverted_aspect = 'n'
model_name = ''
model = None
count2 = 0
curr_position = [0, 0]
TF_Image_size = 128
layers = 0
LR = 0


def load_tfmodel(model_n):
    global model
    convnet = input_data(shape=[None, TF_Image_size, TF_Image_size, 1], name='input')

    for _ in range(layers):
        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists(model_n + '.meta'):
        model.load(model_n)
        print('model loaded!')
    else:
        print('model not found! ' + model_n)


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
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 0:
        return True
    return False


def search():
    global crop_size_width
    global crop_size_height
    global auto_config_search_accuracy
    global auto_config_search_start_accuracy
    global inverted_aspect
    global image

    curr_position = [0, 0]




def search_nearby():
    print("asd")


def move_crop_left(current_position, crop_size, pixels=10):
    result = current_position[0] - pixels
    if result >= 0:
        return result
    else:
        return 0


def move_crop_right(current_position, crop_size, pixels=10):
    result = current_position[0] + pixels
    if result <= image_width:
        return result
    else:
        return image_width


def move_crop_up(current_position, crop_size, pixels=10):
    result = current_position[1] - pixels
    if result >= 0:
        return result
    else:
        return 0


def move_crop_down(current_position, crop_size, pixels=10):
    result = current_position[1] + pixels
    if result <= image_height:
        return result
    else:
        return image_height


def get_next_crop(count, crop_size, accuracy_w=0.1, accuracy_h=0.1):
    global image
    global curr_position
    visualize = True
    visualize_img = None
    if visualize:
        visualize_img = image.copy()
    move_ratio_width = int(crop_size[0] * accuracy_w)
    move_ratio_height = int(crop_size[1] * accuracy_h)
    if curr_position[0] + move_ratio_width <= image_width:
        curr_position = move_crop_right(curr_position, move_ratio_width)
    elif curr_position[1] + crop_size_height <= image_height:
        move_crop_down(curr_position, move_ratio_height)
        curr_position[0] = 0
    else:
        curr_position[0, 0]
    if visualize:
        cv2.rectangle(visualize_img,
                      (curr_position[0], curr_position[1]),
                      (curr_position[0] + curr_position[0] + crop_size[0],
                       curr_position[1] + curr_position[1] + crop_size[1]),
                      (255, 0, 0),
                      2)
        cv2.imshow('visualize_movement', visualize_img)
        cv2.waitKey(1)

    return image[curr_position[1]:curr_position[1] + crop_size_height,
          curr_position[0]:curr_position[0] + crop_size_width]


def draw_heatmap():
    global crop_size_width
    global crop_size_height
    global auto_config_search_accuracy
    global auto_config_search_start_accuracy
    global inverted_aspect
    global image
    visualize = False
    visualize_img = None
    if visualize:
        visualize_img = image.copy()
    crop_size_height = crop_size_width
    curr_position = [0, 0]
    move_ratio_width = int(crop_size_width * 0.1)
    move_ratio_height = int(crop_size_height * 0.1)
    image2 = image.copy()
    if visualize:
        cv2.imshow('main', image)
        cv2.waitKey(1)
    while True:
        if curr_position[1] + crop_size_height >= image_height:
            break
        else:
            curr_position[1] = curr_position[1] + move_ratio_height
            curr_position[0] = 0
        while True:
            crop = image2[curr_position[1]:curr_position[1] + crop_size_height, curr_position[0]:curr_position[0] + crop_size_width]
            if predict(crop):
                cv2.circle(image, (int(curr_position[0] + crop_size_width / 2), int(curr_position[1] + crop_size_height / 2)),
                              2,
                              (255, 0, 0), -1)
                if visualize:
                    cv2.imshow('main', image)
                    cv2.waitKey(1)
                curr_position[0] = int(curr_position[0] + move_ratio_width)
                #curr_position[0] = int(curr_position[0] + crop_size_width / 4)
            else:
                curr_position[0] = int(curr_position[0] + move_ratio_width)
            if visualize:
                cv2.rectangle(visualize_img,
                              (curr_position[0], curr_position[1]),
                              (curr_position[0] + crop_size_width,
                               curr_position[1] + crop_size_height),
                              (255, 0, 0),
                              2)
                cv2.imshow('visualize_movement', visualize_img)
                cv2.waitKey(100)
                visualize_img = image.copy()
            if curr_position[0] + crop_size_width >= image_width:
                break

    if not os.path.exists('heatmaps'):
        os.makedirs('heatmaps')
    os.chdir('heatmaps')
    heatmap_img_file_name = model_name + 'C' + str(crop_size_width) + '.jpg'
    '''
    if os.path.isfile(heatmap_img_file_name):
        i = input('override the old file? (y/n) ')
        if i.lower() == 'n' or i.lower() == 'no':
            img_file_name = input('File ' + heatmap_img_file_name + ' already exists, give a new name (format: L<Layers>R<LearningRate>S<imagesize>): ')
        elif i.lower() == 'y' or i.lower() == 'yes':
            cv2.imwrite(heatmap_img_file_name, image)
    else:
        cv2.imwrite(heatmap_img_file_name, image)
    '''
    cv2.imwrite(heatmap_img_file_name, image)
    os.chdir('..')


def init(argv):
    global image_width
    global image_height
    global crop_size_width
    global image
    global layers
    global LR
    global model_name
    try:
        opts, args = getopt.getopt(argv, "hi:c:m:r:l:", ["image", "crop", "model", "rate", "layers"])
    except getopt.GetoptError:
        print('Usage: -i <image path> -c <crop width> -m <model path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: -i <image path> -c <crop width> -m <model path>')
        elif opt in ('-i', '--image'):
            img_file = arg
        elif opt in ('-c', '--crop'):
            crop_size_width = int(arg)
        elif opt in ('-m', '--model'):
            model_n = arg
            model_name = arg
        elif opt in ('-r', '--rate'):
            LR = float(arg)
        elif opt in ('-l', '--layers'):
            layers = int(arg)
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    image_height, image_width = image.shape
    load_tfmodel(model_n)
    #for i in range(0, len(image)):
    #    for j in range(0, len(image[i])):
    #        if image[i][j] < 80:
    #            image[i][j] = 0
    #        else:
    #            image[i][j] = 100



def main(argv):
    cv2.namedWindow("main", cv2.WINDOW_AUTOSIZE)
    init(argv)
    draw_heatmap()


if __name__ == '__main__':
    main(sys.argv[1:])
