import os
import math




'''
4
5
3
10
0.01
2
200
'''


#
# Best result:
# ===========
# Layers = 4
# LR = 0.001
# Epochs = 20
# Crop = 180
# Shuffle = False
#


LR_iterations = 4
Layer_iterations = 4
epoch_iterations = 4
start_epochs = 10
epochs = 0
start_rate = 0.001
start_layers = 3
image_file = '1.jpg'
train = True
test = True
crop_size = 180
visualize = False


for i in range(0, LR_iterations):
    rate = start_rate / math.pow(10, i)
    for j in range(0, Layer_iterations):
        layers = start_layers + j
        for h in range(0, epoch_iterations):
            epochs = start_epochs + h * 10
            model_name = 'L' + str(layers) + 'R' + str(rate) + 'E' + str(epochs)
            print('Current iteration: ' + ' LR: ' + str(rate) + ', Layers: ' + str(layers) + ', Epochs: ' + str(
                epochs) + ', crop_size: ' + str(crop_size) + ' image: ' + image_file + ' model name: ' + model_name)
            if train:
                result = os.system("python3 TFTrainSinglePark.py -l " + str(layers) + " -r " + str(
                    rate) + " -m " + model_name + " -e " + str(epochs) + ' -v ' + str(visualize))
            if test:
                os.system("python3 ImageRecognition.py -i " + image_file + " -c " + str(
                    crop_size) + " -m " + model_name + " -r " + str(rate) + " -l " + str(layers) + ' -v ' + str(visualize))






