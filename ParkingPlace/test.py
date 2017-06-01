import os
import math


LR_iterations = 4
Layer_iterations = 5
epoch_iterations = 3
start_epochs = 10
epochs = 0
start_rate = 0.01
start_layers = 2
image_file = '1.jpg'
crop_size = 200

for i in range(0, LR_iterations):
    rate = start_rate / math.pow(10, i)
    for j in range(0, Layer_iterations):
        layers = start_layers + j
        for h in range(0, epoch_iterations):
            epochs = start_epochs + h * 10
            model_name = 'L' + str(layers) + 'R' + str(rate) + 'E' + str(epochs)
            print('Current iteration: ' + ' LR: ' + str(rate) + ', Layers: ' + str(layers) + ', Epochs: ' + str(
                epochs) + ', crop_size: ' + str(crop_size) + ' image: ' + image_file + ' model name: ' + model_name)
            #print('Usage: -l <layers count> -r <learning rate> -m <model path> -e <epochs>')
            os.system("python3 TFTrainSinglePark.py -l " + str(layers) + " -r " + str(rate) + " -m " + model_name + " -e " + str(epochs))
            # print('Usage: -i <image path> -c <crop width> -m <model path>')
            os.system("python3 ImageRecognition.py -i " + image_file + " -c " + str(crop_size) + " -m models/" + model_name + " -r " + str(rate) + " -l " + str(layers))






