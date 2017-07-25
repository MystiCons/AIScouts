
# Raspberry Version of park detection

The contents of this folder are used to detect cars on parks using a Raspberry pi.    
We used [picamera](https://picamera.readthedocs.io/en/release-1.13/) to take pictures with raspberry.    
The `rasp_run_detection.py` script also sends the data to thingsboard using [requests](http://docs.python-requests.org/en/master/).   

Dependancies:    
 * Python 3+   
 * picamera 
 * MachineLearning.rasp_model (In this repository)   
 * tqdm   
 * JSON   
 * Pickle   
 * Pillow   
 * TFlearn   
 * Tensorflow [(for raspberry)](https://github.com/samjabrahams/tensorflow-on-raspberry-pi)   

Installation guide can be found [here](https://github.com/MystiCons/AIScouts/blob/master/README.md)    

## ConfigureClient.py
An client (with ui) which can and should be used to configure your installed raspberry. Uses tcp to move information between your config pc and raspberry. 
[User Guide](https://github.com/MystiCons/AIScouts/wiki/Configure-Client-User-Guide).

## stream_server.py
A tcp server for raspberry pi, sends images to configure client and receives points of interest.

## rasp_find_objects_from_image.py
This module contains a class called ObjectRecognition which uses MachineLearning.rasp_model to predict crops from an image.   

## points.poi 
A pickle file containing the points of interest sent from ConfigureClient.

## rasp_run_detection.py
An example of a main loop, detecting cars with raspberry pi.

## Camera.py
Contains a class which can take pictures with picamera. get_frame() takes an picture and returns it.

