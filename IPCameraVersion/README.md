
# IP Camera Version of park detection

The contents of this folder are used to detect cars on parks using an IP Camera.    
We used [Raspberry PI as an IP Camera.](https://github.com/silvanmelchior/RPi_Cam_Web_Interface)    
The `ipcamera_park_detection.py` script also sends the data to thingsboard using requests.   

Dependancies:    
 * OpenCV 2
 * DeepLearning.Model (In this repository)
 * tqdm
 * matplotlib
 * JSON
 * Pickle

## find_objects_from_image.py
This module contains a class called ObjectRecognition which uses DeepLearning.Model to predict crops from an image.   

When `find_objects(img)` is called, the script will allow the user to draw crops on the image which will be used as static crop positions which will be predicted on every image after the drawing. The crop positions can be saved to .poi file by calling `ObjectRecognition.save_poi('path/name')` and loaded with `ObjectRecognition.load_poi('path/name')`    

if ObjectRecognition class is created using `auto_find=True` the script tries to find objects which are in the `interesting_labels` array.    

Hotkeys for the crop drawing:
 * R: Reset points of interest
 * ESC: Stop drawing
 * Mouse 2 (Right button): Undo last crop
 * Mouse 1 (Hold): Draw crop

## Results    
The folder `results` has some image outputs of the auto_find feature using different models. The images are named after the models parameters. (Layers + Learning_rate + Epochs + Size)