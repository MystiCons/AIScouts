# AIScouts

## About AIScouts

**AIScouts** is a team of two members, who try to map the possibilities of machine learning and computer vision.   
The main challenge of the team is to create a sensor which detects cars on a parking place and counts the free parks.   
The parking place challenge is done with [Tensorflow](https://www.tensorflow.org/) and Python 3. To create fast prototypes we use [tflearn](http://tflearn.org/) which is a higher-level API for Tensorflow.   

---

## Directories

There are two versions of the detection system:    
**The First** version ([RaspberryVersion](https://github.com/MystiCons/AIScouts/tree/master/RaspberryVersion)) runs completely on raspberry pi but requires configuration by the user to determine where the parks are. This is done by ConfigureClient.py which allows the user to connect to the raspberry pi through TCP server-client system. ConfigureClient can also collect images from the configured parks (Used as training data). This version uses Pillow to manipulate the images.   
   
**The second** version ([IPCameraVersion](https://github.com/MystiCons/AIScouts/tree/master/IPCameraVersion)) runs on a server, which fetches images from an IP camera. This version uses OpenCV 2 to manipulate the images.   
We used [Raspberry PI as an IP Camera.](https://github.com/silvanmelchior/RPi_Cam_Web_Interface)    

**DeepLearning** directory has our training scripts and model class. The model class describes the neural network model which can be saved and loaded in different scripts without writing the model again.    
There's a simple version (rasp_model.py) and full version (model.py).    
The full version uses OpenCV and simple version uses Pillow. Simple version has neither train method nor test methods. 
The simple version should be used if you only want to load a model and predict, not train it.   

**Robotics** directory has BB-8 (Sphero) toy robots control scripts and a custom robot (Raspberry Pi) control script.   

**Utils** has a toolkit we used to gather data and anything which helped our jobs.   

**InstallScripts** directory has some dependancy installation scripts.   

---

## RaspberryVersion Installation

On raspberry pi:
```
# Install python3
sudo apt-get install python3 python3-dev

# Install build dependancies
sudo apt-get update

sudo apt-get upgrade -y
sudo apt-get install -y build-essential cmake python3-pip python3-dev cmake libjpeg8-dev

# Enable camera
sudo raspi-config # -> Interfacing options -> Camera -> Enable

# install dependancies
sudo pip3 install numpy pickle tqdm json picamera

# install tensorflow (https://github.com/samjabrahams/tensorflow-on-raspberry-pi)
wget https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v1.1.0/tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl
sudo pip3 install tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl

# If you get an error on import tensorflow:
sudo pip3 uninstall mock
sudo pip3 install mock

# Install Pillow and tflearn
sudo pip3 install Pillow
sudo pip3 install git+https://github.com/tflearn/tflearn.git

# Clone this repository
git clone https://github.com/MystiCons/AIScouts

# Run RaspberryVersion/rasp_run_detection.py
cd AIScouts
python3 RaspberryVersion/rasp_run_detection.py

```

On config pc:

```
# Install python3
sudo apt-get install python3 python3-dev

# Install build dependancies
sudo apt-get update

sudo apt-get upgrade -y
sudo apt-get install -y build-essential cmake python3-pip python3-dev cmake libjpeg8-dev

# install dependancies
sudo pip3 install numpy pickle json tqdm

# install tensorflow (https://github.com/samjabrahams/tensorflow-on-raspberry-pi)
wget https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v1.1.0/tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl
sudo pip3 install tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl

# If you get an error on import tensorflow:
sudo pip3 uninstall mock
sudo pip3 install mock

# Install Pillow and tflearn
sudo pip3 install Pillow 
sudo pip3 install git+https://github.com/tflearn/tflearn.git

```

Usage guide for raspberry version [here](https://github.com/MystiCons/AIScouts/blob/master/RaspberryVersion/README.md)
