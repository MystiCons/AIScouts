# AIScouts

**AIScouts** is a team of two members, whom try to map the possibilities of machine learning and computer vision.   
The main challenge of the team is to create a sensor which detects cars on a parking place and counts the free parks.   
The parking place challenge is done with [tensorflow](https://www.tensorflow.org/) and python 3. To create fast prototypes we use [tflearn](http://tflearn.org/) which is a higher-level API for tensorflow.   

---

There are two versions of the detection system:
**The First** version ([RaspberryVersion](https://github.com/MystiCons/AIScouts/tree/master/RaspberryVersion)) runs completely on raspberry pi, but requires configuration by the user to determine where the parks are. This is done by ConfigureClient.py which allows the user to connect to the raspberry pi through tcp server-client system. ConfigureClient can also collect images from the configured parks (Used as training data). This version uses Pillow to manipulate the images.   
   
**The second** version ([IPCameraVersion](https://github.com/MystiCons/AIScouts/tree/master/IPCameraVersion)) runs on a server, which fetches images from an ip camera. This version uses opencv 2 to manipulate the images.  

**DeepLearning** directory has our training scripts and model class. The model class describes the neural network model which can be saved and loaded in different scripts without writing the model again.    
There's a simple version (rasp_model.py) and full version (model.py).    
The full version uses opencv and simple version uses Pillow. Simple version has neither train method nor test methods. 
Simple version should be used if you only want to load a model and predict, not train it.

**Robotics** directory has BB-8 (Sphero) toy robots control scripts and a custom robot (Raspberry pi) control script.   

**Utils** has a toolkit we used to gather data and anything which helped our jobs.   

---
