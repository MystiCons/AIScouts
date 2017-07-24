
# Machine learning
This folder has all the code containing machine learning techniques used in the parking place detection system.    
## model.py and rasp_model.py
These files describe a neural network and it's parameters.   
Rasp_model is a simple version of model.py. The simple version can't train the model or test the model, it should be used only if you want to predict using a trained model, which can be loaded using `load_model()` method.    
The full version `model.py` can be used to train a model from any images. The images should be put into a folder named by the label you want.  
`model.py` also has an implementation of deep convolutional generative adversarial neural network, which is capable of generating new images from a dataset.    
DCGAN during training:    
![DCGAN](https://github.com/MystiCons/AIScouts/blob/master/Images/cars.gif?raw=true)   
The first five images use the same random noise as an input for generator. Which means the network will try to draw the same image over and over.   
The last five use a new random noise for every batch, which means the image will always be a new random image.   
You can find this example [here](generate_images.py).    

Example how to train a new convolutional neural network model: 
```
from MachineLearning.model import Model

# Where the training data folders are. This folder will be used also as an output folder. 
data_folder = "/home/user/parking_place/"

# Folders containing the training data inside data_folder
# These folder names will become the labels for the data
image_folders = ['/Car/', '/Park/'] 

# Create a new instance
mod = Model(image_folders, learning_rate=0.001, layers=4, epochs=10,
           data_folder=data_folder, model_name='park_model')

# train_model() will load the images, train it and save the model to data_folder/models/model_name 
# Also the training data will be saved as .npy binary file partitions, every 40000 images a new partition will be created
# This will make the data loading much faster
mod.train_model()
# If you wish to use old data set you can use
mod.train_model(saved_train_data_path='/some/path/')
# Also if you have seperate validation data, you can create a folder for them required name is image_folder + _validation
# For example Car_validation and Park_validation, these folders should be in the same folder as the training data folders are
# Usage
mod.train_model(saved_train_data_path='/some/path/', separate_validation_data=True)

```

Example how to use a trained model:
```
from MachineLearning.rasp_model import Model

# Loads model in the given path
mod = Model.load_model("/home/user/parking_place/models/park_model22")

# Predicts what the image is using your neural network
label, confidence = mod.predict_with_path("./someimage.jpg")
# Use mod.predict(img) if you want to predict a preloaded Pillow image
# Opencv version of prediction is in model.py

```

Models folder has some trained models, using hundreds of thousands images of parked cars. 
   
