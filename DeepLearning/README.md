
## model.py and rasp_model.py
These files describe a neural network and it's parameters.   
Rasp_model is a simple version of model.py. The simple version can't train the model or test the model, it should be used only if you want to predict using a trained model, which can be loaded using `load_model()` method.    
The full version `model.py` can be used to train a model from any images. The images should be put into a folder named by the label you want.   

Example how to train a new model: 
```
from DeepLearning.model import Model

# Where the training data folders are. This folder will be used also as an output folder. 
data_folder = "/home/user/parking_place/"

# Folders containing the training data inside data_folder
# These folder names will become the labels for the data
image_folders = ['/Car/', '/Park/'] 

# Create a new instance
mod = Model(image_folders, learning_rate=0.001, layers=4, epochs=10,
           data_folder=data_folder, model_name='park_model')

# train_model() will load the images, train it and save the model to data_folder/models/model_name 
mod.train_model()

```

Example how to use a trained model:
```
from DeepLearning.rasp_model import Model

# Loads model in the given path
mod = Model.load_model("/home/user/parking_place/models/park_model22")

# Predicts what the image is using your neural network
label, confidence = mod.predict_with_path("./someimage.jpg")
# Use mod.predict(img) if you want to predict a preloaded Pillow image
# Opencv version of prediction is in model.py

```

Models folder has some trained models, using hundreds of thousands images of parked cars. 
   
