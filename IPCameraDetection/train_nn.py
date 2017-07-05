from model import Model

paths = ['/Car/', '/Park/']
data_folder = "/media/cf2017/levy/tensorflow/parking_place/"
mod = Model(paths, learning_rate=0.001, layers=3, epochs=10,
           data_folder=data_folder, model_name='park_model' + str(19))
mod.train_model()