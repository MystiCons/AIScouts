from model import Model

paths = ['/Car/', '/Park/']
data_folder = "/media/cf2017/levy/tensorflow/parking_place2/"
mod = Model(paths, learning_rate=0.001, layers=4, epochs=10,
           data_folder=data_folder, model_name='park_model' + str(22))
mod.train_model()#saved_train_data_path=data_folder + "park_model19/")
