from model import Model

paths = ['cat/', 'coffee/', 'computer/', 'cup/', 'dog/', 'mouse/']
data_folder = "/media/cf2017/levy/images/"
#mod = Model(paths, learning_rate=0.001, layers=3, epochs=50, data_folder='/media/cf2017/levy/images/')
#mod.train_model()


mod = Model.load_model(data_folder + "models/L3R0.001E50")
#print(mod.predict_with_path(data_folder + paths[0] + '1'))
#print(mod.predict_with_path(data_folder + paths[1] + '2'))
mod.test_model('test_data/')
