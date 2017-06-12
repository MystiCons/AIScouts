from model import Model

paths = ['train_data/true/', 'train_data/false/']
mod = Model(paths, epochs=1, data_folder='/media/cf2017/levy/tensorflow/images/',
                  model_name='testi')

mod.train_model()



