from model import Model
from find_objects_from_image import ObjectRecognition
from capture_ip_camera import IpCamera
import cv2

paths = ['train_data/true/', 'train_data/false/', 'train_data/taken/']
data_folder = "/media/cf2017/levy/tensorflow/images/"
#mod = Model(paths, learning_rate=0.001, layers=4, epochs=40,
#            data_folder='/media/cf2017/levy/tensorflow/images/', model_name='testi1')
#mod.save_settings()
#mod.train_model()

mod = Model.load_model(data_folder + "models/testi1")
#mod.test_model()
camera = IpCamera('http://192.168.51.247/html/cam_pic.php?time=1497446828170&pDelay=40000')
#mod.test_model()

while True:
    objectrec = ObjectRecognition(mod, ['true', 'taken'], visualize=False)
    objectrec.find_objects(camera.get_frame(), 150)



