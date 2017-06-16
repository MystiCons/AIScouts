from model import Model
from find_objects_from_image import ObjectRecognition
from capture_ip_camera import IpCamera
import time
import cv2
import os


paths = ['train_data/true/', 'train_data/false/', 'train_data/taken/']
data_folder = "/media/cf2017/levy/tensorflow/images/"
#mod = Model(paths, learning_rate=0.001, layers=4, epochs=40,
#            data_folder='/media/cf2017/levy/tensorflow/images/', model_name='testi1')
#mod.save_settings()
#mod.train_model()

mod = Model.load_model(data_folder + "models/testi1")
#mod.test_model()
camera1 = IpCamera('http://192.168.51.207/cam_pic.php', user='User', password='Salasana1')
camera2 = IpCamera('http://192.168.51.212/html/cam_pic.php', user='Parkki', password='S4lasana#07')

camera = camera2
#mod.test_model()
objectrec = ObjectRecognition(mod, ['taken'], auto_find=False, visualize=True)
cv2.namedWindow('main', cv2.WINDOW_NORMAL)
cv2.resizeWindow('main', 1280, 720)
if os.path.isfile('./points.poi'):
    objectrec.load_poi('./points')
else:
    frame = camera.get_frame()
    img, counts = objectrec.find_objects(frame, crop_size=[60, 60])
    objectrec.save_poi('./points')
while True:
    t = time.time()
    frame = camera.get_frame()
    #print("Got new image in: " + str(round(t-time.time(), 4)) + " seconds")
    t = time.time()
    img, counts = objectrec.find_objects(frame, crop_size=[60, 60])
    #print("new image processed in: " + str(round(t-time.time(), 4)) + " seconds")
    cv2.imshow('main', img)
    key = cv2.waitKey(1)
    if key == 27:
        print('Closing')
        camera.opener.close()
        exit()
    if key == ord('r'):
        objectrec.reset_poi()
    if key == ord('s'):
        objectrec.toggle_points_of_interest()





