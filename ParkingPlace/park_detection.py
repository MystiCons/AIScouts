from model import Model
from find_objects_from_image import ObjectRecognition
from capture_ip_camera import IpCamera
import time
import cv2
import os
import requests
import json
import datetime


paths = ['/Car/', '/Park/']
data_folder = "/media/cf2017/levy/tensorflow/parking_place/"
#mod = Model(paths, learning_rate=0.001, layers=3, epochs=10,
#           data_folder=data_folder, model_name='park_model13')
#mod.save_settings()
#mod.train_model(saved_train_data="park_model11_train_data")


mod = Model.load_model(data_folder + "models/park_model13")
#mod.test_model()
camera1 = IpCamera('http://192.168.51.207/cam_pic.php', user='User', password='Salasana1')
camera2 = IpCamera('http://192.168.51.205/html/cam_pic.php', user='Parkki', password='S4lasana#07')
camera3 = IpCamera('http://192.168.51.131/html/cam_pic.php', user='Parkki', password='S4lasana#123')

camera = camera3
#mod.test_model()
interesting_labels = ['Car', 'Park']
objectrec = ObjectRecognition(mod, interesting_labels, auto_find=False, visualize=True)
cv2.namedWindow('main', cv2.WINDOW_NORMAL)
cv2.resizeWindow('main', 1280, 720)
if os.path.isfile('./points.poi'):
    objectrec.load_poi('./points')
else:
    frame = camera.get_frame()
    img, counts = objectrec.find_objects(frame, crop_size=[100, 128])
    objectrec.save_poi('./points')

start_time = time.time()
elapsed_time = 0
start_time2 = time.time()
elapsed_time2 = 0

summed_counts = {}
avg_counts = {}
for label in interesting_labels:
    summed_counts.update({label: []})
    avg_counts.update({label: 0})

while True:
    key = cv2.waitKey(1)
    if key == 27:
        print('Closing')
        camera.opener.close()
        exit()
    if key == ord('r'):
        objectrec.reset_poi()
    if key == ord('s'):
        objectrec.toggle_points_of_interest()

    frame = camera.get_frame()
    #print("Got new image in: " + str(round(t-time.time(), 4)) + " seconds")
    t = time.time()
    img, counts = objectrec.find_objects(frame.copy(), crop_size=[100, 128])
    objectrec.save_images_from_poi(frame, data_folder + 'new_training_data/', every_x_s=60)

    for key in counts:
        for v in counts[key]:
            summed_counts[key].append(v)

    #print("new image processed in: " + str(round(t-time.time(), 4)) + " seconds")
    cv2.imshow('main', img)
    elapsed_time = time.time() - start_time
    elapsed_time2 = time.time() - start_time2

    # Get avarage of predictions, and save it for the next x sec
    if elapsed_time2 >= 30:
        elapsed_time2 = 0
        start_time2 = time.time()
        for key in avg_counts:
            avg_counts[key] = 0
        for i in range(len(objectrec.saved_poi)):
            key_counts = {}
            for key in summed_counts:
                key_counts.update({key: summed_counts[key].count(i)})
            avg_counts[max(key_counts, key=key_counts.get)] += 1

    if elapsed_time >= 60:
        elapsed_time = 0
        start_time = time.time()
        try:
            data = {'Cars': avg_counts['Car'], 'Free': avg_counts['Park']}
            r = requests.post('http://192.168.51.140:8080/api/v1/gngqqCwoYPqr5qWmUw8v/telemetry',
                              data=json.dumps(data))
            cv2.imwrite(data_folder + 'temp/' + '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + '.bmp', img)
        except Exception:
            print('Could not connect to thingsboard! ' + str(datetime.datetime.now()))





