from rasp_model import Model
from rasp_find_objects_from_image import ObjectRecognition
from rasp_ip_camera import IpCamera
from Camera import Camera
import time
import requests
import json
import datetime
import os
from PIL import Image

mod = Model.load_model("../ParkingPlace/models/park_model14")
interesting_labels = ['Car', 'Park']
objectrec = ObjectRecognition(mod, interesting_labels, auto_find=False, visualize=False)

#camera = IpCamera('http://192.168.51.131/html/cam_pic.php', user='Parkki', password='S4lasana#123')
camera = Camera()
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

    t = time.time()
    try:
        objectrec.load_poi('../ParkingPlace/points')
    except Exception:
        print('Points of interest couldnt be loaded, trying to auto find')
    img, counts = objectrec.find_objects(camera.get_frame())
    for key in counts:
        for v in counts[key]:
            summed_counts[key].append(v)

    elapsed_time = time.time() - start_time
    elapsed_time2 = time.time() - start_time2

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
        except Exception:
            print('Could not connect to thingsboard! ' + str(datetime.datetime.now()))
    print("Looped in: " + str(round(t - time.time(), 4)) + " seconds")


