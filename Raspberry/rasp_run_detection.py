from rasp_model import Model
from rasp_find_objects_from_image import ObjectRecognition
from Camera import Camera
from stream_server import StreamServer
import time
import requests
import json
import datetime
import os, sys
import threading
import base64
from io import BytesIO

mod = Model.load_model("../ParkingPlace/models/park_model14")
interesting_labels = ['Car', 'Park']
objectrec = ObjectRecognition(mod, interesting_labels, auto_find=False, visualize=False)

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

count = 0
try:
    objectrec.load_poi('../ParkingPlace/points')
except Exception:
    print('Points of interest couldnt be loaded, trying to auto find')

try:
    server = StreamServer()
    server_thread = threading.Thread(target=server.start)
    server_thread.daemon = True
    server_thread.start()
    print("Initialization Successful!")
    while True:
        if server.received_data:
            poi = server.get_poi()
            objectrec.saved_poi = poi
            objectrec.save_poi('../ParkingPlace/points')
        t = time.time()

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
        server.send_data_to_all(img)
        if elapsed_time >= 60:
            for key in summed_counts:
                summed_counts[key].clear()
            elapsed_time = 0
            start_time = time.time()
            try:
                buffer = BytesIO()
                img.save(buffer, format='JPEG')
                img_str = base64.b64encode(buffer.getvalue())
                data = {'Cars': avg_counts['Car'], 'Free': avg_counts['Park']}
                r = requests.post('http://192.168.51.140:8080/api/v1/gngqqCwoYPqr5qWmUw8v/telemetry',
                                  data=json.dumps(data))
                r = requests.post('http://192.168.51.140:8080/api/v1/gngqqCwoYPqr5qWmUw8v/attributes',
                                  data=json.dumps({'image': str(img_str)}))
                print(data)

            except Exception as e:
                print('Could not connect to thingsboard! ' + str(datetime.datetime.now()))
                print(e.with_traceback())
        #print("Looped in: " + str(round(t - time.time(), 4)) + " seconds")
        time.sleep(1)
        count += 1
except (KeyboardInterrupt, SystemExit):
    sys.exit()
finally:
    try:
        server.sock.close()
    except os.error as e:
        print(e.strerror)
