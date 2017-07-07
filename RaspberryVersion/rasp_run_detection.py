file = open('/home/pi/asd', 'a')
file.write('\nStarted')
import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from DeepLearning.rasp_model import Model
from RaspberryVersion.rasp_find_objects_from_image import ObjectRecognition
from RaspberryVersion.Camera import Camera
from RaspberryVersion.stream_server import StreamServer
import time
import requests
import json
import datetime
import os, sys
import threading
import base64
import subprocess
from io import BytesIO
import traceback
file.write('\nImported')
token = 'gngqqCwoYPqr5qWmUw8v'
token2 = 'Eo8KxecNVvn9AVg3VXjS'
token3 = 'gAr2fUXsBYuPUMyCUF7F'

curr_token = token3
try:
    ip = subprocess.getoutput("/sbin/ifconfig wlan0 | grep 'inet addr:' | cut -d: -f2 | awk '{ print $1}'")
    if '192' not in ip and '172' not in ip:
        ip = subprocess.getoutput("/sbin/ifconfig eth0 | grep 'inet addr:' | cut -d: -f2 | awk '{ print $1}'")
    ip2 = subprocess.getoutput("/sbin/ifconfig ppp0 | grep 'inet addr:' | cut -d: -f2 | awk '{ print $1}'")

    r = requests.post('http://192.168.51.140:8080/api/v1/'+curr_token+'/attributes',
                                      data=json.dumps({'ipAddress': ip}))
    r = requests.post('http://192.168.51.140:8080/api/v1/'+curr_token+'/attributes',
                                      data=json.dumps({'vpnAddress': ip2}))
    file.write('\nSend Ips')
    #print(ip)
    mod = Model.load_model("/home/pi/dev/AIScouts/DeepLearning/models/park_model22")
    file.write('\nLoaded model')
    interesting_labels = ['Car', 'Park']
    objectrec = ObjectRecognition(mod, interesting_labels, auto_find=False, visualize=False)
    file.write('\nCreated objectrec')
    camera = Camera()
    file.write('\nCreated Camera')
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
    file.write('\nLoading poi')
    objectrec.load_poi('/home/pi/dev/AIScouts/IPCameraVersion/points')
    file.write('\nLoaded poi')
except Exception as e:
    r = requests.post('http://192.168.51.140:8080/api/v1/' + curr_token + '/attributes',
                      data=json.dumps({'error': traceback.print_exc()}))

file.close()
try:
    server_launched = False
    while not server_launched:
        try:
            server = StreamServer(objectrec.saved_poi)
            server_launched = True
        except os.error as e:
            print(e.strerror)
    server_thread = threading.Thread(target=server.start)
    server_thread.daemon = True
    server_thread.start()
    print("Initialization Successful!")
    while True:
        if server.received_data:
            poi = server.get_poi()
            objectrec.saved_poi = poi
            objectrec.save_poi('/home/pi/dev/AIScouts/IPCameraVersion/points')
        t = time.time()
        img_orig = camera.get_frame()
        img, counts = objectrec.find_objects(img_orig.copy())

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
        server.send_data_to_all(img, img_orig)
        if elapsed_time >= 60:
            elapsed_time = 0
            start_time = time.time()
            try:
                buffer = BytesIO()
                img.save(buffer, format='JPEG')
                img_str = base64.b64encode(buffer.getvalue())
                data = {'NotFree': avg_counts['Car'], 'Free': avg_counts['Park']}
                for key in counts:
                    for park in counts[key]:
                        value = 0
                        if key == 'Park':
                            value = 1
                        data.update({str(park): value})
                r = requests.post('http://192.168.51.140:8080/api/v1/'+curr_token+'/telemetry',
                                  data=json.dumps(data))
                r = requests.post('http://192.168.51.140:8080/api/v1/'+curr_token+'/attributes',
                                  data=json.dumps({'image': str(img_str)}))
                print(data)

            except Exception as e:
                print('Could not connect to thingsboard! ' + str(datetime.datetime.now()))
                print(e.with_traceback())
            finally:
                for key in summed_counts:
                    summed_counts[key].clear()
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
