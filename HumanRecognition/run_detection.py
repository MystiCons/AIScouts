import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from RaspberryVersion.Camera import Camera
from HumanRecognition.Detection import MotionDetection
import requests
from io import BytesIO
import base64
import json

camera = Camera()
detector = MotionDetection(256)

#mod = Model.load_model("/home/pi/dev/HumanRecognition/DeepLearning/models/wimmalab")

while True:
    frame = camera.get_frame()
    image, positions = detector.get_motion_position(frame)
    buffer = BytesIO()
    frame.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue())
    r = requests.post('http://192.168.51.140:8080/api/v1/gAr2fUXsBYuPUMyCUF7F/attributes',
                  data=json.dumps({'image': str(img_str)}))




