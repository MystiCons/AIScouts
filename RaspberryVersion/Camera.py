from picamera import PiCamera
from picamera.array import PiRGBArray
from PIL import Image
import time


class Camera:
    camera = None

    def __init__(self):
        self.camera = PiCamera(resolution=(1024, 768))
        self.camera.awb_mode = 'auto'
        time.sleep(1)

    def __exit__(self, exc_type, exc_value, traceback):
        self.camera.close()

    def get_frame(self, rotate=0):
        frame = PiRGBArray(self.camera)
        self.camera.capture(frame, format='rgb')
        return Image.fromarray(frame.array, 'RGB')
