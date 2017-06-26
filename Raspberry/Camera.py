from picamera import PiCamera
from PIL import Image
from io import BytesIO
import numpy as np
import time


class Camera:

    stream = None
    camera = None

    def __init__(self):
        self.stream = BytesIO()
        self.camera = PiCamera()
        self.camera.start_preview()
        self.camera.capture(self.stream, format='bmp')

    def __exit__(self, exc_type, exc_value, traceback):
        self.camera.close()

    def get_frame(self):
        self.stream.seek(1)
        return Image.open(self.stream)
