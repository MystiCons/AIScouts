from picamera import PiCamera
from PIL import Image
from io import BytesIO
import numpy as np
import time



def get_frame(self):
    stream = BytesIO()
    with PiCamera() as camera:
        camera.start_preview()
        time.sleep(1)
        camera.capture(stream, format='bmp')
    stream.seek(0)
    return Image.open(stream)
