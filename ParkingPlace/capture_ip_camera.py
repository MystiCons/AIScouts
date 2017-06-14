import base64
from urllib.request import Request, urlopen
import cv2
import numpy as np


class IpCamera:
    def __init__(self, url, user=None, password=None):
        self.url = url
        #auth_encoded = base64.encodebytes('%s:%s' % (user, password))[:-1]

        self.req = Request(self.url)
        #self.req.add_header('Authorization', 'Basic %s' % auth_encoded)

    def get_frame(self):
        response = urlopen(self.url)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        return frame
