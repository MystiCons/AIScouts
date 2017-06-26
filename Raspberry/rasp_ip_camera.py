import base64
from urllib.request import HTTPPasswordMgrWithDefaultRealm, HTTPBasicAuthHandler, build_opener
from PIL import Image
from io import BytesIO
import numpy as np


class IpCamera:
    user = None
    password = None
    opener = None
    response = None

    def __init__(self, url, user=None, password=None):
        self.user = user
        self.password = password
        self.url = url
        password_manager = HTTPPasswordMgrWithDefaultRealm()
        password_manager.add_password(None, url, user, password)
        auth_manager = HTTPBasicAuthHandler(password_manager)
        self.opener = build_opener(auth_manager)
        #install_opener(self.opener)


    def get_frame(self):
        self.response = self.opener.open(self.url)
        feed = self.response.read()
        img_array = np.asarray(bytearray(feed), dtype=np.uint8)
        frame = Image.open(BytesIO(img_array))
        return frame
