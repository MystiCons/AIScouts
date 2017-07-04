
from urllib.request import Request, urlopen,\
    HTTPPasswordMgrWithPriorAuth, HTTPBasicAuthHandler, build_opener
import cv2
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
        password_manager = HTTPPasswordMgrWithPriorAuth()
        password_manager.add_password(None, url, user, password, is_authenticated=True)
        auth_manager = HTTPBasicAuthHandler(password_manager)
        self.opener = build_opener(auth_manager)
        #install_opener(self.opener)


    def get_frame(self):
        self.response = self.opener.open(self.url)
        feed = self.response.read()

        img_array = np.asarray(bytearray(feed), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
