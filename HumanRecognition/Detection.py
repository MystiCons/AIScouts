import cv2
from PIL import Image
import numpy as np

class MotionDetection:

    last_frame = None
    curr_frame = None
    image_size = 256

    def __init__(self, frame_size):
        self.image_size = frame_size
        pass

    def get_motion_position(self, new_frame):
        new_frame = cv2.cvtColor(np.array(new_frame), cv2.COLOR_RGB2BGR)
        #frame = cv2.resize(new_frame, (self.image_size, self.image_size))
        gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.last_frame is None:
            self.last_frame = gray
            return
        self.curr_frame = gray
        frame_delta = cv2.absdiff(self.last_frame, self.curr_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        (_, contours, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        ret = []
        if contours:
            for con in contours:
                (x, y, w, h) = cv2.boundingRect(con)
                cv2.rectangle(new_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                ret.append([x, y, w, h])
        return new_frame, ret

