import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import cv2
import numpy as np
import MachineLearning.rasp_model

class MotionDetection:

    first_frame = None
    curr_frame = None
    image_size = 256
    min_area = 500
    model = None

    def __init__(self, frame_size, model=None):
        self.image_size = frame_size
        self.model = model

    def get_motion_position(self, new_frame):
        crop = np.zeros((256, 256, 3), np.uint8)
        new_frame = cv2.cvtColor(np.array(new_frame), cv2.COLOR_RGB2BGR)
        #gray = cv2.resize(new_frame, (self.image_size, self.image_size))
        gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.first_frame is None:
            self.first_frame = gray
            return new_frame, crop, []
        self.curr_frame = gray
        frame_delta = cv2.absdiff(self.first_frame, self.curr_frame)
        thresh = cv2.threshold(frame_delta, 5, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        (_, contours, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        ret = []
        new_frame = cv2.cvtColor(np.array(new_frame), cv2.COLOR_BGR2RGB)
        if len(contours) > 0:
            con = max(contours, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(con)
            if cv2.contourArea(con) < self.min_area:
                #print('small area: ' + str(cv2.contourArea(con)))
                return new_frame, [], []
            crop = np.array(new_frame.copy()[int(y):int(y+h),
                   int(x):int(x+w)])
            cv2.rectangle(new_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            ret.append([x, y, w, h])
            self.first_frame = gray
            return new_frame, crop, ret
        self.first_frame = gray
        return new_frame, crop, []

    def recognize_crop(self, crop):
        label, confidence = self.model.predict(crop)
        return label, confidence

