import os
import cv2
import time
import pickle

class ObjectRecognition:
    crop_size = 128
    image_width = 0
    image_height = 0
    model = None
    curr_position = [0, 0]
    visualize = False
    interesting = []
    #{Point of interest[x,y], cropsize[x,y]}
    saved_poi = []
    auto_find = False
    show_poi = False
    labels_counts = {}
    curr_image = None
    curr_image_gray = None
    #MouseClickCallback variables
    refPtStart = []
    refPtEnd = []
    cropping = False
    setupImage = None
    setupImage2 = None

    elapsed_time = 0
    start_time = 0

    # If interesting labels is None, the script will ask the user to draw his points of interest
    def __init__(self, model, interesting_labels, auto_find=False, visualize=False):
        self.model = model
        self.visualize = visualize
        self.auto_find = auto_find
        self.interesting = interesting_labels
        for label in model.label_folders:
            self.labels_counts.update({label.split('/')[-2]: []})


    def predict_poi(self, crop):
        img = cv2.resize(crop, (self.model.img_size, self.model.img_size))
        label, confidence = self.model.predict(img)
        if label in self.interesting:
            return True
        return False


    def save_poi(self, path):
        f = open(path + '.poi', 'wb')
        pickle.dump(self.saved_poi, f, 2)
        f.close()

    def load_poi(self, path):
        f = open(path + '.poi', 'rb')
        self.saved_poi = pickle.load(f)

    def save_images_from_poi(self, image, path, every_x_s):
        if not os.path.isdir(path):
            os.mkdir(path)
        # if start time hasn't been initialized
        if self.start_time == 0:
            self.start_time = time.time()
        self.elapsed_time = time.time() - self.start_time
        if self.elapsed_time >= every_x_s:
            for key, value in self.saved_poi:
                crop = image[int(key[1] - value[1] / 2):int(key[1] + value[1] / 2),
                       int(key[0] - value[0] / 2):int(key[0] + value[0] / 2)]
                img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                label, confidence = self.model.predict(img)
                if not os.path.isdir(path + label):
                    os.mkdir(path + label)
                dirlen = len(os.listdir(path + label))
                cv2.imwrite(path + label + '/' + str(dirlen + 1) + '.bmp', crop)
            self.start_time = time.time()
            self.elapsed_time = 0

    def find_objects(self, img):
        if isinstance(img, str):
            image = cv2.imread(img, cv2.IMREAD_COLOR)
        else:
            image = img
        self.curr_image = image.copy()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.curr_image_gray = gray_image.copy()
        self.image_height, self.image_width = gray_image.shape
        for key in self.labels_counts:
            self.labels_counts[key].clear()
        i = 0
        for key, value in self.saved_poi:
            crop = gray_image[int(key[1]-value[1]/2):int(key[1] + value[1]/2),
                   int(key[0]-value[0]/2):int(key[0] + value[0]/2)]
            label, confidence = self.model.predict(crop)
            if label in self.interesting:
                if label == self.interesting[0]:
                    color = (0, 255, 0)
                elif label == self.interesting[1]:
                    color = (255, 0, 0)
                elif label == self.interesting[2]:
                    color = (0, 0, 255)
                else:
                    color = (0, 0, 0)
                self.labels_counts[label].append(i)
                if self.visualize:
                    cv2.rectangle(image,
                                  (int(key[0]-value[0]/2), int(key[1]-value[1]/2)),
                                  (int(key[0] + value[0]/2),
                                  int(key[1] + value[1]/2)),
                                  color,
                                  2)
                    text = str(i) + ' C: ' + str(round(confidence, 2))
                    cv2.putText(image, text, (key[0] - len(text) * 3, key[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            else:
                if self.show_poi:
                    cv2.rectangle(image,
                                  (int(key[0]-value[0]/2), int(key[1]-value[1]/2)),
                                  (int(key[0] + value[0]/2),
                                  int(key[1] + value[1]/2)),
                                  (255, 255, 255),
                                  2)
            i += 1
        return image, self.labels_counts
