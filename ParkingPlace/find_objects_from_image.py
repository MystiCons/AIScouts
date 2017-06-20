from model import Model
import numpy as np
import os
import cv2
import time
try:
    import matplotlib.pyplot as plt
    from sklearn.cluster import MeanShift, estimate_bandwidth
except Exception as e:
    pass


import pickle
from random import randint
from pyclustering.cluster.optics import optics
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster import cluster_visualizer


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


    def predict_poi(self, crop):
        img = cv2.resize(crop, (self.model.img_size, self.model.img_size))
        label, confidence = self.model.predict(img)
        if label in self.interesting:
            return True
        return False


    def reset_poi(self):
        self.saved_poi.clear()
        self.refPtStart.clear()
        self.refPtEnd.clear()


    def save_poi(self, path):
        f = open(path + '.poi', 'wb')
        pickle.dump(self.saved_poi, f, 2)
        f.close()

    def load_poi(self, path):
        f = open(path + '.poi', 'rb')
        self.saved_poi = pickle.load(f)

    def toggle_points_of_interest(self):
        self.show_poi = not self.show_poi

    def save_images_from_poi(self, image, path, every_x_s=None):
        if not os.path.isdir(path):
            os.mkdir(path)
        if every_x_s is None:

            for key, value in self.saved_poi:
                crop = image[int(key[1] - value[1] / 2):int(key[1] + value[1] / 2),
                       int(key[0] - value[0] / 2):int(key[0] + value[0] / 2)]
                dir = os.listdir(path)
                count = len(dir)
                cv2.imwrite(path + str(count) + '.bmp', crop)
        else:
            # if start time hasn't been initialized
            if self.start_time == 0:
                self.start_time = time.time()
            self.elapsed_time = time.time() - self.start_time
            if self.elapsed_time >= every_x_s:
                for key, value in self.saved_poi:
                    crop = image[int(key[1] - value[1] / 2):int(key[1] + value[1] / 2),
                           int(key[0] - value[0] / 2):int(key[0] + value[0] / 2)]
                    dir = os.listdir(path)
                    count = len(dir)
                    cv2.imwrite(path + str(count) + '.bmp', crop)
                self.start_time = time.time()
                self.elapsed_time = 0

    def find_objects(self, img, crop_size=None):
        if isinstance(img, str):
            image = cv2.imread(img, cv2.IMREAD_COLOR)
        else:
            image = img
        self.curr_image = image.copy()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.curr_image_gray = gray_image.copy()
        self.image_height, self.image_width = gray_image.shape
        if not self.auto_find:
            # if saved points dictionary is empty
            if not self.saved_poi:
                self.draw_points_of_interest(image)
        else:
            if crop_size is None:
                print("If the class is initialized with auto_find = True, please run find_objects with a crop_size=[x,y] parameter")
                exit(2)
            if not self.saved_poi:
                self.find_points_of_interest(crop_size, gray_image.copy())
        counts = {}
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
                if label in counts:
                    counts[label] += 1
                else:
                    counts[label] = 1

                cv2.rectangle(image,
                              (int(key[0]-value[0]/2), int(key[1]-value[1]/2)),
                              (int(key[0] + value[0]/2),
                              int(key[1] + value[1]/2)),
                              color,
                              2)
                text = str(round(confidence, 2))
                cv2.putText(image, text, (key[0] - len(text) * 3, key[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            else:
                if self.show_poi:
                    cv2.rectangle(image,
                                  (int(key[0]-value[0]/2), int(key[1]-value[1]/2)),
                                  (int(key[0] + value[0]/2),
                                  int(key[1] + value[1]/2)),
                                  (255, 255, 255),
                                  2)
        return image, counts

    def draw_points_of_interest(self, img):
        self.setupImage2 = img.copy()
        self.setupImage = img.copy()
        cv2.namedWindow("setup", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('setup', 1280, 720)
        cv2.setMouseCallback("setup", self.click_and_crop)
        while True:
            cv2.imshow('setup', self.setupImage2)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                self.setupImage = img.copy()
                self.setupImage2 = img.copy()
                self.refPtStart = []
                self.refPtEnd = []
            # if esc is pressed, stop drawing mode
            if key == 27:
                for i in range(len(self.refPtEnd)):
                    crop_size = [self.refPtEnd[i][0] - self.refPtStart[i][0],
                                  self.refPtEnd[i][1] - self.refPtStart[i][1]]

                    middle_x = int(self.refPtStart[i][0] + crop_size[0] / 2)
                    middle_y = int(self.refPtStart[i][1] + crop_size[1] / 2)

                    self.setupImage = img.copy()

                    crop_size = [abs(crop_size[0]), abs(crop_size[1])]
                    middle_point = [middle_x,
                                    middle_y]
                    self.saved_poi.append([middle_point, crop_size])
                cv2.destroyAllWindows()
                self.save_poi('./points')
                break

    def click_and_crop(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPtStart.append([x, y])
            self.cropping = True

        if event == cv2.EVENT_RBUTTONDOWN and self.cropping is False:
            if len(self.refPtEnd) == 0:
                return
            del self.refPtStart[-1]
            del self.refPtEnd[-1]
            self.setupImage = self.curr_image.copy()
            for i in range(len(self.refPtStart)):
                crop = self.curr_image_gray[int(self.refPtStart[i][1]):int(self.refPtEnd[i][1]),
                       int(self.refPtStart[i][0]):int(self.refPtEnd[i][0])]
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
                cv2.rectangle(self.setupImage, (self.refPtStart[i][0], self.refPtStart[i][1]),
                              (self.refPtEnd[i][0], self.refPtEnd[i][1]), color, 2)
                self.setupImage2 = self.setupImage.copy()


        elif event == cv2.EVENT_LBUTTONUP:
            self.refPtEnd.append([x, y])
            self.cropping = False
            crop = self.curr_image_gray[int(self.refPtStart[-1][1]):int(self.refPtEnd[-1][1]),
                   int(self.refPtStart[-1][0]):int(self.refPtEnd[-1][0])]
            cv2.imshow('setup2', crop)
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
            else:
                color = (0, 0, 0)
            cv2.rectangle(self.setupImage, (self.refPtStart[-1][0], self.refPtStart[-1][1])
                                            , (self.refPtEnd[-1][0],self.refPtEnd[-1][1]), color, 2)
            self.setupImage2 = self.setupImage.copy()

        elif self.cropping:
            self.setupImage2 = self.setupImage.copy()
            cv2.rectangle(self.setupImage2, (self.refPtStart[-1][0], self.refPtStart[-1][1]),
                          (x, y), (0, 255, 0), 2)

    def find_points_of_interest(self, crop_size, img):
        poix = []
        poiy = []
        visualize_img = None
        if self.visualize:
            visualize_img = img.copy()
        curr_position = [0, 0]
        move_ratio_width = int(crop_size[0] * 0.1)
        move_ratio_height = int(crop_size[1] * 0.1)

        image_orig = img.copy()
        if self.visualize:
            cv2.imshow('main', img)
            cv2.waitKey(1)
        while True:

            if curr_position[1] + crop_size[1] >= self.image_height:
                break
            else:
                curr_position[1] = curr_position[1] + move_ratio_height
                curr_position[0] = 0
            while True:
                crop = image_orig[curr_position[1]:curr_position[1] + crop_size[1], curr_position[0]:curr_position[0] + crop_size[0]]
                if self.predict_poi(crop):
                    cv2.circle(img, (int(curr_position[0] + crop_size[0] / 2), int(curr_position[1] + crop_size[1] / 2)),
                                  2,
                                  (255, 0, 0), -1)
                    poix.append(curr_position[0] + crop_size[0] / 2)
                    poiy.append(curr_position[1] + crop_size[1] / 2)
                    curr_position[0] = int(curr_position[0] + move_ratio_width)
                else:
                    curr_position[0] = int(curr_position[0] + move_ratio_width)
                if self.visualize:
                    cv2.rectangle(visualize_img,
                                  (curr_position[0], curr_position[1]),
                                  (curr_position[0] + crop_size[0],
                                   curr_position[1] + crop_size[1]),
                                  (255, 0, 0),
                                  2)
                    cv2.imshow('main', visualize_img)
                    cv2.waitKey(1)
                    visualize_img = img.copy()
                if curr_position[0] + crop_size[0] >= self.image_width:
                    break

        if not os.path.exists('heatmaps'):
            os.makedirs('heatmaps')
        heatmap_img_file_name = (self.model.model_name + 'CX' + str(crop_size[0])+ 'CY' + str(crop_size[1]) + '.bmp')
        cv2.imwrite('heatmaps/' + heatmap_img_file_name, img)

        clusters = self.cluster_optics(poix, poiy)
        mid_points = []

        for i in range(len(clusters)):
            avg = 0
            count = 0
            for j in clusters[i]:
                count += 1
                avg += j
            mid_points.append([int(avg[0] / count), int(avg[1] / count)])

        image = image_orig.copy()
        for i in range(len(mid_points)):
             cv2.circle(image, (mid_points[i][0], mid_points[i][1]),
                          2,
                          (255, 0, 0), -1)
        if self.visualize:
            cv2.imshow("main", image)
            cv2.waitKey()

        if not os.path.exists('POIs'):
            os.makedirs('POIs')
        cv2.imwrite('POIs/' + self.model.model_name + 'CX' + str(crop_size[0])+ 'CY' + str(crop_size[1]) + 'CLOP' + '.bmp',
                    image)
        for i in mid_points:
            self.saved_poi.append([i, crop_size])


    def cluster_meanshift(self, xs, ys):
        POI = []
        for i in range(len(xs)):
            POI.append([xs[i], ys[i]])
        POI = np.array(POI)

        bw = estimate_bandwidth(POI, quantile=0.085, n_samples=50)

        ms = MeanShift(bw, bin_seeding=True)
        ms.fit(POI)
        clusters = ms.cluster_centers_.astype(int)

        ret = clusters

        return ret



    def cluster_optics(self, xs, ys):
        POI = []
        for i in range(len(xs)):
            POI.append([xs[i], ys[i]])
        POI = np.array(POI)

        optics_instance = optics(POI, 27, 5)
        optics_instance.process()
        clusters = optics_instance.get_clusters()

        if self.visualize:
            vis = cluster_visualizer()
            vis.append_clusters(clusters, POI)
            vis.show()
        ret = []

        for i in range(len(clusters)):
            ret.append([])
            for j in range(len(clusters[i])):
                ret[i].append(POI[clusters[i][j]])

        return ret

    def cluster_kmeans(self, xs, ys):
        POI = []
        for i in range(len(xs)):
            POI.append([xs[i], ys[i]])
        POI = np.array(POI)

        rand = []
        for i in range(14):
            r = randint(0, len(POI))
            rand.append(POI[r])


        kmeans_instance = kmeans(POI, rand, 4)
        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()

        if self.visualize:
            vis = cluster_visualizer()
            vis.append_clusters(clusters, POI)
            vis.show()
        ret = []

        for i in range(len(clusters)):
            ret.append([])
            for j in range(len(clusters[i])):
                ret[i].append(POI[clusters[i][j]])

        return ret


