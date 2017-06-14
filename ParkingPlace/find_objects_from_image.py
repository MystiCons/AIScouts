from model import Model
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from sklearn.cluster import MeanShift, estimate_bandwidth
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


    def __init__(self, model, interesting_labels, visualize=False):
        self.model = model
        self.visualize = visualize
        self.interesting = interesting_labels


    def predict_poi(self, crop):
        img = cv2.resize(crop, (self.model.img_size, self.model.img_size))
        label, confidence = self.model.predict(img)
        if label in self.interesting:
            return True
        return False



    def find_objects(self, img, crop_size):
        if isinstance(img, str):
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        else:
            image = img
        self.image_height, self.image_width = image.shape
        poi = self.find_points_of_interest(crop_size, image.copy())
        for i in range(len(poi)):
            crop = image[int(poi[i][1]-crop_size/2):int(poi[i][1] + crop_size/2),
                   int(poi[i][0]-crop_size/2):int(poi[i][0] + crop_size/2)]
            label, confidence = self.model.predict(crop)
            if label in self.interesting:
                cv2.rectangle(image,
                              (int(poi[i][0]-crop_size/2), int(poi[i][1]-crop_size/2)),
                              (int(poi[i][0] + crop_size/2),
                              int(poi[i][1] + crop_size/2)),
                              (255, 0, 0),
                              2)
        cv2.imshow('main', image)
        cv2.waitKey(1)


    def find_points_of_interest(self, crop_size, img):
        poix = []
        poiy = []
        visualize_img = None
        if self.visualize:
            visualize_img = img.copy()
        curr_position = [0, 0]
        move_ratio_width = int(crop_size * 0.1)
        move_ratio_height = int(crop_size * 0.1)

        image_orig = img.copy()
        if self.visualize:
            cv2.imshow('main', img)
            cv2.waitKey(1)
        while True:

            if curr_position[1] + crop_size >= self.image_height:
                break
            else:
                curr_position[1] = curr_position[1] + move_ratio_height
                curr_position[0] = 0
            while True:
                crop = image_orig[curr_position[1]:curr_position[1] + crop_size, curr_position[0]:curr_position[0] + crop_size]
                if self.predict_poi(crop):
                    cv2.circle(img, (int(curr_position[0] + crop_size / 2), int(curr_position[1] + crop_size / 2)),
                                  2,
                                  (255, 0, 0), -1)
                    poix.append(curr_position[0] + crop_size / 2)
                    poiy.append(curr_position[1] + crop_size / 2)
                    curr_position[0] = int(curr_position[0] + move_ratio_width)
                else:
                    curr_position[0] = int(curr_position[0] + move_ratio_width)
                if self.visualize:
                    cv2.rectangle(visualize_img,
                                  (curr_position[0], curr_position[1]),
                                  (curr_position[0] + crop_size,
                                   curr_position[1] + crop_size),
                                  (255, 0, 0),
                                  2)
                    cv2.imshow('main', visualize_img)
                    cv2.waitKey(1)
                    visualize_img = img.copy()
                if curr_position[0] + crop_size >= self.image_width:
                    break

        if not os.path.exists('heatmaps'):
            os.makedirs('heatmaps')
        heatmap_img_file_name = (self.model.model_name + 'C' + str(crop_size) + '.jpg')
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
        cv2.imwrite('POIs/' + self.model.model_name + 'C' + str(crop_size) + 'CLOP' + '.jpg', image)
        return mid_points


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


