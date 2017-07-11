import cv2
import os
from tqdm import tqdm
import numpy as np

class DataManipulation:

    data_folder = None

    def __init__(self, data_folder):
        self.data_folder = data_folder

    def try_cluster_training_data(self, path, clusters, img_size=128):
        data = []
        for img in tqdm(os.listdir(path)):
            img_orig = cv2.imread(path + img, cv2.IMREAD_GRAYSCALE)
            img_orig = cv2.resize(img_orig, (img_size, img_size))
            data.append(np.array(img_orig, dtype=np.float32))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        data = np.array(data)
        compactness, labels, centers = cv2.kmeans(data, clusters, None, criteria, 10, flags)
        A = ord('A')
        if not os.path.isdir(self.data_folder + 'clustered_images/'):
            os.mkdir(self.data_folder + 'clustered_images')
        for i in range(clusters):
            images = data[labels.ravel() == i]
            print(labels.ravel())
            if not os.path.isdir(self.data_folder + 'clustered_images/' + chr(A + i)):
                os.mkdir(self.data_folder + 'clustered_images/' + chr(A + i))
            count = 0
            for image in tqdm(images):
                cv2.imwrite(self.data_folder + 'clustered_images/' + chr(A + i) + '/' + str(count) + '.bmp',
                            image)
                count += 1

    def color_quantization(self, imgs_path, colors, img_size, save_images_path=None):
        dir = os.listdir(imgs_path)
        images = []
        for i in tqdm(range(len(dir))):
            img = cv2.resize(cv2.imread(imgs_path + str(i) + '.bmp', cv2.IMREAD_GRAYSCALE),
                             (img_size, img_size))
            images.append(np.array(img))

        new_images = []
        for i in range(len(images)):
            data = images[i].reshape((-1, 1))
            # convert to np.float32
            data = np.float32(data)
            # define criteria, number of clusters(K) and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret, label, center = cv2.kmeans(data, colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            # Now convert back into uint8, and make original image
            center = np.uint8(center)
            res = center[label.flatten()]
            res2 = res.reshape(images[i].shape)
            if save_images_path is None:
                new_images.append(res2)
            else:
                dirlen = len(os.listdir(save_images_path))
                cv2.imwrite(save_images_path + str(dirlen) + '.bmp',
                            res2)

        if save_images_path is not None:
            return new_images


    def flip_images(self, src, dest, img_size):
        dir = os.listdir(src)
        images = []
        for i in tqdm(range(len(dir))):
            img = cv2.imread(src + str(i) + '.bmp', cv2.IMREAD_GRAYSCALE)
            images.append(np.array(img))


        for i in range(len(images)):
            try:
                img = images[i]
                cv2.flip(images[i], 1, img)
                cv2.imwrite(dest + str(i) + '.bmp',
                            img)
            except Exception:
                pass


if __name__ == '__main__':
    inp = "/media/cf2017/levy/backup/temp/"
    #mod = Model.load_model("/media/cf2017/levy/tensorflow/images/" + "models/testi1")

    #mod.try_cluster_training_data("/media/cf2017/levy/tensorflow/images/new_training_data/", 2)
    #mod.color_quantization(mod.data_folder + 'clustered_images/A/', mod.data_folder + 'clustered_images/A/')
    manipulator = DataManipulation("/media/cf2017/levy/tensorflow/parking_place2/")
    manipulator.try_cluster_training_data("/media/cf2017/levy/tensorflow/parking_place2/asd/Park/", 5)
    #manipulator.color_quantization(inp + "Park/", 24, 128, save_images_path=inp + "augmented_parks/")
    #manipulator.color_quantization(inp + "Park/", 16, 128, save_images_path=inp + "augmented_parks/")
    #manipulator.color_quantization(inp + "Car/", 32, 128, save_images_path=inp + "augmented_cars/")
    #manipulator.color_quantization(inp + "Car/", 16, 128, save_images_path=inp + "augmented_cars/")
    #manipulator.color_quantization(inp + "Car/", 12, 128, save_images_path=inp + "augmented_cars/")
    #manipulator.color_quantization(inp + "Car/", 6, 128, save_images_path=inp + "augmented_cars/")


    #manipulator.flip_images(inp + "augmented_parks/", inp + "flipped_parks/", 128)
    #manipulator.flip_images(inp + "Car/", inp + "flipped_cars/", 128)


