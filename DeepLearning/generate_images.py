import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from DeepLearning.model import Model
from DeepLearning.model import denormalize_image
import numpy as np
import cv2
import matplotlib.pyplot as plt

paths = ['Car/']
data_folder = "/media/cf2017/levy/tensorflow/parking_place2/"

mod = Model(paths, data_folder=data_folder,
            epochs=1, img_size=64, model_name='gen_model2',
            learning_rate=0.0002, DCGAN=True)
mod.DCGAN(mnist_data=True)
#mod = Model.load_model(data_folder + 'models/gen_model2')
z = np.random.uniform(0., 1., size=[4, mod.z_dim])
images = mod.predict_DCGAN(z)
images = denormalize_image(images)
new_im = np.vstack(([(np.hstack(([images[i * j] for i in range(1)]))) for j in range(1)]))
cv2.imshow('main', new_im)
cv2.waitKey()


f, a = plt.subplots(4, 10, figsize=(10, 4))
for i in range(10):
    # Noise input.
    z = np.random.uniform(0., 1., size=[4, mod.z_dim])
    images = np.array(mod.predict_DCGAN(z))
    images = denormalize_image(images)
    for j in range(4):
        # Generate image from noise. Extend to 3 channels for matplot figure.
        img = np.reshape(np.repeat(images[j][:, :, np.newaxis], 3, axis=2),
                         newshape=(mod.img_size, mod.img_size, 3))
        a[j][i].imshow(img)
f.show()
plt.draw()
plt.waitforbuttonpress()

'''
for i in range(10):
    # Noise input.
    z = np.random.uniform(-1., 1., size=[1, z_dim])
    images = np.array(gen.predict({'input_gen_noise': z}))
    images = np.array([x * 255 for x in images], dtype='uint8')
    for j in range(4):
        img = images[j]
        cv2.imshow("main", img)
        cv2.waitKey()

'''