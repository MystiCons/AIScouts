from model import Model
from find_objects_from_image import ObjectRecognition
import os
import cv2

mod = Model.load_model("models/park_model14")

objectrec = ObjectRecognition(mod, ['Park', 'Car'])

img, counts = objectrec.find_objects('./2.bmp')
cv2.imshow('main', img)
cv2.waitKey()

objectrec.save_poi('./points')
#os.system('./send_poi.sh')
