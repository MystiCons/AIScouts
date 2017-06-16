from model import Model
from find_objects_from_image import ObjectRecognition
import os
import cv2

mod = Model.load_model("models/testi1")

objectrec = ObjectRecognition(mod, ['true', 'taken', 'false'])
objectrec.toggle_points_of_interest()

img, counts = objectrec.find_objects('./1.jpg')
cv2.imshow('main', img)
cv2.waitKey()

objectrec.save_poi('./poi')
os.system('./send_poi.sh')
