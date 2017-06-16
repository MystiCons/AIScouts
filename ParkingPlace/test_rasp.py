from model import Model
from find_objects_from_image import ObjectRecognition
import time

mod = Model.load_model("models/testi1")

objectrec = ObjectRecognition(mod, ['true', 'taken', 'false'], auto_find=True, visualize=False)
while True:
    t = time.time()
    try:
        objectrec.load_poi('./poi')
    except Exception:
        print('Points of interest couldnt be loaded, trying to auto find')
    img, counts = objectrec.find_objects('./1.jpg', [180, 180])
    print("new image processed in: " + str(round(t-time.time(), 4)) + " seconds")
    print("found " + str(counts))
    input("continue?")


