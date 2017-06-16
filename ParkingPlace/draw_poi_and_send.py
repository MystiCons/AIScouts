from model import Model
from find_objects_from_image import ObjectRecognition
import time

from scp import SCPClient
from paramiko import SSHClient

mod = Model.load_model("models/testi1")
ssh = SSHClient()
#ssh.load_host_keys()

objectrec = ObjectRecognition(mod, ['true', 'taken', 'false'], auto_find=False, visualize=False)
t = time.time()
img, counts = objectrec.find_objects('./1.jpg', [180, 180])
objectrec.save_poi('./poi')
ssh.connect('192.168.51.212', username='pi', password='raspberry')
scp = SCPClient(ssh.get_transport())
scp.put('./poi.poi', '/home/pi/tensorflow_test/AIScouts/ParkingPlace/poi.poi')
scp.close()
