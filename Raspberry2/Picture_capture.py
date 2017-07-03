import os
import picamera
from time import sleep


folder_name = input("Folder name: ")
current_folder = "./"
cap_amount = 0
cap_amount = +int(input("How many pictures: "))




#./ + folder name
def makedir():
  try:
    os.makedirs(current_folder+folder_name)
  except OSError:
    pass
  # let exception propagate if we just can't
  # cd into the specified directory
  os.chdir(folder_name)    
    
makedir()

#Camera adjustment
camera = picamera.PiCamera()
camera.resolution = (1920, 1080)
camera.framerate = 24
counter = 1

while (counter <= cap_amount):
    img_name = str(counter)+'.png'
    camera.capture(img_name)
    counter = counter + 1
    sleep(0.05)

print("Done")

os.chdir("..")    







'''
#Command backlog
camera.hflip = True
camera.vflip = True
camera.start_preview()
camera.stop_preview()
camera.brightness = 70
camera.sharpness = 0
camera.contrast = 0
camera.brightness = 50
camera.saturation = 0
camera.ISO = 0
camera.video_stabilization = False
camera.exposure_compensation = 0
camera.exposure_mode = 'auto'
camera.meter_mode = 'average'
camera.awb_mode = 'auto'
camera.image_effect = 'none'
camera.color_effects = None
camera.rotation = 0
camera.crop = (0.0, 0.0, 1.0, 1.0)
camera.resolution = (100, 100)
camera.framerate = 24

camera.start_recording('video.h264')
camera.stop_recording()
'''
