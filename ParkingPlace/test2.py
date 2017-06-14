from capture_ip_camera import IpCamera

camera = IpCamera('http://192.168.51.247/html/cam_pic.php?time=1497446828170&pDelay=40000')

while True:
    camera.get_frame()