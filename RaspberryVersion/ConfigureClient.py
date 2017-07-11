import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import socket
import threading
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import traceback
from PIL import ImageTk
import tkinter
import pickle

import numpy as np
#from DeepLearning.rasp_model import Model


try:
    import cv2
except ImportError:
    print('Cv2 could not be imported.')


class TCPClient:
    host = ''
    port = 1337
    sock = None
    close = False
    latest_image = None
    latest_orig_image = None
    images_lock = None
    socket_lock = None
    poi_lock = None

    def __init__(self):
        self.init_socket()
        self.images_lock = threading.Lock()
        self.socket_lock = threading.Lock()
        self.poi_lock = threading.Lock()

    def init_socket(self):
        self.sock = socket.socket()
        socket.setdefaulttimeout(30)

    def connect(self, host, port):
        try:
            print('Connecting to ' + host + ':' + str(port))
            self.sock.connect((host, port))
            self.close = False
            return self.receive_poi()
        except socket.error as e:
            print('Connection failed! ' + e.strerror)

    def disconnect(self):
        self.socket_lock.acquire()
        self.close = True
        self.socket_lock.release()

    def receive_poi(self):
        try:
            poi = []
            chunks = []
            chunk = 0
            try:
                chunk = self.sock.recv(2048)
            except socket.error:
                pass
            finally:
                if chunk:
                    chunks.append(chunk)
                    while True:
                        try:
                            self.sock.settimeout(0.5)
                            chunk = self.sock.recv(2048)
                            if not chunk:
                                break
                            chunks.append(chunk)
                        except socket.error:
                            break
                    data = b''.join(chunks)
                    poi = pickle.loads(data)
                    self.sock.settimeout(10)
                    print('Received new points of interest!')
        except socket.error as e:
            traceback.print_exc(file=sys.stdout)
        return poi

    def receive(self):
        try:
            while True:
                chunks = []
                chunk = self.sock.recv(4096)
                if self.close:
                    break
                self.socket_lock.acquire()
                if chunk:
                    chunks.append(chunk)
                    while True:
                        try:
                            self.sock.settimeout(0.5)
                            chunk = self.sock.recv(4096)
                            if not chunk:
                                break
                            chunks.append(chunk)
                        except socket.error:
                            break
                    data = b''.join(chunks)
                    try:
                        images = pickle.loads(data)
                        img = Image.open(BytesIO(base64.b64decode(images[0])))
                        img_orig = Image.open(BytesIO(base64.b64decode(images[1])))
                        self.images_lock.acquire()
                        self.latest_image = img
                        self.latest_orig_image = img_orig
                        self.images_lock.release()
                    except Exception as e:
                        print('Image receiving failed.')
                    finally:
                        self.sock.settimeout(10)
                self.socket_lock.release()
                if self.close:
                    break

        except socket.error as e:
            traceback.print_exc(file=sys.stdout)
        finally:
            self.sock.close()
            self.init_socket()

    def send_data(self, data):
        try:
            print(data)
            data_string = pickle.dumps(data)
            self.sock.sendall(data_string)
        except socket.error as e:
            print(e.strerror)

    def get_next_image(self):
        self.images_lock.acquire()

        img = self.latest_image

        self.images_lock.release()
        return img

    def get_next_image_orig(self):
        self.images_lock.acquire()

        img = self.latest_orig_image

        self.images_lock.release()
        return img


class Client:

    root = None
    tcp_client = None
    panel = None

    image_raw = None
    image_orig = None
    image_draw_mode = None
    image_edited = None

    photo_image = None

    canvas = None
    frame = None

    buttons_label = None
    ip_box = None
    port_box = None
    connect_button = None
    disconnect_button = None
    redraw_button = None
    data_collect_button = None
    full_image_collect_button = None
    cancel_button = None
    connected = False

    poi_lock = threading.Lock()
    image_orig_lock = threading.Lock()

    mouse_start_x = 0
    mouse_start_y = 0

    reconfigure_mode = False
    data_collection_mode = False
    full_image_collection_mode = False
    points_of_interest = []
    points_of_interest_temp = []
    collect_every_ms = 60000
    collect_full_images_every_ms = 1000
    model = None

    def __init__(self, model=None):
        self.model = model
        self.tcp_client = TCPClient()
        self.build_ui()
        self.panel.bind("<Button 1>", self.mouse_down)
        self.panel.bind("<Button 3>", self.mouse2_down)
        self.panel.bind('<B1-Motion>', self.mouse_drag)
        self.panel.bind('<ButtonRelease-1>', self.mouse_up)
        self.root.after(100, self.update)
        self.root.after(self.collect_every_ms, self.collect_data)
        self.root.after(self.collect_full_images_every_ms, self.collect_full_images)

    def build_ui(self):
        self.root = tkinter.Tk()
        self.root.title('Configuration')
        self.root.geometry('1440x900')
        self.root.resizable(True, True)
        self.buttons_label = tkinter.Label(self.root)
        self.buttons_label.grid(sticky=tkinter.E)
        self.panel = tkinter.Label(self.root)
        self.panel.grid(columnspan=40, row=0, column=1, sticky=tkinter.W)

        self.ip_box = tkinter.Entry(self.root)
        self.ip_box.grid(row=1, column=1)
        self.ip_box.delete(0, tkinter.END)
        self.ip_box.insert(0, "192.168.53.")

        self.port_box = tkinter.Entry(self.root)
        self.port_box.grid(row=1, column=2)
        self.port_box.delete(0, tkinter.END)
        self.port_box.insert(0, "1337")

        self.connect_button = tkinter.Button(self.root, text="Connect", width=10, command=self.connect)
        self.connect_button.grid(row=1, column=3)

        self.disconnect_button = tkinter.Button(self.root, text="Disconnect", width=10, command=self.disconnect)
        self.disconnect_button.grid(row=1, column=4)

        self.redraw_button = tkinter.Button(self.root, text="Draw new parks", width=10, command=self.redraw)
        self.redraw_button.grid(row=1, column=5)

        self.data_collect_button = tkinter.Button(self.root, text="Start collecting data", width=20, command=self.collect_toggle)
        self.data_collect_button.grid(row=1, column=6)

        self.full_image_collect_button = tkinter.Button(self.root, text="Start collecting full images", width=20,
                                                  command=self.collect_full_images_toggle)
        self.full_image_collect_button.grid(row=1, column=7)

    def collect_data(self):
        if self.data_collection_mode and not self.connected:
            self.collect_toggle()
        if not self.image_draw_mode:
            self.image_orig_lock.acquire()
            if self.image_orig and self.data_collection_mode:
                self.save_images_from_poi(self.tcp_client.get_next_image_orig(), '/media/cf2017/levy/tensorflow/parking_place2/new_validation_data/')
                print('Collected data')
            self.image_orig_lock.release()
        self.root.after(self.collect_every_ms, self.collect_data)

    def collect_full_images(self):
        try:
            if self.full_image_collection_mode and not self.connected:
                self.collect_full_images_toggle()
            if not self.image_draw_mode:
                path = '/media/cf2017/levy/tensorflow/parking_place2/time_lapse/'
                self.image_orig_lock.acquire()
                if self.image_orig and self.full_image_collection_mode:
                    if not os.path.isdir(path):
                        os.mkdir(path)
                    dirlen = len(os.listdir(path))
                    img = self.tcp_client.get_next_image()
                    if img:
                        img2 = img.crop((0,0, 1024, 640))
                        img2.save(path + str(dirlen + 1) + '.jpg')
                self.image_orig_lock.release()
        except:
            pass
        finally:
            self.root.after(self.collect_full_images_every_ms, self.collect_full_images)

    def collect_full_images_toggle(self):
        if not self.full_image_collection_mode and self.connected:
            self.full_image_collect_button.configure(text="Stop collecting full images")
            self.full_image_collection_mode = True
        else:
            self.full_image_collect_button.configure(text="Start collecting full images")
            self.full_image_collection_mode = False

    def save_images_from_poi(self, image, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        # if start time hasn't been initialized
        gray_image = image.convert('L')
        image_array = np.asarray(gray_image)
        for key, value in self.points_of_interest:
            crop = image_array[int(key[1] - value[1] / 2):int(key[1] + value[1] / 2),
                   int(key[0] - value[0] / 2):int(key[0] + value[0] / 2)]
            img = Image.fromarray(crop, 'L')
            if self.model:
                label, confidence = self.model.predict(img)
                if not os.path.isdir(path + label):
                    os.mkdir(path + label)
                dirlen = len(os.listdir(path + label))
                img.save(path + label + '/' + str(dirlen + 1) + '.bmp')
            else:
                if not os.path.isdir(path + 'unsorted'):
                    os.mkdir(path + 'unsorted')
                dirlen = len(os.listdir(path + 'unsorted'))
                img.save(path + 'unsorted/' + str(dirlen + 1) + '.bmp')


    def run_tk(self):
        self.root.mainloop()
        self.tcp_client.close = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tcp_client.close = True

    def cancel_redraw(self):
        self.redraw_button.configure(text="Draw new parks")
        self.reconfigure_mode = False
        self.points_of_interest_temp.clear()
        self.cancel_button.destroy()

    def collect_toggle(self):
        if not self.data_collection_mode and self.connected:
            self.data_collect_button.configure(text="Stop collecting data")
            self.data_collection_mode = True
        else:
            self.data_collect_button.configure(text="Start collecting data")
            self.data_collection_mode = False

    def redraw(self):
        if not self.reconfigure_mode:
            if not self.image_orig:
                return
            # Start drawing mode
            self.image_draw_mode = self.image_orig.copy()
            self.image_edited = self.image_orig.copy()
            self.reconfigure_mode = True
            self.poi_lock.acquire()
            self.points_of_interest_temp.clear()
            self.poi_lock.release()
            self.redraw_button.configure(text="Stop and send")
            self.cancel_button = tkinter.Button(self.root, text="Cancel drawing", width=10, command=self.cancel_redraw)
            self.cancel_button.grid(column=5, row=2)
        else:
            # Send data
            self.poi_lock.acquire()
            self.points_of_interest.clear()
            self.points_of_interest = self.points_of_interest_temp.copy()
            self.tcp_client.send_data(self.points_of_interest.copy())
            self.cancel_redraw()
            self.poi_lock.release()

    def mouse2_down(self, event):
        if not self.reconfigure_mode:
            return
        self.poi_lock.acquire()
        if len(self.points_of_interest_temp) > 0:
            del self.points_of_interest_temp[-1]
        self.poi_lock.release()

    def mouse_down(self, event):
        if not self.reconfigure_mode:
            return
        self.poi_lock.acquire()
        self.points_of_interest_temp.append([[event.x, event.y], [0, 0]])
        self.poi_lock.release()
        self.mouse_start_x = event.x
        self.mouse_start_y = event.y

    def mouse_up(self, event):
        if not self.reconfigure_mode:
            return
        crop_size = [self.mouse_start_x - event.x,
                     self.mouse_start_y - event.y]
        crop_size = [abs(crop_size[0]), abs(crop_size[1])]
        middle_point = [int(self.mouse_start_x - (self.mouse_start_x - event.x) / 2),
                        int(self.mouse_start_y - (self.mouse_start_y - event.y) / 2)]
        self.poi_lock.acquire()
        self.points_of_interest_temp[-1] = [middle_point, crop_size]
        self.poi_lock.release()

    def mouse_drag(self, event):
        if not self.reconfigure_mode:
            return
        crop_size = [self.mouse_start_x - event.x,
                     self.mouse_start_y - event.y]
        crop_size = [abs(crop_size[0]), abs(crop_size[1])]
        middle_point = [int(self.mouse_start_x - (self.mouse_start_x - event.x) / 2),
                        int(self.mouse_start_y - (self.mouse_start_y - event.y) / 2)]
        self.poi_lock.acquire()
        self.points_of_interest_temp[-1][0] = middle_point
        self.points_of_interest_temp[-1][1] = crop_size
        self.poi_lock.release()

    def connect(self):
        if not self.connected:
            ip = self.ip_box.get()
            port = int(self.port_box.get())
            self.points_of_interest = self.tcp_client.connect(ip, port)
            print('Connected!')
            thread = threading.Thread(target=self.tcp_client.receive)
            thread.daemon = True
            thread.start()
            self.connected = True

    def disconnect(self):
        self.connected = False
        self.tcp_client.disconnect()

    def update(self):
        if self.reconfigure_mode:
            if not self.image_edited:
                self.root.after(50, self.update)
                return
            self.image_edited = self.image_draw_mode.copy()
            img2 = ImageDraw.Draw(self.image_edited)
            self.poi_lock.acquire()
            for point in self.points_of_interest_temp:
                img2.rectangle(((int(point[0][0] - point[1][0] / 2), int(point[0][1] - point[1][1] / 2)),
                                (int(point[0][0] + point[1][0] / 2), int(point[0][1] + point[1][1] / 2))),
                               outline='red')
            self.poi_lock.release()
            self.image_raw = self.image_edited.copy()

        else:
            self.image_orig_lock.acquire()
            self.image_orig = self.tcp_client.get_next_image()
            if self.image_orig:
                self.image_raw = self.image_orig.copy()
            self.image_orig_lock.release()
        if self.image_raw:
            # tkinter requires the image to be saved
            self.photo_image = ImageTk.PhotoImage(self.image_raw)
            self.panel.configure(image=self.photo_image)
            self.panel.size()
        self.root.after(50, self.update)


def main():
    #mod = Model.load_model("/home/cf2017/PycharmProjects/AIScouts/AIScouts/DeepLearning/models/park_model22")
    client = Client()
    client.run_tk()

if __name__ == '__main__':
    main()


