import socket
import threading
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import traceback
import sys
from PIL import ImageTk
import tkinter
import pickle


class TCPClient:
    host = ''
    port = 1337
    sock = None
    close = False
    latest_image = None
    images_lock = None
    socket_lock = None

    def __init__(self):
        self.init_socket()
        self.images_lock = threading.Lock()
        self.socket_lock = threading.Lock()

    def init_socket(self):
        self.sock = socket.socket()
        socket.setdefaulttimeout(30)

    def connect(self, host, port):
        try:
            print('Connecting to ' + host + ':' + str(port))
            self.sock.connect((host, port))
            self.close = False
            print('Connected!')
            return True
        except socket.error as e:
            print('Connection failed! ' + e.strerror)
            return False

    def disconnect(self):
        self.socket_lock.acquire()
        self.close = True
        self.socket_lock.release()

    def receive(self):
        try:
            while True:
                chunks = []
                chunk = self.sock.recv(2048)
                if self.close:
                    break
                self.socket_lock.acquire()
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
                    img = Image.open(BytesIO(base64.b64decode(data)))
                    self.images_lock.acquire()
                    self.latest_image = img
                    self.images_lock.release()
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
            data_string = pickle.dumps(data)
            self.sock.sendall(data_string)
        except socket.error as e:
            print(e.strerror)

    def get_next_image(self):
        self.images_lock.acquire()
        img = self.latest_image

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

    ip_box = None
    port_box = None
    connect_button = None
    disconnect_button = None
    redraw_button = None
    cancel_button = None
    connected = False

    image_edited_lock = threading.Lock()

    mouse_start_x = 0
    mouse_start_y = 0

    reconfigure_mode = False
    points_of_interest = []

    def __init__(self):
        self.tcp_client = TCPClient()
        self.build_ui()
        self.panel.bind("<Button 1>", self.mouse_down)
        self.panel.bind('<B1-Motion>', self.mouse_drag)
        self.panel.bind('<ButtonRelease-1>', self.mouse_up)
        self.root.after(100, self.show_new_image)

    def build_ui(self):
        self.root = tkinter.Tk()
        self.root.title('Configuration')
        self.root.geometry('1440x900')
        self.root.resizable(False, False)
        self.panel = tkinter.Label(self.root)
        self.panel.pack()

        self.ip_box = tkinter.Entry(self.root)
        self.ip_box.pack()
        self.ip_box.delete(0, tkinter.END)
        self.ip_box.insert(0, "192.168.51.131")

        self.port_box = tkinter.Entry(self.root)
        self.port_box.pack()
        self.port_box.delete(0, tkinter.END)
        self.port_box.insert(0, "1337")

        self.connect_button = tkinter.Button(self.root, text="Connect", width=10, command=self.connect)
        self.connect_button.pack()

        self.disconnect_button = tkinter.Button(self.root, text="Disconnect", width=10, command=self.disconnect)
        self.disconnect_button.pack()

        self.redraw_button = tkinter.Button(self.root, text="Draw new parks", width=10, command=self.redraw)
        self.redraw_button.pack()

    def run_tk(self):
        self.root.mainloop()
        self.tcp_client.close = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tcp_client.close = True

    def cancel_redraw(self):
        self.redraw_button.configure(text="Draw new parks")
        self.reconfigure_mode = False
        self.cancel_button.destroy()

    def redraw(self):
        if not self.reconfigure_mode:
            if not self.image_orig:
                return
            self.image_draw_mode = self.image_orig.copy()
            self.reconfigure_mode = True
            self.redraw_button.configure(text="Stop and send")
            self.cancel_button = tkinter.Button(self.root, text="Cancel drawing", width=10, command=self.cancel_redraw)
            self.cancel_button.pack()
        else:
            self.cancel_redraw()
            self.tcp_client.send_data(self.points_of_interest)
            self.points_of_interest = []

    def mouse_down(self, event):
        if not self.reconfigure_mode:
            return
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
        self.points_of_interest.append([middle_point, crop_size])
        img2 = ImageDraw.Draw(self.image_draw_mode)
        img2.rectangle(((self.mouse_start_x, self.mouse_start_y), (event.x, event.y)), outline='red')
        self.image_edited_lock.acquire()
        self.image_edited = self.image_draw_mode.copy()
        self.image_edited_lock.release()

    def mouse_drag(self, event):
        if not self.reconfigure_mode:
            return
        self.image_edited_lock.acquire()
        self.image_edited = self.image_draw_mode.copy()
        img2 = ImageDraw.Draw(self.image_edited)
        img2.rectangle(((self.mouse_start_x, self.mouse_start_y), (event.x, event.y)), outline='red')
        self.image_edited_lock.release()

    def connect(self):
        if not self.connected:
            ip = self.ip_box.get()
            port = int(self.port_box.get())
            if self.tcp_client.connect(ip, port):
                thread = threading.Thread(target=self.tcp_client.receive)
                thread.daemon = True
                thread.start()
                self.connected = True

    def disconnect(self):
        self.connected = False
        self.tcp_client.disconnect()

    def show_new_image(self):
        if self.reconfigure_mode:
            if not self.image_edited:
                self.root.after(50, self.show_new_image)
                return
            self.image_edited_lock.acquire()
            self.image_raw = self.image_edited.copy()
            self.image_edited_lock.release()

        else:
            self.image_orig = self.tcp_client.get_next_image()
            if self.image_orig:
                self.image_raw = self.image_orig.copy()
        if self.image_raw:
            # tkinter requires the image to be saved
            self.photo_image = ImageTk.PhotoImage(self.image_raw)
            self.panel.configure(image=self.photo_image)
            self.panel.size()
        self.root.after(50, self.show_new_image)


def main():
    client = Client()
    client.run_tk()

if __name__ == '__main__':
    main()


