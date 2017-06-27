import socket
import threading
import base64
from io import BytesIO
from PIL import Image
import traceback
import sys
from PIL import ImageFile, ImageTk
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time
import tkinter

class TCPClient:
    host = ''
    port = 1337
    sock = None
    close = False
    latest_image = None
    images_lock = None

    def __init__(self):
        self.sock = socket.socket()
        self.images_lock = threading.Lock()
        socket.setdefaulttimeout(30)

    def connect(self, host, port):
        self.sock.connect((host, port))
        print('Connecting to ' + host + ':' + str(port))

    def receive(self):
        try:
            while True:

                chunks = []
                chunk = self.sock.recv(2048)
                if chunk:
                    chunks.append(chunk)
                    while True:
                        try:
                            self.sock.settimeout(0.5)
                            chunk = self.sock.recv(2048)
                            chunks.append(chunk)
                        except socket.error:
                            break
                    data = b''.join(chunks)
                    img = Image.open(BytesIO(base64.b64decode(data)))
                    self.images_lock.acquire()
                    self.latest_image = img
                    self.images_lock.release()
                    self.sock.settimeout(10)
                if self.close:
                    break
        except socket.error as e:
            traceback.print_exc(file=sys.stdout)

    def get_next_image(self):
        self.images_lock.acquire()
        img = self.latest_image

        self.images_lock.release()
        return img


def button_click_exit_mainloop (event):
    event.widget.quit() # this will cause mainloop to unblock.


def main():
    client = TCPClient()
    client.connect('192.168.51.131', 1337)
    thread = threading.Thread(target=client.receive)
    thread.daemon = True
    thread.start()
    root = tkinter.Tk()
    root.geometry('1280x720')
    canvas = tkinter.Canvas(root, width=1024, height=1024)
    canvas.pack()
    root.after(500, show_new_image, client, canvas, root)
    root.mainloop()
    client.close = True


def show_new_image(client, canvas, root, label_image=None):
    img = client.get_next_image()
    if img:
        if label_image is not None:
            label_image.destroy()
        tkpi = ImageTk.PhotoImage(img)
        label_image = tkinter.Label(root, image=tkpi)
        label_image.place(x=100, y=100, width=img.size[0], height=img.size[1])

    root.after(500, show_new_image, client, canvas, root, label_image)
    root.mainloop()


if __name__ == '__main__':
    main()


