import socket
import sys
import threading
import base64
import datetime
import traceback
import pickle
from io import BytesIO


class StreamServer:
    host = ''
    port = 1337
    sock = None
    connections = {}
    closed = False
    connections_lock = None
    received_data = False
    poi = []
    poi_lock = threading.Lock()

    def __init__(self, poi):
        self.sock = socket.socket()
        self.poi = poi
        self.sock.bind((self.host, self.port))
        # Set accept timeout to 60 sec
        socket.setdefaulttimeout(30)
        self.connections_lock = threading.Lock()
        self.sock.listen(10)

    def __enter__(self):
        self.sock = socket.socket()
        self.sock.bind((self.host, self.port))
        # Set accept timeout to 60 sec
        socket.setdefaulttimeout(30)
        self.connections_lock = threading.Lock()
        self.sock.listen(10)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.closed = True

    def start(self):
        try:
            while True:
                conn, addr = self.sock.accept()
                if conn is not None and addr is not None:
                    self.connections_lock.acquire()
                    self.connections.update({addr[0]: conn})
                    self.connections_lock.release()
                    self.send_data(self.poi)
                    client_thread = threading.Thread(target=self.receive_poi, args=(conn, addr))
                    client_thread.daemon = True
                    client_thread.start()
                    print(str(addr) + ' Connected! ' + str(datetime.datetime.now()))
                if self.closed:
                    break

        except Exception as e:
            traceback.print_exc(file=sys.stdout)
        finally:
            self.sock.close()

    def close(self):
        self.sock.close()

    def send_data(self, data):
        try:
            data_string = pickle.dumps(data)
            self.sock.sendall(data_string)
        except socket.error as e:
            print(e.strerror)

    def send_data_to_all(self, data):
        if len(self.connections) > 0:
            buffer = BytesIO()
            data.save(buffer, format='JPEG')
            data_str = base64.b64encode(buffer.getvalue())
            self.connections_lock.acquire()
            disconnected = []
            for addr in self.connections:
                try:
                    self.connections[addr].sendall(data_str)
                except socket.error as e:
                    # Save key if disconnected
                    disconnected.append(addr)
                    print(str(addr) + ' Disconnected!')
            # Delete disconnected addresses
            if len(disconnected) > 0:
                for addr in disconnected:
                    del self.connections[addr]
            self.connections_lock.release()

    def get_poi(self):
        self.poi_lock.acquire()
        saved_poi = self.poi.copy()
        self.poi = []
        self.received_data = False
        self.poi_lock.release()
        return saved_poi

    def receive_poi(self, conn, addr):
        try:
            while True:
                chunks = []
                chunk = 0
                try:
                    chunk = conn.recv(2048)
                except socket.error:
                    pass
                finally:
                    if self.closed:
                        break
                    if chunk:
                        chunks.append(chunk)
                        while True:
                            try:
                                conn.settimeout(0.5)
                                chunk = conn.recv(2048)
                                if not chunk:
                                    break
                                chunks.append(chunk)
                            except socket.error:
                                break
                        data = b''.join(chunks)
                        self.poi_lock.acquire()
                        self.poi = pickle.loads(data)
                        self.poi_lock.release()
                        self.received_data = True
                        conn.settimeout(10)
                        print('Received new points of interest!')
                    if self.closed:
                        break
        except socket.error as e:
            traceback.print_exc(file=sys.stdout)

