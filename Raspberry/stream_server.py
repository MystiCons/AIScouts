import socket
import sys
import threading
import base64
import datetime
import traceback
from io import BytesIO


class StreamServer:
    host = ''
    port = 1337
    sock = None
    connections = {}
    closed = False
    connections_lock = None

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind((self.host, self.port))
        # Set accept timeout to 60 sec
        socket.setdefaulttimeout(30)
        self.connections_lock = threading.Lock()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.closed = True

    def start(self):
        try:
            self.sock.listen(10)
            while True:
                conn, addr = self.sock.accept()
                if conn is not None and addr is not None:
                    self.connections_lock.acquire()
                    self.connections.update({addr[0]: conn})
                    self.connections_lock.release()
                    print(str(addr) + ' Connected! ' + str(datetime.datetime.now()))
                if self.closed:
                    break

        except Exception as e:
            traceback.print_exc(file=sys.stdout)
        finally:
            self.sock.close()

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

    def close(self):
        pass
