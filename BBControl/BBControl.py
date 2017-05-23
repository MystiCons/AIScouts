#!/usr/bin/python
import socket
import BB8_driver

bb8 = BB8_driver.Sphero()

def main():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bb8.connect()
    bb8.start()

    # Bind the socket to the port
    try:
        server_address = ('192.168.51.207', 1337)
        print ('starting up on port ' + str(server_address))
        sock.bind(server_address)
        # Listen for incoming connections
        sock.listen(1)
        while True:
            # Wait for a connection
            print('waiting for a connection')
            connection, client_address = sock.accept()
            print('connection from' + str(client_address))
            data = connection.recv(100)
            message = str(data).strip()
            print('received ' + message)
            if(message == "forward"):
                bb8.roll(150, 180,  1, False)
            if(message == "backward"):
                bb8.roll(150, 0, 1, False)
            if(message == "left"):
                bb8.roll(150, 90, 1, False)
            if(message == "right"):
                bb8.roll(150, 270, 1, False)
            bb8.set_rgb_led(255,0,0,0,False)
            bb8.join()
    finally:
        connection.close()
        bb8.disconnect()


if __name__ == "__main__":
    main()



