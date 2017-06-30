import socket

def main():
    try:
        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Connect the socket to the port where the server is listening
        server_address = ('192.168.51.207', 1337)
        print ('connecting to port ' + str(server_address))
        sock.connect(server_address)
        # Send data
        message = input('Direction: ')
        print ('sending ' + message)
        sock.sendall(message.encode())
        
    finally:
        print('closing socket')
        sock.close()
    
if __name__ == "__main__":
    main()
