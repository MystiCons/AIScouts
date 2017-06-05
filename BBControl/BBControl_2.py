#!/usr/bin/python
import socket
import BB8_driver
import time
import sys


bb8 = BB8_driver.Sphero()


def main():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bb8.connect()
    bb8.start()

    # Bind the socket to the port
    try:
        server_address = ('192.168.51.247', 1337)
        print ('starting up on port ' + str(server_address))
        sock.bind(server_address)
        # Listen for incoming connections
        sock.listen(1)
        A = 1
        B = 1
        while B == 1:
            # Wait for a connection
            print('waiting for a connection')
            connection, client_address = sock.accept()
            A = 1
            while A == 1:
                print('connection from' + str(client_address))
                data = connection.recv(1024)
                message = str(data).strip()
                print('received ' + message)
                if(message == "forward"):
                    bb8.roll(100, 180,  1, False)
                    time.sleep(0.5)
                    pass
                    
                if(message == "reverse"):
                    bb8.roll(100, 0, 1, False)
                    time.sleep(0.5)
                    pass
                    
                if(message == "left"):
                    bb8.roll(100, 90, 1, False)
                    time.sleep(0.5)
                    pass

                if(message == "right"):
                    bb8.roll(100, 270, 1, False)
                    time.sleep(0.5)
                    pass
                if(message == "stop"):
                    bb8.roll(0, 0, 1, False)
                    bb8.set_rgb_led(0,255,0,0,False)
                    pass

                if(message == "disconnect"):
                    bb8.roll(0, 0, 0, False)
                    bb8.set_rgb_led(0,0,255,0,False)
                    sock.shutdown(1)
                    A = 0
                    print ("Disconnected connection")
                    pass
                    
                if(message == "poweroff"):
                    bb8.roll(0, 0, 0, False)
                    bb8.set_rgb_led(255,255,255,0,False)
                    bb8.disconnect()
                    sock.shutdown(1)
                    sock.close()
                    A = 0
                    B = 0
                    print ("Program closed")
                    break
                    
            
                #bb8.set_rgb_led(255,0,0,0,False)
                bb8.roll(0, 0, 0, False)
                bb8.join()

    except Exception as exc:
        print("Error:"+ str(exc))
        bb8.disconnect()
        sock.shutdown(1)
        sock.close()
            



            #finally:
              #sock.close()
            #bb8.disconnect()
    
    #sock.close()
    #bb8.disconnect()


if __name__ == "__main__":
    main()
