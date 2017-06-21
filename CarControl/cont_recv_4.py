from control2 import Control
import socket
import time


def getipaddr():
    sock_ip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_ip.connect(("195.148.129.49", 80))
    ownip = None
    ownip = (sock_ip.getsockname()[0])
    print("Your IP Address is: ", ownip)
    return ownip


def main():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cont = Control()
    # Bind the socket to the port
    try:

        #server_address = ('192.168.51.212', 1337)
        ip = str(getipaddr())
        server_address = (ip, 1337)
        print ('starting up on port ' + str(server_address))
        sock.bind(server_address)
        # Listen for incoming connections
        sock.listen(1)
        A = 1
        B = 1
        while (A == 1):
            # Wait for a connection
            print('waiting for a connection')
            connection, client_address = sock.accept()
            print('connection from' + str(client_address))
            B = 1
            cont.gpiosetup()
            print("GPIO set")
            while (B == 1):
               
                data = connection.recv(128)
                message = str(data).strip()
                #print('received ' + message)
                #cont.gpiosetup()
                
                if(message.find("forward") != -1):
                    cont.forward()
                    time.sleep(0.3)
                    pass
                    
                if(message.find("backward") != -1):
                    cont.backward()
                    time.sleep(0.3)
                    pass
                    
                if(message.find("turnleft") != -1):
                    cont.turnLeft()
                    time.sleep(0.3)
                    pass

                if(message.find("turnright") != -1):
                    cont.turnRight()
                    time.sleep(0.3)
                    pass
                    
                if(message.find("spinright") != -1):
                    cont.spinRight()
                    time.sleep(0.3)
                    pass
                    
                if(message.find("spinleft") != -1):
                    cont.spinLeft()
                    time.sleep(0.3)
                    pass
                
                if(message.find("reverseleft") != -1):
                    cont.reverseLeft()
                    time.sleep(0.3)
                    pass
                    
                if(message.find("reverselight") != -1):
                    cont.reverseRight()
                    time.sleep(0.3)
                    pass
                
                
                if(message.find("stop") != -1):
                    cont.stop()
                    pass

                if(message.find("disconnect") != -1):
                    cont.stop()
                    cont.cleanup()
                    cont.gpiosetup()
                    sock.shutdown(1)
                    B = 0
                    print ("Disconnected connection")
                    time.sleep(1)
                    continue
                    
                #Equal to Didnt work correctly

                if(message.find("poweroff") != -1):
                    cont.stop()
                    #cont.cleanup()
                    B = 0
                    A = 0
                    sock.shutdown(1)
                    sock.close()
                    print ("Program closed")
                    time.sleep(1)
                    continue
                    
                if(B == 0):
                    sock.shutdown(1)
                    time.sleep(1)
                    message = None
                    connection.shutdown(socket.SHUT_RDWR)
                    print("B 0")
                    continue
                    
                time.sleep(0.2)
                cont.stop()
            

            if(A == 0):
                #sock.shutdown(1)
                #sock.close()
                time.sleep(1)
                connection.shutdown(socket.SHUT_RDWR)
                time.sleep(0.1)
                cont.cleanup()
                print("A 0")
                break
                
            
    except Exception as exc:
        pass
        print("Error:"+ str(exc))
        cont.stop()
        cont.cleanup()
        sock.shutdown(1)
        sock.close()

if __name__ == "__main__":
    main()
