
import socket
import sys, tty, termios
#import RPi.GPIO as GPIO
from time import sleep

#ef main():
command = "stop"
ch = None



def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def forward(): #Etuperin meno
        global command
        print ("Going forward (Eteenpain)")
        command = "forward"


def backward(): # Takaperin meno
        global command
        print ("Going backward (Taaksepain)")
        command = "backward"


def stop(): #Pysahtyminen "(Ei terava)"
        global command
        print ("Now stop (Seis)")
        command = "stop"


def turnLeft():
        global command
        print ("turnLeft (Vasen)")
        command = "turnleft"


def turnRight():
        global command
        print ("turnRight (Oikea)")
        command = "turnright"


def spinLeft():
        global command
        print ("spinLeft (Vasemmalle pyorahdys)")
        command = "spinleft"

       
def spinRight():
        global command
        print ("spinRight (Oikealle pyorahdys)")
        command = "spinright"


def reverseLeft():
        global command
        print ("reverseLeft (Takavasen)")
        command = "reverseleft"



def reverseRight():
        global command
        print ("reverseRight (Takaoikea)")
        command = "reverseright"

def disconnect():
        global command
        print("Going offline")
        command = "disconnect"
        
def poweroff():
        global command
        print("Closing receiver program and controlling program")
        command = "poweroff"


try:
    #Device ip address
    #ipdevice = None
    ipdevice = raw_input("Input IP Address: ")
    #server_address = ('192.168.51.212', 1337)
    server_address = (ipdevice, 1337)
    server_address = server_address
    #print("Device IP Address: " + ipdevice)

    #Creating connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #server_address = ('192.168.51.193', 1337)
    sock.connect(server_address)


    # Instructions printed on console
    print("w/x: acceleration (kiihdytys)")
    print("q/e, a/d, z/c: steering (ohjaus)")
    print("m: exits, closes receiving program as well")
    print("n: exits, only disconnects from receiving program")


    # Infinite loop that will not end until the user presses the
    # exit key

    while True:
        try:
            # Keyboard character retrieval method is called and saved
            # into variable
                char = getch()
                command = "stop"
                stop()




                # The car will drive forward when the "w" key is pressed
                if(char == "w"):
                    command = "forward"
                    forward()
                    #sleep(0.25)
                    pass

            # The car will reverse when the "x" key is pressed
                if(char == "x"):
                    command = "reverse"
                    backward()
                    #sleep(0.25)
                    pass

            # The "q" key will toggle the steering left
                if(char == "q"):
                    command = "left"
                    turnLeft()
                    #sleep(0.3)
                    pass

            # The "e" key will toggle the steering right
                if(char == "e"):
                    command = "right"
                    turnRight()
                    #sleep(0.3)
                    pass
            # The "a" key will toggle the steering left
                if(char == "a"):
                    command = "spinleft"
                    spinLeft()
                    #sleep(0.4)
                    pass

            # The "d" key will toggle the steering right
                if(char == "d"):
                    command = "spinright"
                    spinRight()
                    #sleep(0.4)
                    pass

                if(char == "z"):
                    command = "reverseleft"
                    reverseLeft()
                    #sleep(0.35)
                    pass

                if(char == "c"):
                    command = "reverseright"
                    reverseRight()
                    #sleep(0.35)
                    pass

                # The "s" key will stop the motor
                if(char == "s"):
                    command = "stop"
                    stop()
                    pass

                # The "m" key will break the loop and exit the program
                if(char == "m"):
                    print("Program Ended")
                    command = "stop"
                    sleep (0.10)
                    command = "poweroff"
                    poweroff()
                    sock.sendall(command.encode())
                    sock.shutdown(1)
                    sock.close()
                    sleep(1)
                    break

                # The "n" key will break the loop and exit the program
                if(char == "n"):
                    print("Program Disconnected")
                    command = "stop"
                    sleep (0.10)
                    command = "disconnect"
                    disconnect()
                    sock.sendall(command.encode())
                    sock.shutdown(1)
                    sock.close()
                    sleep (1)
                    break

                sock.sendall(command.encode())

            # The keyboard character variable will be set to blank, ready
            # to save the next key that is pressed
                char = ""
                sleep(0.25)

        except Exception as exc:
                print("Error",  exc)
                command = "disconnect"
                sock.shutdown(1)
                sock.close()
                print("Socket Closed")

except Exception as exc2:
    print("Error",  exc2)
    command = "disconnect"
    #sock.shutdown(1)
    #sock.close()
    print("Program Closed")
            
            
            
            
            
