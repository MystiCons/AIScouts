#import socket
#import sys
import RPi.GPIO as GPIO
#from time import sleep

class Control:

    #Pin configuration

    Motor1A = 16
    Motor1B = 18
    Motor1E = 32

    Motor2A = 23
    Motor2B = 21
    Motor2E = 33

    Motor1PWM = None
    Motor2PWM = None

    # 50% Dutycycle
    dutycycle = 50  # 50%


    #GPIO pinnit "BOARD" numeroilla + L298N pinnit A,B = suunnat ja E = Enable ja GPIO pinnien tilan asetus outputiksi
    #Mahdollisesti tulevaisuudessa E pinnejä voitaisiin käyttää PWM:llä rajoittamaan moottorien nopeutta

    def __init__(self):
        GPIO.setmode(GPIO.BOARD)


        GPIO.setup(self.Motor1A,GPIO.OUT)
        GPIO.setup(self.Motor1B,GPIO.OUT)
        GPIO.setup(self.Motor1E,GPIO.OUT)

        GPIO.setup(self.Motor2A,GPIO.OUT)
        GPIO.setup(self.Motor2B,GPIO.OUT)
        GPIO.setup(self.Motor2E,GPIO.OUT)

        self.Motor1PWM = GPIO.PWM(self.Motor1E, 250)
        self.Motor2PWM = GPIO.PWM(self.Motor2E, 250)

        #0% Dutycycle
        self.Motor1PWM.start(0)
        self.Motor2PWM.start(0)


        self.Motor1PWM.ChangeDutyCycle(self.dutycycle)
        self.Motor2PWM.ChangeDutyCycle(self.dutycycle)

    def gpiosetup(self):
        GPIO.setmode(GPIO.BOARD)

        GPIO.setup(self.Motor1A,GPIO.OUT)
        GPIO.setup(self.Motor1B,GPIO.OUT)
        GPIO.setup(self.Motor1E,GPIO.OUT)

        GPIO.setup(self.Motor2A,GPIO.OUT)
        GPIO.setup(self.Motor2B,GPIO.OUT)
        GPIO.setup(self.Motor2E,GPIO.OUT)

        self.Motor1PWM = GPIO.PWM(self.Motor1E, 250)
        self.Motor2PWM = GPIO.PWM(self.Motor2E, 250)

        #0% Dutycycle
        self.Motor1PWM.start(0)
        self.Motor2PWM.start(0)


        self.Motor1PWM.ChangeDutyCycle(self.dutycycle)
        self.Motor2PWM.ChangeDutyCycle(self.dutycycle)


    def forward(self): #Etuperin meno
        print ("Going forwards (Eteenpain)")
        GPIO.output(self.Motor1E,GPIO.HIGH)
        GPIO.output(self.Motor2E,GPIO.HIGH)
        GPIO.output(self.Motor1A,GPIO.HIGH)
        GPIO.output(self.Motor1B,GPIO.LOW)
        GPIO.output(self.Motor2A,GPIO.HIGH)
        GPIO.output(self.Motor2B,GPIO.LOW)




    def backward(self): # Takaperin meno
        print ("Going backwards (Taaksepain)")


        GPIO.output(self.Motor1A,GPIO.LOW)
        GPIO.output(self.Motor1B,GPIO.HIGH)
        GPIO.output(self.Motor2A,GPIO.LOW)
        GPIO.output(self.Motor2B,GPIO.HIGH)
        GPIO.output(self.Motor1E,GPIO.HIGH)
        GPIO.output(self.Motor2E,GPIO.HIGH)


    def stop(self): #Pysahtyminen "(Ei terävä)"
        print ("Now stop (Seis)")


        GPIO.output(self.Motor1E,GPIO.LOW)
        GPIO.output(self.Motor2E,GPIO.LOW)
        GPIO.output(self.Motor1A,GPIO.LOW)
        GPIO.output(self.Motor1B,GPIO.LOW)
        GPIO.output(self.Motor2A,GPIO.LOW)
        GPIO.output(self.Motor2B,GPIO.LOW)




    def turnLeft(self):
        print ("turnLeft (Vasen)")

        GPIO.output(self.Motor1A,GPIO.LOW)
        GPIO.output(self.Motor1B,GPIO.LOW)
        GPIO.output(self.Motor2A,GPIO.LOW)
        GPIO.output(self.Motor2B,GPIO.HIGH)
        GPIO.output(self.Motor1E,GPIO.HIGH)
        GPIO.output(self.Motor2E,GPIO.HIGH)



    def turnRight(self):
        print ("turnRight (Oikea)")


        GPIO.output(self.Motor1A,GPIO.LOW)
        GPIO.output(self.Motor1B,GPIO.HIGH)
        GPIO.output(self.Motor2A,GPIO.LOW)
        GPIO.output(self.Motor2B,GPIO.LOW)
        GPIO.output(self.Motor1E,GPIO.HIGH)
        GPIO.output(self.Motor2E,GPIO.HIGH)


    def spinLeft(self):
        print ("spinLeft (Vasemmalle pyorahdys)")


        GPIO.output(self.Motor1A,GPIO.HIGH)
        GPIO.output(self.Motor1B,GPIO.LOW)
        GPIO.output(self.Motor2A,GPIO.LOW)
        GPIO.output(self.Motor2B,GPIO.HIGH)
        GPIO.output(self.Motor1E,GPIO.HIGH)
        GPIO.output(self.Motor2E,GPIO.HIGH)

    def spinRight(self):
        print ("spinRight (Oikealle pyorahdys)")
        GPIO.output(self.Motor1A,GPIO.LOW)
        GPIO.output(self.Motor1B,GPIO.HIGH)
        GPIO.output(self.Motor2A,GPIO.HIGH)
        GPIO.output(self.Motor2B,GPIO.LOW)
        GPIO.output(self.Motor1E,GPIO.HIGH)
        GPIO.output(self.Motor2E,GPIO.HIGH)

    def reverseLeft(self):
        print ("reverseLeft (Takavasen)")
        GPIO.output(self.Motor1E,GPIO.HIGH)
        GPIO.output(self.Motor2E,GPIO.HIGH)
        GPIO.output(self.Motor1A,GPIO.LOW)
        GPIO.output(self.Motor1B,GPIO.HIGH)
        GPIO.output(self.Motor2A,GPIO.LOW)
        GPIO.output(self.Motor2B,GPIO.LOW)


    def reverseRight(self):
        print ("reverseRight (Takaoikea)")
        GPIO.output(self.Motor1E,GPIO.HIGH)
        GPIO.output(self.Motor2E,GPIO.HIGH)
        GPIO.output(self.Motor1A,GPIO.LOW)
        GPIO.output(self.Motor1B,GPIO.LOW)
        GPIO.output(self.Motor2A,GPIO.LOW)
        GPIO.output(self.Motor2B,GPIO.HIGH)

    def cleanup(self):
        print("GPIO Cleanup")
        GPIO.setmode(GPIO.BOARD)
        GPIO.cleanup()



    #Pin configuration
    '''Motor1A = 16
    Motor1B = 18
    Motor1E = 22
     
    Motor2A = 23
    Motor2B = 21
    Motor2E = 19'''
