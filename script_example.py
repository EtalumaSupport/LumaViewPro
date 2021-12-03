
'''
To run, you will need to install the following packages:
    numpy
    pyserial
    pypylon
    time

You will also need to install the camera driver from Basler

'''
# Additional LumaViewPro files
from trinamic import *
from ledboard import *
from pyloncamera import *
import time

led = LEDBoard()
xyz = TrinamicBoard()
cam = PylonCamera()

# ----------------------------------------------------
# Controlling an LED
# ----------------------------------------------------
led.led_on(0, 100)  # turn on LED at channel 0 at 100mA
time.sleep(1)       # wait one second
led.led_off()       # turn off all LEDs

# ----------------------------------------------------
# Controlling focus and XY stage
# ----------------------------------------------------
xyz.zhome()         # home position (retracted) objective
xyz.xyhome()        # home position of xy stage
xyz.goto_Z(1000)    # move to absolute position at 1000um (1 mm)

# ----------------------------------------------------
# Controlling the Camera
# ----------------------------------------------------
