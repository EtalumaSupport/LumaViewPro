
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
from PIL import Image

led = LEDBoard()
xyz = TrinamicBoard()
cam = PylonCamera()

'''
# ----------------------------------------------------
# Controlling an LED
# ----------------------------------------------------
led.led_on(0, 50)  # turn on LED at channel 0 at 50mA
time.sleep(1)       # wait one second
led.led_off()       # turn off all LEDs

# ----------------------------------------------------
# Controlling focus and XY stage
# ----------------------------------------------------
xyz.zhome()         # home position (retracted) objective
xyz.xyhome()        # home position of xy stage
xyz.move_abs_pos('X', 5800)    # move to absolute position in um
xyz.move_abs_pos('Y', 3500)    # move to absolute position in um
xyz.move_abs_pos('Z', 3270)    # move to absolute position in um

# ----------------------------------------------------
# Controlling the Camera
# ----------------------------------------------------
if cam.active:
    cam.frame_size(1900,1900)
    cam.grab()
    img = Image.fromarray(cam.array)
    img.show()
'''



# ----------------------------------------------------
# Example
# ----------------------------------------------------
if cam.active:
    cam.frame_size(1900,1900)

    led.led_on(0, 25)  # turn on LED at channel 0 at 50mA
    time.sleep(1)       # wait 1 sec
    cam.grab()
    img = Image.fromarray(cam.array)
    img.show()

    led.led_on(1, 50)  # turn on LED at channel 0 at 50mA
    time.sleep(1)       # wait 1 sec
    cam.grab()
    img = Image.fromarray(cam.array)
    img.show()

    led.led_on(2, 150)  # turn on LED at channel 0 at 50mA
    time.sleep(1)       # wait 1 sec
    cam.grab()
    img = Image.fromarray(cam.array)
    img.show()

    led.led_off()       # turn off all LEDs
