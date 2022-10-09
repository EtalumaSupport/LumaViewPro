#!/usr/bin/python3
'''
To run, you will need to install the following packages:
    numpy
    pyserial
    pypylon
    time

You will also need to install the camera driver from Basler

'''
# Additional LumaViewPro files
from trinamic850 import *
from ledboard import *
from pyloncamera import *
import time
from PIL import Image

led = LEDBoard()
xyz = TrinamicBoard()
cam = PylonCamera()

# ----------------------------------------------------
# Controlling an LED
# ----------------------------------------------------
for i in range(6):
    print("testing LED ", i+1)
    led.led_on(i, 600)  # turn on LED at channel 0 at 50mA
    time.sleep(1)       # wait one second
    led.leds_off()      # turn off all LEDs


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
    # cam.frame_size(1900,1900)
    
    for i in range(3):
        cam.grab()
        img = Image.fromarray(cam.array)
        img.show()


# ----------------------------------------------------
# Example
# ----------------------------------------------------
xyz.xyhome()

xyz.move_abs_pos('X', 5000)    # move to absolute position in um
xyz.move_abs_pos('Y', 5000)    # move to absolute position in um
xyz.move_abs_pos('Z', 3000)    # move to absolute position in um
time.sleep(2)       # wait 1 sec

if cam.active:
    cam.frame_size(1900,1900)

    led.led_on(0, 50)  # turn on LED at channel 0 at 50mA
    time.sleep(1)      # wait 1 sec
    cam.grab()
    img = Image.fromarray(cam.array)
    img.show()

    led.led_on(1, 100)  # turn on LED at channel 0 at 50mA
    time.sleep(1)       # wait 1 sec
    cam.grab()
    #img = Image.fromarray(cam.array)
    #img.show()

    led.led_on(2, 150)  # turn on LED at channel 0 at 50mA
    time.sleep(1)       # wait 1 sec
    cam.grab()
    img = Image.fromarray(cam.array)
    img.show()

    led.leds_off()       # turn off all LEDs