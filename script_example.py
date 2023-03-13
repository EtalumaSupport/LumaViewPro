#!/usr/bin/python3
'''
To run, you will need to install the following packages:
    numpy
    pyserial
    pypylon
    time
    Pillow

You will also need to install the camera driver from Basler

'''
# Additional LumaViewPro files
from motorboard import *
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
    print("testing LED ", i)
    led.led_on(i, 100)  # turn on LED at channel i at 100mA
    time.sleep(1)       # wait one second
    led.leds_off()      # turn off all LEDs


# ----------------------------------------------------
# Controlling focus and XY stage
# ----------------------------------------------------
xyz.xyhome()        # home position of xy stage
for t in range(5):
    time.sleep(1)              # cannot send it new commands to move while its homing
    print(5-t)
xyz.move_abs_pos('X', 60000)    # move to absolute position in um
xyz.move_abs_pos('Y', 40000)    # move to absolute position in um
xyz.move_abs_pos('Z', 7000)     # move to absolute position in um

# # ----------------------------------------------------
# # Controlling the Turret (Not Yet Functional)
# # ----------------------------------------------------
# xyz.thome()
# xyz.move_abs_pos('T', 30.000) # move to absolute position in deg

# ----------------------------------------------------
# Controlling the Camera
# ----------------------------------------------------
if cam.active:
    cam.frame_size(1900,1900)
    
    for i in range(3):
        cam.grab()
        img = Image.fromarray(cam.array)
        img.show()


# ----------------------------------------------------
# Example
# ----------------------------------------------------
xyz.xyhome()

# Homing needs 5 seconds. Test LEDs while homing.
for i in range(6):
    print("testing LED ", i)
    led.led_on(i, 100)  # turn on LED at channel i at 100mA
    time.sleep(1)

led.leds_off()

xyz.move_abs_pos('X', 60000)    # move to absolute position in um
xyz.move_abs_pos('Y', 40000)    # move to absolute position in um
xyz.move_abs_pos('Z', 7000)     # move to absolute position in um
time.sleep(2)                   # wait to arrive

if cam.active:
    cam.frame_size(1900,1900)

    led.led_on(0, 600)
    time.sleep(1)
    cam.grab()
    img = Image.fromarray(cam.array)
    img.show()

    led.led_on(1, 600)
    time.sleep(1)
    cam.grab()
    img = Image.fromarray(cam.array)
    img.show()

    led.led_on(2, 600)
    time.sleep(1)
    cam.grab()
    img = Image.fromarray(cam.array)
    img.show()

    led.leds_off()       # turn off all LEDs