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
import lumascope_api

import time
from PIL import Image

scope = lumascope_api.Lumascope()

# ----------------------------------------------------
# Controlling an LED
# ----------------------------------------------------
for i in range(6):
    print("testing LED ", i)
    scope.led_on(i, 100)  # turn on LED at channel i at 100mA
    time.sleep(1)       # wait one second
    scope.leds_off()      # turn off all LEDs


# ----------------------------------------------------
# Controlling focus and XY stage
# ----------------------------------------------------
scope.xyhome()        # home position of xy stage

i = 0
while scope.is_homing:
    time.sleep(1)              # do not send it new commands to move while its homing
    print(i)
    i += 1

# # ----------------------------------------------------
# # Controlling the Turret (Not Yet Functional)
# # ----------------------------------------------------
# scope.thome()
# scope.move_abs_pos('T', 90.) # move to absolute position in deg

# ----------------------------------------------------
# Controlling the Camera
# ----------------------------------------------------
scope.set_frame_size(1900,1900)
    
for i in range(3):
    img = Image.fromarray(scope.get_image())
    img.show()


# ----------------------------------------------------
# Simple Scripting Example
# ----------------------------------------------------
scope.xyhome()
i = 0
while scope.is_homing:
    time.sleep(1)              # do not send it new commands to move while its homing
    print(i)
    i += 1

# Homing needs 10 seconds. Test LEDs while homing.
for i in range(6):
    print("testing LED ", i)
    scope.led_on(i, 100)  # turn on LED at channel i at 100mA
    time.sleep(1.5)
    scope.leds_off()


scope.move_absolute_position('X', 60000)    # move to absolute position in um
scope.move_absolute_position('Y', 40000)    # move to absolute position in um
scope.move_absolute_position('Z', 7000)     # move to absolute position in um
time.sleep(2)                   # wait to arrive

scope.set_frame_size(1900,1900)

scope.led_on(0, 600)
time.sleep(1)
scope.get_image()
img = Image.fromarray(scope.get_image())
img.show()

scope.led_on(1, 600)
time.sleep(1)
img = Image.fromarray(scope.get_image())
img.show()

scope.led_on(2, 600)
time.sleep(1)
img = Image.fromarray(scope.get_image())
img.show()

scope.leds_off()       # turn off all LEDs
