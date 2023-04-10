#!/usr/bin/python3

# ----------------------------------------------------
# lumascope_api testing. Not intended for customer
# ----------------------------------------------------

# libraries
# ----------------------------------------------------
import lumascope_api
import time
from PIL import Image
scope = lumascope_api.Lumascope()

# home microscope
scope.xyhome()
scope.set_frame_size(1900,1900)

for t in range(10):
    time.sleep(1)              # cannot send it new commands to move while its homing
    print(10-t)

# # Testing 'CAMERA FUNCTIONS'
# # ----------------------------------------------------
# scope.get_image()
# scope.save_live_image()

# Testing 'INTEGRATED SCOPE FUNCTIONS'
# ----------------------------------------------------
# example step values
x = 59.95*1000
y = 37.48*1000
z = 7253.43
af = 1
ch = 4
fc = 0
ill = 10
gain = 1
auto_gain = 0
exp = 30

step  = [x, y, z, af, ch, fc, ill, gain, auto_gain, exp]

# verified functional
scope.goto_step(step)
scope.led_on(3, 1) 
time.sleep(5)

# verified functional
AF_min = 0.5
AF_max = 6.0
AF_range = 15.0
scope.autofocus(AF_min, AF_max, AF_range)
