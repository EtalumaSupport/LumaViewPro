

# -------------------------------------
# Illumination Test
# -------------------------------------
from ledboard import *
import time
'''
led = LEDBoard()

for i in range(4):
    led.led_on(i, 600)  # turn on LED at channel 0 at 50mA
    print('ch', i, 'at 600mA')
    time.sleep(1)       # wait one second
    led.leds_off()       # turn off all LEDs

'''
'''
# -------------------------------------
# Camera Test
# -------------------------------------
from pyloncamera import *
from PIL import Image


cam = PylonCamera()
if cam.active:
    cam.grab()
    img = Image.fromarray(cam.array)
    img.show()
'''


# -------------------------------------
# Motion Test
# -------------------------------------
from trinamic850 import *
import time

xyz = TrinamicBoard()

xyz.xyhome()

for i in range(5):
    print("Moving in 1 sec")
    time.sleep(1)
    xyz.move_abs_pos('X', i*10000)
    xyz.move_rel_pos('Y', 10000)
    time.sleep(6)
    print()


# a = xyz.xy_ustep2um(6333850)
# b = xyz.xy_um2ustep(a)
# c = xyz.xy_ustep2um(b)
# print(a, b, c)
