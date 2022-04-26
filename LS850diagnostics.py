
'''
# -------------------------------------
# Illumination Test
# -------------------------------------
from ledboard import *
import time

led = LEDBoard()

for i in range(4):
    led.led_on(i, 60000)  # turn on LED at channel 0 at 50mA
    time.sleep(1)       # wait one second
    led.led_off()       # turn off all LEDs
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

xyz = TrinamicBoard()
import time

time.sleep(1)
# xyz.move_rel_pos('X', 160000)
# xyz.move_rel_pos('Y', 160000)
xyz.current_pos('X')
xyz.current_pos('Y')
xyz.current_pos('Z')
xyz.target_pos('X')
xyz.target_pos('Y')
xyz.target_pos('Z')

'''
# signed 32 bit hex to dec
if value >=  0x80000000:
    value -= 0x10000000
print(int(value))

# signed dec to 32 bit hex
value = -200000
if value < 0:
    value = 4294967296+value
print(hex(value))
'''
