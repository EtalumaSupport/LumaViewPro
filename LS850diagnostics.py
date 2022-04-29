
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

xyz.xyhome()
print('Start')
xyz.current_pos('X')
xyz.current_pos('Y')
xyz.current_pos('Z')

# Move Approximately Center
xyz.move_abs_pos('X', 1000000)
xyz.move_abs_pos('Y', 1000000)

print('Moving')
xyz.current_pos('X')
xyz.current_pos('Y')
xyz.current_pos('Z')

time.sleep(2)

print('Done')
xyz.current_pos('X')
xyz.current_pos('Y')
xyz.current_pos('Z')
