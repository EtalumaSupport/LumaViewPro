# Non-GUI protocol script  - Example
# ------------------------------------------------------------------------

# Import Python Libraries
import time
import pandas as pd
from PIL import Image

# Import Custom Libraries
from trinamic850 import *
from ledboard import *
from pyloncamera import *

# Initialize hardware
led = LEDBoard()
xyz = TrinamicBoard()
cam = PylonCamera()

# home position of xy-stage
xyz.xyhome()

# Load Script from file

protocol = pd.read_csv('./data/sample_protocol.csv')

# Iterate through lines
for i in range(len(protocol)):

    # Get Info
    name = protocol['Name'].iloc[i]
    x = protocol['X'].iloc[i]
    y = protocol['Y'].iloc[i]
    z = protocol['Z'].iloc[i]
    ch = int(protocol['Channel'].iloc[i])
    ill = protocol['Illumination'].iloc[i]
    gain = protocol['Gain'].iloc[i]
    exp = protocol['Exposure'].iloc[i]
    AF = protocol['AF'].iloc[i]
    dt = protocol['dt'].iloc[i]

    print(name)

    # Go to XYZ position
    xyz.move_abs_pos('X', x)
    xyz.move_abs_pos('Y', y)
    xyz.move_abs_pos('Z', z)

    while  not(xyz.target_status('X') and xyz.target_status('Y') and xyz.target_status('Z')):
        time.sleep(0.1)

    # turn on LED
    led.led_on(ch, ill)
    time.sleep(1)
    # Adjust Camera Settings
    # cam.gain(gain)
    # cam.exposure_t(exp)
    # cam.frame_size(400,400)

    # Capture Image
    cam.grab()
    img = Image.fromarray(cam.array)
    # img.show()

    # Turn off LEDs
    led.leds_off()
