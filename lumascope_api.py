#!/usr/bin/python3

'''
MIT License

Copyright (c) 2023 Etaluma, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyribackground_downght notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```
This open source software was developed for use with Etaluma microscopes.

AUTHORS:
Kevin Peter Hickerson, The Earthineering Company
Anna Iwaniec Hickerson, Keck Graduate Institute

MODIFIED:
March 2, 2023
'''

# Import Lumascope Hardware files
from trinamic850 import TrinamicBoard
from ledboard import LEDBoard
from pyloncamera import PylonCamera

class Lumascope():

    def __init__(self):
        """Initialize Microscope"""

        # LED Control Board
        try:
            self.led = LEDBoard()

        except:
            self.led = False
            print('[SCOPE API ] LED False')
        
        # Motion Control Board
        try:
            self.motion = TrinamicBoard()
        except:
            self.motion = False
            print('[SCOPE API ] Motion False')

        # Camera
        try:
            self.camera = PylonCamera()
        except:
            self.camera = False
            print('[SCOPE API ] Camera False')


    ########################################################################
    # LED BOARD FUNCTIONS
    ########################################################################

    def leds_enable(self):
        """ Enable all LEDS"""
        if not self.led: return
        self.led.leds_enable()

    def leds_disable(self):
        """ Disable all LEDS"""
        if not self.led: return
        self.led.leds_disable()

    def led_on(self, channel, mA):
        """ Turn on LED at channel number at mA power """
        if not self.led: return
        self.led.led_on(channel, mA)

    def led_off(self, channel):
        """ Turn off LED at channel number """
        if not self.led: return
        self.led.led_off(channel)

    def leds_off(self):
        """ Turn off all LEDs """
        if not self.led: return
        self.led.leds_off()

    def ch2color(self, color):
        if not self.led: return
        return self.led.ch2color(color)

    def color2ch(self, color):
        if not self.led: return
        return self.led.color2ch(color)

    ########################################################################
    # CAMERA FUNCTIONS
    ########################################################################
    def get_image(self):
        """Get last image grabbed by camera"""
        if self.camera.grab():
            array = self.camera.array
            return array
        else:
            return False

    def save_image(self, save_folder = './capture', file_root = 'img_', append = 'ms', color = 'BF'):
        pass
        # if not lumaview.scope.camera:
        #     return

        # img = np.zeros((lumaview.scope.camera.array.shape[0], lumaview.scope.camera.array.shape[1], 3))

        # if color == 'Blue':
        #     img[:,:,0] = lumaview.scope.camera.array
        # elif color == 'Green':
        #     img[:,:,1] = lumaview.scope.camera.array
        # elif color == 'Red':
        #     img[:,:,2] = lumaview.scope.camera.array
        # else:
        #     img[:,:,0] = lumaview.scope.camera.array
        #     img[:,:,1] = lumaview.scope.camera.array
        #     img[:,:,2] = lumaview.scope.camera.array

        # img = np.flip(img, 0)

        # # set filename options
        # if append == 'ms':
        #     append = str(int(round(time.time() * 1000)))
        # elif append == 'time':
        #     append = time.strftime("%Y%m%d_%H%M%S")
        # else:
        #     append = ''

        # # generate filename string
        # filename = file_root + append + '.tiff'

        # try:
        #     cv2.imwrite(save_folder+'/'+filename, img.astype(np.uint8))
        #     # cv2.imwrite(filename, img.astype(np.uint8))
        # except:
        #     error_log("Error: Unable to save. Perhaps save folder does not exist?")


    def set_frame_size(self, w, h):
        """Set frame size of camera to w by h"""

        if not self.camera: return
        self.camera.frame_size(w, h)

    def set_gain(self, gain):
        """Set camera gain"""

        if not self.camera: return
        self.camera.gain(gain)

    def set_auto_gain(self, state=True):
        """ Enable / Disable camera auto_gain with the value of 'state'
        It will be continueously updating based on the current image """

        if not self.camera: return
        self.camera.auto_gain(state)

    def set_exposure_time(self, t):
        """ Set exposure time in the camera hardware t (msec)"""

        if not self.camera: return
        self.camera.exposure_t(t)

    def get_exposure_time(self):
        """ Get exposure time in the camera hardware
         Returns t (msec), or -1 if the camera is inactive"""

        if not self.camera: return -1
        exposure = self.camera.get_exposure_t()
        return exposure
        
    def set_auto_exposure_time(self, state = True):
        """ Enable / Disable camera auto_exposure with the value of 'state'
        It will be continueously updating based on the current image """

        if not self.camera: return
        self.camera.auto_exposure_t(state)

    ########################################################################
    # MOTION CONTROL FUNCTIONS
    ########################################################################

    def zhome(self):
        """Home the z-axis (i.e. focus)"""
        if not self.motion: return
        self.motion.zhome()

    def xyhome(self):
        """Home the xy-axes (i.e. stage). Note: z-axis will always home first"""
        if not self.motion: return
        self.motion.xyhome()

    def xycenter(self):
        """Move Stage to the center."""

        if not self.motion: return
        self.motion.xycenter()

    def thome(self):
        """Home the Turret"""

        if not self.motion: return
        self.motion.thome()

    def get_target_position(self, axis):
        """Get the value of the target position of the axis relative to home
        Returns position (um), or 0 if the motion board is inactive
        values of axis 'X', 'Y', 'Z', and 'T' """

        if not self.motion: return 0
        target_position = self.motion.target_pos(axis)
        return target_position
        
    def get_current_position(self, axis):
        """Get the value of the current position of the axis relative to home
        Returns position (um), or 0 if the motion board is inactive
        values of axis 'X', 'Y', 'Z', and 'T' """

        if not self.motion: return 0
        target_position = self.motion.current_pos(axis)
        return target_position
        
    def move_absolute_position(self, axis, pos):
        """ Move to absolute position (in um) of axis"""

        if not self.motion: return
        self.motion.move_abs_pos(axis, pos)

    def move_relative_position(self, axis, um):
        """ Move to relative distance (in um) of axis"""

        if not self.motion: return
        self.motion.move_rel_pos(axis, um)

    def get_home_status(self, axis):
        """ Return True if axis is in home position"""
 
        if not self.motion: return False
        status = self.motion.home_status(axis)
        return status

    def get_target_status(self, axis):
        """ Return True if axis is at target position"""

        if not self.motion: return False
        status = self.motion.target_status(axis)
        return status
        
    # Get all reference status register bits as 32 character string (32-> 0)
    def get_reference_status(self, axis):
        """ Get all reference status register bits as 32 character string (32-> 0) """

        if not self.motion: return False
        status = self.motion.reference_status(axis)
        return status

    def get_overshoot(self):
        """Is z-axis (focus) currently in overshoot mode?"""

        if not self.motion: return False
        return self.motion.overshoot
