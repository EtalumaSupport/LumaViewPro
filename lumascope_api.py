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
March 16, 2023
'''

# Import Lumascope Hardware files
from motorboard import MotorBoard
from ledboard import LEDBoard
from pyloncamera import PylonCamera
import time
import cv2
import numpy as np

class Lumascope():

    def __init__(self):
        """Initialize Microscope"""

        # LED Control Board
        try:
            self.led = LEDBoard()

        except:
            print('[SCOPE API ] LED Board Not Initialized')
        
        # Motion Control Board
        try:
            self.motion = MotorBoard()
        except:
            print('[SCOPE API ] Motion Board Not Initialized')

        # Camera
        try:
            self.camera = PylonCamera()
        except:
            print('[SCOPE API ] Camera Board Not Initialized')


    ########################################################################
    # LED BOARD FUNCTIONS
    ########################################################################

    def leds_enable(self):
        """ LED BOARD FUNCTIONS
        Enable all LEDS"""
        if not self.led: return
        self.led.leds_enable()

    def leds_disable(self):
        """ LED BOARD FUNCTIONS
        Disable all LEDS"""
        if not self.led: return
        self.led.leds_disable()

    def led_on(self, channel, mA):
        """ LED BOARD FUNCTIONS
        Turn on LED at channel number at mA power """
        if not self.led: return
        self.led.led_on(channel, mA)

    def led_off(self, channel):
        """ LED BOARD FUNCTIONS
        Turn off LED at channel number """
        if not self.led: return
        self.led.led_off(channel)

    def leds_off(self):
        """ LED BOARD FUNCTIONS
        Turn off all LEDs """
        if not self.led: return
        self.led.leds_off()

    def ch2color(self, color):
        """ LED BOARD FUNCTIONS
        Convert channel number to string representing color
         0 -> Blue: Fluorescence
         1 -> Green: Fluorescence
         2 -> Red: Fluorescence
         3 -> BF: Brightfield
         4 -> PC: Phase Contrast
         5 -> EP: Extended Phase Contrast
           """
        if not self.led: return
        return self.led.ch2color(color)

    def color2ch(self, color):
        """ LED BOARD FUNCTIONS
        Convert string representing color to channel number
         Blue: Fluorescence Channel 0   -> 0
         Green: Fluorescence Channel 1  -> 1
         Red: Fluorescence Channel 2    -> 2
         BF: Brightfield                -> 3
         PC: Phase Contrast             -> 4
         EP: Extended Phase Contrast    -> 5
           """
        if not self.led: return
        
        return self.led.color2ch(color)

    ########################################################################
    # CAMERA FUNCTIONS
    ########################################################################

    def get_image(self):
        """ CAMERA FUNCTIONS
        Grab and return image from camera"""
        if self.camera.grab():
            array = self.camera.array
            return array
        else:
            return False

    def save_image(self, save_folder = './capture', file_root = 'img_', append = 'ms', color = 'BF'):
        """CAMERA FUNCTIONS
        Grab the current live image and save to file
        """
        if not self.camera:
            return
        
        array = self.get_image()
        if array is False:
            return 
        img = np.zeros((array.shape[0], array.shape[1], 3))

        if color == 'Blue':
            img[:,:,0] = array
        elif color == 'Green':
            img[:,:,1] = array
        elif color == 'Red':
            img[:,:,2] = array
        else:
            img[:,:,0] = array
            img[:,:,1] = array
            img[:,:,2] = array

        img = np.flip(img, 0)

        # set filename options
        if append == 'ms':
            append = str(int(round(time.time() * 1000)))
        elif append == 'time':
            append = time.strftime("%Y%m%d_%H%M%S")
        else:
            append = ''

        # generate filename string
        filename = file_root + append + '.tiff'

        try:
            cv2.imwrite(save_folder+'/'+filename, img.astype(np.uint8))
            print("[SCOPE API ] Saving Image to",save_folder+'/'+filename )
        except:
            print("[SCOPE API ] Error: Unable to save. Perhaps save folder does not exist?")

    def get_max_width(self):
        """CAMERA FUNCTIONS
        Grab the max pixel width of camera
        """
        if not self.camera: return 0
        return self.camera.active.Width.Max

    def get_max_height(self):
        """CAMERA FUNCTIONS
        Grab the max pixel height of camera
        """
        if not self.camera: return 0
        return self.camera.active.Height.Max
      
    def get_width(self):
        """CAMERA FUNCTIONS
        Grab the current pixel width setting of camera
        """
        if not self.camera: return 0
        return self.camera.active.Width.GetValue()

    def get_height(self):
        """CAMERA FUNCTIONS
        Grab the current pixel height setting of camera
        """
        if not self.camera: return 0
        return self.camera.active.Height.GetValue()

    def set_frame_size(self, w, h):
        """CAMERA FUNCTIONS
        Set frame size (pixel width by picel height
        of camera to w by h"""

        if not self.camera: return
        self.camera.frame_size(w, h)

    def set_gain(self, gain):
        """CAMERA FUNCTIONS
        Set camera gain"""

        if not self.camera: return
        self.camera.gain(gain)

    def set_auto_gain(self, state=True):
        """CAMERA FUNCTIONS
        Enable / Disable camera auto_gain with the value of 'state'
        It will be continueously updating based on the current image """

        if not self.camera: return
        self.camera.auto_gain(state)

    def set_exposure_time(self, t):
        """CAMERA FUNCTIONS
         Set exposure time in the camera hardware t (msec)"""

        if not self.camera: return
        self.camera.exposure_t(t)

    def get_exposure_time(self):
        """CAMERA FUNCTIONS
         Get exposure time in the camera hardware
         Returns t (msec), or -1 if the camera is inactive"""

        if not self.camera: return -1
        exposure = self.camera.get_exposure_t()
        return exposure
        
    def set_auto_exposure_time(self, state = True):
        """CAMERA FUNCTIONS
         Enable / Disable camera auto_exposure with the value of 'state'
        It will be continueously updating based on the current image """

        if not self.camera: return
        self.camera.auto_exposure_t(state)

    ########################################################################
    # MOTION CONTROL FUNCTIONS
    ########################################################################

    def zhome(self):
        """MOTION CONTROL FUNCTIONS
        Home the z-axis (i.e. focus)"""
        if not self.motion: return
        self.motion.zhome()

    def xyhome(self):
        """MOTION CONTROL FUNCTIONS
        Home the xy-axes (i.e. stage). Note: z-axis and turret will always home first"""
        if not self.motion: return
        self.motion.xyhome()

    def xycenter(self):
        """MOTION CONTROL FUNCTIONS
        Move Stage to the center."""

        if not self.motion: return
        self.motion.xycenter()

    def thome(self):
        """MOTION CONTROL FUNCTIONS
        Home the Turret"""

        if not self.motion: return
        self.motion.thome()

    def get_target_position(self, axis):
        """MOTION CONTROL FUNCTIONS
        Get the value of the target position of the axis relative to home
        Returns position (um), or 0 if the motion board is inactive
        values of axis 'X', 'Y', 'Z', and 'T' """

        if not self.motion.driver: return 0
        target_position = self.motion.target_pos(axis)
        return target_position
        
    def get_current_position(self, axis):
        """MOTION CONTROL FUNCTIONS
        Get the value of the current position of the axis relative to home
        Returns position (um), or 0 if the motion board is inactive
        values of axis 'X', 'Y', 'Z', and 'T' """

        if not self.motion.driver: return 0
        target_position = self.motion.current_pos(axis)
        return target_position
        
    def move_absolute_position(self, axis, pos):
        """MOTION CONTROL FUNCTIONS
         Move to absolute position (in um) of axis"""

        if not self.motion: return
        self.motion.move_abs_pos(axis, pos)

    def move_relative_position(self, axis, um):
        """MOTION CONTROL FUNCTIONS
         Move to relative distance (in um) of axis"""

        if not self.motion: return
        self.motion.move_rel_pos(axis, um)

    def get_home_status(self, axis):
        """MOTION CONTROL FUNCTIONS
         Return True if axis is in home position"""
 
        if not self.motion: return False
        status = self.motion.home_status(axis)
        return status

    def get_target_status(self, axis):
        """MOTION CONTROL FUNCTIONS
         Return True if axis is at target position"""

        if not self.motion: return False
        status = self.motion.target_status(axis)
        return status
        
    # Get all reference status register bits as 32 character string (32-> 0)
    def get_reference_status(self, axis):
        """MOTION CONTROL FUNCTIONS
         Get all reference status register bits as 32 character string (32-> 0) """

        if not self.motion: return False
        status = self.motion.reference_status(axis)
        return status

    def get_overshoot(self):
        """MOTION CONTROL FUNCTIONS
        Is z-axis (focus) currently in overshoot mode?"""

        if not self.motion: return False
        return self.motion.overshoot

    ########################################################################
    # INTEGRATED SCOPE FUNCTIONS
    ########################################################################

    ## IN PROGRESS - DO NOT USE ##
    def autofocus(self, dt=0):
        """INTEGRATED SCOPE FUNCTIONS
        begin autofocus functionality (not yet ported from LVP)"""
        
        # These should be passed but its not working
        AF_range = 15.0
        AF_min =   0.5
        AF_max =   6.0
        
        print('[SCOPE API ] Lumascope.autofocus()')
        self.is_complete = False
        self.is_autofocus = True

        if self.camera.active == False:
            print('[SCOPE API ] Error: VerticalControl.autofocus()')
            self.is_autofocus = False
            return

        center = self.get_current_position('Z')

        self.z_min = max(0, center-AF_range)      # starting minimum z-height for autofocus
        self.z_max = center+AF_range              # starting maximum z-height for autofocus
        self.resolution = AF_max                  # starting step size for autofocus
        self.exposure = self.get_exposure_time()  # camera exposure to determine 'wait' time

        self.positions = []       # List of positions to step through
        self.focus_measures = []  # Measure focus score at each position
        self.last_focus = 0       # Last / Previous focus score
        self.last = False         # Are we on the last scan for autofocus?

        # Start the autofocus process at z-minimum
        self.move_absolute_position('Z', self.z_min)

        while self.is_autofocus:
            time.sleep(0.01)

            # If the z-height has reached its target
            if self.get_target_status('Z') and not self.get_overshoot():
                
                # Wait two exposure lengths
                time.sleep(2*self.exposure/1000+0.2) # msec into sec

                # observe the image 
                image = self.get_image()
                rows, cols = image.shape

                # Use center quarter of image for focusing
                image = image[int(rows/4):int(3*rows/4),int(cols/4):int(3*cols/4)]

                # calculate the position and focus measure
                try:
                    current = self.get_current_position('Z')
                    focus = self.focus_function(image)
                    next_target = self.get_target_position('Z') + self.resolution
                except:
                    print('[SCOPE API ] Error talking to motion controller.')
                    raise

                # append to positions and focus measures
                self.positions.append(current)
                self.focus_measures.append(focus)

                # if (focus < self.last_focus) or (next_target > self.z_max):
                if next_target > self.z_max:

                    # Calculate new step size for resolution
                    prev_resolution = self.resolution
                    self.resolution = prev_resolution / 3 # SELECT DESIRED RESOLUTION FRACTION

                    if self.resolution < AF_min:
                        self.resolution = AF_min

                    # As long as the step size is larger than or equal to the minimum and not the last pass
                    if self.resolution >= AF_min and not self.last:

                        # compute best focus
                        focus = self.focus_best(self.positions, self.focus_measures)

                        # assign new z_min, z_max, resolution, and sweep
                        self.z_min = focus-prev_resolution 
                        self.z_max = focus+prev_resolution 

                        # reset positions and focus measures
                        self.positions = []
                        self.focus_measures = []

                        # go to new z_min
                        self.move_absolute_position('Z', self.z_min)

                        if self.resolution == AF_min:
                            self.last = True
                            print('self.last = True')
                    else: # self.resolution >= AF_min and not self.last
                        # compute best focus
                        focus = self.focus_best(self.positions, self.focus_measures)
                        
                        # go to best focus
                        self.move_absolute_position('Z', focus) # move to absolute target
                        
                        # end autofocus sequence
                        self.is_autofocus = False
                        
                        self.is_complete = True

                else:
                    # move to next position
                    self.move_relative_position('Z', self.resolution)

                # update last focus
                self.last_focus = focus

    # Algorithms for estimating the quality of the focus
    def focus_function(self, image, algorithm = 'vollath4'):
        """INTEGRATED SCOPE FUNCTIONS
        assess focus value at specific position for autofocus function
        (not yet ported from LVP)"""

        print('[LVP Main  ] VerticalControl.focus_function()')
        w = image.shape[0]
        h = image.shape[1]

        # Journal of Microscopy, Vol. 188, Pt 3, December 1997, pp. 264–272
        if algorithm == 'vollath4': # pg 266
            image = np.double(image)
            sum_one = np.sum(np.multiply(image[:w-1,:h], image[1:w,:h])) # g(i, j).g(i+1, j)
            sum_two = np.sum(np.multiply(image[:w-2,:h], image[2:w,:h])) # g(i, j).g(i+2, j)
            print('[LVP Main  ] Focus Score Vollath: ' + str(sum_one - sum_two))
            return sum_one - sum_two

        elif algorithm == 'skew':
            hist = np.histogram(image, bins=256,range=(0,256))
            hist = np.asarray(hist[0], dtype='int')
            max_index = hist.argmax()

            edges = np.histogram_bin_edges(image, bins=1)
            white_edge = edges[1]

            skew = white_edge-max_index
            print('[LVP Main  ] Focus Score Skew: ' + str(skew))
            return skew

        elif algorithm == 'pixel_variation':
            sum = np.sum(image)
            ssq = np.sum(np.square(image))
            var = ssq*w*h-sum**2
            print('[LVP Main  ] Focus Score Pixel Variation: ' + str(var))
            return var
        
            '''
        elif algorithm == 'convolve2D':
            # Bueno-Ibarra et al. Optical Engineering 44(6), 063601 (June 2005)
            kernel = np.array([ [0, -1, 0],
                                [-1, 4,-1],
                                [0, -1, 0]], dtype='float') / 6
            n = 9
            a = 1
            kernel = np.zeros([n,n])
            for i in range(n):
                for j in range(n):
                    r2 = ((i-(n-1)/2)**2 + (j-(n-1)/2)**2)/a**2
                    kernel[i,j] = 2*(1-r2)*np.exp(-0.5*r2)/np.sqrt(3*a)
            print('[LVP Main  ] kernel\t' + str(kernel))
            convolve = signal.convolve2d(image, kernel, mode='valid')
            sum = np.sum(convolve)
            print('[LVP Main  ] sum\t' + str(sum))
            return sum
            '''
        else:
            return 0

    def focus_best(self, positions, values, algorithm='direct'):
        """INTEGRATED SCOPE FUNCTIONS
        select best focus position for autofocus function
        (not yet ported from LVP)"""

        print('[LVP Main  ] VerticalControl.focus_best()')
        if algorithm == 'direct':
            max_value = max(values)
            max_index = values.index(max_value)
            return positions[max_index]

        elif algorithm == 'mov_avg':
            avg_values = np.convolve(values, [.5, 1, 0.5], 'same')
            max_index = avg_values.argmax()
            return positions[max_index]

        else:
            return positions[0]

