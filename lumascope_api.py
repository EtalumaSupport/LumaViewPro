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
Gerard Decker, The Earthineering Company

MODIFIED:
April 21, 2023
'''

# Import Lumascope Hardware files
from motorboard import MotorBoard
from ledboard import LEDBoard
from pyloncamera import PylonCamera

# Import additional libraries
from lvp_logger import logger
from threading import Timer
import time
import cv2
import numpy as np

class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)

class Lumascope():

    def __init__(self):
        """Initialize Microscope"""

        # LED Control Board
        try:
            self.led = LEDBoard()

        except:
            logger.exception('[SCOPE API ] LED Board Not Initialized')
        
        # Motion Control Board
        try:
            self.motion = MotorBoard()
        except:
            logger.exception('[SCOPE API ] Motion Board Not Initialized')

        # Camera
        try:
            self.camera = PylonCamera()
        except:
            logger.exception('[SCOPE API ] Camera Board Not Initialized')

        # Initialize scope status booleans
        self.is_homing = False           # Is the microscope currently moving to home position

        self.is_capturing = False        # Is the microscope currently attempting image capture (with illumination)
        self.capture_return = False      # Will be image if capture is ready to pull, else False

        self.is_focusing = False         # Is the microscope currently attempting autofocus
        self.autofocus_return = False    # Will be z-position if focus is ready to pull, else False

        # self.is_stepping = False         # Is the microscope currently attempting to capture a step
        # self.step_capture_return = False # Will be image at step settings if ready to pull, else False

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
            return self.camera.array
        else:
            return False

    def save_image(self, array, save_folder = './capture', file_root = 'img_', append = 'ms', color = 'BF'):
        """CAMERA FUNCTIONS
        save image (as array) to file
        """
        
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
        path = save_folder + '/' + filename

        try:
            cv2.imwrite(path, img.astype(np.uint8))
            logger.info(f'[SCOPE API ] Saving Image to {path}')
        except:
            logger.exception("[SCOPE API ] Error: Unable to save. Perhaps save folder does not exist?")

    def save_live_image(self, save_folder = './capture', file_root = 'img_', append = 'ms', color = 'BF'):
        """CAMERA FUNCTIONS
        Grab the current live image and save to file
        """
        array = self.get_image()
        if array is False:
            return 
        self.save_image(array, save_folder, file_root, append, color)
 
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

        if not self.camera: return 0
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
        self.is_homing = True
        self.motion.xyhome()

        self.xyhome_timer = RepeatTimer(0.01, self.xyhome_iterate)
        self.xyhome_timer.start()

    def xyhome_iterate(self):
        if not self.is_moving():
            self.is_homing = False
            self.xyhome_timer.cancel()

    def xycenter(self):
        """MOTION CONTROL FUNCTIONS
        Move Stage to the center."""

        if not self.motion: return
        self.motion.xycenter()

    def thome(self):
        """MOTION CONTROL FUNCTIONS
        Home the Turret"""

        if not self.motion:
            return
        self.motion.thome()

    def tmove(self, degrees):
        """MOTION CONTROL FUNCTIONS
        Move turret to position in degrees"""

        if not self.motion: return
        # MUST home move objective home first to prevent crash
        self.zhome()
        #self.xyhome()
        #self.xycenter()

        self.is_turreting = True
        self.move_absolute_position('T', degrees)
        self.tmove_timer = RepeatTimer(0.01, self.tmove_iterate, args=(degrees,))
        #self.tmove_timer = RepeatTimer(2, self.tmove_iterate, args=(degrees,))
        self.tmove_timer.start()

    def tmove_iterate(self, degrees):
        if not self.is_moving():
            self.is_turreting = False
            self.tmove_timer.cancel()

    def get_target_position(self, axis):
        """MOTION CONTROL FUNCTIONS
        Get the value of the target position of the axis relative to home
        Returns position (um), or 0 if the motion board is inactive
        values of axis 'X', 'Y', 'Z', and 'T' """

        if not self.motion.driver: return 0
        try:
            target_position = self.motion.target_pos(axis)
            return target_position
        except:
            raise
        
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
         Return True if axis is in home position or motionboard is """
 
        if not self.motion: return True
        status = self.motion.home_status(axis)
        return status

    def get_target_status(self, axis):
        """MOTION CONTROL FUNCTIONS
         Return True if axis is at target position"""

        if not self.motion: return True
        status = self.motion.target_status(axis)
        return status
        
    # Get all reference status register bits as 32 character string (32-> 0)
    def get_reference_status(self, axis):
        """MOTION CONTROL FUNCTIONS
         Get all reference status register bits as 32 character string (32-> 0) """

        if not self.motion: return
        status = self.motion.reference_status(axis)
        return status

    def get_overshoot(self):
        """MOTION CONTROL FUNCTIONS
        Is z-axis (focus) currently in overshoot mode?"""

        if not self.motion: return False
        return self.motion.overshoot

    def is_moving(self):
        # If not communicating with motor board
        if not self.motion.driver: return False

        # Check each axis
        x_status = self.get_target_status('X')
        y_status = self.get_target_status('Y')
        z_status = self.get_target_status('Z')

        if x_status and y_status and z_status and not self.get_overshoot():
            return False
        else:
            return True

    '''
    ########################################################################
    # COORDINATES
    ########################################################################

    # INCOMPLETE
    def plate_to_stage(self, px, py):
        # plate coordinates in mm from top left
        # stage coordinates in um from bottom right

        # Get labware dimensions
        x_max = 127.76 # in mm
        y_max = 85.48 # in mm

        # Convert coordinates
        sx = x_max - 3.88 - px
        sy = y_max - 2.74 - py

        # Convert from mm to um
        sx = sx*1000
        sy = sy*1000

        # return
        return sx, sy
    
    # INCOMPLETE
    def stage_to_plate(self, sx, sy):
        # stage coordinates in um from bottom right
        # plate coordinates in mm from top left

        # Get labware dimensions
        x_max = 127.76 # in mm
        y_max = 85.48 # in mm

        # Convert coordinates
        px = x_max - (3880 + sx)/1000
        py = y_max - (2740 + sy)/1000
 
        return px, py
    '''
    
    ########################################################################
    # INTEGRATED SCOPE FUNCTIONS
    ########################################################################

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ILLUMINATE AND CAPTURE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def capture(self):
        """INTEGRATED SCOPE FUNCTIONS
        Capture image with illumination"""       

        if not self.led: return
        if not self.camera: return

        # Set capture states
        self.is_capturing = True
        self.capture_return = False

        # Wait time for exposure and rolling shutter
        wait_time = 2*self.get_exposure_time()/1000+0.2

        # Start thread to wait until capture is complete
        capture_timer = Timer(wait_time, self.capture_complete)
        capture_timer.start()

    def capture_complete(self):
        self.capture_return = self.get_image() # Grab image
        self.is_capturing = False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # AUTOFOCUS Functionality
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Functional, but not integrated with LVP, just for scripting at the moment.
    
    def autofocus(self, AF_min, AF_max, AF_range):
        """INTEGRATED SCOPE FUNCTIONS
        begin autofocus functionality"""

        # Check all hardware required
        if not self.led: return
        if not self.motion: return
        if not self.camera: return

        # Check if hardware is actively responding
        if self.led.driver is False: return
        if self.motion.driver is False: return
        if self.camera.active is False: return
        
        # Set autofocus states
        self.is_focusing = True          # Is the microscope currently attempting autofocus
        self.autofocus_return = False    # Will be z-position if focus is ready to pull, else False

        # Determine center of AF
        center = self.get_current_position('Z')

        self.z_min = max(0, center-AF_range)      # starting minimum z-height for autofocus
        self.z_max = center+AF_range              # starting maximum z-height for autofocus
        self.resolution = AF_max                  # starting step size for autofocus

        self.AF_positions = []       # List of positions to step through
        self.focus_measures = []     # Measure focus score at each position
        self.last_focus_score = 0    # Last / Previous focus score
        self.last_focus_pass = False # Are we on the last scan for autofocus?

        # Start the autofocus process at z-minimum
        self.move_absolute_position('Z', self.z_min)

        # Start thread to wait until autofocus is complete
        self.autofocus_timer = RepeatTimer(0.01, self.autofocus_iterate, args=(AF_min,))
        self.autofocus_timer.start()

    def autofocus_iterate(self, AF_min):
        """INTEGRATED SCOPE FUNCTIONS
        iterate autofocus functionality"""

        # Ignore steps until conditions are met
        if self.is_moving(): return   # needs to be in position
        if self.is_capturing: return  # needs to have completed capture with illumination

        # Is there a previous capture result to pull?
        if self.capture_return is False:
            # No -> start a capture event
            self.capture()
            return
            
        else:
            # Yes -> pull the capture result and clear
            image = self.capture_return
            self.capture_return = False
            
        if image is False:
            # Stop thread image can't be acquired
            self.autofocus_timer.cancel()
            return
   
        # observe the image
        rows, cols = image.shape

        # Use center quarter of image for focusing
        image = image[int(rows/4):int(3*rows/4),int(cols/4):int(3*cols/4)]

        # calculate the position and focus measure
        try:
            current = self.get_current_position('Z')
            focus = self.focus_function(image)
            next_target = self.get_target_position('Z') + self.resolution
        except:
            logger.exception('[SCOPE API ] Error talking to motion controller.')

        # append to positions and focus measures
        self.AF_positions.append(current)
        self.focus_measures.append(focus)

        if next_target <= self.z_max:
            self.move_relative_position('Z', self.resolution)
            return

        # Adjust future steps if next_target went out of bounds
        # Calculate new step size for resolution
        prev_resolution = self.resolution
        self.resolution = prev_resolution / 3 # SELECT DESIRED RESOLUTION FRACTION

        if self.resolution < AF_min:
            self.resolution = AF_min
            self.last_focus_pass = True

        # compute best focus
        focus = self.focus_best(self.AF_positions, self.focus_measures)

        if not self.last_focus_pass:
            # assign new z_min, z_max, resolution, and sweep
            self.z_min = focus-prev_resolution 
            self.z_max = focus+prev_resolution 

            # reset positions and focus measures
            self.AF_positions = []
            self.focus_measures = []

            # go to new z_min
            self.move_absolute_position('Z', self.z_min)
                
        else:
            # go to best focus
            self.move_absolute_position('Z', focus) # move to absolute target
            
            # end autofocus sequence
            self.autofocus_return = focus
            self.is_focusing = False

            # Stop thread image when autofocus is complete
            self.autofocus_timer.cancel()

    # Algorithms for estimating the quality of the focus
    def focus_function(self, image, algorithm = 'vollath4'):
        """INTEGRATED SCOPE FUNCTIONS
        assess focus value at specific position for autofocus function"""

        logger.info('[SCOPE API ] Lumascope.focus_function()')

        w = image.shape[0]
        h = image.shape[1]

        # Journal of Microscopy, Vol. 188, Pt 3, December 1997, pp. 264–272
        if algorithm == 'vollath4': # pg 266
            image = np.double(image)
            sum_one = np.sum(np.multiply(image[:w-1,:h], image[1:w,:h])) # g(i, j).g(i+1, j)
            sum_two = np.sum(np.multiply(image[:w-2,:h], image[2:w,:h])) # g(i, j).g(i+2, j)
            logger.info('[SCOPE API ] Focus Score Vollath: ' + str(sum_one - sum_two))
            return sum_one - sum_two

        elif algorithm == 'skew':
            hist = np.histogram(image, bins=256,range=(0,256))
            hist = np.asarray(hist[0], dtype='int')
            max_index = hist.argmax()

            edges = np.histogram_bin_edges(image, bins=1)
            white_edge = edges[1]

            skew = white_edge-max_index
            logger.info('[SCOPE API ] Focus Score Skew: ' + str(skew))
            return skew

        elif algorithm == 'pixel_variation':
            sum = np.sum(image)
            ssq = np.sum(np.square(image))
            var = ssq*w*h-sum**2
            logger.info('[SCOPE API ] Focus Score Pixel Variation: ' + str(var))
            return var
        
        else:
            return 0
    
    def focus_best(self, positions, values, algorithm='direct'):
        """INTEGRATED SCOPE FUNCTIONS
        select best focus position for autofocus function"""

        logger.info('[SCOPE API ] Lumascope.focus_best()')
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


