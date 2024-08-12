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
June 1, 2023
'''

# Import Lumascope Hardware files
from motorboard import MotorBoard
from ledboard import LEDBoard
from pyloncamera import PylonCamera

# Import additional libraries
from lvp_logger import logger
import modules.common_utils as common_utils
import modules.coord_transformations as coord_transformations
import modules.objectives_loader as objectives_loader
import datetime
import pathlib
import time
import threading
import os
import contextlib
import cv2
import numpy as np

import image_utils


class Lumascope():

    def __init__(self):
        """Initialize Microscope"""
        self._coordinate_transformer = coord_transformations.CoordinateTransformer()
        self._objectives_loader = objectives_loader.ObjectiveLoader()

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
        self.image_buffer = None
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
        self.last_focus_score = None

        # self.is_stepping = False         # Is the microscope currently attempting to capture a step
        # self.step_capture_return = False # Will be image at step settings if ready to pull, else False

        self._labware = None             # The labware currently installed
        self._objective = None           # The objective currently installed
        self._stage_offset = None        # The stage offset for the microscope
        if self.camera:
            self._binning_size = self.camera.get_binning_size()
        else:
            self._binning_size = 1

        self._scale_bar = {
            'enabled': False
        }


    ########################################################################
    # SCOPE CONFIGURATION FUNCTIONS
    ########################################################################
    def set_labware(self, labware):
        self._labware = labware

    def set_objective(self, objective_id):
        self._objective = self._objectives_loader.get_objective_info(objective_id=objective_id)

    def set_scale_bar(self, enabled: bool):
        self._scale_bar['enabled'] = enabled

    def set_stage_offset(self, stage_offset):
        self._stage_offset = stage_offset

    def set_binning_size(self, size):
        self._binning_size = size

        if self.camera:
            self.camera.set_binning_size(size=size)

    def get_binning_size(self) -> int:
        if not self.camera:
            return 1

        return self.camera.get_binning_size()
    

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

    def get_led_ma(self, color: str):
        """ LED BOARD FUNCTIONS
        Get LED illumination (mA)"""
        if not self.led: return -1

        return self.led.get_led_ma(color=color)
    

    def get_led_state(self, color: str):
        """ LED BOARD FUNCTIONS
        Get a dictionary containing the LED state and illumination (mA)"""
        if not self.led: return -1

        return self.led.get_led_state(color=color)
    

    def get_led_states(self):
        """ LED BOARD FUNCTIONS
        Get a dictionary of dictionaries containing the LED states and illumination (mA)"""
        if not self.led: return -1

        return self.led.get_led_states()
    

    def led_on(self, channel, mA):
        """ LED BOARD FUNCTIONS
        Turn on LED at channel number at mA power """
        if not self.led: return

        if type(channel) == str:
            channel = self.color2ch(color=channel)

        self.led.led_on(channel, mA)

    def led_off(self, channel):
        """ LED BOARD FUNCTIONS
        Turn off LED at channel number """
        if not self.led: return

        if type(channel) == str:
            channel = self.color2ch(color=channel)

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

    def get_image(
        self,
        force_to_8bit: bool = True,
        earliest_image_ts: datetime.datetime | None = None,
        timeout: datetime.timedelta = datetime.timedelta(seconds=0),
        all_ones_check: bool = False,
    ):
        """ CAMERA FUNCTIONS
        Grab and return image from camera
        # If use_host_buffer set to true, it will return the results already stored in the
        # host array. It will not wait for the next capture.
        """
        start_time = datetime.datetime.now()
        stop_time = start_time + timeout
        
        while True:
            all_ones_failed = False
            grab_status, grab_image_ts = self.camera.grab()

            if grab_status == True:
                self.image_buffer = self.camera.array.copy()

                if all_ones_check == True:

                    if np.all(self.image_buffer == np.iinfo(self.image_buffer.dtype).max):
                        all_ones_failed = True
                        logger.warn(f"[SCOPE API ] get_image all_ones_check failed")

                if all_ones_failed == False:
                    if earliest_image_ts is None:
                        break
                    
                    if grab_image_ts >= earliest_image_ts:
                        break

                    logger.warn(f"[SCOPE API ] get_image earliest_image_time {earliest_image_ts} not met -> Image TS: {grab_image_ts}")

            # In case of timeout, if we hit the timeout because of the all ones check, then just let it continue and return the all ones image
            if datetime.datetime.now() > stop_time: 
                if all_ones_failed == False:
                    logger.error(f"[SCOPE API ] get_image timeout stop_time ({stop_time}) exceeded")
                    return False
                else:
                    logger.warn(f"[SCOPE API ] get_image timeout stop_time ({stop_time}) exceeded with all_ones_failed")
                    break
            
            if grab_status == False:
                logger.error(f"[SCOPE API ] get_image grab failed, retrying")

            time.sleep(0.05)

        use_scale_bar = self._scale_bar['enabled']
        if self._objective is None:
            use_scale_bar = False

        if use_scale_bar:
            self.image_buffer = image_utils.add_scale_bar(
                image=self.image_buffer,
                objective=self._objective,
                binning_size=self._binning_size,
            )

        if force_to_8bit and self.image_buffer.dtype != np.uint8:
            self.image_buffer = image_utils.convert_12bit_to_8bit(self.image_buffer)

        return self.image_buffer

        
    def get_next_save_path(self, path):
        """ GETS THE NEXT SAVE PATH GIVEN AN EXISTING SAVE PATH

            :param path of the format './{save_folder}/{well_label}_{color}_{file_id}.tiff'   
            :returns the next save path './{save_folder}/{well_label}_{color}_{file_id + 1}.tiff'   

        """

        NUM_SEQ_DIGITS = 6
        # Handle both .tiff and .ome.tiff by detecting multiple extensions if present
        # pathlib doesn't seem to handle multiple extensions natively
        path2 = pathlib.Path(path)
        extension = ''.join(path2.suffixes)
        stem = path2.name[:len(path2.name)-len(extension)]
        seq_separator_idx = stem.rfind('_')
        stem_base = stem[:seq_separator_idx]
        seq_num_str = stem[seq_separator_idx+1:]
        seq_num = int(seq_num_str)

        next_seq_num = seq_num + 1
        next_seq_num_str = f"{next_seq_num:0>{NUM_SEQ_DIGITS}}"
        
        new_path = path2.parent / f"{stem_base}_{next_seq_num_str}{extension}"
        return str(new_path)
    

    def generate_image_save_path(self, save_folder, file_root, append, tail_id_mode, output_format):
        if type(save_folder) == str:
            save_folder = pathlib.Path(save_folder)

        if file_root is None:
            file_root = ""

        if output_format == 'OME-TIFF':
            file_extension = ".ome.tiff"
        else:
            file_extension = ".tiff"

        # generate filename and save path string
        if tail_id_mode == "increment":
            initial_id = '_000001'
            filename =  f"{file_root}{append}{initial_id}{file_extension}"
            path = save_folder / filename

            # Obtain next save path if current directory already exists
            while os.path.exists(path):
                path = self.get_next_save_path(path)

        elif tail_id_mode == None:
            filename =  f"{file_root}{append}{file_extension}"
            path = save_folder / filename
        
        else:
            raise Exception(f"tail_id_mode: {tail_id_mode} not implemented")
        
        return path


    def generate_image_metadata(self, color) -> dict:
        def _validate():
            if self._objective is None:
                raise Exception(f"[SCOPE API ] Objective not set")
            
            if 'focal_length' not in self._objective:
                raise Exception(f"[SCOPE API ] Objective focal length not provided")

            if self._labware is None:
                raise Exception(f"[SCOPE API ] Labware not set")
            
            if self._stage_offset is None:
                raise Exception(f"[SCOPE API ] Stage offset not set")
            
        _validate()

        px, py = self._coordinate_transformer.stage_to_plate(
            labware=self._labware,
            stage_offset=self._stage_offset,
            sx=self.get_current_position(axis='X'),
            sy=self.get_current_position(axis='Y')
        )
        z = self.get_current_position(axis='Z')

        px = round(px, common_utils.max_decimal_precision('x'))
        py = round(py, common_utils.max_decimal_precision('y'))
        z  = round(z,  common_utils.max_decimal_precision('z'))

        pixel_size_um = round(
            common_utils.get_pixel_size(
                focal_length=self._objective['focal_length'],
                binning_size=self._binning_size,
            ),
            common_utils.max_decimal_precision('pixel_size'),
        )
        
        metadata = {
            'channel': color,
            'focal_length': self._objective['focal_length'],
            'plate_pos_mm': {'x': px, 'y': py},
            'z_pos_um': z,
            'exposure_time_ms': round(self.get_exposure_time(), common_utils.max_decimal_precision('exposure')),
            'gain_db': round(self.get_gain(), common_utils.max_decimal_precision('gain')),
            'illumination_ma': round(self.get_led_ma(color=color), common_utils.max_decimal_precision('illumination')),
            'binning_size': self._binning_size,
            'pixel_size_um': pixel_size_um,
        }

        return metadata

    def prepare_image_for_saving(
        self,
        array: np.ndarray,
        save_folder: str,
        file_root: str,
        append: str,
        color: str,
        tail_id_mode: str,
        output_format: str,
        true_color: str,
    ):
        metadata = self.generate_image_metadata(color=true_color)

        if array.dtype == np.uint16:
            array = image_utils.convert_12bit_to_16bit(array)

        img = image_utils.add_false_color(array=array, color=color)
        img = np.flip(img, 0)

        path = self.generate_image_save_path(
            save_folder=save_folder,
            file_root=file_root,
            append=append,
            tail_id_mode=tail_id_mode,
            output_format=output_format
        )

        metadata['file_loc'] = path

        return {
            'image': img,
            'metadata': metadata,
        }


    def save_image(
        self,
        array,
        save_folder = './capture',
        file_root = 'img_',
        append = 'ms',
        color = 'BF',
        tail_id_mode = "increment",
        output_format: str = "TIFF",
        true_color: str = 'BF',
    ):
        """CAMERA FUNCTIONS
        save image (as array) to file
        """

        image_data = self.prepare_image_for_saving(
            array=array,
            save_folder=save_folder,
            file_root=file_root,
            append=append,
            color=color,
            tail_id_mode=tail_id_mode,
            output_format=output_format,
            true_color=true_color,
        )

        image = image_data['image']
        metadata = image_data['metadata']
        file_loc = metadata['file_loc']

        try:
            if output_format == 'OME-TIFF':
                image_utils.write_ome_tiff(
                    data=image,
                    file_loc=file_loc,
                    metadata=metadata,
                )
            else:
                cv2.imwrite(str(file_loc), image.astype(array.dtype))

            logger.info(f'[SCOPE API ] Saving Image to {file_loc}')
        except:
            logger.exception("[SCOPE API ] Error: Unable to save. Perhaps save folder does not exist?")

        return file_loc
    

    def save_live_image(
            self,
            save_folder = './capture',
            file_root = 'img_',
            append = 'ms',
            color = 'BF',
            tail_id_mode = "increment",
            force_to_8bit: bool = True,
            output_format: str = "TIFF",
            true_color: str = 'BF',
            earliest_image_ts: datetime.datetime | None = None,
            timeout: datetime.timedelta = datetime.timedelta(seconds=0),
            all_ones_check: bool = False,
        ):

        """CAMERA FUNCTIONS
        Grab the current live image and save to file
        """
        array = self.get_image(
            force_to_8bit=force_to_8bit,
            earliest_image_ts=earliest_image_ts,
            timeout=timeout,
            all_ones_check=all_ones_check,
        )
        if array is False:
            return 
        return self.save_image(array, save_folder, file_root, append, color, tail_id_mode, output_format=output_format, true_color=true_color)
 

    def get_max_width(self):
        """CAMERA FUNCTIONS
        Grab the max pixel width of camera
        """
        if (not self.camera) or (not self.camera.active): return 0
        return self.camera.active.Width.Max

    def get_max_height(self):
        """CAMERA FUNCTIONS
        Grab the max pixel height of camera
        """
        if (not self.camera) or (not self.camera.active): return 0
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
        Set frame size (pixel width by pixel height
        of camera to w by h"""

        if not self.camera: return
        self.camera.set_frame_size(w, h)

    def get_frame_size(self):
        """CAMERA FUNCTIONS
        Get frame size (pixel width by pixel height
        of camera to w by h"""

        if not self.camera: return
        return self.camera.get_frame_size()


    def get_gain(self):
        """CAMERA FUNCTIONS
        Get camera gain"""

        if not self.camera: return -1
        return self.camera.get_gain()
    
    def set_gain(self, gain):
        """CAMERA FUNCTIONS
        Set camera gain"""

        if not self.camera: return
        self.camera.gain(gain)

    def set_auto_gain(self, state: bool, settings: dict):
        """CAMERA FUNCTIONS
        Enable / Disable camera auto_gain with the value of 'state'
        It will be continueously updating based on the current image """

        if not self.camera: return
        self.camera.auto_gain(
            state,
            target_brightness=settings['target_brightness'],
            min_gain=settings['min_gain'],
            max_gain=settings['max_gain'],
        )

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

    
    def camera_is_connected(self) -> bool:
        if not self.camera:
            return False
        
        if not self.camera.active:
            return False
    
        return True

    ########################################################################
    # MOTION CONTROL FUNCTIONS
    ########################################################################
    @contextlib.contextmanager
    def reference_position_logger(self):
        before = self.get_limit_switch_status_all_axes()
        logger.info(f"Limit switch status before homing: {before}")
        yield
        after = self.get_limit_switch_status_all_axes()
        logger.info(f"Limit switch status after homing: {after}")


    def zhome(self):
        """MOTION CONTROL FUNCTIONS
        Home the z-axis (i.e. focus)"""
        #if not self.motion: return
        with self.reference_position_logger():
            self.motion.zhome()

    def xyhome(self):
        """MOTION CONTROL FUNCTIONS
        Home the xy-axes (i.e. stage). Note: z-axis and turret will always home first"""
        #if not self.motion: return
        with self.reference_position_logger():
            self.is_homing = True
            self.motion.xyhome()

        return

        #while self.is_moving():
        #    time.sleep(0.01)
        #self.is_homing = False
    
    def has_xyhomed(self):
        """MOTION CONTROL FUNCTIONS
        Indicate if the xy-axes (i.e. stage) has been homed since startup"""
        return self.motion.has_xyhomed()

    def xyhome_iterate(self):
        if not self.is_moving():
            self.is_homing = False
            self.xyhome_timer.cancel()

    def xycenter(self):
        """MOTION CONTROL FUNCTIONS
        Move Stage to the center."""

        #if not self.motion: return
        self.motion.xycenter()


    @contextlib.contextmanager
    def safe_turret_mover(self):
        # Save off current Z position before moving Z to 0
        logger.info('[SCOPE API ] Moving Z to 0')
        initial_z = self.get_current_position(axis='Z')
        self.move_absolute_position('Z', pos=0, wait_until_complete=True)
        self.is_turreting = True
        yield
        self.is_turreting = False
        # Restore Z position
        logger.info(f'[SCOPE API ] Restoring Z to {initial_z}')
        self.move_absolute_position('Z', pos=initial_z, wait_until_complete=True)


    def thome(self):
        """MOTION CONTROL FUNCTIONS
        Home the Turret"""

        #if not self.motion:
        #    return

        # Move turret
        with self.reference_position_logger():
            with self.safe_turret_mover():
                self.motion.thome()


    def tmove(self, degrees):
        """MOTION CONTROL FUNCTIONS
        Move turret to position in degrees"""
        # MUST home move objective home first to prevent crash
        #self.zhome()
        #self.move_absolute_position('Z', self.z_min)

        with self.safe_turret_mover():
            logger.info(f'[SCOPE API ] Moving T to {degrees}')
            self.move_absolute_position('T', degrees, wait_until_complete=True)


    def get_target_position(self, axis=None):
        """MOTION CONTROL FUNCTIONS
        Get the value of the target position of the axis relative to home
        Returns position (um), or 0 if the motion board is inactive
        values of axis 'X', 'Y', 'Z', and 'T' """

        if not self.motion.driver: return 0

        if axis is None:
            positions = {}
            for ax in ('X', 'Y', 'Z', 'T'):
                positions[ax] = self.motion.target_pos(axis=ax)
            return positions
        
        position = self.motion.target_pos(axis)
        return position
        
    def get_current_position(self, axis=None):
        """MOTION CONTROL FUNCTIONS
        Get the value of the current position of the axis relative to home
        Returns position (um), or 0 if the motion board is inactive
        values of axis 'X', 'Y', 'Z', and 'T' """

        if not self.motion.driver: return 0

        if axis is None:
            positions = {}
            for ax in ('X', 'Y', 'Z', 'T'):
                positions[ax] = self.motion.current_pos(axis=ax)
            return positions
        
        position = self.motion.current_pos(axis)
        return position


    def move_absolute_position(self, axis, pos, wait_until_complete=False, overshoot_enabled: bool = True):
        """MOTION CONTROL FUNCTIONS
         Move to absolute position (in um) of axis"""

        #if not self.motion: return
        self.motion.move_abs_pos(axis, pos, overshoot_enabled=overshoot_enabled)
        
        if wait_until_complete is True:
            self.wait_until_finished_moving()


    def move_relative_position(self, axis, um, wait_until_complete=False, overshoot_enabled: bool = True):
        """MOTION CONTROL FUNCTIONS
         Move to relative distance (in um) of axis"""

        #if not self.motion: return
        self.motion.move_rel_pos(axis, um, overshoot_enabled=overshoot_enabled)

        if wait_until_complete is True:
            self.wait_until_finished_moving()


    def get_home_status(self, axis):
        """MOTION CONTROL FUNCTIONS
         Return True if axis is in home position or motionboard is """
 
        #if not self.motion: return True
        status = self.motion.home_status(axis)
        return status

    def get_target_status(self, axis):
        """MOTION CONTROL FUNCTIONS
         Return True if axis is at target position"""

        #if not self.motion: return True

        # Handle case where we want to know if turret has reached its target, but there is no turret
        if (axis == 'T') and (self.motion.has_turret == False):
            return True
        
        status = self.motion.target_status(axis)
        return status
        
    # Get all reference status register bits as 32 character string (32-> 0)
    def get_reference_status(self, axis):
        """MOTION CONTROL FUNCTIONS
         Get all reference status register bits as 32 character string (32-> 0) """

        #if not self.motion: return
        return self.motion.reference_status(axis=axis)
    

    def get_limit_switch_status(self, axis):
        """
        MOTION CONTROL FUNCTIONS
        Get limit switch status for an axis
        """
        return self.motion.limit_switch_status(axis=axis)


    def get_limit_switch_status_all_axes(self):
        """
        MOTION CONTROL FUNCTIONS
        Get limit switch status for all axes
        """
        resp = {}
        for axis in ('X','Y','Z','T'):
            resp[axis] = self.get_limit_switch_status(axis=axis)
        return resp


    def get_overshoot(self):
        """MOTION CONTROL FUNCTIONS
        Is z-axis (focus) currently in overshoot mode?"""

        #if not self.motion: return False
        return self.motion.overshoot

    def is_moving(self):
        # If not communicating with motor board
        if not self.motion.driver: return False

        # Check each axis
        x_status = self.get_target_status('X')
        y_status = self.get_target_status('Y')
        z_status = self.get_target_status('Z')
        t_status = self.get_target_status('T')

        if x_status and y_status and z_status and t_status and not self.get_overshoot():
            return False
        else:
            return True
        

    def wait_until_finished_moving(self):

        if not self.motion.driver: return

        while self.is_moving():
            time.sleep(0.05)
        
        return
    

    def get_microscope_model(self):
        if not self.motion.driver:
            return None
        
        return self.motion.get_microscope_model()
    
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
        #print("Wait time = ", wait_time)

        # Start thread to wait until capture is complete
        capture_timer = threading.Timer(wait_time, self.capture_complete)
        capture_timer.start()

    def capture_complete(self):
        self.capture_return = self.get_image() # Grab image
        self.is_capturing = False

    
    def capture_blocking(self):
        if not self.led: return
        if not self.camera: return

        wait_time = 2*self.get_exposure_time()/1000+0.2
        time.sleep(wait_time)
        return self.get_image()

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
        self.last_focus_score = None    # Last / Previous focus score
        self.last_focus_pass = False # Are we on the last scan for autofocus?

        # Start the autofocus process at z-minimum
        self.move_absolute_position('Z', self.z_min)

        while not self.autofocus_iterate(AF_min):
            time.sleep(0.01)

    def autofocus_iterate(self, AF_min):
        """INTEGRATED SCOPE FUNCTIONS
        iterate autofocus functionality"""
        done=False

        # Ignore steps until conditions are met
        if self.is_moving(): return done  # needs to be in position
        if self.is_capturing: return done # needs to have completed capture with illumination

        # Is there a previous capture result to pull?
        if self.capture_return is False:
            # No -> start a capture event
            self.capture()
            return done
            
        else:
            # Yes -> pull the capture result and clear
            image = self.capture_return
            self.capture_return = False
            
        if image is False:
            # Stop thread image can't be acquired
            done = True
            return done
   
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
            return done

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
            self.last_focus_score = focus

            # Stop thread image when autofocus is complete
            done=True
        return done

    # Algorithms for estimating the quality of the focus
    def focus_function(self, image, algorithm = 'vollath4', include_logging: bool = True):
        """INTEGRATED SCOPE FUNCTIONS
        assess focus value at specific position for autofocus function"""

        if include_logging:
            logger.info('[SCOPE API ] Lumascope.focus_function()')

        w = image.shape[0]
        h = image.shape[1]

        # Journal of Microscopy, Vol. 188, Pt 3, December 1997, pp. 264–272
        if algorithm == 'vollath4': # pg 266
            image = np.double(image)
            sum_one = np.sum(np.multiply(image[:w-1,:h], image[1:w,:h])) # g(i, j).g(i+1, j)
            sum_two = np.sum(np.multiply(image[:w-2,:h], image[2:w,:h])) # g(i, j).g(i+2, j)
            if include_logging:
                logger.info('[SCOPE API ] Focus Score Vollath: ' + str(sum_one - sum_two))
            return sum_one - sum_two

        elif algorithm == 'skew':
            hist = np.histogram(image, bins=256,range=(0,256))
            hist = np.asarray(hist[0], dtype='int')
            max_index = hist.argmax()

            edges = np.histogram_bin_edges(image, bins=1)
            white_edge = edges[1]

            skew = white_edge-max_index
            if include_logging:
                logger.info('[SCOPE API ] Focus Score Skew: ' + str(skew))
            return skew

        elif algorithm == 'pixel_variation':
            sum = np.sum(image)
            ssq = np.sum(np.square(image))
            var = ssq*w*h-sum**2
            if include_logging:
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


