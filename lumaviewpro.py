#!/usr/bin/python3

'''
MIT License

Copyright (c) 2020 Etaluma, Inc.

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
Bryan Tiedemann, The Earthineering Company

MODIFIED:
Sept. 10, 2021
'''

# General
import sys
import numpy as np
import time
import os
import json
import serial
import serial.tools.list_ports as list_ports
# from scipy.optimized import curve_fit

# Kivy
import kivy
from kivy.app import App
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.graphics import RenderContext
from kivy.properties import StringProperty, ObjectProperty, BooleanProperty, NumericProperty
from kivy.properties import BoundedNumericProperty, ColorProperty, OptionProperty, ListProperty
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.animation import Animation
from kivy.graphics import Line, Color, Rectangle
# from kivy.config import Config
# Config.set('graphics', 'width', '1920')
# Config.set('graphics', 'height', '1080')

# User Interface
from kivy.uix.accordion import Accordion, AccordionItem
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.scatter import Scatter
from kivy.uix.widget import Widget
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.switch import Switch
from kivy.uix.slider import Slider
from kivy.uix.dropdown import DropDown
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button

# From KivyMD
from kivymd.uix.behaviors import HoverBehavior, TouchBehavior

# Video Related
from kivy.graphics.texture import Texture
import cv2
from scipy import signal

# Pylon Camera Related
from pypylon import pylon

global lumaview
global protocol
with open('./data/protocol.json', "r") as read_file:
    protocol = json.load(read_file)

class PylonCamera(Image):
    record = ObjectProperty(None)
    record = False

    def __init__(self, **kwargs):
        super(PylonCamera,self).__init__(**kwargs)
        self.camera = False
        self.play = True
        self.array = []
        self.connect()
        self.start()


    def connect(self):
        try:
            # Create an instant camera object with the camera device found first.
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
            self.camera.Width.SetValue(self.camera.Width.Max)
            self.camera.Height.SetValue(self.camera.Height.Max)
            # self.camera.PixelFormat.SetValue('PixelFormat_Mono12')
            self.camera.GainAuto.SetValue('Off')
            self.camera.ExposureAuto.SetValue('Off')
            self.camera.ReverseX.SetValue(True);
            # Grabbing Continusely (video) with minimal delay
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            print("LumaViewPro compatible camera or scope is now connected.")

        except:
            if self.camera == False:
                print("It looks like a LumaViewPro compatible camera or scope is not connected.")
                print("Error: PylonCamera.connect() exception")
            self.camera = False

    def start(self, fps = 14):
        self.fps = fps
        self.frame_event = Clock.schedule_interval(self.update, 1.0 / self.fps)

    def stop(self):
        if self.frame_event:
            Clock.unschedule(self.frame_event)

    def update(self, dt):
        if self.camera == False:
            self.connect()
            if self.camera == False:
                self.source = "./data/icons/camera to USB.png"
                return
        try:
            global lumaview
            if self.camera.IsGrabbing():
                grabResult = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():
                    image = grabResult.GetArray()
                    self.array = image
                    image_texture = Texture.create(size=(image.shape[1],image.shape[0]), colorfmt='luminance')
                    image_texture.blit_buffer(image.flatten(), colorfmt='luminance', bufferfmt='ubyte')                    # display image from the texture
                    self.texture = image_texture

                self.lastGrab = pylon.PylonImage()
                self.lastGrab.AttachGrabResultBuffer(grabResult)

            if self.record == True:
                lumaview.capture(0)

            grabResult.Release()

        except:
            if self.camera == False:
                print("A LumaViewPro compatible camera or scope was disconnected.")
                print("Error: PylonCamera.update() exception")
            self.camera = False

    def frame_size(self, w, h):
        if self.camera == False:
            print("A LumaViewPro compatible camera or scope is not connected.")
            print("Error: PylonCamera.frame_size() self.camera == False")
            return

        width = int(min(int(w), self.camera.Width.Max)/2)*2
        height = int(min(int(h), self.camera.Height.Max)/2)*2
        offset_x = int((self.camera.Width.Max-width)/4)*2
        offset_y = int((self.camera.Height.Max-height)/4)*2

        self.camera.StopGrabbing()
        self.camera.Width.SetValue(width)
        self.camera.Height.SetValue(height)
        self.camera.OffsetX.SetValue(offset_x)
        self.camera.OffsetY.SetValue(offset_y)
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def gain(self, gain):
        if self.camera == False:
            print("A LumaViewPro compatible camera or scope is not connected.")
            print("Error: PylonCamera.gain() self.camera == False")
            return

        self.camera.StopGrabbing()
        self.camera.Gain.SetValue(gain)
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def auto_gain(self, state = True):
        if self.camera == False:
            print("A LumaViewPro compatible camera or scope is not connected.")
            print("Error: PylonCamera.gain() self.camera == False")
            return

        self.camera.StopGrabbing()
        if state == True:
            self.camera.GainAuto.SetValue('Once') # 'Off' 'Once' 'Continuous'
        else:
            self.camera.GainAuto.SetValue('Off')

        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def exposure_t(self, t):
        if self.camera == False:
            print("A LumaViewPro compatible camera or scope is not connected.")
            print("Error: PylonCamera.exposure_t() self.camera == False")
            return

        self.stop()
        self.camera.StopGrabbing()
        self.camera.ExposureTime.SetValue(t*1000) # (t*1000) in microseconds; therefore t  in milliseconds

        # # DEBUG:
        # print(camera.ExposureTime.Min)
        # print(camera.ExposureTime.Max)
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        # Adjust Frame Rate based on exposure time
        fps = 1000/t # maximum based on exposure time
        fps = min(14, fps) # camera limit is 14 fps regardless of exposure
        self.start(fps)

    def auto_exposure_t(self):
        if self.camera == False:
            print("A LumaViewPro compatible camera or scope is not connected.")
            print("Error: PylonCamera.gain() self.camera == False")
            return

        self.camera.StopGrabbing()
        self.camera.ExposureAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

class LEDBoard:
    def __init__(self, **kwargs):

        # self.port="/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_205835435736-if00"
        # self.port = "/dev/ttyS0"

        ports = serial.tools.list_ports.comports(include_links = True)
        for port in ports:
            if (port.vid == 1155) and (port.pid == 22336):
                print('LED Control Board identified at', port.device)
                self.port = port.device
                protocol['port'] = port.device

        self.baudrate=115200
        self.bytesize=serial.EIGHTBITS
        self.parity=serial.PARITY_NONE
        self.stopbits=serial.STOPBITS_ONE
        self.timeout=.5 # seconds
        self.driver = True
        self.connect()

    def __del__(self):
        if self.driver != False:
            self.driver.close()

    def connect(self):
        try:
            self.driver = serial.Serial(port=self.port, baudrate=self.baudrate, bytesize=self.bytesize, parity=self.parity, stopbits=self.stopbits, timeout=self.timeout)
            self.driver.close()
            self.driver.open()
        except:
            if self.driver != False:
                print("It looks like a Lumaview compatible LED driver board is not plugged in.")
                print("Error: LEDBoard.connect() exception")
            self.driver = False

    def color2ch(self, color):
        if color == 'Blue':
            return 0
        elif color == 'Green':
            return 1
        elif color == 'Red':
            return 2
        else: # BF
            return 3

    def led_cal(self, channel):
        command = '{CAL,'+ str(channel) + '}00'
        if self.driver != False:
            self.driver.write(command.encode('utf-8')+b"\r\n")

    def led_on(self, channel, mA):
        command = '{TON,'+ str(channel) + ',H,' + str(mA) + '}00'
        if self.driver != False:
            self.driver.write(command.encode('utf-8')+b"\r\n")

    def led_off(self):
        command = '{TOF}00'
        if self.driver != False:
            self.driver.write(command.encode('utf-8')+b"\r\n")

class TrinamicBoard:
    # Trinamic 3230 board (preferred)
    # Trinamic 6110 board (now) Address
    z_microstep = 0.05489864865 # microns per microstep at 32x microstepping
                                # via Eric Weiner 6/24/2021
    xy_microstep = 0.15625      # microns per microstep at 32x microstepping

    addr = 255

    # Commmand set available in direct mode
    cmnd = {
        'ROR':1,    # Rotate Right
        'ROL':2,    # Rotate Left
        'MST':3,    # Motor Stop
        'MVP':4,    # Move to Position
        'SAP':5,    # Set Axis Parameter
        'GAP':6,    # Get Axis Parameter
        'STAP':7,   # Store Axis Parameter
        'RSAP':8,   # Restore Axis Parameter
        'SGP':9,    # Set Global Parameter
        'GGP':10,   # Get Global Parameter
        'STGP':11,  # Store Global Parameter
        'RSGP':12,  # Restore Global Parameter
        'RFS':13,   # Reference Search
        'SIO':14,   # Set Output
        'GIO':15,   # Get Input / Output
        'SCO':30,   # Set Coordinate
        'GCO':31,   # Get Coordinate
        'CCO':32    # Capture Coordinate
    }

    axis = {
        'X':0, # Left to Right is positive
        'Y':1, # Front to Back is positive
        'Z':2  # Down is positive
    }

    #----------------------------------------------------------
    # Set up Serial Port connection
    #----------------------------------------------------------
    def __init__(self, **kwargs):
        # find all available serial ports
        ports = serial.tools.list_ports.comports(include_links = True)
        for port in ports:
            if (port.vid == 10812) and (port.pid == 256):
                print('Trinamic Motor Control Board identified at', port.device)
                self.port = port.device

        self.reset = 0
        self.baudrate = 9600
        self.bytesize = serial.EIGHTBITS
        self.parity = serial.PARITY_NONE
        self.stopbits = serial.STOPBITS_ONE
        self.timeout = 5 # seconds
        self.driver = True
        try:
            self.connect()
        except:
            print("Could not connect to Trinamic Motor Control Board")
            self.driver = False

    def __del__(self):
        if self.driver != False:
            self.driver.close()

    def connect(self):
        self.driver = serial.Serial(port=self.port, baudrate=self.baudrate, bytesize=self.bytesize, parity=self.parity, stopbits=self.stopbits, timeout=self.timeout)

        # check user variable not stored in EEPROM
        self.reset = self.SendGram('GGP', 255, 'Z', 0) # This is bank 2 ('Z' is just to get at it)
        if self.reset == 0:
            print('Homing Z-axis')
            self.zhome()
            self.reset = self.SendGram('SGP', 255, 'Z', 1) # This is bank 2 ('Z' is just to get at it)

        self.driver.close()
        self.driver.open()

    #----------------------------------------------------------
    # Receive Datagram
    #----------------------------------------------------------
    def GetGram(self, verbose = True):

        # receive the datagram
        datagram = self.driver.read(9)

        checksum = 0
        for i in range(8):         # compare checksum
            checksum =  (checksum + datagram[i]) & 0xff

        if checksum != datagram[8]:
            print('Return Checksum Error')
            return False

        Address=datagram[0]
        Status=datagram[2]
        if Status != 100:
            print('Return Status Error')
        Value = int.from_bytes(datagram[4:8], byteorder='big', signed=True)
        if verbose == True:
            print('Status: ', Status)
            print('Value:  ', Value)
        return Value

    #----------------------------------------------------------
    # Send Datagram
    #----------------------------------------------------------
    def SendGram(self, Command, Type, Motor, Value):

        datagram = bytearray(9)
        datagram[0] = self.addr
        datagram[1] = self.cmnd[Command]
        datagram[2] = Type
        datagram[3] = self.axis[Motor]
        datagram[4] = (Value >> 24) & 0xff # shift by 24 bits i.e. divide by 2, 24 times
        datagram[5] = (Value >> 16) & 0xff # shift by 16 bits i.e. divide by 2, 16 times
        datagram[6] = (Value >> 8) & 0xff  # shift by 8 bits i.e. divide by 2, 8 times
        datagram[7] = Value & 0xff # bitwise and with 0xff to get last 8 byte

        for i in range(8):         # generate checksum
            datagram[8] =  (datagram[8] + datagram[i]) & 0xff

        if self.driver != False:
            self.driver.write(datagram)
            return self.GetGram(verbose = False)
        else:
            print('Trinamic Motor Control Board is not connected')
            return False

    def z_ustep2um(self, ustep):
        um = float(ustep)*self.z_microstep
        return um

    def z_um2ustep(self, um):
        ustep = int(um/self.z_microstep)
        return ustep

    def zhome(self):
        # TODO all the initialization should be in __init__
        #----------------------------------------------------------
        # Z-Axis Initialization
        #----------------------------------------------------------
        self.SendGram('SAP', 6, 'Z', 16)      # Maximum current
        self.SendGram('SAP', 7, 'Z', 8)       # Standby current
        self.SendGram('SAP', 2, 'Z', 1000)    # Target Velocity
        self.SendGram('SAP', 5, 'Z', 500)     # Acceleration
        self.SendGram('SAP', 140, 'Z', 5)     # 32X Microstepping
        self.SendGram('SAP', 153, 'Z', 9)     # Ramp Divisor 9
        self.SendGram('SAP', 154, 'Z', 3)     # Pulse Divisor 3
        self.SendGram('SAP', 163, 'Z', 0)     # Constant TOff Mode (spreadcycle)


        # Parameters for Limit Switches
        #----------------------------------------------------------
        self.SendGram('SAP', 12, 'Z', 0)      # enable Right Limit switch
        self.SendGram('SAP', 13, 'Z', 0)      # enable Left Limit switch

        # Parameters for Homing
        #----------------------------------------------------------
        self.SendGram('SAP', 193, 'Z', 65)    # Search Right Stop Switch (Down)
        self.SendGram('SAP', 194, 'Z', 1000)  # Reference search speed
        self.SendGram('SAP', 195, 'Z', 10)    # Reference switch speed (was 10X less than search speed in LumaView)

        # Start the Trinamic Homing Procedure
        #----------------------------------------------------------
        self.SendGram('RFS', 0, 'Z', 0)       # Home to the Right Limit switch (Down)
        self.zhome_event = Clock.schedule_interval(self.check_zhome, 0.05)

    # Test if reference function is complete (z homing)
    def check_zhome(self, dt):
        value = self.SendGram('RFS', 2, 'Z', 0)
        if value == 0:
            Clock.unschedule(self.zhome_event)
        # print('still scheduled?')

    def xy_ustep2um(self, ustep):
        um = float(ustep)*self.xy_microstep
        return um

    def xy_um2ustep(self, um):
        ustep = int(um/self.xy_microstep)
        return ustep

    def xyhome(self):

        self.zhome()
        #----------------------------------------------------------
        # X-Axis Initialization
        #----------------------------------------------------------
        self.SendGram('SAP', 6, 'X', 16)     # Maximum current
        self.SendGram('SAP', 7, 'X', 8)      # standby current
        self.SendGram('SAP', 2, 'X', 1000)   # Target Velocity
        self.SendGram('SAP', 5, 'X', 500)    # Acceleration
        self.SendGram('SAP', 140, 'X', 5)    # 32X Microstepping
        self.SendGram('SAP', 153, 'X', 9)    # Ramp Divisor 9
        self.SendGram('SAP', 154, 'X', 3)    # Pulse Divisor 3
        self.SendGram('SAP', 163, 'X', 0)    # Constant TOff Mode (spreadcycle)

        # Parameters for Limit Switches
        #----------------------------------------------------------
        self.SendGram('SAP', 12, 'X', 0)     # enable Right Limit switch
        self.SendGram('SAP', 13, 'X', 0)     # enable Left Limit switch

        # Parameters for Homing
        #----------------------------------------------------------
        # 'Set Axis Parameter', 'Reference Search Mode', 'X-axis', '1+64 = Search Right Stop Switch Only'
        self.SendGram('SAP', 193, 'X', 65)   # Search Right Stop switch Only
        self.SendGram('SAP', 194, 'X', 1000) # Reference search speed
        self.SendGram('SAP', 195, 'X', 10)   # Reference switch speed (was 10X less than search speed in LumaView)

        # Start the Trinamic Homing Procedure
        # ----------------------------------------------------------
        self.SendGram('RFS', 0, 'X', 0)      # Home to the Right Limit switch (Right)

        #----------------------------------------------------------
        # Y-Axis Initialization
        #----------------------------------------------------------
        self.SendGram('SAP', 6, 'Y', 16)    # Maximum current
        self.SendGram('SAP', 7, 'Y', 8)     # Standby current
        self.SendGram('SAP', 2, 'Y', 1000)  # Target Velocity
        self.SendGram('SAP', 5, 'Y', 500)   # Acceleration
        self.SendGram('SAP', 140, 'Y', 5)   # 32X Microstepping
        self.SendGram('SAP', 153, 'Y', 9)   # Ramp Divisor 9
        self.SendGram('SAP', 154, 'Y', 3)   # Pulse Divisor 3
        self.SendGram('SAP', 163, 'Y', 0)   # Constant TOff Mode (spreadcycle)

        # Parameters for Limit Switches
        #----------------------------------------------------------
        self.SendGram('SAP', 12, 'Y', 0)    # enable Right Limit switch
        self.SendGram('SAP', 13, 'Y', 0)    # enable Left Limit switch

        # Parameters for Homing
        #----------------------------------------------------------
        self.SendGram('SAP', 193, 'Y', 65)   # Search Right Stop switch Only (Back)
        self.SendGram('SAP', 194, 'Y', 1000) # Reference search speed
        self.SendGram('SAP', 195, 'Y', 10)   # Reference switch speed (was 10X less than search speed in LumaView)

        # Start the Trinamic Homing Procedure
        #----------------------------------------------------------
        self.SendGram('RFS', 0, 'Y', 0)      # Home to the Right Limit switch (Back)
        self.xyhome_event = Clock.schedule_interval(self.check_xyhome, 0.05)

    # Test if reference function is complete (xy homing)
    def check_xyhome(self, dt):
        x = self.SendGram('RFS', 2, 'X', 0)
        y = self.SendGram('RFS', 2, 'Y', 0)
        if (x == 0) and (y == 0):
            Clock.unschedule(self.xyhome_event)

# -------------------------------------------------------------------------
# MAIN DISPLAY of LumaViewPro App
# -------------------------------------------------------------------------
class MainDisplay(FloatLayout):
    led_board = ObjectProperty(None)
    led_board = LEDBoard()
    motion = ObjectProperty(None)
    motion = TrinamicBoard()

    def choose_folder(self):
        content = LoadDialog(load=self.load,
                             cancel=self.dismiss_popup,
                             path=protocol['live_folder'])
        self._popup = Popup(title="Select Save Folder",
                            content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path):
        protocol['live_folder'] = path
        self.dismiss_popup()

    def dismiss_popup(self):
        self._popup.dismiss()

    def cam_toggle(self):
        microscope = self.ids['viewer_id'].ids['microscope_camera']
        if microscope.camera == False:
            return

        if microscope.play == True:
            microscope.play = False
            self.led_board.led_off()
            microscope.stop()
        else:
            microscope.play = True
            microscope.start()

    def capture(self, dt, save_folder = './capture/', file_root = 'live_', append = 'ms', color = 'BF'):
        microscope = self.ids['viewer_id'].ids['microscope_camera']
        if microscope.camera == False:
            return

        img = np.zeros((microscope.array.shape[0], microscope.array.shape[1], 3))

        if self.ids['mainsettings_id'].currentLayer != 'protocol':
            color = self.ids['mainsettings_id'].currentLayer

        if color == 'Blue':
            img[:,:,0] = microscope.array
        elif color == 'Green':
            img[:,:,1] = microscope.array
        elif color == 'Red':
            img[:,:,2] = microscope.array
        else:
            img[:,:,0] = microscope.array
            img[:,:,1] = microscope.array
            img[:,:,2] = microscope.array

        img = np.flip(img, 0)

        folder = protocol['live_folder']

        # set filename options
        if append == 'time':
            append = time.strftime("%Y%m%d_%H%M%S")
        elif append == 'ms':
            append = str(int(round(time.time() * 1000)))
        else:
            append =''

        # generate filename string
        filename =  file_root + append + '.tiff'

        try:
            cv2.imwrite(save_folder+'/'+filename, img.astype(np.uint8))

        except:
            print("Error: Unable to save. Save folder does not exist?")

    def record(self):
        microscope = self.ids['viewer_id'].ids['microscope_camera']
        if microscope.camera == False:
            return
        microscope.record = not microscope.record

    def composite(self, dt):
        microscope = self.ids['viewer_id'].ids['microscope_camera']
        if microscope.camera == False:
            return

        folder = protocol['live_folder']
        img = np.zeros((protocol['frame_height'], protocol['frame_width'], 3))

        layers = ['BF', 'Blue', 'Green', 'Red']
        for layer in layers:
            # multicolor image stack

            if protocol[layer]['acquire'] == True:
                lumaview.ids['mainsettings_id'].ids[layer].goto_focus()

                # Wait for focus to be reached (approximate)
                set_value = lumaview.motion.SendGram('GAP', 0, 'Z', 10)  # Get target value
                get_value = lumaview.motion.SendGram('GAP', 1, 'Z', 0)   # Get current value

                while set_value != get_value:
                    time.sleep(.01)
                    get_value = lumaview.motion.SendGram('GAP', 1, 'Z', 0)   # Get current value

                # set the gain and exposure
                gain = protocol[layer]['gain']
                microscope.gain(gain)
                exposure = protocol[layer]['exp']
                microscope.exposure_t(exposure)
                # turn on the LED
                # update illumination to currently selected settings
                illumination = protocol[layer]['ill']
                led_board = lumaview.led_board

                # Dark field capture
                led_board.led_off()
                time.sleep(exposure/1000)  # Should be replaced with Clock
                microscope.update(0)
                darkfield = microscope.array
                # Florescent capture
                led_board.led_on(led_board.color2ch(layer), illumination) #self.layer??
                time.sleep(exposure/1000)  # Should be replaced with Clock
                microscope.update(0)
                #corrected = np.max(microscope.array - darkfield, np.zeros(like=darkfield))
                corrected = microscope.array - np.minimum(microscope.array,darkfield)
                # buffer the images
                if layer == 'Blue':
                    img[:,:,0] = corrected
                elif layer == 'Green':
                    img[:,:,1] = corrected
                elif layer == 'Red':
                    img[:,:,2] = corrected
                # # if Brightfield is included
                # else:
                #     a = 0.3
                #     img[:,:,0] = img[:,:,0]*a + corrected*(1-a)
                #     img[:,:,1] = img[:,:,1]*a + corrected*(1-a)
                #     img[:,:,2] = img[:,:,2]*a + corrected*(1-a)

            led_board.led_off()
            lumaview.ids['mainsettings_id'].ids[layer].ids['apply_btn'].state = 'normal'

        img = np.flip(img, 0)

        filename = 'composite_' + str(int(round(time.time() * 1000))) + '.tiff'
        cv2.imwrite(folder+'/'+filename, img.astype(np.uint8))
        # TODO save file in 16 bit TIFF, OMETIFF, and others
        # cv2.imwrite(folder+'/'+filename, img.astype(np.uint16)) # This works

        # # TODO display captured composite
        # microscope.stop()
        # microscope.source = filename
        # time.sleep(5) #Needs to be user selected
        # microscope.start()

    def fit_image(self):
        microscope = self.ids['viewer_id'].ids['microscope_camera']
        if microscope.camera == False:
            return
        self.ids['viewer_id'].scale = 1
        self.ids['viewer_id'].pos = (0,0)

    def one2one_image(self):
        microscope = self.ids['viewer_id'].ids['microscope_camera']
        if microscope.camera == False:
            return
        w = self.width
        h = self.height
        scale_hor = float(microscope.camera.Width.GetValue()) / float(w)
        scale_ver = float(microscope.camera.Height.GetValue()) / float(h)
        scale = max(scale_hor, scale_ver)
        self.ids['viewer_id'].scale = scale
        self.ids['viewer_id'].pos = (int((w-scale*w)/2),int((h-scale*h)/2))

# -----------------------------------------------------------------------------
# Shader code
# Based on code from the kivy example Live Shader Editor found at:
# kivy.org/doc/stable/examples/gen__demo__shadereditor__main__py.html
# -----------------------------------------------------------------------------
fs_header = '''
#ifdef GL_ES
precision highp float;
#endif

/* Outputs from the vertex shader */
varying vec4 frag_color;
varying vec2 tex_coord0;

/* uniform texture samplers */
uniform sampler2D texture0;

/* fragment attributes
attribute float red_gain;
attribute float green_gain;
attribute float blue_gain; */

/* custom one */
uniform vec2 resolution;
uniform float time;
uniform vec4 black_point;
uniform vec4 white_point;
'''

vs_header = '''
#ifdef GL_ES
precision highp float;
#endif

/* Outputs to the fragment shader */
varying vec4 frag_color;
varying vec2 tex_coord0;

/* vertex attributes */
attribute vec2     vPosition;
attribute vec2     vTexCoords0;

/* uniform variables */
uniform mat4       modelview_mat;
uniform mat4       projection_mat;
uniform vec4       color;
'''

# global illumination_vals
# illumination_vals = (0., )*4

# global gain_vals
# gain_vals = (1., )*4

class ShaderViewer(Scatter):
    black = ObjectProperty(0.)
    white = ObjectProperty(1.)

    fs = StringProperty('''
void main (void) {
	gl_FragColor =
    white_point *
    frag_color *
    texture2D(texture0, tex_coord0)
    - black_point;
    //gl_FragColor = pow(glFragColor.rgb, 1/gamma)
}
''')
    vs = StringProperty('''
void main (void) {
  frag_color = color;
  tex_coord0 = vTexCoords0;
  gl_Position =
  projection_mat *
  modelview_mat *
  vec4(vPosition.xy, 0.0, 1.0);
}
''')


    def __init__(self, **kwargs):
        self.canvas = RenderContext()
        super(ShaderViewer, self).__init__(**kwargs)
        self.canvas.shader.fs = fs_header + self.fs
        self.canvas.shader.vs = vs_header + self.vs

    def on_touch_down(self, touch):
        # Override Scatter's `on_touch_down` behavior for mouse scroll
        if touch.is_mouse_scrolling:
            if touch.button == 'scrolldown':
                if self.scale < 100:
                    self.scale = self.scale * 1.1
            elif touch.button == 'scrollup':
                if self.scale > 0.1:
                    self.scale = self.scale * 0.8
        # If some other kind of "touch": Fall back on Scatter's behavior
        else:
            super(ShaderViewer, self).on_touch_down(touch)

    def update_shader(self, false_color):

        c = self.canvas
        c['projection_mat'] = Window.render_context['projection_mat']
        c['time'] = Clock.get_boottime()
        c['resolution'] = list(map(float, self.size))
        c['black_point'] = (self.black, )*4
        c['gamma'] = 2.2

        if false_color == 'Red':
            c['white_point'] = (self.white, 0., 0., 1.)
        elif false_color == 'Green':
            c['white_point'] = (0., self.white, 0., 1.)
        elif false_color == 'Blue':
            c['white_point'] = (0., 0., self.white, 1.)
        else:
            c['white_point'] = (self.white, )*4

        # print('b:', self.black, 'w:', self.white)

    def on_fs(self, instance, value):
        self.canvas.shader.fs = value

    def on_vs(self, instance, value):
        self.canvas.shader.vs = value

Factory.register('ShaderViewer', cls=ShaderViewer)


class MotionSettings(BoxLayout):
    settings_width = dp(300)
    isOpen = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_draw=self.check_settings)

    # Hide (and unhide) motion settings
    def toggle_settings(self):
        global lumaview
        microscope = lumaview.ids['viewer_id'].ids['microscope_camera']
        microscope.stop()
        self.ids['verticalcontrol_id'].update_event = Clock.schedule_interval(self.ids['verticalcontrol_id'].update_gui, 0.5)
        self.ids['xy_stagecontrol_id'].update_event = Clock.schedule_interval(self.ids['xy_stagecontrol_id'].update_gui, 0.5)

        # move position of motion control
        if self.isOpen:
            self.ids['toggle_motionsettings'].state = 'normal'
            self.pos = -self.settings_width+30, 0
            self.isOpen = False
        else:
            self.ids['toggle_motionsettings'].state = 'down'
            self.pos = 0, 0
            self.isOpen = True

        if microscope.play == True:
            microscope.start()

    def check_settings(self, *args):
        # global lumaview
        if not self.isOpen:
            self.ids['toggle_motionsettings'].state = 'normal'
            self.pos = -self.settings_width+30, 0
        else:
            self.ids['toggle_motionsettings'].state = 'down'
            self.pos = 0, 0

class ShaderEditor(BoxLayout):
    fs = StringProperty('''
void main (void){
	gl_FragColor =
    white_point *
    frag_color *
    texture2D(texture0, tex_coord0)
    - black_point;
}
''')
    vs = StringProperty('''
void main (void) {
  frag_color = color;
  tex_coord0 = vTexCoords0;
  gl_Position =
  projection_mat *
  modelview_mat *
  vec4(vPosition.xy, 0.0, 1.0);
}
''')

    viewer = ObjectProperty(None)
    hide_editor = ObjectProperty(None)
    hide_editor = True


    def __init__(self, **kwargs):
        super(ShaderEditor, self).__init__(**kwargs)
        self.test_canvas = RenderContext()
        s = self.test_canvas.shader
        self.trigger_compile = Clock.create_trigger(self.compile_shaders, -1)
        self.bind(fs=self.trigger_compile, vs=self.trigger_compile)

    def compile_shaders(self, *largs):
        print('ShaderEditor.compile_shaders()')
        if not self.viewer:
            print('ShaderEditor.compile_shaders() Fail')
            return

        # we don't use str() here because it will crash with non-ascii char
        fs = fs_header + self.fs
        vs = vs_header + self.vs

        print('-->', fs)
        self.viewer.fs = fs
        print('-->', vs)
        self.viewer.vs = vs

    # Hide (and unhide) Shader settings
    def toggle_editor(self):
        # update protocol
        if self.hide_editor == False:
            self.hide_editor = True
            self.pos = -285, 0
        else:
            self.hide_editor = False
            self.pos = 0, 0

class MainSettings(BoxLayout):
    settings_width = dp(300)
    isOpen = BooleanProperty(False)
    notCollapsing = BooleanProperty(True)
    currentLayer = StringProperty('microscope')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_draw=self.check_settings)

    # Hide (and unhide) main settings
    def toggle_settings(self):
        global lumaview
        microscope = lumaview.ids['viewer_id'].ids['microscope_camera']
        microscope.stop()

        # move position of settings
        if self.isOpen:
            self.ids['toggle_mainsettings'].state = 'normal'
            self.pos = lumaview.width - 30, 0
            self.isOpen = False
        else:
            self.ids['toggle_mainsettings'].state = 'down'
            self.pos = lumaview.width - self.settings_width, 0
            self.isOpen = True

        if microscope.play == True:
            microscope.start()

    def accordion_collapse(self, layer):
        global lumaview
        microscope = lumaview.ids['viewer_id'].ids['microscope_camera']
        microscope.stop()
        lumaview.led_board.led_off()
        self.currentLayer = layer
        self.notCollapsing = not(self.notCollapsing)
        if self.notCollapsing:
            if microscope.play == True:
                microscope.start()

    def check_settings(self, *args):
        global lumaview
        if not self.isOpen:
            self.ids['toggle_mainsettings'].state = 'normal'
            self.pos = lumaview.width - 30, 0
        else:
            self.ids['toggle_mainsettings'].state = 'down'
            self.pos = lumaview.width - self.settings_width, 0


class Histogram(Widget):
    bg_color = ObjectProperty(None)
    layer = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Histogram, self).__init__(**kwargs)
        if self.bg_color is None:
            self.bg_color = (1, 1, 1, 1)

        self.event = Clock.schedule_interval(self.histogram, 1)
        self.hist_range_set = False
        self.edges = [0,255]
        self.stablize = 0.3

    def histogram(self, *args):
        global lumaview

        camera = lumaview.ids['viewer_id'].ids['microscope_camera']
        if camera.camera != False:
            image = camera.array
            hist = np.histogram(image, bins=256,range=(0,256))
            if self.hist_range_set:
                edges = self.edges
            else:
                edges = np.histogram_bin_edges(image, bins=1)
                edges[0] = self.stablize*self.edges[0] + (1 - self.stablize)*edges[0]
                edges[1] = self.stablize*self.edges[1] + (1 - self.stablize)*edges[1]

            # mean = np.mean(hist[1],hist[0])
            lumaview.ids['viewer_id'].black = float(edges[0])/255.
            lumaview.ids['viewer_id'].white = float(edges[1])/255.

            # UPDATE SHADER

            self.canvas.clear()
            r, b, g, a = self.bg_color
            self.hist = hist
            self.edges = edges
            with self.canvas:
                x = self.x
                y = self.y
                w = self.width
                h = self.height
                Color(r, b, g, a/12)
                Rectangle(pos=(x, y), size=(256, h))
                Color(r, b, g, a/4)
                Rectangle(pos=(x + edges[0], y), size=(edges[1]-edges[0], h))
                Color(r, b, g, a/2)
                #self.color = Color(rgba=self.color)
                logHistogram = lumaview.ids['mainsettings_id'].ids[self.layer].ids['logHistogram_id'].active
                if logHistogram:
                    maxheight = np.log(np.max(hist[0])+1)
                else:
                    maxheight = np.max(hist[0])
                if maxheight > 0:
                    scale=h/maxheight
                    for i in range(len(hist[0])):
                        if logHistogram:
                            counts = scale*np.log(hist[0][i] + 1)
                        else:
                            counts = np.ceil(scale*hist[0][i])
                        self.pos = self.pos
                        self.line = Line(points=(x+i, y, x+i, y+counts), width=1)
        # else:
        #     print("Can't find image.")

    # def on_touch_down(self, touch):
    #     x = touch.x - self.x
    #     print(x)
    #     if abs(x - self.edges[0]) < 20:
    #         self.edges[0] = x
    #         self.hist_range_set = True
    #     if abs(x - self.edges[1]) < 20:
    #         self.edges[1] = x
    #         self.hist_range_set = True
    #     '''if touch.is_mouse_scrolling:
    #         if touch.button == 'scrolldown':
    #             if self.scale < 100:
    #                 self.scale = self.scale * 1.1
    #         elif touch.button == 'scrollup':
    #             if self.scale > 0.1:
    #                 self.scale = self.scale * 0.8
    #                 '''
    #     # If some other kind of "touch": Fall back on Scatter's behavior
    #     #else:
    #         #super(ShaderViewer, self).on_touch_down(touch)


class VerticalControl(BoxLayout):

    def course_up(self):
        course = protocol['objective']['step_course']
        dist = lumaview.motion.z_um2ustep(course)       # 10 um
        lumaview.motion.SendGram('MVP', 1, 'Z', -dist)  # Move UP relative
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def fine_up(self):
        fine = protocol['objective']['step_fine']
        dist = lumaview.motion.z_um2ustep(fine)         # 1 um
        lumaview.motion.SendGram('MVP', 1, 'Z', -dist)  # Move UP
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def fine_down(self):
        fine = protocol['objective']['step_fine']
        dist = lumaview.motion.z_um2ustep(fine)         # 1 um
        lumaview.motion.SendGram('MVP', 1, 'Z', dist)   # Move DOWN
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def course_down(self):
        course = protocol['objective']['step_course']
        dist = lumaview.motion.z_um2ustep(course)       # 10 um
        lumaview.motion.SendGram('MVP', 1, 'Z', dist)   # Move DOWN
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def set_position(self, pos):
        usteps = -lumaview.motion.z_um2ustep(float(pos))   # position on slider is in mm
        lumaview.motion.SendGram('MVP', 0, 'Z', usteps)  # Move to absolute position
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def set_bookmark(self):
        usteps = lumaview.motion.SendGram('GAP', 1, 'Z', 0)  # Get current z height in usteps
        height = -lumaview.motion.z_ustep2um(usteps)
        protocol['z_bookmark'] = height

    def goto_bookmark(self):
        height = protocol['z_bookmark']
        usteps = -lumaview.motion.z_um2ustep(height)
        lumaview.motion.SendGram('MVP', 0, 'Z', usteps)  # set current z height in usteps
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def home(self):
        lumaview.motion.zhome()
        self.ids['home_id'].text = 'Homing...'
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def update_gui(self, dt):
        set_value = lumaview.motion.SendGram('GAP', 0, 'Z', 10)  # Get target value
        get_value = lumaview.motion.SendGram('GAP', 1, 'Z', 0)   # Get current value

        z_home = lumaview.motion.SendGram('RFS', 2, 'Z', 0)
        if z_home != 0:
            self.ids['obj_position'].value = 0
            self.ids['set_position_id'].text = '0.00'
        else:
            set_pos = -lumaview.motion.z_ustep2um(set_value)
            self.ids['obj_position'].value = set_pos
            self.ids['set_position_id'].text = format(set_pos, '.2f')

        get_pos = -lumaview.motion.z_ustep2um(get_value)
        self.ids['get_position_id'].text = format(get_pos, '.2f')

        if set_value == get_value:
            Clock.unschedule(self.update_event)
            self.ids['home_id'].text = 'Home'
            self.ids['home_id'].state = 'normal'

    # User selected the autofocus function
    def autofocus(self):
        camera = lumaview.ids['viewer_id'].ids['microscope_camera']

        if camera == False:
            print("Error: VerticalControl.autofocus() self.camera == False")
            return

        # TODO Needs to be set by the user
        center = protocol['z_bookmark']
        range =  protocol['objective']['AF_range']
        fine =   protocol['objective']['AF_min']
        course = protocol['objective']['AF_max']

        self.z_min = -lumaview.motion.z_um2ustep(center-range)
        self.z_max = -lumaview.motion.z_um2ustep(center+range)
        self.z_step = -int(lumaview.motion.z_um2ustep(fine*2))

        dt = 1 # TODO change this based on focus and exposure time

        self.positions = [0]
        self.focus_measures = [0]

        if self.ids['autofocus_id'].state == 'down':
            self.ids['autofocus_id'].text = 'Focusing...'
            lumaview.motion.SendGram('MVP', 0, 'Z', self.z_min) # Go to z_min
            self.autofocus_event = Clock.schedule_interval(self.focus_iterate, dt)

    def focus_iterate(self, dt):
        camera = lumaview.ids['viewer_id'].ids['microscope_camera']
        image = camera.array

        #target = lumaview.motion.SendGram('GAP', 0, 'Z', 10)  # Get target value
        target = lumaview.motion.SendGram('GAP', 1, 'Z', 0)  # Get current value

        self.positions.append(target)
        self.focus_measures.append(self.focus_function(image))

        lumaview.motion.SendGram('MVP', 1, 'Z', self.z_step) # move by z_step

        if self.ids['autofocus_id'].state == 'normal':
            self.ids['autofocus_id'].text = 'Autofocus'
            Clock.unschedule(self.autofocus_event)
            print("autofocus cancelled")

        elif target <= self.z_max:
            self.ids['autofocus_id'].state = 'normal'
            self.ids['autofocus_id'].text = 'Autofocus'
            Clock.unschedule(self.autofocus_event)

            focus = self.focus_best(self.positions, self.focus_measures)
            # print(self.positions, '\t', self.focus_measures)
            print("Focus Position:", -lumaview.motion.z_ustep2um(focus))
            lumaview.motion.SendGram('MVP', 0, 'Z', focus) # move to absolute target

        self.update_gui(0)

    # Algorithms for estimating the quality of the focus
    def focus_function(self, image, algorithm = 'vollath4'):
        w = image.shape[0]
        h = image.shape[1]

        # Journal of Microscopy, Vol. 188, Pt 3, December 1997, pp. 264–272
        if algorithm == 'vollath4': # pg 266
            image = np.double(image)
            sum_one = np.sum(np.multiply(image[:w-1,:h], image[1:w,:h])) # g(i, j).g(i+1, j)
            sum_two = np.sum(np.multiply(image[:w-2,:h], image[2:w,:h])) # g(i, j).g(i+2, j)
            print(sum_one - sum_two)
            return sum_one - sum_two

        elif algorithm == 'skew':
            hist = np.histogram(image, bins=256,range=(0,256))
            hist = np.asarray(hist[0], dtype='int')
            max_index = hist.argmax()

            edges = np.histogram_bin_edges(image, bins=1)
            white_edge = edges[1]

            skew = white_edge-max_index
            print(skew)
            return skew

        elif algorithm == 'pixel_variation':
            sum = np.sum(image)
            ssq = np.sum(np.square(image))
            var = ssq*w*h-sum**2
            print('pixel_variation:', var)
            return var

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
            print(kernel)
            convolve = signal.convolve2d(image, kernel, mode='valid')
            sum = np.sum(convolve)
            print(sum)
            return sum

        else:
            return 0

    def focus_best(self, positions, values, algorithm='mov_avg'):
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

    # Image Sharpening using Richardson-Lucy deconvolution algorithm
    def Richardson_Lucy(self, image):
        # https://scikit-image.org/docs/dev/auto_examples/filters/
        # plot_deconvolution.html#sphx-glr-download-auto-examples-filters-plot-deconvolution-py
        pass

class XYStageControl(BoxLayout):

    def course_left(self):
        global lumaview
        lumaview.motion.SendGram('MVP', 1, 'X', -10000)  # Move LEFT relative by 10000

    def fine_left(self):
        global lumaview
        lumaview.motion.SendGram('MVP', 1, 'X', -1000)  # Move LEFT by 1000

    def fine_right(self):
        global lumaview
        lumaview.motion.SendGram('MVP', 1, 'X', 1000)  # Move RIGHT by 1000

    def course_right(self):
        global lumaview
        lumaview.motion.SendGram('MVP', 1, 'X', 10000)  # Move RIGHT by 10000

    def course_back(self):
        global lumaview
        lumaview.motion.SendGram('MVP', 1, 'Y', -10000)  # Move BACK relative by 10000

    def fine_back(self):
        global lumaview
        lumaview.motion.SendGram('MVP', 1, 'Y', -1000)  # Move BACK by 1000

    def fine_fwd(self):
        global lumaview
        lumaview.motion.SendGram('MVP', 1, 'Y', 1000)  # Move FORWARD by 1000

    def course_fwd(self):
        global lumaview
        lumaview.motion.SendGram('MVP', 1, 'Y', 10000)  # Move FORWARD by 10000

    def set_xposition(self, pos):
        global lumaview
        usteps = -lumaview.motion.xy_um2ustep(float(pos)*1000)   # position in text is in mm
        lumaview.motion.SendGram('MVP', 0, 'X', usteps)  # Move to absolute position
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def set_yposition(self, pos):
        global lumaview
        usteps = -lumaview.motion.xy_um2ustep(float(pos)*1000)   # position in text is in mm
        lumaview.motion.SendGram('MVP', 0, 'Y', usteps)  # Move to absolute position
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def set_xbookmark(self):
        global lumaview
        usteps = lumaview.motion.SendGram('GAP', 1, 'X', 0)  # Get current x position in usteps
        x_pos = -lumaview.motion.xy_ustep2um(usteps)
        protocol['x_bookmark'] = x_pos

    def goto_xbookmark(self):
        global lumaview
        x_pos = protocol['x_bookmark']
        usteps = -lumaview.motion.xy_um2ustep(x_pos)
        lumaview.motion.SendGram('MVP', 0, 'X', usteps)  # set current x position in usteps
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def set_ybookmark(self):
        global lumaview
        usteps = lumaview.motion.SendGram('GAP', 1, 'Y', 0)  # Get current y position in usteps
        y_pos = -lumaview.motion.z_ustep2um(usteps)
        protocol['y_bookmark'] = y_pos

    def goto_ybookmark(self):
        global lumaview
        y_pos = protocol['y_bookmark']
        usteps = -lumaview.motion.z_um2ustep(y_pos)
        lumaview.motion.SendGram('MVP', 0, 'Y', usteps)  # set current y position in usteps
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)

    def home(self):
        global lumaview
        lumaview.motion.xyhome()
        self.ids['home_id'].text = 'Homing...'
        self.update_event = Clock.schedule_interval(self.update_gui, 0.5)
        # self.ids['x_pos_id'].text = format(0, '.2f')
        # self.ids['y_pos_id'].text = format(0, '.2f')

    def update_gui(self, dt):
        global lumaview
        x_target = lumaview.motion.SendGram('GAP', 0, 'X', 10)  # Get target value
        y_target = lumaview.motion.SendGram('GAP', 0, 'Y', 10)  # Get target value

        x_value = lumaview.motion.SendGram('GAP', 1, 'X', 0)  # Get current value
        y_value = lumaview.motion.SendGram('GAP', 1, 'Y', 0)  # Get current value

        x_home = lumaview.motion.SendGram('RFS', 2, 'X', 0)
        y_home = lumaview.motion.SendGram('RFS', 2, 'Y', 0)

        if (x_home != 0) or (y_home != 0):
            self.ids['x_pos_id'].text = '0.00'
            self.ids['y_pos_id'].text = '0.00'
        else:
            x_pos = -lumaview.motion.xy_ustep2um(x_target)
            y_pos = -lumaview.motion.xy_ustep2um(y_target)
            self.ids['x_pos_id'].text = format(x_pos/1000, '.2f')
            self.ids['y_pos_id'].text = format(y_pos/1000, '.2f')

        if (x_target == x_value) and (y_target == y_value):
            Clock.unschedule(self.update_event)
            self.ids['home_id'].text = 'Home'
            self.ids['home_id'].state = 'normal'

class MicroscopeSettings(BoxLayout):

    def frame_size(self):
        global lumaview
        global protocol

        w = int(self.ids['frame_width'].text)
        h = int(self.ids['frame_height'].text)

        camera = lumaview.ids['viewer_id'].ids['microscope_camera'].camera

        width = int(min(int(w), camera.Width.Max)/2)*2
        height = int(min(int(h), camera.Height.Max)/2)*2

        protocol['frame_width'] = width
        protocol['frame_height'] = height

        self.ids['frame_width'].text = str(width)
        self.ids['frame_height'].text = str(height)

        lumaview.ids['viewer_id'].ids['microscope_camera'].frame_size(width, height)

# Pass-through class for microscope selection drop-down menu, defined in .kv file
# -------------------------------------------------------------------------------
class MicroscopeDropDown(DropDown):
    pass


# First line of Microscope Settings control panel, to select model of microscope
# ------------------------------------------------------------------------------
class ScopeSelect(BoxLayout):
    # The text displayed in the button
    scope_str = StringProperty(protocol['microscope'])

    def __init__(self, **kwargs):
        super(ScopeSelect, self).__init__(**kwargs)

        # Create label and button here so DropDown menu works properly
        self.mainlabel = Label(text = 'Lumascope Model',
                               size_hint_x = None, width = '150dp', font_size = '12sp')
        self.mainbutton = Button(text = self.scope_str,
                                 size_hint_y = None, height = '30dp', font_size = '12sp')

        # Group widgets together
        self.dropdown = MicroscopeDropDown()
        self.add_widget(self.mainlabel)
        self.add_widget(self.mainbutton)

        # Add actions - mainbutton opens dropdown menu
        self.mainbutton.bind(on_release = self.dropdown.open)

        # Dropdown buttons do stuff based on their text through microscope_selectFN
        self.dropdown.bind(on_select = lambda instance,
                                       scope: setattr(self.mainbutton, 'text', scope))
        self.dropdown.bind(on_select = self.microscope_selectFN)


    def microscope_selectFN(self, instance, scope):
        global protocol
        self.scope_str = scope
        self.parent.ids['image_of_microscope'].source = './data/scopes/'+scope+'.png'
        protocol['microscope'] = scope
        print("Selected microscope: {0}" . format(scope))


# Pass-through class for objective settings drop down menu, defined in .kv file
# ----------------------------------------------------------------------------
class ObjectiveDropDown(DropDown):
    objectives = ObjectProperty()

    def __init__(self, **kwargs):
        super(ObjectiveDropDown, self).__init__(**kwargs)

        with open('./data/objectives.json', "r") as read_file:
            self.objectives = json.load(read_file)

        for obj in self.objectives:
            button = Button(text = obj,
                            size_hint_y = None, height = '30dp', font_size = '12sp')
            self.add_widget(button)
            button.bind(on_release = self.objective_select)

    def objective_select(self, instance):
        global lumaview
        global protocol

        print(instance.text)
        protocol['objective'] = self.objectives[instance.text]
        protocol['objective']['ID'] = instance.text
        microscope_settings_id = lumaview.ids['mainsettings_id'].ids['microscope_settings_id']
        microscope_settings_id.ids['select_obj_id'].mainbutton.text = protocol['objective']['ID']
        microscope_settings_id.ids['magnification_id'].text = str(protocol['objective']['magnification'])
        self.dismiss()

# Second line of Microscope Settings control panel, to select objective
# ---------------------------------------------------------------------
class ObjectiveSelect(BoxLayout):
    global protocol

    def __init__(self, **kwargs):
        super(ObjectiveSelect, self).__init__(**kwargs)

        # Create label and button here so DropDown menu works properly
        self.mainlabel = Label(text = 'Objective',
                               size_hint_x = None, width = '150dp', font_size = '12sp')
        self.mainbutton = Button(text = str(protocol['objective']['ID']),
                                 size_hint_y = None, height = '30dp', font_size = '12sp')

        # Group widgets together
        self.dropdown = ObjectiveDropDown()
        self.add_widget(self.mainlabel)
        self.add_widget(self.mainbutton)

        # Add actions - mainbutton opens dropdown menu
        self.mainbutton.bind(on_release = self.dropdown.open)

# Modified Slider Class to enable on_release event
# ---------------------------------------------------------------------
class ModSlider(Slider):
    def __init__(self, **kwargs):
        self.register_event_type('on_release')
        super(ModSlider, self).__init__(**kwargs)

    def on_release(self):
        pass

    def on_touch_up(self, touch):
        super(ModSlider, self).on_touch_up(touch)
        if touch.grab_current == self:
            self.dispatch('on_release')
            return True

# LayerControl Layout class
# ---------------------------------------------------------------------
class LayerControl(BoxLayout):
    layer = StringProperty(None)
    bg_color = ObjectProperty(None)
    global protocol

    def __init__(self, **kwargs):
        super(LayerControl, self).__init__(**kwargs)
        if self.bg_color is None:
            self.bg_color = (0.5, 0.5, 0.5, 0.5)

    def ill_slider(self):
        illumination = self.ids['ill_slider'].value
        protocol[self.layer]['ill'] = illumination
        self.apply_settings()

    def ill_text(self):
        try:
            illumination = float(self.ids['ill_text'].text)
            protocol[self.layer]['ill'] = illumination
            self.ids['ill_slider'].value = illumination
            self.apply_settings()
        except:
            print('Illumination value is not in acceptable range of 0 to 600 mA.')

    def gain_auto(self):
        if self.ids['gain_auto'].state == 'down':
            state = True
        else:
            state = False
        protocol[self.layer]['gain_auto'] = state
        self.apply_settings()

    def gain_slider(self):
        gain = self.ids['gain_slider'].value
        protocol[self.layer]['gain'] = gain
        self.apply_settings()

    def gain_text(self):
        try:
            gain = float(self.ids['gain_text'].text)
            protocol[self.layer]['gain'] = gain
            self.ids['gain_slider'].value = gain
            self.apply_settings()
        except:
            print('Gain value is not in acceptable range of 0 to 24 dB.')

    def exp_slider(self):
        exposure = self.ids['exp_slider'].value
        # exposure = 10 ** self.ids['exp_slider'].value # slider is log_10(ms)
        protocol[self.layer]['exp'] = exposure        # protocol is ms
        self.apply_settings()

    def exp_text(self):
        try:
            exposure = float(self.ids['exp_text'].text)
            protocol[self.layer]['exp'] = exposure
            self.ids['exp_slider'].value = exposure
            # self.ids['exp_slider'].value = float(np.log10(exposure)) # convert slider to log_10
            self.apply_settings()
        except:
            print('Exposure value is not in acceptable range of 0.01 to 1000ms.')

    def choose_folder(self):
        content = LoadDialog(load=self.load,
                             cancel=self.dismiss_popup,
                             path=protocol[self.layer]['save_folder'])
        self._popup = Popup(title="Select Save Folder",
                            content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path):
        protocol[self.layer]['save_folder'] = path
        if len(path) > 30:
            self.ids['folder_btn'].text = '... '+path[-30:]
        else:
            self.ids['folder_btn'].text = path
        self.dismiss_popup()

    def dismiss_popup(self):
        self._popup.dismiss()

    def root_text(self):
        protocol[self.layer]['file_root'] = self.ids['root_text'].text

    def false_color(self):
        protocol[self.layer]['false_color'] = self.ids['false_color'].active
        self.apply_settings()

    def update_acquire(self):
        protocol[self.layer]['acquire'] = self.ids['acquire'].active

    def save_focus(self):
        global lumaview
        usteps = lumaview.motion.SendGram('GAP', 1, 'Z', 0)  # Get current z height in usteps
        height = -lumaview.motion.z_ustep2um(usteps)
        protocol[self.layer]['focus'] = height

    def goto_focus(self):
        global lumaview
        height = protocol[self.layer]['focus']
        usteps = -lumaview.motion.z_um2ustep(height)
        lumaview.motion.SendGram('MVP', 0, 'Z', usteps)  # set current z height in usteps
        control = lumaview.ids['motionsettings_id'].ids['verticalcontrol_id']
        control.update_event = Clock.schedule_interval(control.update_gui, 0.5)


    def apply_settings(self):
        global lumaview
        # global gain_vals

        # update illumination to currently selected settings
        # -----------------------------------------------------
        illumination = protocol[self.layer]['ill']
        led_board = lumaview.led_board
        if self.ids['apply_btn'].state == 'down': # if the button is down
            # In active channel,turn on LED
            led_board.led_on(led_board.color2ch(self.layer), illumination)
            #  turn the state of remaining channels to 'normal' and text to 'OFF'
            layers = ['BF', 'Blue', 'Green', 'Red']
            for layer in layers:
                if(layer != self.layer):
                    lumaview.ids['mainsettings_id'].ids[layer].ids['apply_btn'].state = 'normal'

        else: # if the button is 'normal' meaning not active
            # In active channel, and turn off LED
            led_board.led_off()

        # update gain to currently selected settings
        # -----------------------------------------------------
        state = protocol[self.layer]['gain_auto']
        lumaview.ids['viewer_id'].ids['microscope_camera'].auto_gain(state)

        if not(state):
            gain = protocol[self.layer]['gain']
            lumaview.ids['viewer_id'].ids['microscope_camera'].gain(gain)

        # update exposure to currently selected settings
        # -----------------------------------------------------
        exposure = protocol[self.layer]['exp']
        lumaview.ids['viewer_id'].ids['microscope_camera'].exposure_t(exposure)

        # choose correct active toggle button image based on color
        # -----------------------------------------------------
        if self.ids['apply_btn'].state == 'down':
            if(self.layer) == 'Red':
                self.ids['apply_btn'].background_down = './data/icons/ToggleRR.png'
            elif(self.layer) == 'Green':
                self.ids['apply_btn'].background_down = './data/icons/ToggleRG.png'
            elif(self.layer) == 'Blue':
                self.ids['apply_btn'].background_down = './data/icons/ToggleRB.png'
        else:
            self.ids['apply_btn'].background_down = './data/icons/ToggleR.png'

        # Remove 'Colorize' option in brightfield control
        # -----------------------------------------------------
        if self.layer == 'BF':
            self.ids['false_color_label'].text = ''
            self.ids['false_color'].color = (0., )*4

        # update false color to currently selected settings and shader
        # -----------------------------------------------------
        for i in np.arange(0.1, 3, 0.1):
            Clock.schedule_once(self.update_shader, i)

    def update_shader(self, dt):
        if self.ids['false_color'].active:
            lumaview.ids['viewer_id'].update_shader(self.layer)
        else:
            lumaview.ids['viewer_id'].update_shader('none')


class TimeLapseSettings(BoxLayout):
    record = ObjectProperty(None)
    record = False
    movie_folder = StringProperty(None)
    n_captures = ObjectProperty(None)

    def update_period(self):
        # if self.ids['capture_period'].text.isnumeric(): # Did not allow for floating point numbers
        try:
            protocol['period'] = float(self.ids['capture_period'].text)
        except:
            print('Update Period is not an acceptable value')

    def update_duration(self):
        # if self.ids['capture_dur'].text.isnumeric():  # Did not allow for floating point numbers
        try:
            protocol['duration'] = float(self.ids['capture_dur'].text)
        except:
            print('Update Duration is not an acceptable value')

    # load protocol from JSON file
    def load_protocol(self, file="./data/protocol.json"):
        global lumaview

        # load protocol JSON file
        with open(file, "r") as read_file:
            global protocol
            protocol = json.load(read_file)
            # update GUI values from JSON data:
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['select_scope_id'].scope_str = protocol['microscope']
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['select_obj_id'].mainbutton.text = protocol['objective']['ID']
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['magnification_id'].text = str(protocol['objective']['magnification'])
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['frame_width'].text = str(protocol['frame_width'])
            lumaview.ids['mainsettings_id'].ids['microscope_settings_id'].ids['frame_height'].text = str(protocol['frame_height'])

            self.ids['capture_period'].text = str(protocol['period'])
            self.ids['capture_dur'].text = str(protocol['duration'])

            layers = ['BF', 'Blue', 'Green', 'Red']
            for layer in layers:
                lumaview.ids['mainsettings_id'].ids[layer].ids['ill_slider'].value = protocol[layer]['ill']
                lumaview.ids['mainsettings_id'].ids[layer].ids['gain_slider'].value = protocol[layer]['gain']
                lumaview.ids['mainsettings_id'].ids[layer].ids['exp_slider'].value = protocol[layer]['exp']
                # lumaview.ids['mainsettings_id'].ids[layer].ids['exp_slider'].value = float(np.log10(protocol[layer]['exp']))
                if len(protocol[layer]['save_folder']) > 30:
                    lumaview.ids['mainsettings_id'].ids[layer].ids['folder_btn'].text = '... ' + protocol[layer]['save_folder'][-30:]
                else:
                    lumaview.ids['mainsettings_id'].ids[layer].ids['folder_btn'].text = protocol[layer]['save_folder'][-30:]
                lumaview.ids['mainsettings_id'].ids[layer].ids['root_text'].text = protocol[layer]['file_root']
                lumaview.ids['mainsettings_id'].ids[layer].ids['false_color'].active = protocol[layer]['false_color']
                lumaview.ids['mainsettings_id'].ids[layer].ids['acquire'].active = protocol[layer]['acquire']

            lumaview.ids['viewer_id'].ids['microscope_camera'].frame_size(protocol['frame_width'], protocol['frame_height'])

    # Save protocol to JSON file
    def save_protocol(self, file="./data/protocol.json"):
        global protocol
        with open(file, "w") as write_file:
            json.dump(protocol, write_file, indent = 4)

    # Run the timed process of capture event
    def run_protocol(self):
        global protocol

        # number of capture events remaining
        ## duration is in hours, period is in minutes
        self.n_captures = int(float(protocol['duration'])*60 / float(protocol['period']))

        # update protocol
        if self.record == False:
            self.record = True

            hrs = np.floor(self.n_captures * protocol['period']/60)
            minutes = np.floor((self.n_captures*protocol['period']/60-hrs)*60)
            hrs = '%02d' % hrs
            minutes = '%02d' % minutes
            self.ids['protocol_btn'].text = hrs+':'+minutes+' remaining'

            self.dt = protocol['period']*60 # frame events are measured in seconds
            self.frame_event = Clock.schedule_interval(self.capture, self.dt)
        else:
            self.record = False
            self.ids['protocol_btn'].text = 'Run Protocol'

            if self.frame_event:
                Clock.unschedule(self.frame_event)

    # One procotol capture event
    def capture(self, dt):
        global lumaview
        camera = lumaview.ids['viewer_id'].ids['microscope_camera']
        if camera == False:
            return
        try:
            self.n_captures = self.n_captures-1
        except:
            print('Capturing a Single Composite Image')

        layers = ['BF', 'Blue', 'Green', 'Red']
        for layer in layers:
            if protocol[layer]['acquire'] == True:
                # global lumaview

                # set the gain and exposure
                gain = protocol[layer]['gain']
                camera.gain(gain)
                exposure = protocol[layer]['exp']
                camera.exposure_t(exposure)
                camera.update(0)

                # turn on the LED
                # update illumination to currently selected settings
                illumination = protocol[layer]['ill']
                led_board = lumaview.led_board
                led_board.led_on(led_board.color2ch(layer), illumination)

                # capture the image
                save_folder = protocol[layer]['save_folder']
                file_root = protocol[layer]['file_root']
                lumaview.capture(0, save_folder, file_root, color = layer)
                # turn off the LED
                led_board.led_off()
            lumaview.ids['mainsettings_id'].ids[layer].ids['apply_btn'].state = 'normal'

    # def convert_to_avi(self):
    #
    #     # self.choose_folder()
    #     save_location = './capture/movie.avi'
    #
    #     img_array = []
    #     for filename in glob.glob('./capture/*.tiff'):
    #         img = cv2.imread(filename)
    #         height, width, layers = img.shape
    #         size = (width,height)
    #         img_array.append(img)
    #
    #     out = cv2.VideoWriter(save_location,cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
    #
    #     for i in range(len(img_array)):
    #         out.write(img_array[i])
    #     out.release()

    def choose_folder(self):
        content = LoadDialog(load=self.load,
                             cancel=self.dismiss_popup,
                             path='./data/')
        self._popup = Popup(title="Select Protocol",
                            content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path):
        # protocol[self.layer]['save_folder'] = path
        # if len(path) > 30:
        #     self.ids['folder_btn'].text = '... '+path[-30:]
        # else:
        #     self.ids['folder_btn'].text = path
        self.load_protocol()
        self.dismiss_popup()

    def dismiss_popup(self):
        self._popup.dismiss()

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    path = ObjectProperty(None)


# ------------------------------------------------------------------------
# TOOLTIPS
# ------------------------------------------------------------------------
class Tooltip(HoverBehavior, TouchBehavior):
    # Tooltip background color
    tooltip_bg_color = ColorProperty(None)

    # Tooltip text color
    tooltip_text_color = ColorProperty(None)

    # Tooltip text to display
    tooltip_text = StringProperty(None)

    # Radius of rounded rectangle's corner
    tooltip_radius = ListProperty([dp(5),])

    # Tooltip display delay, defaults to 0 and max of 4
    tooltip_display_delay = BoundedNumericProperty(0, min=0, max=4)

    # Y-offset of tooltip text, defaults to 0
    shift_y = NumericProperty(0)

    _tooltip = None


    def delete_clock(self, widget, touch, *args):
        if self.collide_point(touch.x, touch.y) and touch.grab_current:
            try:
                Clock.unschedule(touch.ud["event"])
            except KeyError:
                pass
            self.on_leave()

    # Returns the coordinates of the tooltio that fit in screen borders
    def adjust_tooltip_position(self, x, y):
        # If tooltip position is outside the right border of the screen:
        if x + self._tooltip.width > Window.width:
            x = Window.width - (self._tooltip.width + dp(10))
        elif x < 0:
            # If the tooltip position is outside the left boder of the screen
            x = '10dp'

        # If the tooltip position is below the bottom border:
        if y < 0:
            y = dp(10)
        elif Window.height - self._tooltip.height < y:
            y - Window.height - (self._tooltip.height + dp(10))

        return x, y

    # Display the tooltip using an animated routine after a display delay defined by user
    def display_tooltip(self, interval):
        if not self._tooltip:
            return

        Window.add_widget(self._tooltip)
        pos = self.to_window(self.center_x, self.center_y)
        x = pos[0] - self._tooltip.width / 2

        if not self.shift_y:
            y = pos[1] - self._tooltip.height / 2 - self.height / 2 - dp(20)
        else:
            y = pos[1] - self._tooltip.height / 2 - self.height + self.shift_y

        x, y = self.adjust_tooltip_position(x, y)
        self._tooltip.pos = (x, y)

        Clock.schedule_once(self.animation_tooltip_show, self.tooltip_display_delay)

    # Method that displays tooltip in an animated way
    def animation_tooltip_show(self, interval):
        if not self._tooltip:
            return

        (Animation(_scale_x = 1, _scale_y = 1, d = 0.1)
                + Animation(opacity = 1, d = 0.2)).start(self._tooltip)

    # Makes tooltip disappear
    def remove_tooltip(self, *args):
        Window.remove_widget(self._tooltip)

    def on_long_touch(self, touch, *args):
        return

    def on_enter(self, *args):
        if not self.tooltip_text:
            return

        self._tooltip = TooltipViewClass(
                tooltip_bg_color = self.tooltip_bg_color,
                tooltip_text_color = self.tooltip_text_color,
                tooltip_text = self.tooltip_text,
                tooltip_radius = self.tooltip_radius)
        Clock.schedule_once(self.display_tooltip, -1)

    def on_leave(self):
        if self._tooltip:
            Window.remove_widget(self._tooltip)
            self._tooltip = None


# Holder layout for the tooltip
class TooltipViewClass(BoxLayout):
    tooltip_bg_color = ColorProperty(None)
    tooltip_text_color = ColorProperty(None)
    tooltip_text = StringProperty()
    tooltip_radius = ListProperty()

    _scale_x = NumericProperty(0)
    _scale_y = NumericProperty(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.padding = [dp(8), dp(4), dp(8), dp(4)]


# Button-derivative with tooltips
class TooltipButton(Button, Tooltip):
    def __init__(self, **kwargs):
        super(TooltipButton, self).__init__(**kwargs)
        self.tooltip_bg_color = (0.1, 0.1, 0.1, 0.7)
        self.tooltip_display_delay = 0.
        self.shift_y = dp(80)


# Toggle Button-derivative with tooltips
class TooltipToggleButton(ToggleButton, Tooltip):
    def __init__(self, **kwargs):
        super(TooltipToggleButton, self).__init__(**kwargs)
        self.tooltip_bg_color = (0.1, 0.1, 0.1, 0.7)
        self.tooltip_display_delay = 0.
        self.shift_y = dp(80)


# -------------------------------------------------------------------------
# RUN LUMAVIEWPRO APP
# -------------------------------------------------------------------------
class LumaViewProApp(App):
    def build(self):
        Window.size = (1280, 800)
        self.icon = './data/icons/icon32x.png'
        global lumaview
        lumaview = MainDisplay()
        # lumaview.ids['motionsettings_id'].ids['verticalcontrol_id'].ids['obj_position'].value
        lumaview.ids['mainsettings_id'].ids['time_lapse_id'].load_protocol("./data/protocol.json")
        lumaview.ids['mainsettings_id'].ids['BF'].apply_settings()
        lumaview.led_board.led_off()
        # Window.minimum_width = 800
        # Window.minimum_height = 600
        return lumaview

    def on_stop(self):
        global lumaview
        lumaview.led_board.led_off()
        lumaview.ids['mainsettings_id'].ids['time_lapse_id'].save_protocol("./data/protocol.json")

LumaViewProApp().run()
