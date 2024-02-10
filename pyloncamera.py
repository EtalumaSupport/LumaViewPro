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
March 20, 2023
'''

import numpy as np
from pypylon import pylon
from lvp_logger import logger

class PylonCamera:

    def __init__(self, **kwargs):
        logger.info('[CAM Class ] PylonCamera.__init__()')
        self.active = False
        self.error_report_count = 0
        self.array = np.array([])
        self.connect()

    def __delete__(self):
        try:
            self.active.close()
        except:
            logger.exception('[CAM Class ] exception')

    def connect(self):
        """ Try to connect to the first available basler camera"""
        try:
            # Create an instant active object with the camera device found first.
            self.active = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.active.Open()
            self.active.Width.SetValue(self.active.Width.Max)
            self.active.Height.SetValue(self.active.Height.Max)
            self.active.BslCenterX.Execute()
            self.active.BslCenterY.Execute()
            self.active.PixelFormat.SetValue('Mono8')
            self.active.GainAuto.SetValue('Off')
            self.active.ExposureAuto.SetValue('Off')
            self.active.ReverseX.SetValue(True)
            self.init_auto_gain_focus()
            # Grabbing Continuously (video) with minimal delay
            self.active.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.error_report_count = 0
            logger.info('[CAM Class ] PylonCamera.connect() succeeded)')

        except:
            self.active = False
            if (self.error_report_count < 6):
                logger.exception('[CAM Class ] PylonCamera.connect() failed')
            self.error_report_count += 1


    def set_pixel_format(self, pixel_format: str):

        if pixel_format not in self.get_supported_pixel_formats():
            logger.exception(f"[CAM Class ] Unsupported pixel format: {pixel_format}")
        
        self.active.StopGrabbing()
        self.active.PixelFormat.SetValue(pixel_format)
        self.active.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
 

    def get_pixel_format(self) -> str:
        return self.active.PixelFormat.GetValue()


    def get_supported_pixel_formats(self) -> tuple:
        return self.active.PixelFormat.GetSymbolics()
    

    def init_auto_gain_focus(self, auto_target_brightness: float=0.5):
        # margin_px = 8
        # self.active.AutoFunctionROIOffsetX.SetValue(margin_px)
        # self.active.AutoFunctionROIOffsetY.SetValue(margin_px)
        self.active.AutoFunctionROIWidth.SetValue(self.active.Width.Max - 2*self.active.AutoFunctionROIOffsetX.GetValue())
        self.active.AutoFunctionROIHeight.SetValue(self.active.Height.Max - 2*self.active.AutoFunctionROIOffsetY.GetValue())
        self.active.AutoFunctionROIUseBrightness = True
        self.active.AutoTargetBrightness.SetValue(auto_target_brightness)
        self.active.AutoFunctionROISelector.SetValue('ROI1')
        self.active.AutoGainLowerLimit.SetValue(self.active.AutoGainLowerLimit.Min)
        self.active.AutoGainUpperLimit.SetValue(self.active.AutoGainUpperLimit.Max)
        self.active.AutoFunctionProfile.SetValue('MinimizeGain')

        # self.set_test_pattern('Testimage2')


    def update_auto_gain_target_brightness(self, auto_target_brightness: float):
        self.active.AutoTargetBrightness.SetValue(auto_target_brightness)


    def grab(self):
        """ Grab last available frame from camera and save it to self.array
        returns True if successful
        returns False if unsuccessful
        access the image using camera.array where camera is the instance of the class"""

        if self.active == False:
            self.connect()
        try:
            if self.active.IsGrabbing():
                grabResult = self.active.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)

                if grabResult.GrabSucceeded():
                    self.array = grabResult.GetArray()

            grabResult.Release()
            self.error_report_count = 0
            # logger.info('[CAM Class ] PylonCamera.grab() succeeded')
            return True

        except:
            if self.error_report_count < 6:
                logger.exception('[CAM Class ] PylonCamera.grab() failed')
            self.error_report_count += 1
            self.active = False
            return False

    def frame_size(self, w, h):
        """ Set camera frame size to w by h and keep centered """

        if self.active != False:

            width = int(min(int(w), self.active.Width.Max)/4)*4
            height = int(min(int(h), self.active.Height.Max)/4)*4

            self.active.StopGrabbing()
            self.active.Width.SetValue(width)
            self.active.Height.SetValue(height)
            self.active.BslCenterX.Execute()
            self.active.BslCenterY.Execute()
            self.active.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            logger.info('[CAM Class ] PylonCamera.frame_size('+str(w)+','+str(h)+')'+'; succeeded')
        else:
            logger.warning('[CAM Class ] PylonCamera.frame_size('+str(w)+','+str(h)+')'+'; inactive')


    def gain(self, gain):
        """ Set gain value in the camera hardware"""

        if self.active != False:
            self.active.Gain.SetValue(gain)
            logger.info('[CAM Class ] PylonCamera.gain('+str(gain)+')'+': succeeded')
        else:
            logger.warning('[CAM Class ] PylonCamera.gain('+str(gain)+')'+': inactive camera')

    def auto_gain(self, state = True, target_brightness: float = 0.5):
        """ Enable / Disable camera auto_gain with the value of 'state'
        It will be continueously updating based on the current image """

        if self.active == False:
            logger.warning('[CAM Class ] PylonCamera.auto_gain('+str(state)+')'+': inactive camera')
            return

        if state == True:
            self.update_auto_gain_target_brightness(auto_target_brightness=target_brightness)
            self.active.GainAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
            self.active.ExposureAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
        else:
            self.active.GainAuto.SetValue('Off')
            self.active.ExposureAuto.SetValue('Off')
        logger.info('[CAM Class ] PylonCamera.auto_gain('+str(state)+')'+': succeeded')
            
            
    def exposure_t(self, t):
        """ Set exposure time in the camera hardware t (msec)"""

        if self.active != False:
            # (t*1000) in microseconds; therefore t  in milliseconds
            self.active.ExposureTime.SetValue(max(t*1000, self.active.ExposureTime.Min))
            logger.info('[CAM Class ] PylonCamera.exposure_t('+str(t)+')'+': succeeded')
        else:
            logger.warning('[CAM Class ] PylonCamera.exposure_t('+str(t)+')'+': inactive camera')

    def get_exposure_t(self):
        """ Set exposure time in the camera hardware
         Returns t (msec), or -1 if the camera is inactive"""

        if self.active != False:
            microsec = self.active.ExposureTime.GetValue() # get current exposure time in microsec
            millisec = microsec/1000 # convert exposure time to millisec
            logger.info('[CAM Class ] PylonCamera.get_exposure_t(): succeeded')
            return millisec
        else:
            logger.warning('[CAM Class ] PylonCamera.get_exposure_t(): inactive camera')
            return -1

    def auto_exposure_t(self, state = True):
        """ Enable / Disable camera auto_exposure with the value of 'state'
        It will be continueously updating based on the current image """

        if self.active != False:
            if state == True:
                self.active.ExposureAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
            else:
                self.active.ExposureAuto.SetValue('Off')
            logger.info('[CAM Class ] PylonCamera.auto_exposure_t('+str(state)+')'+': succeeded')
        else:
            logger.warning('[CAM Class ] PylonCamera.auto_exposure_t('+str(state)+')'+': inactive camera')


    def set_test_pattern(self, enabled: bool = False, pattern: str = 'Black'):
        if not self.active:
            return
        
        #if not enabled:
        #    self.active # TODO
        
        self.active.TestPattern.SetValue(pattern)
        self.grab()
