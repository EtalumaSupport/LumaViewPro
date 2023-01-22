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

MODIFIED:
January 22, 2023
'''

import numpy as np
from pypylon import pylon

class PylonCamera:

    def __init__(self, **kwargs):
        self.active = False
        self.error_report_count = 0
        self.message = 'PylonCamera.__init__()'
        self.array = np.array([])
        self.connect()

    def __delete__(self):
        try:
            self.active.close()
        except:
            print('exception')

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
            # self.active.PixelFormat.SetValue('PixelFormat_Mono12')
            self.active.GainAuto.SetValue('Off')
            self.active.ExposureAuto.SetValue('Off')
            self.active.ReverseX.SetValue(True)
            # Grabbing Continuously (video) with minimal delay
            self.active.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.error_report_count = 0
            self.message = 'PylonCamera.connect() succeeded'            

        except:
            self.active = False
            if (self.error_report_count < 6):
                print('Error: Cannot connect to camera')
            self.error_report_count += 1
            self.message = 'PylonCamera.connect() failed'            

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
            self.message = 'PylonCamera.grab() succeeded'            
            return True

        except:
            if self.error_report_count < 6:
                print('Error: Cannot grab texture from camera')
            self.error_report_count += 1
            self.active = False
            self.message = 'PylonCamera.grab() failed'            
            return False

    def frame_size(self, w, h):
        """ Set camera frame size to w by h and keep centered """

        print(self.active)
        if self.active != False:

            width = int(min(int(w), self.active.Width.Max)/4)*4
            height = int(min(int(h), self.active.Height.Max)/4)*4

            self.active.StopGrabbing()
            self.active.Width.SetValue(width)
            self.active.Height.SetValue(height)
            self.active.BslCenterX.Execute()
            self.active.BslCenterY.Execute()
            self.active.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.message = 'PylonCamera.frame_size('+str(w)+','+str(h)+')'+'; succeeded' 
        else:
            self.message = 'PylonCamera.frame_size('+str(w)+','+str(h)+')'+'; inactive' 


    def gain(self, gain):
        """ Set gain value in the camera hardware"""

        print(self.active)
        if self.active != False:
            self.active.Gain.SetValue(gain)
            self.message = 'PylonCamera.gain('+str(gain)+')'+': succeeded' 
        else:
            self.message = 'PylonCamera.gain('+str(gain)+')'+': inactive camera' 

    def auto_gain(self, state = True):
        """ Enable / Disable camera auto_gain with the value of 'state'
        It will be continueously updating based on the current image """

        print(self.active)
        if self.active != False:
            if state == True:
                self.active.GainAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
            else:
                self.active.GainAuto.SetValue('Off')
            self.message = 'PylonCamera.auto_gain('+str(state)+')'+': succeeded' 
        else:
            self.message = 'PylonCamera.auto_gain('+str(state)+')'+': inactive camera' 
            
    def exposure_t(self, t):
        """ Set exposure time in the camera hardware t (msec)"""

        if self.active != False:
            # (t*1000) in microseconds; therefore t  in milliseconds
            self.active.ExposureTime.SetValue(max(t*1000, self.active.ExposureTime.Min))
            self.message = 'PylonCamera.exposure_t('+str(t)+')'+': succeeded' 
        else:
            self.message = 'PylonCamera.exposure_t('+str(t)+')'+': inactive camera' 

    def get_exposure_t(self):
        """ Set exposure time in the camera hardware
         Returns t (msec), or -1 if the camera is inactive"""

        if self.active != False:
            microsec = self.active.ExposureTime.GetValue() # get current exposure time in microsec
            millisec = microsec/1000 # convert exposure time to millisec
            self.message = 'PylonCamera.get_exposure_t(): succeeded' 
            return millisec
        else:
            self.message = 'PylonCamera.get_exposure_t(): inactive camera' 
            return -1

    def auto_exposure_t(self, state = True):
        """ Enable / Disable camera auto_exposure with the value of 'state'
        It will be continueously updating based on the current image """

        if self.active != False:
            if state == True:
                self.active.ExposureAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
            else:
                self.active.ExposureAuto.SetValue('Off')
            self.message = 'PylonCamera.auto_exposure_t('+str(state)+')'+': succeeded' 
        else:
            self.message = 'PylonCamera.auto_exposure_t('+str(state)+')'+': inactive camera' 

