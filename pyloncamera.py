'''
MIT License

Copyright (c) 2024 Etaluma, Inc.

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
'''

import contextlib
import datetime
import os

import numpy as np
from pypylon import pylon, genicam
from lvp_logger import logger

class PylonCamera:

    def __init__(self, **kwargs):
        logger.info('[CAM Class ] PylonCamera.__init__()')
        self.active = False
        self.error_report_count = 0
        self.array = np.array([])
        self.cam_image_handler = None

        if os.getenv("PYLON_CAMEMU", None) != None:
            logger.info('[CAM Class ] PylonCamera.connect() detected request to use camera emulation')
            self._use_camera_emulation = True
        else:
            self._use_camera_emulation = False

        self.connect()

    def __delete__(self):
        try:
            self.active.close()
        except:
            logger.exception('[CAM Class ] exception')

    @contextlib.contextmanager
    def update_camera_config(self):
        camera = self.active
        was_grabbing = camera.IsGrabbing()

        if was_grabbing:
            self.stop_grabbing()

        yield

        if was_grabbing:
            self.start_grabbing()


    def stop_grabbing(self):
        camera = self.active
        camera.StopGrabbing()


    def start_grabbing(self):
        camera = self.active
        camera.StartGrabbing(
            pylon.GrabStrategy_LatestImageOnly,
            pylon.GrabLoop_ProvidedByInstantCamera
        )

    def connect(self):
        """ Try to connect to the first available basler camera"""
        try:
            p_device = pylon.TlFactory.GetInstance().CreateFirstDevice()
            self.active = pylon.InstantCamera(p_device)
            camera = self.active
            camera.RegisterConfiguration(
                pylon.AcquireContinuousConfiguration(),
                pylon.RegistrationMode_ReplaceAll,
                pylon.Cleanup_Delete
            )
            # camera.RegisterConfiguration(
            #     ConfigurationEventPrinter(),
            #     pylon.RegistrationMode_Append,
            #     pylon.Cleanup_Delete
            # )

            self.cam_image_handler = ImageHandler()
            camera.RegisterImageEventHandler(
                self.cam_image_handler,
                pylon.RegistrationMode_Append,
                pylon.Cleanup_Delete
            )

            camera.Open()
            self.init_camera_config()
            self.start_grabbing()

            self.error_report_count = 0
            logger.info('[CAM Class ] PylonCamera.connect() succeeded')
        
        except genicam.RuntimeException as ex:
            # Handles when the device is already open in another application
            logger.error(f'[CAM Class ] PylonCamera.connect() failed -> {ex}')
            self.active = False
            self.error_report_count += 1
        except:
            logger.exception('[CAM Class ] PylonCamera.connect() failed')
            self.active = False
            self.error_report_count += 1
    
    
    def init_camera_config(self):
        camera = self.active
        if camera == False:
            return
        
        with self.update_camera_config():
            camera.UserSetSelector = "Default"
            camera.UserSetLoad.Execute()
            self.set_pixel_format(pixel_format='Mono8')
            self.auto_gain(state=False)
            camera.ReverseX.SetValue(True)
            if not self._use_camera_emulation:
                self.init_auto_gain_focus()
            self.exposure_t(t=10)
            self.set_frame_size(w=1900, h=1900)


    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float=1.0):
        self.active.AcquisitionFrameRateEnable.Value = enabled

        if enabled:
            self.active.AcquisitionFrameRate.Value = fps


    def set_pixel_format(self, pixel_format: str) -> bool:
        if not self.active:
            return False

        if pixel_format not in self.get_supported_pixel_formats():
            logger.exception(f"[CAM Class ] Unsupported pixel format: {pixel_format}")
            return False
        
        with self.update_camera_config():
            self.active.PixelFormat.SetValue(pixel_format)
        
        return True
 

    def get_pixel_format(self) -> str:
        return self.active.PixelFormat.GetValue()


    def get_supported_pixel_formats(self) -> tuple:
        return self.active.PixelFormat.GetSymbolics()
    

    def set_binning_size(self, size: int) -> bool:
        if not self.active:
            return False
        
        if size < 1 or size > 4:
            logger.exception(f"[CAM Class ] Unsupported bin size: {size}")
            return False
        
        logger.debug(f"Binning size before update: {self.get_binning_size()}")
        logger.debug(f"Frame size size before update: {self.get_frame_size()}")
        with self.update_camera_config():
            self.active.BinningVertical.SetValue(size)
            self.active.BinningVerticalMode.SetValue('Sum')
            self.active.BinningHorizontal.SetValue(size)
            self.active.BinningVerticalMode.SetValue('Sum')

        logger.debug(f"Binning size after update: {self.get_binning_size()}")
        logger.debug(f"Frame size size after update: {self.get_frame_size()}")
                
        return True
    

    def get_binning_size(self) -> int:
        if not self.active:
            return 1
        
        vert_bin = self.active.BinningVertical.GetValue()
        horiz_bin = self.active.BinningHorizontal.GetValue()

        if horiz_bin != vert_bin:
            logger.exception(f"[CAM Class ] Binning mismatch detected between horizontal ({horiz_bin}) and vertical ({vert_bin})")
        
        return vert_bin


    def init_auto_gain_focus(
        self,
        auto_target_brightness: float=0.5,
        min_gain: float | None = None,
        max_gain: float | None = None,
    ):
        self.active.AutoFunctionROIWidth.SetValue(self.active.Width.Max - 2*self.active.AutoFunctionROIOffsetX.GetValue())
        self.active.AutoFunctionROIHeight.SetValue(self.active.Height.Max - 2*self.active.AutoFunctionROIOffsetY.GetValue())
        self.active.AutoFunctionROIUseBrightness = True
        self.active.AutoTargetBrightness.SetValue(auto_target_brightness)
        self.active.AutoFunctionROISelector.SetValue('ROI1')

        if min_gain is None:
            min_gain = self.active.AutoGainLowerLimit.Min
        
        if max_gain is None:
            max_gain = self.active.AutoGainUpperLimit.Max

        self.active.AutoGainLowerLimit.SetValue(min_gain)
        self.active.AutoGainUpperLimit.SetValue(max_gain)
        self.active.AutoFunctionProfile.SetValue('MinimizeExposureTime')


    def update_auto_gain_target_brightness(self, auto_target_brightness: float):
        with self.update_camera_config():
            self.active.AutoTargetBrightness.SetValue(auto_target_brightness)


    def update_auto_gain_min_max(self, min_gain: float | None, max_gain: float | None):
        if not self.active:
            return
        
        if min_gain is None:
            min_gain = self.active.AutoGainLowerLimit.Min
        
        if max_gain is None:
            max_gain = self.active.AutoGainUpperLimit.Max

        with self.update_camera_config():
            self.active.AutoGainLowerLimit.SetValue(min_gain)
            self.active.AutoGainUpperLimit.SetValue(max_gain)


    def grab(self):
        """ Grab last available frame from camera and save it to self.array
        returns True if successful
        returns False if unsuccessful
        access the image using camera.array where camera is the instance of the class"""
        if not self.cam_image_handler:
            return False, None
        
        try:
            result, image, image_ts = self.cam_image_handler.GetLastImage()
            if result is False:
                return False, None
            
            self.array = image
            return True, image_ts

        except Exception as ex:
            logger.exception(f"Failed to grab image: {ex}")
            return False, None
  

    def set_frame_size(self, w, h):
        """ Set camera frame size to w by h and keep centered """
        camera = self.active
        if camera == False:
            logger.warning('[CAM Class ] PylonCamera.set_frame_size('+str(w)+','+str(h)+')'+'; inactive')
            return

        width = int(min(int(w), camera.Width.Max)/4)*4
        height = int(min(int(h), camera.Height.Max)/4)*4

        with self.update_camera_config():
            camera.Width.SetValue(width)
            camera.Height.SetValue(height)
            camera.BslCenterX.Execute()
            camera.BslCenterY.Execute()

        logger.info('[CAM Class ] PylonCamera.set_frame_size('+str(w)+','+str(h)+')'+'; succeeded')

    
    def get_min_frame_size(self) -> dict:
        camera = self.active
        if camera == False:
            return {}
        
        return {
            'width': camera.Width.GetMin(),
            'height': camera.Height.GetMin(),
        }
    

    def get_max_frame_size(self) -> dict:
        camera = self.active
        if camera == False:
            return {}
        
        return {
            'width': camera.Width.GetMax(),
            'height': camera.Height.GetMax(),
        }
 

    def get_frame_size(self):
        camera = self.active
        if camera == False:
            return
        
        width = camera.Width.GetValue()
        height = camera.Height.GetValue()

        return {
            'width': width,
            'height': height,
        }
    

    def get_gain(self):
        if self.active == False:
            logger.warning('[CAM Class ] PylonCamera.get_gain(): inactive camera')
            return -1
        
        return float(self.active.Gain.GetValue())


    def gain(self, gain):
        """ Set gain value in the camera hardware"""
        if self.active == False:
            logger.warning('[CAM Class ] PylonCamera.gain('+str(gain)+')'+': inactive camera')
            return

        self.active.Gain.SetValue(float(gain))
        logger.info('[CAM Class ] PylonCamera.gain('+str(gain)+')'+': succeeded')


    def auto_gain(
        self,
        state = True,
        target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None
    ):
        """ Enable / Disable camera auto_gain with the value of 'state'
        It will be continueously updating based on the current image """

        if self.active == False:
            logger.warning('[CAM Class ] PylonCamera.auto_gain('+str(state)+')'+': inactive camera')
            return

        if state == True:
            self.update_auto_gain_target_brightness(auto_target_brightness=target_brightness)
            self.update_auto_gain_min_max(min_gain=min_gain, max_gain=max_gain)
            self.active.GainAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
            self.active.ExposureAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
        else:
            self.active.GainAuto.SetValue('Off')
            self.active.ExposureAuto.SetValue('Off')
        logger.info('[CAM Class ] PylonCamera.auto_gain('+str(state)+')'+': succeeded')

    def auto_gain_once(
        self,
        state = True,
        target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None
    ):
        """ Enable / Disable camera auto_gain with the value of 'state'
        Auto Gain/Exposure executed one time """

        if self.active == False:
            logger.warning('[CAM Class ] PylonCamera.auto_gain_once('+str(state)+')'+': inactive camera')
            return

        if state == True:
            self.update_auto_gain_target_brightness(auto_target_brightness=target_brightness)
            self.update_auto_gain_min_max(min_gain=min_gain, max_gain=max_gain)
            self.active.GainAuto.SetValue('Once') # 'Off' 'Once' 'Continuous'
            self.active.ExposureAuto.SetValue('Once') # 'Off' 'Once' 'Continuous'
        else:
            self.active.GainAuto.SetValue('Off')
            self.active.ExposureAuto.SetValue('Off')
        logger.info('[CAM Class ] PylonCamera.auto_gain_once('+str(state)+')'+': succeeded')
            
            
    def exposure_t(self, t):
        """ Set exposure time in the camera hardware t (msec)"""
        if self.active == False:
            logger.warning('[CAM Class ] PylonCamera.exposure_t('+str(t)+')'+': inactive camera')
            return
        
        # (t*1000) in microseconds; therefore t  in milliseconds
        self.active.ExposureTime.SetValue(max(float(t)*1000, self.active.ExposureTime.Min))
        logger.info('[CAM Class ] PylonCamera.exposure_t('+str(t)+')'+': succeeded')


    def get_exposure_t(self):
        """ Set exposure time in the camera hardware
         Returns t (msec), or -1 if the camera is inactive"""

        if self.active == False:
            logger.warning('[CAM Class ] PylonCamera.get_exposure_t(): inactive camera')
            return -1

        microsec = self.active.ExposureTime.GetValue() # get current exposure time in microsec
        millisec = microsec/1000 # convert exposure time to millisec
        logger.info('[CAM Class ] PylonCamera.get_exposure_t(): succeeded')
        return millisec
            

    def auto_exposure_t(self, state = True):
        """ Enable / Disable camera auto_exposure with the value of 'state'
        It will be continueously updating based on the current image """

        if self.active == False:
            logger.warning('[CAM Class ] PylonCamera.auto_exposure_t('+str(state)+')'+': inactive camera')
            return
        
        if state == True:
            self.active.ExposureAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
        else:
            self.active.ExposureAuto.SetValue('Off')

        logger.info('[CAM Class ] PylonCamera.auto_exposure_t('+str(state)+')'+': succeeded')


    def set_test_pattern(self, enabled: bool = False, pattern: str = 'Black'):
        if self.active == False:
            return
        
        #if not enabled:
        #    self.active # TODO
        
        self.active.TestPattern.SetValue(pattern)
        self.grab()


class ImageHandler(pylon.ImageEventHandler):

    def __init__(self):
        super().__init__()
        self.last_result = False
        self.last_img = None
        self._failed_grabs = 0
        self._last_img_ts = None
        

    def OnImageGrabbed(self, camera, grabResult):
        try:
            self.last_result = grabResult.GrabSucceeded()
            if self.last_result:
                self.last_img = grabResult.GetArray()
                self._last_img_ts = datetime.datetime.now()
            else:
                self._failed_grabs += 1
                logger.exception(f"Grab Failed -> result: {self.last_result}, {self._failed_grabs} failed grabs")
        except Exception as e:
            logger.exception(e)

    def GetLastImage(self):
        if self.last_result is False:
            return False, None, None
        
        return self.last_result, self.last_img.copy(), self._last_img_ts
    

# class ConfigurationEventPrinter(pylon.ConfigurationEventHandler):
#     def OnAttach(self, camera):
#         print("OnAttach event")

#     def OnAttached(self, camera):
#         print("OnAttached event for device ", camera.GetDeviceInfo().GetModelName())

#     def OnOpen(self, camera):
#         print("OnOpen event for device ", camera.GetDeviceInfo().GetModelName())

#     def OnOpened(self, camera):
#         print("OnOpened event for device ", camera.GetDeviceInfo().GetModelName())

#     def OnGrabStart(self, camera):
#         print("OnGrabStart event for device ", camera.GetDeviceInfo().GetModelName())

#     def OnGrabStarted(self, camera):
#         print("OnGrabStarted event for device ", camera.GetDeviceInfo().GetModelName())

#     def OnGrabStop(self, camera):
#         print("OnGrabStop event for device ", camera.GetDeviceInfo().GetModelName())

#     def OnGrabStopped(self, camera):
#         print("OnGrabStopped event for device ", camera.GetDeviceInfo().GetModelName())

#     def OnClose(self, camera):
#         print("OnClose event for device ", camera.GetDeviceInfo().GetModelName())

#     def OnClosed(self, camera):
#         print("OnClosed event for device ", camera.GetDeviceInfo().GetModelName())

#     def OnDestroy(self, camera):
#         print("OnDestroy event for device ", camera.GetDeviceInfo().GetModelName())

#     def OnDestroyed(self, camera):
#         print("OnDestroyed event")

#     def OnDetach(self, camera):
#         print("OnDetach event for device ", camera.GetDeviceInfo().GetModelName())

#     def OnDetached(self, camera):
#         print("OnDetached event for device ", camera.GetDeviceInfo().GetModelName())

#     def OnGrabError(self, camera, errorMessage):
#         print("OnGrabError event for device ", camera.GetDeviceInfo().GetModelName())
#         print("Error Message: ", errorMessage)

#     def OnCameraDeviceRemoved(self, camera):
#         print("OnCameraDeviceRemoved event for device ", camera.GetDeviceInfo().GetModelName())
