# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import datetime
import os
import threading

import numpy as np
from pypylon import pylon, genicam
from lvp_logger import logger

import queue

from drivers.camera import Camera


class PylonCamera(Camera):

    def __init__(self, **kwargs):

        if os.getenv("PYLON_CAMEMU", None) is not None:
            logger.info('[CAM Class ] PylonCamera.connect() detected request to use camera emulation')
            self._use_camera_emulation = True
        else:
            self._use_camera_emulation = False

        super().__init__()

    def _mark_disconnected(self):
        """Atomically mark camera as disconnected. Sets both flags together
        to avoid inconsistent state between _device_removed and active."""
        self._device_removed = True
        self.active = None

    def _get_max_exposure_models(self) -> dict:
        return {
            "daA3840-45um": 1_000,
            "a2A3536-31umBAS": 10_000,
        }

    def disconnect(self) -> bool:
        try:
            if self.active is not None:
                try:
                    if self.is_grabbing():
                        self.stop_grabbing()
                except Exception:
                    pass
                self.active.Close()
                self.active = None
                logger.info('[CAM Class ] Disconnected from Pylon camera')
                return True
            else:
                logger.info('[CAM Class ] Pylon camera not connected')
        except Exception as e:
            logger.exception(f'[CAM Class ] Pylon camera disconnect failed: {e}')
        return False

    def __delete__(self):
        try:
            self.active.close()
        except Exception:
            logger.exception('[CAM Class ] exception')

    def stop_grabbing(self):
        camera = self.active
        try:
            camera.StopGrabbing()
        except Exception as e:
            logger.warning(f'[CAM Class ] stop_grabbing ignored error: {e}')


    def start_grabbing(self):
        camera = self.active
        try:
            camera.StartGrabbing(
                pylon.GrabStrategy_LatestImageOnly,
                pylon.GrabLoop_ProvidedByInstantCamera
            )
        except Exception as e:
            logger.warning(f'[CAM Class ] start_grabbing ignored error: {e}')

    def is_grabbing(self):
        return self.active.IsGrabbing()

    def connect(self) -> bool:
        """ Try to connect to the first available basler camera"""
        try:
            p_device = pylon.TlFactory.GetInstance().CreateFirstDevice()
            self.active = pylon.InstantCamera(p_device)
            camera = self.active
            # Ensure previous removal flag does not persist across a new connection
            self._device_removed = False
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
            # Register a minimal removal handler that only sets an internal flag
            try:
                camera.RegisterConfiguration(
                    _CameraRemovalHandler(self),
                    pylon.RegistrationMode_Append,
                    pylon.Cleanup_Delete
                )
            except Exception:
                # If registration isn't supported on this platform/TL, ignore
                pass

            self.cam_image_handler = ImageHandler(self)
            camera.RegisterImageEventHandler(
                self.cam_image_handler,
                pylon.RegistrationMode_Append,
                pylon.Cleanup_Delete
            )

            camera.Open()
            
            self._device_removed = False

            # Store device identity if possible
            try:
                dev_info = camera.GetDeviceInfo()
                self.model_name = dev_info.GetModelName()
                try:
                    self._device_serial = dev_info.GetSerialNumber()
                except Exception:
                    # Some transports may not provide a serial accessor
                    self._device_serial = None

                try:
                    nm = camera.GetNodeMap()
                    
                    ds_n = nm.GetNode("DeviceSerialNumber")
                    device_serial = nm.GetNode("DeviceSerialNumber").ToString()
                    logger.info(f'[CAM Class ] Camera Serial Number: {device_serial}')

                    firmware = nm.GetNode("DeviceFirmwareVersion").ToString()
                    logger.info(f'[CAM Class ] Camera Firmware Version: {firmware}')

                    temps = self.get_all_temperatures()
                    for name, temp in temps.items():
                        logger.info(f'[CAM Class ] Camera {name} Temperature : {temp:.2f} °C')

                except Exception as e:
                    logger.error(f'[CAM Class ] Failed to read device info nodes: {e}', exc_info=True)

            except Exception:
                self.model_name = None
                self._device_serial = None

            # Set max exposure based on detected model
            if self.model_name:
                self.set_max_exposure_time()

            # Ensure no stale queued frames or state
            try:
                self.cam_image_handler.reset()
            except Exception:
                pass

            self.init_camera_config()
            self.start_grabbing()

            self.error_report_count = 0
            logger.info('[CAM Class ] Connected to Pylon camera')
            return True

        except genicam.RuntimeException as ex:
            logger.error(f'[CAM Class ] Pylon camera connect failed (may be open in another application): {ex}')
            self.active = None
            self.error_report_count += 1
        except Exception:
            logger.exception('[CAM Class ] Pylon camera connect failed')
            self.active = None
            self.error_report_count += 1

        return False

    def find_model_name(self):
        if not self.active:
            logger.warning('[CAM Class ] Cannot read model name: camera inactive')
            return

        dev_info = self.active.GetDeviceInfo()
        self.model_name = dev_info.GetModelName()
        logger.info(f'[CAM Class ] Camera model: {self.model_name}')
    
    def get_all_temperatures(self):
        """
        Returns dict like:
            {'FpgaCore': 43.2, 'SomethingElse': 40.1, ...}
        """
        # Camera Must be open prior to calling function
        if not self.active:
            logger.warning('[CAM Class ] get_all_temperatures(): inactive camera')
            return {}

        try:
            nodemap = self.active.GetNodeMap()

            selector = nodemap.GetNode("DeviceTemperatureSelector")
            temp = nodemap.GetNode("DeviceTemperature")

            if selector is None or temp is None:
                return {}

            temps: dict[str, float] = {}

            # Iterate all available selector entries
            for entry in selector.GetEntries():

                name = entry.GetSymbolic()       # e.g. "FpgaCore"
                value = entry.GetValue()         # enum integer value

                # Select this temperature source
                selector.SetIntValue(value)

                # Read temperature
                if genicam.IsReadable(temp):
                    temps[name] = temp.GetValue()

            return temps
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Failed to read camera temperatures: {e}')
            self._mark_disconnected()
            return {}
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error reading temperatures: {e}')
            return {}

    def init_camera_config(self):
        camera = self.active
        if camera is None:
            return

        try:
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
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Camera communication error during init_camera_config: {e}')
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in init_camera_config: {e}')


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

        try:
            with self.update_camera_config():
                self.active.PixelFormat.SetValue(pixel_format)
            return True
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Camera communication error during set_pixel_format({pixel_format}): {e}')
            self._mark_disconnected()
            return False
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in set_pixel_format: {e}')
            return False
 

    def get_pixel_format(self) -> str:
        if not self.active:
            return ""

        try:
            return self.active.PixelFormat.GetValue()
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Failed to read pixel format: Camera may be disconnected - {e}')
            self._mark_disconnected()
            return ""
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error reading pixel format: {e}')
            return ""


    def get_supported_pixel_formats(self) -> tuple:
        return self.active.PixelFormat.GetSymbolics()
    

    def set_binning_size(self, size: int) -> bool:
        if not self.active:
            return False

        if size < 1 or size > 4:
            logger.exception(f"[CAM Class ] Unsupported bin size: {size}")
            return False

        try:
            logger.debug(f"[CAM Class ] Binning {self.get_binning_size()} -> {size}, frame {self.get_frame_size()}")
            with self.update_camera_config():
                self.active.BinningVertical.SetValue(size)
                self.active.BinningVerticalMode.SetValue('Sum')
                self.active.BinningHorizontal.SetValue(size)
                self.active.BinningVerticalMode.SetValue('Sum')

            logger.debug(f"[CAM Class ] Binning set to {self.get_binning_size()}, frame now {self.get_frame_size()}")

            return True
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Camera communication error during set_binning_size({size}): {e}')
            self._mark_disconnected()
            return False
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in set_binning_size: {e}')
            return False
    

    def get_binning_size(self) -> int:
        if not self.active:
            return 1

        try:
            vert_bin = self.active.BinningVertical.GetValue()
            horiz_bin = self.active.BinningHorizontal.GetValue()

            if horiz_bin != vert_bin:
                logger.exception(f"[CAM Class ] Binning mismatch detected between horizontal ({horiz_bin}) and vertical ({vert_bin})")

            return vert_bin
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Failed to read binning size: Camera may be disconnected - {e}')
            self._mark_disconnected()
            return 1
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error reading binning size: {e}')
            return 1


    def init_auto_gain_focus(
        self,
        auto_target_brightness: float=0.5,
        min_gain: float | None = None,
        max_gain: float | None = None,
    ):
        try:
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
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Camera communication error during init_auto_gain_focus: {e}')
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in init_auto_gain_focus: {e}')


    def update_auto_gain_target_brightness(self, auto_target_brightness: float):
        try:
            with self.update_camera_config():
                self.active.AutoTargetBrightness.SetValue(auto_target_brightness)
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Camera communication error during update_auto_gain_target_brightness({auto_target_brightness}): {e}')
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in update_auto_gain_target_brightness: {e}')


    def update_auto_gain_min_max(self, min_gain: float | None, max_gain: float | None):
        if not self.active:
            return

        try:
            if min_gain is None:
                min_gain = self.active.AutoGainLowerLimit.Min

            if max_gain is None:
                max_gain = self.active.AutoGainUpperLimit.Max

            with self.update_camera_config():
                self.active.AutoGainLowerLimit.SetValue(min_gain)
                self.active.AutoGainUpperLimit.SetValue(max_gain)
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Camera communication error during update_auto_gain_min_max(min={min_gain}, max={max_gain}): {e}')
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in update_auto_gain_min_max: {e}')


    def grab(self):
        """ Grab last available frame from camera and save it to self.array
        returns True if successful
        returns False if unsuccessful
        access the image using camera.array where camera is the instance of the class"""
        # Check if camera is still active before attempting grab
        if self.active is None:
            return False, None
            
        if self._device_removed:
            return False, None
            
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
    
    def grab_new_capture(self, timeout: float):
        """
        Blocks for timeout waiting for new frame. Saves from to self.array when received
        returns bool, ts
        """
        if not self.cam_image_handler:
            return False, None
        
        try:
            try:
                self.cam_image_handler._frame_queue.get_nowait()
            except queue.Empty:
                pass
            result, image, image_ts = self.cam_image_handler._frame_queue.get(block=True, timeout=timeout)
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
        if camera is None:
            logger.warning(f'[CAM Class ] Cannot set frame size {w}x{h}: camera inactive')
            return

        try:
            width = int(min(int(w), camera.Width.Max)/4)*4
            height = int(min(int(h), camera.Height.Max)/4)*4

            with self.update_camera_config():
                camera.Width.SetValue(width)
                camera.Height.SetValue(height)
                camera.BslCenterX.Execute()
                camera.BslCenterY.Execute()

            logger.info(f'[CAM Class ] Frame size set to {width}x{height}')
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Camera communication error during set_frame_size({w}x{h}): {e}')
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in set_frame_size: {e}')

    
    def get_min_frame_size(self) -> dict:
        camera = self.active
        if camera is None:
            return {}
        
        return {
            'width': camera.Width.GetMin(),
            'height': camera.Height.GetMin(),
        }
    

    def get_max_frame_size(self) -> dict:
        camera = self.active
        if camera is None:
            return {}
        
        return {
            'width': camera.Width.GetMax(),
            'height': camera.Height.GetMax(),
        }
 

    def get_frame_size(self):
        camera = self.active
        if camera is None:
            return

        try:
            width = camera.Width.GetValue()
            height = camera.Height.GetValue()

            return {
                'width': width,
                'height': height,
            }
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Failed to read frame size: Camera may be disconnected - {e}')
            self._mark_disconnected()
            return None
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error reading frame size: {e}')
            return None
    

    def get_gain(self):
        if self.active is None:
            logger.warning('[CAM Class ] Cannot read gain: camera inactive')
            return -1

        try:
            return float(self.active.Gain.GetValue())
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Failed to read gain value: Camera may be disconnected - {e}')
            self._mark_disconnected()
            return -1
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error reading gain: {e}')
            return -1

    def is_connected(self) -> bool:
        """Return True if the current camera is considered connected.
        Uses internal removal flag and, if available, the SDK's device-removed query.
        Avoids transport-layer enumeration to reduce risk of native-side instability.
        """
        if self._device_removed:
            self._mark_disconnected()
            return False
        if self.active is None:
            self._mark_disconnected()
            return False
        # if getattr(self, "_use_camera_emulation", False):
        #     return True
        # try:
        #     if self.active:
        #         if hasattr(self.active, "IsCameraDeviceRemoved"):
        #             if self.active.IsCameraDeviceRemoved():
        #                 self._device_removed = True
        #                 logger.warning('[CAM Class ] Camera device removed')
        #                 return False
        # except Exception:
        #     return False
        return True


    def gain(self, gain):
        """ Set gain value in the camera hardware"""
        if self.active is None:
            logger.warning(f'[CAM Class ] Cannot set gain {gain}: camera inactive')
            return

        try:
            self.active.Gain.SetValue(float(gain))
            logger.info(f'[CAM Class ] Gain set to {gain}')
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Camera communication error during gain({gain}): {e}')
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in gain: {e}')


    def auto_gain(
        self,
        state = True,
        target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None
    ):
        """ Enable / Disable camera auto_gain with the value of 'state'
        It will be continueously updating based on the current image """

        if self.active is None:
            logger.warning(f'[CAM Class ] Cannot set auto_gain({state}): camera inactive')
            return

        try:
            if state:
                self.update_auto_gain_target_brightness(auto_target_brightness=target_brightness)
                self.update_auto_gain_min_max(min_gain=min_gain, max_gain=max_gain)
                self.active.GainAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
                self.active.ExposureAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
            else:
                self.active.GainAuto.SetValue('Off')
                self.active.ExposureAuto.SetValue('Off')
            logger.info(f'[CAM Class ] Auto gain {"enabled" if state else "disabled"}')
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Auto gain({state}) failed: {e}')
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in auto_gain: {e}')

    def auto_gain_once(
        self,
        state = True,
        target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None
    ):
        """ Enable / Disable camera auto_gain with the value of 'state'
        Auto Gain/Exposure executed one time """

        if self.active is None:
            logger.warning(f'[CAM Class ] Cannot set auto_gain_once({state}): camera inactive')
            return

        try:
            if state:
                self.update_auto_gain_target_brightness(auto_target_brightness=target_brightness)
                self.update_auto_gain_min_max(min_gain=min_gain, max_gain=max_gain)
                self.active.GainAuto.SetValue('Once') # 'Off' 'Once' 'Continuous'
                self.active.ExposureAuto.SetValue('Once') # 'Off' 'Once' 'Continuous'
            else:
                self.active.GainAuto.SetValue('Off')
                self.active.ExposureAuto.SetValue('Off')
            logger.info(f'[CAM Class ] Auto gain once {"enabled" if state else "disabled"}')
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Auto gain once({state}) failed: {e}')
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in auto_gain_once: {e}')
            
            
    def exposure_t(self, t):
        """ Set exposure time in the camera hardware t (msec)"""
        if self.active is None:
            logger.warning(f'[CAM Class ] Cannot set exposure {t}ms: camera inactive')
            return

        if t > self.max_exposure:
            logger.warning(f'[CAM Class ] Exposure {t}ms exceeds max ({self.max_exposure}ms)')
            return

        # Pylon takes time in microseconds, so pass t*1000 to convert to us
        try:
            self.active.ExposureTime.SetValue(max(float(t)*1000, self.active.ExposureTime.Min))
            logger.info(f'[CAM Class ] Exposure set to {t}ms')
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Camera communication error during exposure_t({t}ms): {e}')
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in exposure_t: {e}')


    def get_exposure_t(self):
        """ Get exposure time in the camera hardware
         Returns t (msec), or -1 if the camera is inactive"""

        if self.active is None:
            logger.warning('[CAM Class ] Cannot read exposure: camera inactive')
            return -1

        try:
            microsec = self.active.ExposureTime.GetValue() # get current exposure time in microsec
            millisec = microsec/1000 # convert exposure time to millisec
            return millisec
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Failed to read exposure time: Camera may be disconnected - {e}')
            self._mark_disconnected()
            return -1
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error reading exposure time: {e}')
            return -1
            

    def auto_exposure_t(self, state = True):
        """ Enable / Disable camera auto_exposure with the value of 'state'
        It will be continueously updating based on the current image """

        if self.active is None:
            logger.warning(f'[CAM Class ] Cannot set auto_exposure({state}): camera inactive')
            return

        try:
            if state:
                self.active.ExposureAuto.SetValue('Continuous') # 'Off' 'Once' 'Continuous'
            else:
                self.active.ExposureAuto.SetValue('Off')
            logger.info(f'[CAM Class ] Auto exposure {"enabled" if state else "disabled"}')
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Auto exposure({state}) failed: {e}')
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in auto_exposure_t: {e}')


    def set_test_pattern(self, enabled: bool = False, pattern: str = 'Black'):
        if self.active is None:
            return

        try:
            self.active.TestPattern.SetValue(pattern)
            self.grab()
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Camera communication error during set_test_pattern({pattern}): {e}')
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in set_test_pattern: {e}')


class ImageHandler(pylon.ImageEventHandler):

    def __init__(self, parent_cam: PylonCamera):
        super().__init__()
        self.last_result = False
        self.last_img = None
        self._failed_grabs = 0
        self._last_img_ts = None
        self._frame_queue = queue.Queue(maxsize=1)
        self._parent = parent_cam
        
    
    def OnImageGrabbed(self, camera, grabResult):
        try:
            # Set thread name for dummy threads
            if "Dummy" in threading.current_thread().name:
                threading.current_thread().name = "PylonImageGrab"
            
            # Check if parent camera was removed before processing
            if self._parent._device_removed:
                logger.debug('[CAM Class ] OnImageGrabbed called but device already marked as removed, ignoring')
                return
            
            # Check if parent camera is still active
            if self._parent.active is None:
                logger.debug('[CAM Class ] OnImageGrabbed called but camera is inactive, ignoring')
                self._parent._device_removed = True
                return
            
            if not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            # Safely check grab result - this can throw native exceptions
            try:
                self.last_result = grabResult.GrabSucceeded()
            except Exception as e:
                logger.warning(f'[CAM Class ] GrabSucceeded() failed: {e}, assuming device removed')
                self._parent._device_removed = True
                return
            
            if self.last_result:
                try:
                    # GetArray() can cause native crash if device is disconnected
                    self.last_img = grabResult.GetArray()
                    self._last_img_ts = datetime.datetime.now()
                    self._frame_queue.put((self.last_result, self.last_img, self._last_img_ts))
                    self._failed_grabs = 0
                except Exception as e:
                    logger.warning(f'[CAM Class ] GetArray() failed: {e}, marking device as removed')
                    self._parent._device_removed = True
                    self.last_result = False
            else:
                self._failed_grabs += 1
                # Rate-limit noisy grab failure logs
                if self._failed_grabs % 5 == 1:
                    logger.warning(f"Grab Failed -> result: {self.last_result}, {self._failed_grabs} failed grabs")
                # If we've failed a lot in a row, stop grabbing to avoid potential underlying SDK exit
                if self._failed_grabs >= 128:
                    try:
                        logger.error('[CAM Class ] Too many grab failures; stopping acquisition')
                        if self._parent.active and self._parent.is_grabbing():
                            self._parent.stop_grabbing()
                        self._parent._device_removed = True
                    except Exception:
                        pass
        except Exception as e:
            logger.exception(e)

    def reset(self):
        try:
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            self.last_result = False
            self.last_img = None
            self._last_img_ts = None
            self._failed_grabs = 0
        except Exception:
            pass

    def GetLastImage(self):
        # Check if parent camera is still valid before returning image
        try:
            if self._parent._device_removed:
                return False, None, None

            if self._parent.active is None:
                return False, None, None
        except Exception:
            return False, None, None
        
        if self.last_result is False:
            return False, None, None
        
        # Safely copy the image - could fail if camera disconnected mid-operation
        try:
            return self.last_result, self.last_img.copy(), self._last_img_ts
        except Exception as e:
            logger.warning(f'[CAM Class ] GetLastImage copy failed: {e}')
            return False, None, None
    
    

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

# Handle camera removal events to flag device disconnect
class _CameraRemovalHandler(pylon.ConfigurationEventHandler):
    def __init__(self, parent_cam: PylonCamera):
        super().__init__()
        self._parent = parent_cam

    def OnCameraDeviceRemoved(self, camera):
        # CRITICAL: This runs in a native Pylon SDK thread during device removal.
        # RACE CONDITION: Do NOT modify self._parent.active here!
        # Other Python threads may be accessing it simultaneously, causing crashes.
        # ONLY set the boolean flag - Python's boolean assignment is atomic.
        self._parent._device_removed = True
        # Note: We do NOT set self._parent.active = False here to avoid race conditions.
        # The is_connected() checks will see _device_removed and handle cleanup safely.
