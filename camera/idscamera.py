import datetime
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
import ids_peak_ipl

from lvp_logger import logger
from camera.camera import Camera
import threading


class IDSCamera(Camera):
    def __init__(self):
        logger.info('[CAM Class ] IDSCamera.__init__()')
        
        self.device_manager = None
        self.data_stream = None
        self.remote_nodemap = None

        super().__init__()

    def connect(self) -> bool:
        try:
            #Initialize device manager
            ids_peak.Library.Initialize()
            self.device_manager = ids_peak.DeviceManager.Instance()
            self.device_manager.Update()

            #Search for devices
            if self.device_manager.Devices().empty():
                raise ConnectionError("Could not find IDS camera")
            
            self.active = self.device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
            self.data_stream = self.active.DataStreams()[0].OpenDataStream()
            self.remote_nodemap = self.active.RemoteDevice().NodeMaps()[0]
            self._device_removed = False


            try:
                self.model_name = self.active.ModelName()
                self._device_serial = self.active.SerialNumber()
                logger.info(f'[CAM Class ] Camera Model: {self.model_name}')
                logger.info(f'[CAM Class ] Camera Serial Number: {self._device_serial}')
                logger.info(f'[CAM Class ] Camera Firmware Version: {self.remote_nodemap.FindNode('DeviceFirmwareVersion').Value()}')
            except:
                logger.warning(f'[CAM CLASS ] Could not read all camera information')

            self.init_camera_config()      
            self.start_grabbing() 

            self.cam_image_handler = ImageHandler(self.data_stream)

            self.error_report_count = 0
            logger.info('[CAM Class ] IDSCamera.connect() succeeded')    
            return True

        except ConnectionError as er:
            logger.warning(f'[CAM Class ] IDSCamera.connect() failed -> {er}')
        except Exception as ex:
            logger.exception(f'[CAM Class ] IDSCamera.connect() failed -> {ex}')

        return False

    def disconnect(self) -> bool:
        logger.info('[CAM Class ] Disconnecting from camera...')
        try:
            if self.active:
                if self.is_grabbing():
                    self.stop_grabbing()
                ids_peak.Library.Close()
                self.remote_nodemap = None
                self.data_stream = None
                self.device_manager = None
                self.active = None
                logger.info('[CAM Class ] IDSCamera.disconnect() succeeded')
                return True
            else:
                logger.info('[CAM Class ] IDSCamera.disconnect() failed: Camera not connected')
        except Exception as e:
            logger.exception(f'[CAM Class ] IDSCamera.disconnect() failed: {e}')
        return False
    
    def is_connected(self) -> bool:
        if self.active in (False, None):
            self._device_removed = True
            return False
        if self._device_removed:
            return False
        return True
    
    def __delete__(self):
        if self.active:
            self.disconnect()

    def init_camera_config(self):
        if not self.active:
            return
        
        with self.update_camera_config():
            self.remote_nodemap.FindNode("UserSetSelector").SetCurrentEntry("Default")
            self.remote_nodemap.FindNode("UserSetLoad").Execute()
            self.remote_nodemap.FindNode("UserSetLoad").WaitUntilDone()
            self.set_pixel_format("Mono10g40IDS") #NOTE: Should be converted to Mono8 for Pixel Format Destination
            #TODO: auto gain
            self.remote_nodemap.FindNode("ReverseX").SetValue(True)
            self.exposure_t(10)
            self.set_frame_size(1920,1528)

    def is_grabbing(self):
        if not self.data_stream:
            return False
        
        return self.data_stream.IsGrabbing()

    def stop_grabbing(self):
        try:
            if self.cam_image_handler:
                self.cam_image_handler.stop()

            self.remote_nodemap.FindNode("AcquisitionStop").Execute()
            self.remote_nodemap.FindNode("AcquisitionStop").WaitUntilDone()
            self.data_stream.StopAcquisition()

            # self.data_stream.KillWait()
            self.data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
            for buffer in self.data_stream.AnnouncedBuffers():
                self.data_stream.RevokeBuffer(buffer)

            # self.remote_nodemap.FindNode("TLParamsLocked").SetValue(0)
        except Exception as e:
            logger.warning(f'[CAM Class ] stop_grabbing ignored error: {e}')

    def start_grabbing(self):
        try:
            # self.remote_nodemap.FindNode("TLParamsLocked").SetValue(1)

            # self.data_stream.FlushPendingKillWaits()

            #Allocate buffers
            payload_size = self.remote_nodemap.FindNode("PayloadSize").Value()
            for _ in range(self.data_stream.NumBuffersAnnouncedMinRequired()):
                buffer = self.data_stream.AllocAndAnnounceBuffer(payload_size)
                self.data_stream.QueueBuffer(buffer)

            self.data_stream.StartAcquisition()
            self.remote_nodemap.FindNode("AcquisitionStart").Execute()
            self.remote_nodemap.FindNode("AcquisitionStart").WaitUntilDone()

            if self.cam_image_handler:
                self.cam_image_handler.start()

            logger.info('[CAM Class ] start_grabbing succeeded')
        except Exception as e:
            logger.warning(f'[CAM Class ] start_grabbing ignored error: {e}')

    def set_frame_size(self, w, h):
        mins = self.get_min_frame_size()
        maxs = self.get_max_frame_size()

        #Convert w and h to closest valid values
        width = int(max(mins['width'], min(maxs['width'], w)) / 48) * 48
        height = int(max(mins['height'], min(maxs['height'], h)) / 4) * 4

        with self.update_camera_config():
            self.remote_nodemap.FindNode("Width").SetValue(width)
            self.remote_nodemap.FindNode("Height").SetValue(height)

    def get_min_frame_size(self) -> dict:
        if not self.active:
            return {}
        
        return {
            'width': self.remote_nodemap.FindNode("Width").Minimum(),
            'height': self.remote_nodemap.FindNode("Height").Minimum(),
        }
    

    def get_max_frame_size(self) -> dict:
        if not self.active:
            return {}
        
        return {
            'width': self.remote_nodemap.FindNode("Width").Maximum(),
            'height': self.remote_nodemap.FindNode("Height").Maximum(),
        }
 

    def get_frame_size(self):
        if not self.active:
            return
        
        width = self.remote_nodemap.FindNode("Width").Value()
        height = self.remote_nodemap.FindNode("Height").Value()

        return {
            'width': width,
            'height': height,
        }

    def set_pixel_format(self, pixel_format):
        if not self.active:
            return False
        
        if pixel_format not in self.get_supported_pixel_formats():
            logger.exception(f"[CAM Class ] Unsupported pixel format: {pixel_format}")
            return False
        
        with self.update_camera_config():
            self.remote_nodemap.FindNode("PixelFormat").SetCurrentEntry(pixel_format)

        return True
    
    def get_pixel_format(self):
        return self.remote_nodemap.FindNode("PixelFormat").CurrentEntry().SymbolicValue()
    
    def get_supported_pixel_formats(self):
        return tuple(pf.SymbolicValue() for pf in self.remote_nodemap.FindNode("PixelFormat").AvailableEntries())
    
    def exposure_t(self, t):
        if self.active == False:
            logger.warning('[CAM Class ] IDSCamera.exposure_t('+str(t)+')'+': inactive camera')
            return
        
        if t > self.max_exposure:
            logger.warning(f'[CAM Class ] IDSCamera.exposure_t(Exposure of {t}ms > camera maximum ({self.max_exposure}ms))')
            return
        
        # IDS takes time in microseconds, so pass t*1000 to convert to us
        try:
            with self.update_camera_config():
                self.remote_nodemap.FindNode("ExposureTime").SetValue(float(t)*1000)
            logger.info('[CAM Class ] IDSCamera.exposure_t('+str(t)+')'+': succeeded')
        except Exception as e:
            logger.error('[CAM class ] IDSCamera.exposure_t(FAILED; Exposure likely out of bounds) {e}')

    def get_exposure_t(self):
        if self.active == False:
            logger.warning('[CAM Class ] IDSCamera.get_exposure_t(): inactive camera')
            return -1

        microsec = self.remote_nodemap.FindNode("ExposureTime").Value() # get current exposure time in microsec
        millisec = microsec/1000 # convert exposure time to millisec
        return millisec
    
    def auto_exposure_t(self, state = True):
        #TODO: Implement for IDS cameras that support auto exposure
        return self.remote_nodemap.HasNode("ExposureAuto")
    
    def find_model_name(self):
        if not self.active:
            logger.warning('[CAM Class ] find_model_name(): inactive camera')
            return

        self.model_name = self.active.ModelName()
        logger.info(f'[CAM Class ] Connected camera model detected as "{self.model_name}"')
        return
    
    def get_all_temperatures(self):
        return {} #TODO: Implement for IDS cameras that support temperature readings
    
    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float=1.0):
        if not self.active:
            logger.warning('[CAM Class ] set_max_acquisition_frame_rate(): inactive camera')
            return
        
        with self.update_camera_config():
            self.remote_nodemap.FindNode("AcquisitionFrameRateTargetEnable").SetValue(enabled)

            if enabled:
                self.remote_nodemap.FindNode("AcquisitionFrameRateTarget").SetValue(fps)

    def set_binning_size(self, size: int) -> bool:
        if not self.active:
            return False
        
        if size < 1 or size > 2:
            logger.exception(f"[CAM Class ] Unsupported bin size: {size}")
            return False
        
        logger.debug(f"Binning size before update: {self.get_binning_size()}")
        logger.debug(f"Frame size size before update: {self.get_frame_size()}")
        with self.update_camera_config():
            # self.remote_nodemap.FindNode("BinningSelector").SetCurrentEntry("Region0")
            self.remote_nodemap.FindNode("BinningVertical").SetValue(size)
            # self.remote_nodemap.FindNode("BinningVerticalMode").SetCurrentEntry("Sum")
            self.remote_nodemap.FindNode("BinningHorizontal").SetValue(size)
            # self.remote_nodemap.FindNode("BinningHorizontalMode").SetCurrentEntry("Sum")

        logger.debug(f"Binning size after update: {self.get_binning_size()}")
        logger.debug(f"Frame size size after update: {self.get_frame_size()}")
                
        return True
    
    def get_binning_size(self) -> int:
        if not self.active:
            return 1
        
        vert_bin = self.remote_nodemap.FindNode("BinningVertical").Value()
        horiz_bin = self.remote_nodemap.FindNode("BinningHorizontal").Value()

        if horiz_bin != vert_bin:
            logger.exception(f"[CAM Class ] Binning mismatch detected between horizontal ({horiz_bin}) and vertical ({vert_bin})")
        
        return vert_bin
    
    def grab(self):
        if not self.cam_image_handler:
            return False, None
        
        try:
            result, image, image_ts = self.cam_image_handler.get_last_image()
            if result is False:
                return False, None
            
            self.array = image
            return True, image_ts

        except Exception as ex:
            logger.exception(f"Failed to grab image: {ex}")
            return False, None
        
    def grab_new_capture(self, timeout):
        if not self.cam_image_handler:
            return False, None
        
        try:
            buffer = self.data_stream.WaitForFinishedBuffer(timeout)
            result = not buffer.IsIncomplete()
            if not result:
                self.data_stream.QueueBuffer(buffer)
                return False, None
            
            img = ids_peak_ipl_extension.BufferToImage(buffer)
            img = img.ConvertTo(ids_peak_ipl.PixelFormatName_Mono8)
            img = img.get_numpy().copy()
            img_ts = datetime.datetime.now()
            self.data_stream.QueueBuffer(buffer)

            self.array = img
            return True, img_ts
            
        except Exception as e:
            return False, None
        
    def update_auto_gain_target_brightness(self, auto_target_brightness: float):
        #TODO: Implement for IDS cameras that support auto gain
        return self.remote_nodemap.HasNode("GainAuto") # Return False if IDS camera does not support auto gain

    def update_auto_gain_min_max(self, min_gain: float | None, max_gain: float | None):
        #TODO: Implement for IDS cameras that support auto gain
        return self.remote_nodemap.HasNode("GainAuto") # Return False if IDS camera does not support auto gain

    def get_gain(self):
        if self.active == False:
            logger.warning('[CAM Class ] idscamera.get_gain(): inactive camera')
            return -1
        
        try:
            value = self.remote_nodemap.FindNode("Gain").Value()
            return float(value)
        except Exception as e:
            logger.error(f'[CAM class ] idscamera.get_gain(FAILED) {e}')
            return -1

    def gain(self, gain):
        if self.active == False:
            logger.warning('[CAM Class ] idscamera.gain('+str(gain)+')'+': inactive camera')
            return

        try:
            self.remote_nodemap.FindNode("GainSelector").SetCurrentEntry("AnalogAll")
            self.remote_nodemap.FindNode("Gain").SetValue(gain)
            logger.info('[CAM Class ] idscamera.gain('+str(gain)+')'+': succeeded')
        except Exception as e:
            logger.error(f'[CAM class ] idscamera.gain(FAILED; Gain likely out of bounds) {e}')
            return
            

    def auto_gain(
        self,
        state = True,
        target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None
    ):
        #TODO: Implement functionality for IDS cameras that support auto gain
        return self.remote_nodemap.HasNode("GainAuto") # Return False if IDS camera does not support auto gain

    def auto_gain_once(
        self,
        state = True,
        target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None
    ):
        #TODO: Implement functionality for IDS cameras that support auto gain
        return self.remote_nodemap.HasNode("GainAuto") # Return False if IDS camera does not support auto gain

    def set_test_pattern(self, enabled: bool = False, pattern: str = 'Black'):
        #TODO: Implement
        pass

class ImageHandler:
    def __init__(self, data_stream: ids_peak.DataStream):
        self.data_stream = data_stream
        self.last_result = False
        self.last_img = None
        self.last_img_ts = None
        self._grab_thread = None
        self._stop_event = threading.Event()

    def start(self):
        if self._grab_thread is None:
            self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
            self._stop_event.clear()
            self._grab_thread.start()

    def stop(self):
        if self._grab_thread is not None:
            self._stop_event.set()
            self._grab_thread.join()
            self._grab_thread = None

    def _grab_loop(self):
        while not self._stop_event.is_set():
            try:
                buffer = self.data_stream.WaitForFinishedBuffer(1000)
                self.last_result = not buffer.IsIncomplete()
                if self.last_result:
                    img = ids_peak_ipl_extension.BufferToImage(buffer)
                    img = img.ConvertTo(ids_peak_ipl.PixelFormatName_Mono8)
                    self.last_img = img.get_numpy().copy()
                    self.last_img_ts = datetime.datetime.now()
                self.data_stream.QueueBuffer(buffer)
            except Exception as e:
                self.last_result = False
                logger.exception(f'[CAM Class ] ImageHandler grab loop exception: {e}')

    def get_last_image(self):
        if not self.last_result:
            return False, None, None
        
        return self.last_result, self.last_img, self.last_img_ts
                    