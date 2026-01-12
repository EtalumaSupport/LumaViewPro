from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension

from lvp_logger import logger
from camera.camera import Camera


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

            #Allocate buffers
            payload_size = self.remote_nodemap.FindNode("PayloadSize").Value()
            for _ in range(self.data_stream.NumBuffersAnnouncedMinRequired()):
                buffer = self.data_stream.AllocAndAnnounceBuffer(payload_size)
                self.data_stream.QueueBuffer(buffer)

            try:
                self.model_name = self.active.ModelName()
                self._device_serial = self.active.SerialNumber()
                logger.info(f'[CAM Class ] Camera Model: {self.model_name}')
                logger.info(f'[CAM Class ] Camera Serial Number: {self._device_serial}')
                logger.info(f'[CAM Class ] Camera Firmware Version: {self.remote_nodemap.FindNode('DeviceFirmwareVersion').Value()}')
            except:
                logger.warning(f'[CAM CLASS ] Could not read all camera information')

            super().connect()        

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
            self.remote_nodemap.FindNode("AcquisitionStop").Execute()
            self.remote_nodemap.FindNode("AcquisitionStop").WaitUntilDone()
            self.data_stream.StopAcquisition()
        except Exception as e:
            logger.warning(f'[CAM Class ] stop_grabbing ignored error: {e}')

    def start_grabbing(self):
        try:
            self.data_stream.StartAcquisition()
            self.remote_nodemap.FindNode("AcquisitionStart").Execute()
            self.remote_nodemap.FindNode("AcquisitionStart").WaitUntilDone()
        except Exception as e:
            logger.warning(f'[CAM Class ] start_grabbing ignored error: {e}')

    def set_frame_size(self, w, h):
        mins = self.get_min_frame_size()
        maxs = self.get_max_frame_size()

        #Convert w and h to closest valid values
        width = int(max(mins['width'], min(maxs['width'], w)) / 48) * 48
        height = int(max(mins['height'], min(maxs['height'], h)) / 4) * 4

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
    
    def find_model_name(self):
        if not self.active:
            logger.warning('[CAM Class ] find_model_name(): inactive camera')
            return

        self.model_name = self.active.ModelName()
        logger.info(f'[CAM Class ] Connected camera model detected as "{self.model_name}"')
        return
    
    def get_all_temperatures(self):
        return {} #NOTE: Device does not support temperature reading
    
    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float=1.0):
        if not self.active:
            logger.warning('[CAM Class ] set_max_acquisition_frame_rate(): inactive camera')
            return
        
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