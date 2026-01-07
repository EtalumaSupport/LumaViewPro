from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension

from lvp_logger import logger
from camera.camera import Camera


class IDSCamera(Camera):
    def __init__(self):
        logger.info('[CAM Class ] IDSCamera.__init__()')
        
        self.device_manager = None
        self.remote_nodemap = None

        super().__init__()

    def connect(self):
        try:
            #Initialize device manager
            ids_peak.Library.Initialize()
            self.device_manager = ids_peak.DeviceManager.Instance()
            self.device_manager.Update()

            #Search for devices
            if self.device_manager.Devices().empty():
                raise ConnectionError("Could not find IDS camera")
            
            self.active = self.device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
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

            super().connect()        

            self.error_report_count = 0
            logger.info('[CAM Class ] IDSCamera.connect() succeeded')    

        except ConnectionError as er:
            logger.warning(f'[CAM Class ] IDSCamera.connect() failed -> {er}')
        except Exception as ex:
            logger.exception(f'[CAM Class ] IDSCamera.connect() failed -> {ex}')

    def init_camera_config(self):
        if not self.active:
            return
        
        with self.update_camera_config():
            self.set_pixel_format("Mono10g40IDS") #NOTE: Should be converted to Mono8 on for Pixel Format Destination
            #TODO: auto gain
            self.remote_nodemap.FindNode("ReverseX").SetValue(True)
            self.exposure_t(10)
            self.set_frame_size(1920,1536)

    def is_grabbing(self):
        return False

    def stop_grabbing(self):
        print('stop grab')

    def start_grabbing(self):
        print('start grab')

    def set_frame_size(self, w, h):
        #TODO: update input sizes to closest valid size
        self.remote_nodemap.FindNode("Width").SetValue(w)
        self.remote_nodemap.FindNode("Height").SetValue(h)

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