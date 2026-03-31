# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import atexit
import datetime
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
import ids_peak_ipl

from lvp_logger import logger
from drivers.camera import Camera, ImageHandlerBase
import threading

# IDS Library.Close() shuts down the entire SDK (not per-device).
# Defer to atexit so it only runs once at process exit.
_ids_library_initialized = False


def _ids_library_cleanup():
    global _ids_library_initialized
    if _ids_library_initialized:
        try:
            ids_peak.Library.Close()
        except Exception:
            pass
        _ids_library_initialized = False


atexit.register(_ids_library_cleanup)


class IDSCamera(Camera):
    def __init__(self):

        self.device_manager = None
        self.data_stream = None
        self.remote_nodemap = None

        super().__init__()

    def connect(self) -> bool:
        global _ids_library_initialized
        try:
            #Initialize device manager
            ids_peak.Library.Initialize()
            _ids_library_initialized = True
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
                logger.info(f'[CAM Class ] Camera Firmware Version: {self.remote_nodemap.FindNode("DeviceFirmwareVersion").Value()}')
            except Exception:
                logger.warning('[CAM Class ] Could not read all IDS camera information')

            # Load camera profile and query dynamic capabilities
            self._load_profile()
            self._query_dynamic_capabilities()

            self.cam_image_handler = ImageHandler(self.data_stream, parent_cam=self)

            self.init_camera_config()
            self.start_grabbing()

            self.error_report_count = 0
            logger.info('[CAM Class ] Connected to IDS camera')
            return True

        except ConnectionError as er:
            logger.warning(f'[CAM Class ] IDS camera connect failed: {er}')
        except Exception as ex:
            logger.exception(f'[CAM Class ] IDS camera connect failed: {ex}')
            # Clean up partial state on failure
            self.active = None
            self.remote_nodemap = None
            self.data_stream = None

        return False

    def disconnect(self) -> bool:
        try:
            if self.active:
                try:
                    if self.is_grabbing():
                        self.stop_grabbing()
                except Exception:
                    pass
                self.active = None
                self.remote_nodemap = None
                self.data_stream = None
                self.device_manager = None
                # Library.Close() deferred to atexit — don't call here
                logger.info('[CAM Class ] Disconnected from IDS camera')
                return True
            else:
                logger.info('[CAM Class ] IDS camera not connected')
        except Exception as e:
            logger.exception(f'[CAM Class ] IDS camera disconnect failed: {e}')
        return False

    def is_connected(self) -> bool:
        if self.active in (False, None):
            self._device_removed = True
            return False
        if self._device_removed:
            return False
        return True

    def _query_dynamic_capabilities(self):
        """Query IDS SDK for gain/exposure ranges and merge into profile."""
        if not self.active or not self.remote_nodemap:
            return

        try:
            # Gain range
            try:
                gain_node = self.remote_nodemap.FindNode("Gain")
                self.profile.gain.total_min_db = gain_node.Minimum()
                self.profile.gain.total_max_db = gain_node.Maximum()
                logger.info(f'[CAM Class ] Gain range: {self.profile.gain.total_min_db:.1f} - '
                            f'{self.profile.gain.total_max_db:.1f} dB')
            except Exception as e:
                logger.debug(f'[CAM Class ] Could not query gain range: {e}')

            # Exposure range
            try:
                exp_node = self.remote_nodemap.FindNode("ExposureTime")
                self.profile.exposure_min_us = exp_node.Minimum()
                self.profile.exposure_max_us = exp_node.Maximum()
                logger.info(f'[CAM Class ] Exposure range: {self.profile.exposure_min_us:.0f} - '
                            f'{self.profile.exposure_max_us:.0f} us')
            except Exception as e:
                logger.debug(f'[CAM Class ] Could not query exposure range: {e}')

        except Exception as e:
            logger.warning(f'[CAM Class ] _query_dynamic_capabilities failed: {e}')

    def init_camera_config(self):
        if not self.active:
            return

        try:
            with self.update_camera_config():
                self.remote_nodemap.FindNode("UserSetSelector").SetCurrentEntry("Default")
                self.remote_nodemap.FindNode("UserSetLoad").Execute()
                self.remote_nodemap.FindNode("UserSetLoad").WaitUntilDone()
                # Use lowest-bandwidth format from profile (Mono8 if available,
                # otherwise first listed). Software ConvertTo handles the rest.
                if self.profile.pixel_formats:
                    preferred = 'Mono8' if 'Mono8' in self.profile.pixel_formats else self.profile.pixel_formats[0]
                else:
                    preferred = 'Mono10g40IDS'
                self.set_pixel_format(preferred)
                #TODO: auto gain
                self.remote_nodemap.FindNode("ReverseX").SetValue(True)
                # UserSetDefault caps AcquisitionFrameRate at 10 fps.
                # Must disable the target limiter AND maximize the actual frame rate.
                try:
                    self.remote_nodemap.FindNode("AcquisitionFrameRateTargetEnable").SetValue(False)
                    logger.info('[CAM Class ] Disabled AcquisitionFrameRateTargetEnable')
                except Exception as e:
                    logger.debug(f'[CAM Class ] AcquisitionFrameRateTargetEnable not available: {e}')
                # Switch throughput limit from Sensor to Link mode.
                # In Sensor mode, the limit applies to raw sensor readout (full res)
                # even when using a smaller ROI, capping fps artificially.
                # In Link mode, the limit applies to actual USB transfer rate.
                try:
                    comp = self.remote_nodemap.FindNode("DeviceLinkThroughputLimitComponent")
                    comp.SetCurrentEntry("Link")
                    logger.info('[CAM Class ] DeviceLinkThroughputLimitComponent set to Link')
                except Exception as e:
                    logger.debug(f'[CAM Class ] DeviceLinkThroughputLimitComponent not available: {e}')
                # Maximize USB throughput limit
                try:
                    node = self.remote_nodemap.FindNode("DeviceLinkThroughputLimit")
                    node.SetValue(node.Maximum())
                    logger.info(f'[CAM Class ] DeviceLinkThroughputLimit set to {node.Maximum()} B/s')
                except Exception as e:
                    logger.debug(f'[CAM Class ] DeviceLinkThroughputLimit not available: {e}')
                # Maximize AcquisitionFrameRate (capped by throughput limit)
                try:
                    fr = self.remote_nodemap.FindNode("AcquisitionFrameRate")
                    fr.SetValue(fr.Maximum())
                    logger.info(f'[CAM Class ] AcquisitionFrameRate set to max: {fr.Maximum():.1f} fps')
                except Exception as e:
                    logger.debug(f'[CAM Class ] AcquisitionFrameRate not available: {e}')
                self.exposure_t(10)
                self.set_frame_size(1920,1528)
        except Exception as e:
            logger.error(f'[CAM Class ] init_camera_config failed: {e}')

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

            self.data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
            for buffer in self.data_stream.AnnouncedBuffers():
                self.data_stream.RevokeBuffer(buffer)
        except Exception as e:
            logger.warning(f'[CAM Class ] stop_grabbing ignored error: {e}')

    def start_grabbing(self):
        try:
            # Allocate buffers — minimum + 3 extra to prevent starvation during
            # frame conversion. With only min (2-3), the camera runs out of
            # buffers while ConvertTo holds one, capping throughput at ~10 fps.
            payload_size = self.remote_nodemap.FindNode("PayloadSize").Value()
            num_buffers = self.data_stream.NumBuffersAnnouncedMinRequired() + 3
            for _ in range(num_buffers):
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
        try:
            mins = self.get_min_frame_size()
            maxs = self.get_max_frame_size()

            if not mins or not maxs:
                logger.error('[CAM Class ] set_frame_size: could not read frame size limits')
                return

            #Convert w and h to closest valid values
            width = int(max(mins['width'], min(maxs['width'], w)) / 48) * 48
            height = int(max(mins['height'], min(maxs['height'], h)) / 4) * 4

            with self.update_camera_config():
                self.remote_nodemap.FindNode("Width").SetValue(width)
                self.remote_nodemap.FindNode("Height").SetValue(height)
        except Exception as e:
            logger.error(f'[CAM Class ] set_frame_size failed: {e}')

    def get_min_frame_size(self) -> dict:
        if not self.active:
            return {}

        try:
            return {
                'width': self.remote_nodemap.FindNode("Width").Minimum(),
                'height': self.remote_nodemap.FindNode("Height").Minimum(),
            }
        except Exception as e:
            logger.error(f'[CAM Class ] get_min_frame_size failed: {e}')
            return {}


    def get_max_frame_size(self) -> dict:
        if not self.active:
            return {}

        try:
            return {
                'width': self.remote_nodemap.FindNode("Width").Maximum(),
                'height': self.remote_nodemap.FindNode("Height").Maximum(),
            }
        except Exception as e:
            logger.error(f'[CAM Class ] get_max_frame_size failed: {e}')
            return {}


    def get_frame_size(self):
        if not self.active:
            return

        try:
            width = self.remote_nodemap.FindNode("Width").Value()
            height = self.remote_nodemap.FindNode("Height").Value()
            return {
                'width': width,
                'height': height,
            }
        except Exception as e:
            logger.error(f'[CAM Class ] get_frame_size failed: {e}')
            return None

    def set_pixel_format(self, pixel_format):
        if not self.active:
            return False

        if pixel_format not in self.get_supported_pixel_formats():
            logger.error(f"[CAM Class ] Unsupported pixel format: {pixel_format}")
            return False

        try:
            with self.update_camera_config():
                self.remote_nodemap.FindNode("PixelFormat").SetCurrentEntry(pixel_format)
            return True
        except Exception as e:
            logger.error(f'[CAM Class ] set_pixel_format({pixel_format}) failed: {e}')
            self._mark_disconnected()
            return False

    def get_pixel_format(self):
        try:
            return self.remote_nodemap.FindNode("PixelFormat").CurrentEntry().SymbolicValue()
        except Exception as e:
            logger.error(f'[CAM Class ] get_pixel_format failed: {e}')
            return None

    def get_supported_pixel_formats(self):
        try:
            return tuple(pf.SymbolicValue() for pf in self.remote_nodemap.FindNode("PixelFormat").AvailableEntries())
        except Exception as e:
            logger.error(f'[CAM Class ] get_supported_pixel_formats failed: {e}')
            return ()

    def exposure_t(self, t):
        if not self.active:
            logger.warning(f'[CAM Class ] Cannot set exposure {t}ms: camera inactive')
            return

        if t > self.max_exposure:
            logger.warning(f'[CAM Class ] Exposure {t}ms exceeds max ({self.max_exposure}ms)')
            return

        # IDS allows changing exposure while acquisition is running —
        # no need for update_camera_config() stop/start cycle.
        try:
            self.remote_nodemap.FindNode("ExposureTime").SetValue(float(t)*1000)
            self._last_exposure_ms = float(t)
            # Update grab timeout so long exposures don't cause perpetual timeouts
            if self.cam_image_handler:
                self.cam_image_handler.timeout_ms = max(2000, int(t * 2 + 500))
            logger.info(f'[CAM Class ] Exposure set to {t}ms')
        except Exception as e:
            logger.error(f'[CAM Class ] Exposure set failed (likely out of bounds): {e}')

    def get_exposure_t(self):
        if not self.active:
            logger.warning('[CAM Class ] Cannot read exposure: camera inactive')
            return -1

        try:
            microsec = self.remote_nodemap.FindNode("ExposureTime").Value()
            millisec = microsec / 1000
            return millisec
        except Exception as e:
            logger.error(f'[CAM Class ] get_exposure_t failed: {e}')
            return -1

    def auto_exposure_t(self, state = True):
        #TODO: Implement for IDS cameras that support auto exposure
        try:
            return self.remote_nodemap.HasNode("ExposureAuto")
        except Exception as e:
            logger.error(f'[CAM Class ] auto_exposure_t failed: {e}')
            return False

    def find_model_name(self):
        if not self.active:
            logger.warning('[CAM Class ] Cannot read model name: camera inactive')
            return

        try:
            self.model_name = self.active.ModelName()
            logger.info(f'[CAM Class ] Camera model: {self.model_name}')
        except Exception as e:
            logger.error(f'[CAM Class ] find_model_name failed: {e}')

    def get_all_temperatures(self):
        return {} #TODO: Implement for IDS cameras that support temperature readings

    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float=1.0):
        if not self.active:
            logger.warning('[CAM Class ] set_max_acquisition_frame_rate(): inactive camera')
            return

        try:
            with self.update_camera_config():
                self.remote_nodemap.FindNode("AcquisitionFrameRateTargetEnable").SetValue(enabled)

                if enabled:
                    self.remote_nodemap.FindNode("AcquisitionFrameRateTarget").SetValue(fps)
        except Exception as e:
            logger.error(f'[CAM Class ] set_max_acquisition_frame_rate failed: {e}')

    def set_binning_size(self, size: int) -> bool:
        if not self.active:
            return False

        if size < 1 or size > 2:
            logger.error(f"[CAM Class ] Unsupported bin size: {size}")
            return False

        try:
            logger.debug(f"[CAM Class ] Binning {self.get_binning_size()} -> {size}, frame {self.get_frame_size()}")
            with self.update_camera_config():
                self.remote_nodemap.FindNode("BinningVertical").SetValue(size)
                self.remote_nodemap.FindNode("BinningHorizontal").SetValue(size)

            logger.debug(f"[CAM Class ] Binning set to {self.get_binning_size()}, frame now {self.get_frame_size()}")
            return True
        except Exception as e:
            logger.error(f'[CAM Class ] set_binning_size failed: {e}')
            return False

    def get_binning_size(self) -> int:
        if not self.active:
            return 1

        try:
            vert_bin = self.remote_nodemap.FindNode("BinningVertical").Value()
            horiz_bin = self.remote_nodemap.FindNode("BinningHorizontal").Value()

            if horiz_bin != vert_bin:
                logger.error(f"[CAM Class ] Binning mismatch detected between horizontal ({horiz_bin}) and vertical ({vert_bin})")

            return vert_bin
        except Exception as e:
            logger.error(f'[CAM Class ] get_binning_size failed: {e}')
            return 1

    # grab() inherited from Camera base class

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
            if img.PixelFormat() != ids_peak_ipl.PixelFormatName_Mono8:
                img = img.ConvertTo(ids_peak_ipl.PixelFormatName_Mono8)
            img = img.get_numpy().copy()
            img_ts = datetime.datetime.now()
            self.data_stream.QueueBuffer(buffer)

            self.array = img
            return True, img_ts

        except Exception as e:
            logger.warning(f'[CAM Class ] grab_new_capture failed: {e}')
            return False, None

    def update_auto_gain_target_brightness(self, auto_target_brightness: float):
        #TODO: Implement for IDS cameras that support auto gain
        try:
            return self.remote_nodemap.HasNode("GainAuto")
        except Exception as e:
            logger.error(f'[CAM Class ] update_auto_gain_target_brightness failed: {e}')
            return False

    def update_auto_gain_min_max(self, min_gain: float | None, max_gain: float | None):
        #TODO: Implement for IDS cameras that support auto gain
        try:
            return self.remote_nodemap.HasNode("GainAuto")
        except Exception as e:
            logger.error(f'[CAM Class ] update_auto_gain_min_max failed: {e}')
            return False

    def get_gain(self):
        if not self.active:
            logger.warning('[CAM Class ] Cannot read gain: camera inactive')
            return -1

        try:
            value = self.remote_nodemap.FindNode("Gain").Value()
            return float(value)
        except Exception as e:
            logger.error(f'[CAM Class ] Read gain failed: {e}')
            return -1

    def gain(self, gain):
        if not self.active:
            logger.warning(f'[CAM Class ] Cannot set gain {gain}: camera inactive')
            return

        try:
            self.remote_nodemap.FindNode("GainSelector").SetCurrentEntry("AnalogAll")
            self.remote_nodemap.FindNode("Gain").SetValue(gain)
            logger.info(f'[CAM Class ] Gain set to {gain}')
        except Exception as e:
            logger.error(f'[CAM Class ] Gain set failed (likely out of bounds): {e}')
            return


    def auto_gain(
        self,
        state = True,
        target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None
    ):
        #TODO: Implement functionality for IDS cameras that support auto gain
        try:
            return self.remote_nodemap.HasNode("GainAuto")
        except Exception as e:
            logger.error(f'[CAM Class ] auto_gain failed: {e}')
            return False

    def auto_gain_once(
        self,
        state = True,
        target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None
    ):
        #TODO: Implement functionality for IDS cameras that support auto gain
        try:
            return self.remote_nodemap.HasNode("GainAuto")
        except Exception as e:
            logger.error(f'[CAM Class ] auto_gain_once failed: {e}')
            return False

    def set_test_pattern(self, enabled: bool = False, pattern: str = 'Black'):
        #TODO: Implement
        pass

class ImageHandler(ImageHandlerBase):
    """IDS camera image handler — polls for frames on a background thread."""

    # Override base class: 10 failures × 1s timeout = ~10s disconnect detection
    MAX_CONSECUTIVE_FAILURES = 10

    def __init__(self, data_stream: ids_peak.DataStream, parent_cam: 'IDSCamera'):
        super().__init__()
        self.data_stream = data_stream
        self.timeout_ms = 2000  # Updated by exposure_t() for long exposures
        self._parent = parent_cam
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
            self._grab_thread.join(timeout=5)
            self._grab_thread = None

    def _grab_loop(self):
        # Pre-create converter for Mono10→Mono8 (reuse avoids per-frame alloc)
        try:
            converter = ids_peak_ipl.ImageConverter()
            converter.PreAllocateConversion(
                ids_peak_ipl.PixelFormatName_Mono8, 1920, 1528)
        except Exception:
            converter = None  # Fall back to per-frame ConvertTo

        while not self._stop_event.is_set():
            try:
                buffer = self.data_stream.WaitForFinishedBuffer(self.timeout_ms)
                if buffer.IsIncomplete():
                    self.data_stream.QueueBuffer(buffer)
                    should_stop = self._record_failure()
                    if should_stop:
                        logger.error('[CAM Class ] Too many grab failures; marking device as removed')
                        self._parent._mark_disconnected()
                        break
                    continue

                # BufferToImage copies pixel data out of the SDK buffer.
                # Return the buffer IMMEDIATELY so the camera can reuse it
                # while we do the (slower) format conversion + numpy copy.
                img = ids_peak_ipl_extension.BufferToImage(buffer)
                self.data_stream.QueueBuffer(buffer)

                if img.PixelFormat() != ids_peak_ipl.PixelFormatName_Mono8:
                    if converter:
                        img = converter.Convert(img, ids_peak_ipl.PixelFormatName_Mono8)
                    else:
                        img = img.ConvertTo(ids_peak_ipl.PixelFormatName_Mono8)
                frame = img.get_numpy().copy()
                ts = datetime.datetime.now()
                self._store_frame(frame, ts)
            except Exception as e:
                # WaitForFinishedBuffer timeout is normal — not a failure
                err_str = str(e).lower()
                if 'abort' in err_str or 'removed' in err_str or 'device' in err_str:
                    logger.warning(f'[CAM Class ] Device removal detected in grab loop: {e}')
                    self._parent._mark_disconnected()
                    break
                should_stop = self._record_failure()
                if should_stop:
                    logger.error('[CAM Class ] Too many grab exceptions; marking device as removed')
                    self._parent._mark_disconnected()
                    break
                if self._failed_grabs % 5 == 1:
                    logger.warning(f'[CAM Class ] ImageHandler grab loop exception: {e}')
