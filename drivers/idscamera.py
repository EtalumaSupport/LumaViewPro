# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import atexit
import datetime
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
import ids_peak_ipl

from lvp_logger import logger
try:
    from lvp_logger import camera_logger as _cam_log
except ImportError:
    _cam_log = None
from drivers.camera import Camera, ImageHandlerBase
from drivers.exceptions import HardwareError
from drivers.registry import camera_registry
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


@camera_registry.register('ids', priority=80)
class IDSCamera(Camera):
    # IDS cameras drive the converter pipeline at 8-bit Mono (Mono8 forced
    # at the SDK boundary so downstream code never has to handle Mono10 /
    # Mono12 packed formats). Override the base default so capability
    # consumers (buffer sizing, save-format selection) treat IDS frames as
    # 8-bit from the start.
    native_bit_depth = 8

    def __init__(self):

        self.device_manager = None
        self.data_stream = None
        self.remote_nodemap = None

        # Cache of the active PixelFormat. PixelFormat only changes through
        # set_pixel_format(), so the cache is refreshed there and cleared on
        # disconnect; get_pixel_format() serves from it to avoid a live
        # node-map read on the per-frame image-metadata path.
        self._pixel_format_cache = None

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
                _cam_log.warning('[CAM Class ] Could not read all IDS camera information')

            # Load camera profile and query dynamic capabilities
            self._load_profile()
            self._query_dynamic_capabilities()

            self.cam_image_handler = ImageHandler(self.data_stream, parent_cam=self)

            self.init_camera_config()
            self.start_grabbing()

            logger.info('[CAM Class ] Connected to IDS camera')
            return True

        except ConnectionError as er:
            _cam_log.warning(f'[CAM Class ] IDS camera connect failed: {er}')
        except Exception as ex:
            _cam_log.exception(f'[CAM Class ] IDS camera connect failed: {ex}')
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
                self._pixel_format_cache = None
                # Library.Close() deferred to atexit -- don't call here
                logger.info('[CAM Class ] Disconnected from IDS camera')
                return True
            else:
                logger.info('[CAM Class ] IDS camera not connected')
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] IDS camera disconnect failed: {e}')
        return False

    def is_connected(self) -> bool:
        if self.active in (False, None):
            self._device_removed = True
            return False
        return not self._device_removed

    def read_diagnostic_snapshot(
        self,
        duration_s: float = 3.0,
        drain_camera_side_errors: bool = True,
    ) -> dict:
        """Stub: IDS path not yet supported by the diagnostic probe API.

        The IDS Peak SDK exposes a different node-map structure and
        statistics surface from Pylon. A separate implementation is
        required; not provided in this commit. The stub returns a
        structured "supported=False" response so the API layer can
        report the gap without raising.
        """
        return {
            'connected': self.active not in (False, None),
            'supported': False,
            'reason': 'IDS Peak diagnostic probe not yet implemented; '
                      'Pylon driver only for now.',
            'errors': [],
        }

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
            _cam_log.warning(f'[CAM Class ] _query_dynamic_capabilities failed: {e}')

    def init_camera_config(self):
        if not self.active:
            return

        try:
            with self.update_camera_config():
                self.remote_nodemap.FindNode("UserSetSelector").SetCurrentEntry("Default")
                self.remote_nodemap.FindNode("UserSetLoad").Execute()
                self.remote_nodemap.FindNode("UserSetLoad").WaitUntilDone()
                # Log the camera's actual PixelFormat options once at init --
                # the supported list is camera-specific (IDS uses names like
                # Mono10g40IDS / Mono12g24IDS, and not all sensors expose
                # Mono8 -- e.g. Sony IMX676 in U3-34L0XCP-M is Mono10/12 only).
                # Operators need this in the log to diagnose any future
                # logical-to-camera mismatch.
                supported = self.get_supported_pixel_formats()
                logger.info(
                    f'[CAM Class ] Supported PixelFormat entries: {list(supported)}')
                # Pick the lowest-bandwidth entry the profile lists (cameras
                # with Mono8 stay Mono8; cameras like IMX676 fall through to
                # Mono10g40IDS). set_pixel_format resolves logical names
                # ('Mono8') to camera-specific entries when applicable.
                if self.profile.pixel_formats:
                    preferred = ('Mono8' if 'Mono8' in self.profile.pixel_formats
                                 else self.profile.pixel_formats[0])
                else:
                    preferred = 'Mono10g40IDS'
                self.set_pixel_format(preferred)
                self.remote_nodemap.FindNode("ReverseX").SetValue(True)
                # Ensure freerun mode (no external trigger)
                try:
                    self.remote_nodemap.FindNode("TriggerMode").SetCurrentEntry("Off")
                except Exception:
                    pass
                # Disable frame rate target limiter (UserSetDefault caps at 10 fps)
                try:
                    self.remote_nodemap.FindNode("AcquisitionFrameRateTargetEnable").SetValue(False)
                    logger.info('[CAM Class ] Disabled AcquisitionFrameRateTargetEnable')
                except Exception as e:
                    logger.debug(f'[CAM Class ] AcquisitionFrameRateTargetEnable not available: {e}')
                # Maximize USB throughput limit
                try:
                    node = self.remote_nodemap.FindNode("DeviceLinkThroughputLimit")
                    node.SetValue(node.Maximum())
                    logger.info(f'[CAM Class ] DeviceLinkThroughputLimit set to {node.Maximum()} B/s')
                except Exception as e:
                    logger.debug(f'[CAM Class ] DeviceLinkThroughputLimit not available: {e}')
                # Set resolution and exposure BEFORE maximizing frame rate --
                # AcquisitionFrameRate.Maximum() depends on current resolution,
                # pixel format, and exposure time.
                self.exposure_t(10)
                self.set_frame_size(1920, 1528)
                # NOW maximize frame rate (after resolution is set)
                try:
                    fr = self.remote_nodemap.FindNode("AcquisitionFrameRate")
                    fr.SetValue(fr.Maximum())
                    logger.info(f'[CAM Class ] AcquisitionFrameRate set to max: {fr.Maximum():.1f} fps')
                except Exception as e:
                    logger.debug(f'[CAM Class ] AcquisitionFrameRate not available: {e}')
        except Exception as e:
            _cam_log.error(f'[CAM Class ] init_camera_config failed: {e}')

    def is_grabbing(self):
        if not self.data_stream:
            return False

        return self.data_stream.IsGrabbing()

    def stop_grabbing(self):
        if _cam_log is not None:
            _cam_log.info('ids AcquisitionStop + StopAcquisition + Flush + RevokeBuffers')
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
            if _cam_log is not None:
                _cam_log.warning(f'ids stop_grabbing FAILED: {e}')
            _cam_log.warning(f'[CAM Class ] stop_grabbing ignored error: {e}')

    def start_grabbing(self):
        if _cam_log is not None:
            _cam_log.info('ids start_grabbing: alloc buffers + StartAcquisition + AcquisitionStart')
        try:
            # Allocate buffers -- minimum + 3 extra to prevent starvation during
            # frame conversion. With only min (2-3), the camera runs out of
            # buffers while ConvertTo holds one, capping throughput at ~10 fps.
            payload_size = self.remote_nodemap.FindNode("PayloadSize").Value()
            num_buffers = self.data_stream.NumBuffersAnnouncedMinRequired() + 3
            for _ in range(num_buffers):
                buffer = self.data_stream.AllocAndAnnounceBuffer(payload_size)
                self.data_stream.QueueBuffer(buffer)

            # Re-maximize frame rate -- stop/start cycles reset it.
            # Must be done AFTER resolution is set (max depends on frame size).
            try:
                fr = self.remote_nodemap.FindNode("AcquisitionFrameRate")
                old_val = fr.Value()
                fr.SetValue(fr.Maximum())
                logger.info(f'[CAM Class ] AcquisitionFrameRate {old_val:.1f} -> {fr.Value():.1f} (max={fr.Maximum():.1f})')
            except Exception as e:
                _cam_log.warning(f'[CAM Class ] Failed to re-maximize AcquisitionFrameRate: {e}')

            self.data_stream.StartAcquisition()
            self.remote_nodemap.FindNode("AcquisitionStart").Execute()
            self.remote_nodemap.FindNode("AcquisitionStart").WaitUntilDone()

            if self.cam_image_handler:
                self.cam_image_handler.start()

            logger.info('[CAM Class ] start_grabbing succeeded')
        except Exception as e:
            _cam_log.warning(f'[CAM Class ] start_grabbing ignored error: {e}')

    def set_frame_size(self, w, h):
        try:
            mins = self.get_min_frame_size()
            maxs = self.get_max_frame_size()

            if not mins or not maxs:
                _cam_log.error('[CAM Class ] set_frame_size: could not read frame size limits')
                return

            #Convert w and h to closest valid values
            width = int(max(mins['width'], min(maxs['width'], w)) / 48) * 48
            height = int(max(mins['height'], min(maxs['height'], h)) / 4) * 4

            with self.update_camera_config():
                self.remote_nodemap.FindNode("Width").SetValue(width)
                self.remote_nodemap.FindNode("Height").SetValue(height)
        except Exception as e:
            _cam_log.error(f'[CAM Class ] set_frame_size failed: {e}')

    def get_min_frame_size(self) -> dict:
        if not self.active:
            return {}

        try:
            return {
                'width': self.remote_nodemap.FindNode("Width").Minimum(),
                'height': self.remote_nodemap.FindNode("Height").Minimum(),
            }
        except Exception as e:
            _cam_log.error(f'[CAM Class ] get_min_frame_size failed: {e}')
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
            _cam_log.error(f'[CAM Class ] get_max_frame_size failed: {e}')
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
            _cam_log.error(f'[CAM Class ] get_frame_size failed: {e}')
            return None

    @staticmethod
    def _resolve_logical_format_name(logical: str, supported) -> str | None:
        """Pure-logic resolver: map a logical PixelFormat name to a camera-native
        SymbolicValue from the given supported list.

        Factored out from _resolve_logical_format for unit-testability without
        an SDK connection. Caller passes the camera's actual supported tuple
        (from get_supported_pixel_formats); this returns the chosen name or
        None.

        Mapping rules:
          'Mono8'  -> first entry whose SymbolicValue starts with 'Mono8'
                      (catches Mono8, Mono8g, Mono8p).
          'Mono12' -> first starting with 'Mono12', falling back to 'Mono10'
                      (sensors that max out at 10-bit substitute for 12-bit).
          Anything else: matches verbatim only.
        """
        if not supported:
            return None
        if logical in supported:
            return logical
        prefixes = {
            'Mono8': ('Mono8',),
            'Mono12': ('Mono12', 'Mono10'),
        }
        for prefix in prefixes.get(logical, ()):
            for entry in supported:
                if entry.startswith(prefix):
                    return entry
        return None

    def _resolve_logical_format(self, logical: str) -> str | None:
        """Map a logical PixelFormat name to the camera's actual SymbolicValue.

        IDS Peak cameras don't expose a literal "Mono8" entry on every model:
        they use sensor-specific suffixed names like "Mono10g40IDS" /
        "Mono12g24IDS". Pylon accepts "Mono8" / "Mono12" as wire names
        directly. To keep callers (UI, characterization tool) driver-agnostic
        we accept the same logical names on both and map per-driver here.
        """
        return self._resolve_logical_format_name(
            logical, self.get_supported_pixel_formats())

    def set_pixel_format(self, pixel_format: str) -> bool:
        """Set the camera pixel format.

        Args:
            pixel_format: Symbolic pixel format name (e.g. 'Mono8'). May be
                a logical name; resolves to the camera-specific entry.

        Returns:
            bool: True on success. False only when the camera is inactive
                or the format cannot be resolved (caller-correctable
                guards). Hardware-level failure raises HardwareError.

        Raises:
            HardwareError: SDK call failed. Marks the camera disconnected
                before raising.
        """
        if not self.active:
            return False

        resolved = self._resolve_logical_format(pixel_format)
        if resolved is None:
            supported = self.get_supported_pixel_formats()
            _cam_log.error(
                f"[CAM Class ] Unsupported pixel format: {pixel_format} "
                f"(camera supports: {list(supported)})")
            return False

        if resolved != pixel_format:
            logger.info(
                f'[CAM Class ] Pixel format {pixel_format} -> {resolved} '
                f'(logical-to-camera mapping)')

        try:
            with self.update_camera_config():
                self.remote_nodemap.FindNode("PixelFormat").SetCurrentEntry(resolved)
            self._pixel_format_cache = resolved
            return True
        except Exception as e:
            _cam_log.error(f'[CAM Class ] set_pixel_format({resolved}) failed: {e}')
            self._mark_disconnected()
            raise HardwareError(
                f'set_pixel_format({resolved}) failed: {type(e).__name__}: {e}'
            ) from e

    def get_pixel_format(self):
        # Served from the cache populated on first read and on every
        # set_pixel_format(); PixelFormat only changes through that setter.
        # get_camera_info() reads this once per saved frame, so a live
        # node-map read here would touch the SDK on every capture.
        if self._pixel_format_cache is not None:
            return self._pixel_format_cache
        try:
            value = self.remote_nodemap.FindNode("PixelFormat").CurrentEntry().SymbolicValue()
            self._pixel_format_cache = value
            return value
        except Exception as e:
            _cam_log.error(f'[CAM Class ] get_pixel_format failed: {e}')
            return None

    def get_supported_pixel_formats(self):
        try:
            return tuple(pf.SymbolicValue() for pf in self.remote_nodemap.FindNode("PixelFormat").AvailableEntries())
        except Exception as e:
            _cam_log.error(f'[CAM Class ] get_supported_pixel_formats failed: {e}')
            return ()

    def exposure_t(self, exposure_ms):
        if not self.active:
            _cam_log.warning(f'[CAM Class ] Cannot set exposure {exposure_ms}ms: camera inactive')
            return

        if exposure_ms > self.max_exposure:
            _cam_log.warning(f'[CAM Class ] Exposure {exposure_ms}ms exceeds max ({self.max_exposure}ms)')
            return

        # IDS allows changing exposure while acquisition is running --
        # no need for update_camera_config() stop/start cycle.
        try:
            us_value = float(exposure_ms)*1000
            if _cam_log is not None:
                _cam_log.info(f'ids ExposureTime.SetValue({us_value:.0f}us) (={exposure_ms}ms)')
            self.remote_nodemap.FindNode("ExposureTime").SetValue(us_value)
            self._last_exposure_ms = float(exposure_ms)
            # Update grab timeout so long exposures don't cause perpetual timeouts
            if self.cam_image_handler:
                self.cam_image_handler.timeout_ms = max(2000, int(exposure_ms * 2 + 500))
            logger.info(f'[CAM Class ] Exposure set to {exposure_ms}ms')
        except Exception as e:
            if _cam_log is not None:
                _cam_log.error(f'ids ExposureTime.SetValue({exposure_ms}ms) FAILED: {e}')
            _cam_log.error(f'[CAM Class ] Exposure set failed (likely out of bounds): {e}')

    def get_exposure_t(self):
        if not self.active:
            _cam_log.warning('[CAM Class ] Cannot read exposure: camera inactive')
            return -1

        try:
            microsec = self.remote_nodemap.FindNode("ExposureTime").Value()
            millisec = microsec / 1000
            return millisec
        except Exception as e:
            _cam_log.error(f'[CAM Class ] get_exposure_t failed: {e}')
            return -1

    def auto_exposure_t(self, state = True):
        try:
            return self.remote_nodemap.HasNode("ExposureAuto")
        except Exception as e:
            _cam_log.error(f'[CAM Class ] auto_exposure_t failed: {e}')
            return False

    def get_all_temperatures(self):
        return {}

    def set_device_link_throughput_limit(
        self,
        mode: str,
        value_bps: int | None = None,
    ) -> bool:
        """IDS Peak SDK exposes only DeviceLinkThroughputLimit (no
        DeviceLinkThroughputLimitMode) per the per-IDS-camera nodemap.
        Setting the value to its Maximum() at init disables effective
        throttling -- which is what init_camera_config already does
        at line 199-203. Mode='Off' here maps to "set to Maximum()";
        Mode='On' with value_bps maps to "set to value_bps".

        Returns True on success, False if camera inactive or the node
        is not present on this IDS body. Does not raise.
        """
        if not self.active or self.remote_nodemap is None:
            return False
        try:
            node = self.remote_nodemap.FindNode("DeviceLinkThroughputLimit")
            if node is None:
                return False
            if mode == 'Off':
                node.SetValue(node.Maximum())
            elif mode == 'On':
                if value_bps is not None:
                    node.SetValue(int(value_bps))
                # if value_bps is None we leave the existing value alone
            else:
                return False
            return True
        except Exception as e:
            _cam_log.warning(
                f'[CAM Class ] IDS set_device_link_throughput_limit('
                f'{mode}, {value_bps}) failed: {e}'
            )
            return False

    def set_acquisition_stop_mode(self, mode: str) -> bool:
        """IDS Peak SDK does not expose BslAcquisitionStopMode (Basler-
        specific node). Stub returns False on all input so the API
        method's hasattr-and-call shape is identical for both drivers.
        """
        return False

    def set_bandwidth_reserve_mode(self, mode: str) -> bool:
        """IDS does not expose Pylon BandwidthReserveMode. Stub False."""
        return False

    def set_gev_packet_size(self, size_bytes: int) -> bool:
        """IDS does not expose Pylon GevSCPSPacketSize. Stub False."""
        return False

    def set_gev_inter_packet_delay(self, delay_ticks: int) -> bool:
        """IDS does not expose Pylon GevSCPD. Stub False."""
        return False

    def set_max_transfer_size(self, value_bytes: int) -> bool:
        """IDS does not expose Pylon StreamGrabber MaxTransferSize. Stub False."""
        return False

    def set_num_max_queued_urbs(self, value: int) -> bool:
        """IDS does not expose Pylon StreamGrabber NumMaxQueuedUrbs. Stub False."""
        return False

    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float=1.0):
        if not self.active:
            _cam_log.warning('[CAM Class ] set_max_acquisition_frame_rate(): inactive camera')
            return

        # IDS allows changing AcquisitionFrameRateTargetEnable +
        # AcquisitionFrameRateTarget while acquisition is running (same
        # runtime-parameter class as ExposureTime, see exposure_t above).
        # Previous wrap in update_camera_config() forced an unnecessary
        # stop_grabbing/start_grabbing cycle on every call (same class as
        # STALL-1's per-step wrapper).
        try:
            if _cam_log is not None:
                _cam_log.info(
                    f'ids AcquisitionFrameRateTargetEnable.SetValue({enabled})'
                    + (f' AcquisitionFrameRateTarget.SetValue({fps})' if enabled else ''))
            self.remote_nodemap.FindNode("AcquisitionFrameRateTargetEnable").SetValue(enabled)
            if enabled:
                self.remote_nodemap.FindNode("AcquisitionFrameRateTarget").SetValue(fps)
        except Exception as e:
            if _cam_log is not None:
                _cam_log.error(f'ids AcquisitionFrameRateTarget*({enabled}, {fps}) FAILED: {e}')
            _cam_log.error(f'[CAM Class ] set_max_acquisition_frame_rate failed: {e}')

    def set_binning_size(self, size: int) -> bool:
        """Set camera pixel binning size.

        Args:
            size: Binning factor (1 or 2; IDS bodies cap at 2x2).

        Returns:
            bool: True on success. False only when the camera is inactive
                or size is out of range (caller-correctable guards).
                Hardware-level failure raises HardwareError.

        Raises:
            HardwareError: SDK call failed.
        """
        if not self.active:
            return False

        if size < 1 or size > 2:
            _cam_log.error(f"[CAM Class ] Unsupported bin size: {size}")
            return False

        try:
            logger.debug(f"[CAM Class ] Binning {self.get_binning_size()} -> {size}, frame {self.get_frame_size()}")
            with self.update_camera_config():
                self.remote_nodemap.FindNode("BinningVertical").SetValue(size)
                self.remote_nodemap.FindNode("BinningHorizontal").SetValue(size)

            logger.debug(f"[CAM Class ] Binning set to {self.get_binning_size()}, frame now {self.get_frame_size()}")
            return True
        except Exception as e:
            _cam_log.error(f'[CAM Class ] set_binning_size failed: {e}')
            raise HardwareError(
                f'set_binning_size({size}) failed: {type(e).__name__}: {e}'
            ) from e

    def get_binning_size(self) -> int:
        if not self.active:
            return 1

        try:
            vert_bin = self.remote_nodemap.FindNode("BinningVertical").Value()
            horiz_bin = self.remote_nodemap.FindNode("BinningHorizontal").Value()

            if horiz_bin != vert_bin:
                _cam_log.error(f"[CAM Class ] Binning mismatch detected between horizontal ({horiz_bin}) and vertical ({vert_bin})")

            return vert_bin
        except Exception as e:
            _cam_log.error(f'[CAM Class ] get_binning_size failed: {e}')
            return 1

    # grab() inherited from Camera base class

    def grab_new_capture(self, timeout_s):
        if not self.cam_image_handler:
            return False, None

        try:
            buffer = self.data_stream.WaitForFinishedBuffer(timeout_s)
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
            _cam_log.warning(f'[CAM Class ] grab_new_capture failed: {e}')
            return False, None

    def update_auto_gain_target_brightness(self, auto_target_brightness: float):
        try:
            return self.remote_nodemap.HasNode("GainAuto")
        except Exception as e:
            _cam_log.error(f'[CAM Class ] update_auto_gain_target_brightness failed: {e}')
            return False

    def update_auto_gain_min_max(self, min_gain_db: float | None, max_gain_db: float | None):
        try:
            return self.remote_nodemap.HasNode("GainAuto")
        except Exception as e:
            _cam_log.error(f'[CAM Class ] update_auto_gain_min_max failed: {e}')
            return False

    def get_gain(self):
        if not self.active:
            _cam_log.warning('[CAM Class ] Cannot read gain: camera inactive')
            return -1

        try:
            value = self.remote_nodemap.FindNode("Gain").Value()
            return float(value)
        except Exception as e:
            _cam_log.error(f'[CAM Class ] Read gain failed: {e}')
            return -1

    def gain(self, gain):
        if not self.active:
            if _cam_log is not None:
                _cam_log.warning(f'ids Gain.SetValue({gain}) SKIPPED: active=None')
            _cam_log.warning(f'[CAM Class ] Cannot set gain {gain}: camera inactive')
            return

        try:
            if _cam_log is not None:
                _cam_log.info(f'ids GainSelector=AnalogAll Gain.SetValue({float(gain):.3f})')
            self.remote_nodemap.FindNode("GainSelector").SetCurrentEntry("AnalogAll")
            self.remote_nodemap.FindNode("Gain").SetValue(gain)
            logger.info(f'[CAM Class ] Gain set to {gain}')
        except Exception as e:
            if _cam_log is not None:
                _cam_log.error(f'ids Gain.SetValue({gain}) FAILED: {e}')
            _cam_log.error(f'[CAM Class ] Gain set failed (likely out of bounds): {e}')
            return


    def auto_gain(
        self,
        state = True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
        ae_max_exposure_ms: float | None = None
    ):
        try:
            return self.remote_nodemap.HasNode("GainAuto")
        except Exception as e:
            _cam_log.error(f'[CAM Class ] auto_gain failed: {e}')
            return False

    def auto_gain_once(
        self,
        state = True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
        ae_max_exposure_ms: float | None = None
    ):
        try:
            return self.remote_nodemap.HasNode("GainAuto")
        except Exception as e:
            _cam_log.error(f'[CAM Class ] auto_gain_once failed: {e}')
            return False

    def set_test_pattern(self, enabled: bool = False, pattern: str = 'Black'):
        pass

class ImageHandler(ImageHandlerBase):
    """IDS camera image handler -- polls for frames on a background thread."""

    # Override base class: 10 failures x 1s timeout = ~10s disconnect detection
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
        # Pre-create converter for Mono10->Mono8 (reuse avoids per-frame alloc)
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
                    # Log every incomplete buffer with fill info. Partial-fill
                    # extent is the only signal we get for USB packet loss /
                    # bandwidth saturation; throttling loses the cause
                    # distribution.
                    try:
                        bsize = buffer.SizeFilled() if hasattr(buffer, 'SizeFilled') else None
                        bcap = buffer.Size() if hasattr(buffer, 'Size') else None
                    except Exception as _bintrospect:
                        bsize, bcap = None, f'<introspect failed: {_bintrospect!r}>'
                    _cam_log.warning(
                        f'[CAM Class ] IDS buffer.IsIncomplete()=True '
                        f'filled={bsize} capacity={bcap}'
                    )
                    self.data_stream.QueueBuffer(buffer)
                    should_stop = self._record_failure()
                    if should_stop:
                        _cam_log.error('[CAM Class ] Too many grab failures; marking device as removed')
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
                err_str = str(e).lower()
                if 'abort' in err_str or 'removed' in err_str or 'device' in err_str:
                    _cam_log.warning(f'[CAM Class ] Device removal detected in grab loop: {e}')
                    self._parent._mark_disconnected()
                    break
                # Log every grab-loop exception. Type + message may vary
                # between failures (timeout vs malformed buffer vs SDK-internal
                # fault); throttling loses the distribution. !r on `e` so
                # any non-ASCII in the SDK message is escaped at format time.
                _cam_log.warning(
                    f'[CAM Class ] ImageHandler grab loop exception: '
                    f'{type(e).__name__}: {e!r}'
                )
                should_stop = self._record_failure()
                if should_stop:
                    _cam_log.error('[CAM Class ] Too many grab exceptions; marking device as removed')
                    self._parent._mark_disconnected()
                    break
