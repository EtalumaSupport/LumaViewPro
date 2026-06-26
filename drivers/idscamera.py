# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import atexit
import datetime
import math
import re
import threading

from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
import ids_peak_ipl

from lvp_logger import logger

try:
    from lvp_logger import camera_logger as _cam_log
except ImportError:
    # Fall back to the main logger so every _cam_log call site stays
    # safe -- the dedicated camera log is an enhancement, not a
    # dependency, and dozens of call sites use _cam_log unguarded.
    _cam_log = logger
from drivers.camera import Camera, ImageHandlerBase
from drivers.exceptions import HardwareError
from drivers.registry import camera_registry

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


def ids_significant_bits(wire_format_name: str) -> int:
    """Payload depth of a frame delivered in the given wire PixelFormat.

    Derived from the GenICam format name so the depth tracks the sensor's
    real format rather than the (16-bit) container the unpacked frame rides
    in: Mono12* -> 12, Mono10* -> 10, Mono8* -> 8. Falls back to the leading
    bit count in the name, then to 8. Pure logic -- no SDK call -- so it is
    unit-testable without a camera.
    """
    if wire_format_name.startswith('Mono12'):
        return 12
    if wire_format_name.startswith('Mono10'):
        return 10
    if wire_format_name.startswith('Mono8'):
        return 8
    match = re.search(r'Mono(\d+)', wire_format_name or '')
    return int(match.group(1)) if match else 8


def _ids_ipl_target(wire_format_name: str):
    """IPL ConvertTo target that unpacks a wire format to its native depth.

    Mono10/Mono12 unpack to a right-aligned uint16; Mono8 stays 8-bit. The
    SDK does the unpack (faster than a hand-unpacker and the alignment is the
    SDK's own); drivers/ids_unpack.py is the bench cross-check for the target.
    """
    if wire_format_name.startswith('Mono12'):
        return ids_peak_ipl.PixelFormatName_Mono12
    if wire_format_name.startswith('Mono10'):
        return ids_peak_ipl.PixelFormatName_Mono10
    return ids_peak_ipl.PixelFormatName_Mono8


@camera_registry.register('ids', priority=80)
class IDSCamera(Camera):
    """IDS Peak driver for the U3-34L0XCP-M (Sony IMX676, packed Mono10/12).

    Delivers each frame at the sensor's native depth in a right-aligned uint16
    container (significant_bits derived from the active wire format), so the
    container width is the base default 16 and the payload depth is per-frame.
    """

    # The camera free-runs at its full sensor/USB rate -- no software fps cap.
    # The two-stage grab pipeline drains buffers as fast as they arrive and the
    # converter only ever unpacks the newest frame, so a high acquisition rate
    # cannot exhaust the buffer pool (the crash an earlier soft cap papered
    # over). See _configure_free_run for the throttles that get removed.

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
            # Initialize device manager
            ids_peak.Library.Initialize()
            _ids_library_initialized = True
            self.device_manager = ids_peak.DeviceManager.Instance()
            self.device_manager.Update()

            # Search for devices
            if self.device_manager.Devices().empty():
                raise ConnectionError('Could not find IDS camera')

            self.active = self.device_manager.Devices()[0].OpenDevice(
                ids_peak.DeviceAccessType_Control
            )
            self.data_stream = self.active.DataStreams()[0].OpenDataStream()
            self.remote_nodemap = self.active.RemoteDevice().NodeMaps()[0]
            self._device_removed = False

            try:
                self.model_name = self.active.ModelName()
                self._device_serial = self.active.SerialNumber()
                logger.info(f'[CAM Class ] Camera Model: {self.model_name}')
                logger.info(f'[CAM Class ] Camera Serial Number: {self._device_serial}')
                logger.info(
                    f'[CAM Class ] Camera Firmware Version: {self.remote_nodemap.FindNode("DeviceFirmwareVersion").Value()}'
                )
            except Exception:
                _cam_log.warning('[CAM Class ] Could not read all IDS camera information')

            # Load camera profile and query dynamic capabilities
            self._load_profile()
            self._query_dynamic_capabilities()

            self.cam_image_handler = ImageHandler(self.data_stream, parent_cam=self)

            self.init_camera_config()
            # connect() returns CONFIGURED but NOT grabbing; the single
            # start fires later via open_and_start() (the start gate).

            logger.info('[CAM Class ] Connected to IDS camera')
            return True

        except ConnectionError as er:
            _cam_log.warning(f'[CAM Class ] IDS camera connect failed: {er}')
        except Exception as ex:
            # No-device / no-GenTL-path is the common case here and is an
            # expected probe outcome; log the type + message, not the stack.
            _cam_log.error(f'[CAM Class ] IDS camera connect failed: {type(ex).__name__}: {ex}')
            # Clean up partial state on failure
            self.active = None
            self.remote_nodemap = None
            self.data_stream = None
            self._pixel_format_cache = None

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
            'reason': 'IDS Peak diagnostic probe not yet implemented; Pylon driver only for now.',
            'errors': [],
        }

    def _query_dynamic_capabilities(self):
        """Query IDS SDK for gain/exposure ranges and merge into profile."""
        if not self.active or not self.remote_nodemap:
            return

        try:
            # Gain range. The IDS Gain node is a linear multiplier (min ~1.0x),
            # but LVP's gain model is dB (shared with the Pylon driver), so the
            # profile range is reported in dB: dB = 20 * log10(factor).
            try:
                gain_node = self.remote_nodemap.FindNode('Gain')
                min_factor = gain_node.Minimum()
                max_factor = gain_node.Maximum()
                self.profile.gain.total_min_db = (
                    20.0 * math.log10(min_factor) if min_factor > 0 else 0.0
                )
                self.profile.gain.total_max_db = (
                    20.0 * math.log10(max_factor) if max_factor > 0 else 0.0
                )
                logger.info(
                    f'[CAM Class ] Gain range: {self.profile.gain.total_min_db:.1f} - '
                    f'{self.profile.gain.total_max_db:.1f} dB '
                    f'({min_factor:.2f}-{max_factor:.2f}x)'
                )
            except Exception as e:
                logger.debug(f'[CAM Class ] Could not query gain range: {e}')

            # Exposure range
            try:
                exp_node = self.remote_nodemap.FindNode('ExposureTime')
                self.profile.exposure_min_us = exp_node.Minimum()
                self.profile.exposure_max_us = exp_node.Maximum()
                logger.info(
                    f'[CAM Class ] Exposure range: {self.profile.exposure_min_us:.0f} - '
                    f'{self.profile.exposure_max_us:.0f} us'
                )
            except Exception as e:
                logger.debug(f'[CAM Class ] Could not query exposure range: {e}')

        except Exception as e:
            _cam_log.warning(f'[CAM Class ] _query_dynamic_capabilities failed: {e}')

    def init_camera_config(self):
        if not self.active:
            return

        try:
            with self.update_camera_config():
                self.remote_nodemap.FindNode('UserSetSelector').SetCurrentEntry('Default')
                self.remote_nodemap.FindNode('UserSetLoad').Execute()
                self.remote_nodemap.FindNode('UserSetLoad').WaitUntilDone()
                # UserSetLoad reset the hardware PixelFormat to the user-set
                # default; drop the cache so the set_pixel_format() below
                # actually applies -- its same-value short-circuit must not
                # match a stale cached value against freshly-reset hardware.
                self._pixel_format_cache = None
                # Log the camera's actual PixelFormat options once at init --
                # the supported list is camera-specific (IDS uses names like
                # Mono10g40IDS / Mono12g24IDS, and not all sensors expose
                # Mono8 -- e.g. Sony IMX676 in U3-34L0XCP-M is Mono10/12 only).
                # Operators need this in the log to diagnose any future
                # logical-to-camera mismatch.
                supported = self.get_supported_pixel_formats()
                logger.info(f'[CAM Class ] Supported PixelFormat entries: {list(supported)}')
                # Pick the lowest-bandwidth entry the profile lists (cameras
                # with Mono8 stay Mono8; cameras like IMX676 fall through to
                # Mono10g40IDS). set_pixel_format resolves logical names
                # ('Mono8') to camera-specific entries when applicable.
                if self.profile.pixel_formats:
                    preferred = (
                        'Mono8'
                        if 'Mono8' in self.profile.pixel_formats
                        else self.profile.pixel_formats[0]
                    )
                else:
                    preferred = 'Mono10g40IDS'
                self.set_pixel_format(preferred)
                self.remote_nodemap.FindNode('ReverseX').SetValue(True)
                # Ensure freerun mode (no external trigger)
                try:
                    self.remote_nodemap.FindNode('TriggerMode').SetCurrentEntry('Off')
                except Exception:
                    pass
                # Set geometry and exposure BEFORE configuring the rate -- the
                # throughput ceiling and the AcquisitionFrameRate max both
                # depend on the active resolution, so the rate config has to
                # follow the ROI set. 1900x1900 is the standard LVP camera
                # resolution (the Pylon driver sets the same), matching the
                # scope's centered square field; the prior 1920x1528 was an
                # arbitrary driver-local divergence.
                self.exposure_t(10)
                self.set_frame_size(1900, 1900)
                self._configure_free_run()
                # One-shot diagnostic so a bundle shows why the free-run rate is
                # what it is (esp. whether the Component=Link keystone applied).
                self._log_free_run_state()
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

            self.remote_nodemap.FindNode('AcquisitionStop').Execute()
            self.remote_nodemap.FindNode('AcquisitionStop').WaitUntilDone()
            self.data_stream.StopAcquisition()

            # Release the transport-layer parameter lock taken in start_grabbing
            # (IDS brackets acquisition with TLParamsLocked 1/0).
            try:
                self.remote_nodemap.FindNode('TLParamsLocked').SetValue(0)
            except Exception as e:
                logger.debug(f'[CAM Class ] TLParamsLocked=0 not available: {e}')

            self.data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
            for buffer in self.data_stream.AnnouncedBuffers():
                self.data_stream.RevokeBuffer(buffer)
        except Exception as e:
            if _cam_log is not None:
                _cam_log.warning(f'ids stop_grabbing FAILED: {e}')
            _cam_log.warning(f'[CAM Class ] stop_grabbing ignored error: {e}')

    def start_grabbing(self):
        if self.is_grabbing():
            if _cam_log is not None:
                _cam_log.info('ids start_grabbing SKIPPED: already grabbing')
            return
        if _cam_log is not None:
            _cam_log.info('ids start_grabbing: alloc buffers + StartAcquisition + AcquisitionStart')
        try:
            # Allocate buffers -- minimum + extra so the pool never starves while
            # the unpack worker holds a buffer through ConvertTo. In flight at
            # once: the worker's buffer, the newest-wins slot's buffer, and the
            # poll thread's just-grabbed buffer; the margin keeps free buffers
            # for the camera to fill at the (now uncapped) acquisition rate.
            payload_size = self.remote_nodemap.FindNode('PayloadSize').Value()
            num_buffers = self.data_stream.NumBuffersAnnouncedMinRequired() + 5
            for _ in range(num_buffers):
                buffer = self.data_stream.AllocAndAnnounceBuffer(payload_size)
                self.data_stream.QueueBuffer(buffer)

            # Re-assert free-run -- a stop/start cycle resets the rate config,
            # and AcquisitionFrameRate's max depends on the current resolution.
            self._configure_free_run()

            # Lock transport-layer params for the streaming session, then start
            # the host stream before the device (IDS example ordering).
            try:
                self.remote_nodemap.FindNode('TLParamsLocked').SetValue(1)
            except Exception as e:
                logger.debug(f'[CAM Class ] TLParamsLocked=1 not available: {e}')

            self.data_stream.StartAcquisition()
            self.remote_nodemap.FindNode('AcquisitionStart').Execute()
            self.remote_nodemap.FindNode('AcquisitionStart').WaitUntilDone()

            if self.cam_image_handler:
                self.cam_image_handler.start()

            logger.info('[CAM Class ] start_grabbing succeeded')
        except Exception as e:
            _cam_log.warning(f'[CAM Class ] start_grabbing ignored error: {e}')

    def _configure_free_run(self):
        """Remove the throttles that cap the IDS frame rate so the camera runs
        at its full sensor/USB rate. The two-stage grab pipeline drains buffers
        as fast as they arrive and the converter sees only the newest frame, so
        no software cap is needed to keep the buffer pool from exhausting.

        Three throttles, set in order (each is an independent ceiling; the
        lowest wins, so all must be lifted):
          - DeviceLinkThroughputLimitComponent = 'Link': the default 'Sensor'
            mode computes the limit against the full raw sensor readout even for
            a smaller ROI, capping fps; 'Link' applies it to the actual USB
            transfer instead. This is the keystone -- without it the limit
            throttles the rate no matter how high the limit value is.
          - DeviceLinkThroughputLimit = Maximum(): no bandwidth throttle.
          - AcquisitionFrameRateTarget disabled, then AcquisitionFrameRate
            maximized: the UserSetDefault enables a low rate-target limiter;
            disabling it and pushing the rate to its max lets the camera
            free-run at whatever the throughput ceiling allows.
        """
        if not self.active or self.remote_nodemap is None:
            return

        try:
            comp = self.remote_nodemap.FindNode('DeviceLinkThroughputLimitComponent')
            comp.SetCurrentEntry('Link')
            logger.info('[CAM Class ] DeviceLinkThroughputLimitComponent set to Link')
        except Exception as e:
            logger.debug(f'[CAM Class ] DeviceLinkThroughputLimitComponent not available: {e}')

        try:
            node = self.remote_nodemap.FindNode('DeviceLinkThroughputLimit')
            node.SetValue(node.Maximum())
            logger.info(f'[CAM Class ] DeviceLinkThroughputLimit set to {node.Maximum()} B/s')
        except Exception as e:
            logger.debug(f'[CAM Class ] DeviceLinkThroughputLimit not available: {e}')

        # Drop the rate-target limiter, then push the acquisition rate to max.
        self.set_max_acquisition_frame_rate(False)
        try:
            fr = self.remote_nodemap.FindNode('AcquisitionFrameRate')
            fr.SetValue(fr.Maximum())
            logger.info(f'[CAM Class ] AcquisitionFrameRate set to max: {fr.Maximum()} fps')
        except Exception as e:
            logger.debug(f'[CAM Class ] AcquisitionFrameRate not available: {e}')

    def _log_free_run_state(self):
        """One-shot diagnostic of the throttles that govern the free-run rate, so
        a log bundle is self-sufficient about why the rate is what it is. The
        keystone is DeviceLinkThroughputLimitComponent='Link': if it did not
        apply (node absent, or no 'Link' entry on this body), the limit is
        computed against the full raw readout and the rate stays Sensor-throttled
        well below the wire ceiling -- so its failure is logged at WARNING with
        the node's available entries, not swallowed."""
        if not self.active or self.remote_nodemap is None:
            return

        try:
            comp = self.remote_nodemap.FindNode('DeviceLinkThroughputLimitComponent')
            entries = [e.SymbolicValue() for e in comp.AvailableEntries()]
            current = comp.CurrentEntry().SymbolicValue()
            if current == 'Link':
                logger.info(
                    f'[CAM Class ] Free-run: ThroughputLimitComponent=Link applied '
                    f'(available={entries})'
                )
            else:
                _cam_log.warning(
                    f'[CAM Class ] Free-run: ThroughputLimitComponent={current}, NOT Link '
                    f'(available={entries}) -- rate stays Sensor-throttled below the wire ceiling'
                )
        except Exception as e:
            _cam_log.warning(
                f'[CAM Class ] Free-run: DeviceLinkThroughputLimitComponent node '
                f'unavailable on this body ({type(e).__name__}: {e}) -- cannot switch to Link'
            )

        def _value_and_max(name):
            try:
                node = self.remote_nodemap.FindNode(name)
                return node.Value(), node.Maximum()
            except Exception as e:
                return None, f'<unavailable: {type(e).__name__}>'

        dltl, dltl_max = _value_and_max('DeviceLinkThroughputLimit')
        rate, rate_max = _value_and_max('AcquisitionFrameRate')
        frame = self.get_frame_size() or {}
        logger.info(
            f'[CAM Class ] Free-run state: frame={frame.get("width")}x{frame.get("height")} '
            f'pixel_format={self.get_pixel_format()} '
            f'DeviceLinkThroughputLimit={dltl}/{dltl_max} B/s '
            f'AcquisitionFrameRate={rate}/{rate_max} fps'
        )

    def benchmark_unpack(self, n_frames: int = 200) -> dict:
        """Decode each packed buffer by BOTH the SDK ConvertTo and the numpy
        ids_unpack path, comparing them for correctness and speed.

        The IMX676 body delivers packed Mono10g40IDS / Mono12g24IDS; every frame
        must be unpacked to a right-aligned uint16 before anything can use it.
        The SDK ConvertTo is the host throughput bottleneck that holds live
        display below the sensor rate; a hand-rolled numpy unpack (drivers/
        ids_unpack) is the candidate replacement. ConvertTo is the correctness
        ORACLE: IDS gives the packed layout only as a figure with no per-bit
        text, so the numpy decode is derived -- a bit-for-bit match against
        ConvertTo on real frames proves the layout and the right-alignment. The
        per-frame timings then say whether numpy is actually faster on this host.

        Pauses the unpack worker and drives the data stream directly so each
        finished buffer is decoded by both paths before it is re-queued, then
        restores the normal grab pipeline. Returns a results dict; the live
        unpack path is untouched.
        """
        import time
        import numpy as np

        from drivers import ids_unpack

        results = {
            'n_requested': n_frames,
            'n_compared': 0,
            'mismatches': 0,
            'first_mismatch': None,
            'packed_dtype': None,
            'wire_format': None,
            'width': None,
            'height': None,
            'available_formats': [],
            'icv': {},
        }
        if not self.active or not self.data_stream:
            results['error'] = 'camera not connected / no data stream'
            return results

        # The device's real PixelFormat menu: a newer SDK could expose an
        # unpacked format that removes the need for any host unpack at all.
        try:
            results['available_formats'] = [
                e.SymbolicValue()
                for e in self.remote_nodemap.FindNode('PixelFormat').AvailableEntries()
            ]
        except Exception as e:
            _cam_log.warning(f'[CAM Class ] benchmark: PixelFormat enum read failed: {e}')

        # Yes/no datapoint on whether the newer IDS ICV conversion library is
        # importable on this install -- not itself a benchmark.
        for _icv_import in ('ids_peak_icv', 'ids_peak.ids_peak_icv'):
            try:
                mod = __import__(_icv_import, fromlist=['_'])
                results['icv'] = {
                    'importable': True,
                    'as': _icv_import,
                    'version': getattr(mod, '__version__', '?'),
                }
                break
            except Exception as e:
                results['icv'] = {'importable': False, 'error': f'{type(e).__name__}: {e}'}

        wire = self.get_pixel_format()
        frame = self.get_frame_size() or {}
        width, height = frame.get('width'), frame.get('height')
        results.update(wire_format=wire, width=width, height=height)
        if not width or not height:
            results['error'] = 'could not read frame size'
            return results
        target = _ids_ipl_target(wire)

        # Pause the worker so this loop owns the finished-buffer flow, and clear
        # any KillWait the stop posted so the first wait is not aborted on arrival.
        handler = self.cam_image_handler
        if handler is not None:
            handler.stop()
        try:
            self.data_stream.FlushPendingKillWaits()
        except Exception as e:
            logger.debug(f'[CAM Class ] benchmark: FlushPendingKillWaits unavailable: {e}')

        convert_ms, numpy_ms = [], []
        try:
            for _ in range(n_frames):
                try:
                    buffer = self.data_stream.WaitForFinishedBuffer(2000)
                except Exception as e:
                    _cam_log.warning(f'[CAM Class ] benchmark: WaitForFinishedBuffer: {e}')
                    continue
                try:
                    if buffer.IsIncomplete():
                        continue
                    img = ids_peak_ipl_extension.BufferToImage(buffer)

                    # ConvertTo path (the current production unpack + the oracle).
                    # Bind the converted image to a local before get_numpy():
                    # get_numpy() returns a view that does NOT keep the IPL image
                    # alive, so a chained ConvertTo(...).get_numpy().copy() frees
                    # the converted buffer the instant get_numpy() returns and
                    # copy() then reads freed memory (access violation). Use a
                    # SEPARATE name from img -- img stays the packed source the
                    # numpy path reads just below.
                    t0 = time.perf_counter()
                    conv_img = img.ConvertTo(target)
                    conv = conv_img.get_numpy().copy()
                    convert_ms.append((time.perf_counter() - t0) * 1000.0)

                    # numpy path: the raw packed wire bytes (uint8 since SDK 2.21)
                    # decoded by our own unpacker. ConvertTo does not mutate img,
                    # so the original is still the packed source here.
                    packed = img.get_numpy_1D()
                    if results['packed_dtype'] is None:
                        results['packed_dtype'] = str(packed.dtype)
                    t1 = time.perf_counter()
                    np_arr = ids_unpack.unpack(wire, packed, width, height)
                    numpy_ms.append((time.perf_counter() - t1) * 1000.0)

                    results['n_compared'] += 1
                    if not np.array_equal(conv, np_arr):
                        results['mismatches'] += 1
                        if results['first_mismatch'] is None:
                            diff = np.argwhere(conv != np_arr)
                            results['first_mismatch'] = {
                                'differing_pixels': int(diff.shape[0]),
                                'first_at': diff[0].tolist() if diff.shape[0] else None,
                            }
                finally:
                    self.data_stream.QueueBuffer(buffer)
        finally:
            if handler is not None:
                handler.start()

        def _stat(xs):
            if not xs:
                return None
            s = sorted(xs)
            mean = sum(xs) / len(xs)
            return {
                'mean_ms': round(mean, 3),
                'median_ms': round(s[len(s) // 2], 3),
                'min_ms': round(s[0], 3),
                'max_ms': round(s[-1], 3),
                'implied_fps': round(1000.0 / mean, 1) if mean else None,
            }

        results['convert'] = _stat(convert_ms)
        results['numpy'] = _stat(numpy_ms)
        return results

    def set_frame_size(self, w, h):
        try:
            mins = self.get_min_frame_size()
            maxs = self.get_max_frame_size()

            if not mins or not maxs:
                _cam_log.error('[CAM Class ] set_frame_size: could not read frame size limits')
                return

            # Convert w and h to closest valid values
            width = int(max(mins['width'], min(maxs['width'], w)) / 48) * 48
            height = int(max(mins['height'], min(maxs['height'], h)) / 4) * 4

            with self.update_camera_config():
                self.remote_nodemap.FindNode('Width').SetValue(width)
                self.remote_nodemap.FindNode('Height').SetValue(height)
        except Exception as e:
            _cam_log.error(f'[CAM Class ] set_frame_size failed: {e}')

    def get_min_frame_size(self) -> dict:
        if not self.active:
            return {}

        try:
            return {
                'width': self.remote_nodemap.FindNode('Width').Minimum(),
                'height': self.remote_nodemap.FindNode('Height').Minimum(),
            }
        except Exception as e:
            _cam_log.error(f'[CAM Class ] get_min_frame_size failed: {e}')
            return {}

    def get_max_frame_size(self) -> dict:
        if not self.active:
            return {}

        try:
            return {
                'width': self.remote_nodemap.FindNode('Width').Maximum(),
                'height': self.remote_nodemap.FindNode('Height').Maximum(),
            }
        except Exception as e:
            _cam_log.error(f'[CAM Class ] get_max_frame_size failed: {e}')
            return {}

    def get_frame_size(self):
        if not self.active:
            return

        try:
            width = self.remote_nodemap.FindNode('Width').Value()
            height = self.remote_nodemap.FindNode('Height').Value()
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
        return self._resolve_logical_format_name(logical, self.get_supported_pixel_formats())

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
            # Caller-correctable (returns False; callers fall back to a
            # supported format), so WARNING not ERROR -- e.g. an 8-bit image
            # mode asks for Mono8 on the IMX676, which exposes only Mono10/12.
            _cam_log.warning(
                f'[CAM Class ] Unsupported pixel format: {pixel_format} '
                f'(camera supports: {list(supported)})'
            )
            return False

        if resolved == self._pixel_format_cache:
            # Already at this format: skip the grab-loop stop/realloc/start that
            # update_camera_config() would otherwise force for a no-op write.
            return True

        if resolved != pixel_format:
            logger.info(
                f'[CAM Class ] Pixel format {pixel_format} -> {resolved} '
                f'(logical-to-camera mapping)'
            )

        try:
            with self.update_camera_config():
                self.remote_nodemap.FindNode('PixelFormat').SetCurrentEntry(resolved)
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
            value = self.remote_nodemap.FindNode('PixelFormat').CurrentEntry().SymbolicValue()
            self._pixel_format_cache = value
            return value
        except Exception as e:
            _cam_log.error(f'[CAM Class ] get_pixel_format failed: {e}')
            return None

    def get_supported_pixel_formats(self):
        try:
            return tuple(
                pf.SymbolicValue()
                for pf in self.remote_nodemap.FindNode('PixelFormat').AvailableEntries()
            )
        except Exception as e:
            _cam_log.error(f'[CAM Class ] get_supported_pixel_formats failed: {e}')
            return ()

    def exposure_t(self, exposure_ms) -> bool:
        """Set exposure. Returns True on success, False on a confirmed
        hardware rejection -- per-frame chunk metadata is not yet wired, so a
        swallowed write failure here would stream frames at the stale exposure
        with no downstream backstop; the caller needs the failure signal."""
        if not self.active:
            _cam_log.warning(f'[CAM Class ] Cannot set exposure {exposure_ms}ms: camera inactive')
            return False

        if exposure_ms > self.max_exposure:
            _cam_log.warning(
                f'[CAM Class ] Exposure {exposure_ms}ms exceeds max ({self.max_exposure}ms)'
            )
            return False

        # IDS allows changing exposure while acquisition is running --
        # no need for update_camera_config() stop/start cycle.
        try:
            us_value = float(exposure_ms) * 1000
            if _cam_log is not None:
                _cam_log.info(f'ids ExposureTime.SetValue({us_value:.0f}us) (={exposure_ms}ms)')
            self.remote_nodemap.FindNode('ExposureTime').SetValue(us_value)
            self._last_exposure_ms = float(exposure_ms)
            # Update grab timeout so long exposures don't cause perpetual timeouts
            if self.cam_image_handler:
                self.cam_image_handler.timeout_ms = max(2000, int(exposure_ms * 2 + 500))
            logger.debug(f'[CAM Class ] Exposure set to {exposure_ms}ms')
            return True
        except Exception as e:
            if _cam_log is not None:
                _cam_log.error(f'ids ExposureTime.SetValue({exposure_ms}ms) FAILED: {e}')
            _cam_log.error(f'[CAM Class ] Exposure set failed (likely out of bounds): {e}')
            return False

    def get_exposure_t(self):
        if not self.active:
            _cam_log.warning('[CAM Class ] Cannot read exposure: camera inactive')
            return -1

        try:
            microsec = self.remote_nodemap.FindNode('ExposureTime').Value()
            millisec = microsec / 1000
            return millisec
        except Exception as e:
            _cam_log.error(f'[CAM Class ] get_exposure_t failed: {e}')
            return -1

    def auto_exposure_t(self, state=True):
        try:
            return self.remote_nodemap.HasNode('ExposureAuto')
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
        throttling -- which is what init_camera_config already does.
        Mode='Off' here maps to "set to Maximum()"; Mode='On' with
        value_bps maps to "set to value_bps".

        Returns True on success, False if camera inactive or the node
        is not present on this IDS body. Does not raise.
        """
        if not self.active or self.remote_nodemap is None:
            return False
        try:
            node = self.remote_nodemap.FindNode('DeviceLinkThroughputLimit')
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

    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float = 1.0):
        if not self.active:
            _cam_log.warning('[CAM Class ] set_max_acquisition_frame_rate(): inactive camera')
            return

        # IDS allows changing AcquisitionFrameRateTargetEnable +
        # AcquisitionFrameRateTarget while acquisition is running (same
        # runtime-parameter class as ExposureTime, see exposure_t above).
        try:
            if _cam_log is not None:
                _cam_log.info(
                    f'ids AcquisitionFrameRateTargetEnable.SetValue({enabled})'
                    + (f' AcquisitionFrameRateTarget.SetValue({fps})' if enabled else '')
                )
            self.remote_nodemap.FindNode('AcquisitionFrameRateTargetEnable').SetValue(enabled)
            if enabled:
                self.remote_nodemap.FindNode('AcquisitionFrameRateTarget').SetValue(fps)
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
            _cam_log.error(f'[CAM Class ] Unsupported bin size: {size}')
            return False

        try:
            logger.debug(
                f'[CAM Class ] Binning {self.get_binning_size()} -> {size}, frame {self.get_frame_size()}'
            )
            with self.update_camera_config():
                self.remote_nodemap.FindNode('BinningVertical').SetValue(size)
                self.remote_nodemap.FindNode('BinningHorizontal').SetValue(size)

            logger.debug(
                f'[CAM Class ] Binning set to {self.get_binning_size()}, frame now {self.get_frame_size()}'
            )
            return True
        except Exception as e:
            _cam_log.error(f'[CAM Class ] set_binning_size failed: {e}')
            raise HardwareError(f'set_binning_size({size}) failed: {type(e).__name__}: {e}') from e

    def get_binning_size(self) -> int:
        if not self.active:
            return 1

        try:
            vert_bin = self.remote_nodemap.FindNode('BinningVertical').Value()
            horiz_bin = self.remote_nodemap.FindNode('BinningHorizontal').Value()

            if horiz_bin != vert_bin:
                _cam_log.error(
                    f'[CAM Class ] Binning mismatch detected between horizontal ({horiz_bin}) and vertical ({vert_bin})'
                )

            return vert_bin
        except Exception as e:
            _cam_log.error(f'[CAM Class ] get_binning_size failed: {e}')
            return 1

    # grab() inherited from Camera base class

    def grab_new_capture(self, timeout_s):
        if not self.cam_image_handler:
            return False, None

        try:
            # WaitForFinishedBuffer wants an integer millisecond timeout
            # (peak::core::Timeout); the caller passes float seconds. The live
            # grab loop already uses ms -- convert here too, or the SWIG call
            # rejects the float and every capture-path grab fails.
            buffer = self.data_stream.WaitForFinishedBuffer(int(timeout_s * 1000))
            result = not buffer.IsIncomplete()
            if not result:
                self.data_stream.QueueBuffer(buffer)
                return False, None

            # Convert to the unpacked native depth (uint16 for Mono10/Mono12),
            # NOT Mono8: the still-capture path reads this frame's depth from
            # last_significant_bits, which now reflects native depth, so an
            # 8-bit array here would be downconverted against a 10/12-bit depth.
            wire = self.get_pixel_format()
            target = _ids_ipl_target(wire)
            img = ids_peak_ipl_extension.BufferToImage(buffer)
            if img.PixelFormat() != target:
                img = img.ConvertTo(target)
            # ConvertTo copied the data out; copy the numpy view, THEN re-queue
            # so the SDK cannot refill the buffer while the array still aliases it.
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
            return self.remote_nodemap.HasNode('GainAuto')
        except Exception as e:
            _cam_log.error(f'[CAM Class ] update_auto_gain_target_brightness failed: {e}')
            return False

    def update_auto_gain_min_max(self, min_gain_db: float | None, max_gain_db: float | None):
        try:
            return self.remote_nodemap.HasNode('GainAuto')
        except Exception as e:
            _cam_log.error(f'[CAM Class ] update_auto_gain_min_max failed: {e}')
            return False

    def get_gain(self):
        """Return gain in dB. The IDS Gain node is a linear factor; convert via
        dB = 20 * log10(factor) (factor >= 1.0 on this body, so dB >= 0) to
        match LVP's dB gain model."""
        if not self.active:
            _cam_log.warning('[CAM Class ] Cannot read gain: camera inactive')
            return -1

        try:
            factor = self.remote_nodemap.FindNode('Gain').Value()
            return 20.0 * math.log10(factor) if factor > 0 else 0.0
        except Exception as e:
            _cam_log.error(f'[CAM Class ] Read gain failed: {e}')
            return -1

    def gain(self, value) -> bool:
        """Set gain. `value` is in dB (LVP's gain unit, shared with the Pylon
        driver). The IDS Gain node is a linear multiplier, so convert
        factor = 10 ** (dB / 20) before writing -- the app's 0 dB floor maps to
        the node's 1.0x unity minimum, which is why the previous unconverted
        write of 0.0 was rejected as out-of-range. The valid dB range is
        published through the profile (see _query_dynamic_capabilities), so a
        caller that honors it stays in range; a genuinely out-of-range request
        still surfaces as a False return, not a silent clamp.

        Returns True on success, False on a confirmed hardware rejection --
        per-frame chunk metadata is not yet wired, so a swallowed write failure
        here would stream frames at the stale gain with no downstream backstop;
        the caller needs the failure signal."""
        if not self.active:
            if _cam_log is not None:
                _cam_log.warning(f'ids Gain.SetValue({value}) SKIPPED: active=None')
            _cam_log.warning(f'[CAM Class ] Cannot set gain {value}: camera inactive')
            return False

        try:
            factor = 10.0 ** (float(value) / 20.0)
            if _cam_log is not None:
                _cam_log.info(
                    f'ids GainSelector=AnalogAll Gain.SetValue({factor:.3f}) (={value} dB)'
                )
            self.remote_nodemap.FindNode('GainSelector').SetCurrentEntry('AnalogAll')
            self.remote_nodemap.FindNode('Gain').SetValue(factor)
            logger.debug(f'[CAM Class ] Gain set to {value} dB ({factor:.3f}x)')
            return True
        except Exception as e:
            if _cam_log is not None:
                _cam_log.error(f'ids Gain.SetValue({value} dB) FAILED: {e}')
            _cam_log.error(f'[CAM Class ] Gain set failed (likely out of bounds): {e}')
            return False

    def auto_gain(
        self,
        state=True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
        ae_max_exposure_ms: float | None = None,
    ):
        try:
            return self.remote_nodemap.HasNode('GainAuto')
        except Exception as e:
            _cam_log.error(f'[CAM Class ] auto_gain failed: {e}')
            return False

    def auto_gain_once(
        self,
        state=True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
        ae_max_exposure_ms: float | None = None,
    ):
        try:
            return self.remote_nodemap.HasNode('GainAuto')
        except Exception as e:
            _cam_log.error(f'[CAM Class ] auto_gain_once failed: {e}')
            return False

    def set_test_pattern(self, enabled: bool = False, pattern: str = 'Black'):
        pass


def _exc_is(exc: Exception, type_name: str, *substrings: str) -> bool:
    """True when `exc` is the named ids_peak exception type (if this SDK build
    exposes it) or, for builds that don't, its message matches any substring.

    Typed-first so an intentional teardown abort or a host stall is recognized
    by class, not by message text alone; the substring fallback keeps the
    classification working where the typed name is absent. When ids_peak is the
    test MagicMock the attribute is not a real type, so only the substring path
    runs -- which is what the unit tests exercise.
    """
    typ = getattr(ids_peak, type_name, None)
    if isinstance(typ, type) and isinstance(exc, typ):
        return True
    message = str(exc).lower()
    return any(s in message for s in substrings)


class _LatestBufferSlot:
    """Newest-wins handoff of finished SDK buffers from the poll thread to the
    unpack worker. A put() that lands on an unconsumed buffer displaces the
    older one and re-queues it immediately (returns it to the pool unread), so
    the converter -- the host's throughput bottleneck -- only ever unpacks the
    freshest buffer, and no displaced buffer is stranded out of the pool. The
    live display path is itself newest-wins (it polls the latest stored frame),
    so dropping intermediate live frames changes nothing a consumer can observe;
    protocol capture runs on the separate grab_new_capture() path.

    get() blocks until a buffer arrives or the slot is stopped, draining any
    pending buffer before it reports the stop sentinel. stop() re-queues a
    still-held buffer so teardown leaks nothing.
    """

    _STOP = object()

    def __init__(self, requeue):
        # requeue(buffer): return a buffer to the SDK pool. Called for every
        # buffer newest-wins displaces, so a dropped buffer never leaks.
        self._requeue = requeue
        self._cond = threading.Condition()
        self._buffer = None
        self._stopped = False
        self.dropped = 0

    def put(self, buffer):
        with self._cond:
            stale = self._buffer
            self._buffer = buffer
            if stale is not None:
                self.dropped += 1
            self._cond.notify()
        # Re-queue the displaced buffer OUTSIDE the lock (QueueBuffer may block).
        if stale is not None:
            self._requeue(stale)

    def get(self, timeout):
        """Return the latest buffer, None on timeout, or _STOP when stopped+empty."""
        with self._cond:
            if self._buffer is None and not self._stopped:
                self._cond.wait(timeout)
            if self._buffer is not None:
                buffer, self._buffer = self._buffer, None
                return buffer
            return self._STOP if self._stopped else None

    def stop(self):
        with self._cond:
            self._stopped = True
            stale = self._buffer
            self._buffer = None
            self._cond.notify_all()
        if stale is not None:
            self._requeue(stale)


class ImageHandler(ImageHandlerBase):
    """IDS image handler -- a two-stage poll/unpack pipeline.

    Stage A (poll thread): WaitForFinishedBuffer, hand the finished SDK buffer
    to a newest-wins slot. Cheap; it never runs the unpack, so it keeps draining
    at the full acquisition rate.

    Stage B (unpack worker): take the freshest buffer, unpack it in place with
    BufferToImage + ConvertTo (the SDK's own, bench-proven conversion -- no
    intermediate byte copy or image-from-buffer reconstruction), store the
    result, and re-queue the buffer. A single worker matches the host
    converter's sustained rate; newest-wins means it never spends the converter
    on a superseded buffer.

    Buffer lifecycle: every finished buffer is re-queued EXACTLY once -- the
    worker re-queues the buffer it unpacks (success or failure, via finally), the
    slot re-queues any buffer newest-wins displaces, and the poll loop re-queues
    an incomplete buffer or one still in hand when stop fires. All QueueBuffer
    calls route through _requeue under a single lock, because the poll and worker
    threads both return buffers to the pool and the SDK does not promise
    QueueBuffer is concurrency-safe.
    """

    # 10 failures x the (exposure-scaled) timeout before declaring the device
    # gone -- preserves the prior disconnect-detection threshold.
    MAX_CONSECUTIVE_FAILURES = 10

    # The worker wakes at least this often to re-check the stop request even
    # when no frames are arriving (a stalled stream must still shut down).
    _WORKER_POLL_S = 0.5

    def __init__(self, data_stream: ids_peak.DataStream, parent_cam: 'IDSCamera'):
        super().__init__()
        self.data_stream = data_stream
        self.timeout_ms = 2000  # Updated by exposure_t() for long exposures
        self._parent = parent_cam
        self._stop_event = threading.Event()
        self._requeue_lock = threading.Lock()
        self._slot = _LatestBufferSlot(self._requeue)
        self._poll_thread = None
        self._worker_thread = None

    def start(self):
        if self._poll_thread is not None:
            return
        self._stop_event.clear()
        self._slot = _LatestBufferSlot(self._requeue)
        # Clear any KillWait left pending by a previous stop() so the first
        # WaitForFinishedBuffer of this session is not aborted on arrival.
        try:
            self.data_stream.FlushPendingKillWaits()
        except Exception as e:
            logger.debug(f'[CAM Class ] FlushPendingKillWaits unavailable: {e}')
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name='IDSUnpackWorker', daemon=True
        )
        self._poll_thread = threading.Thread(target=self._poll_loop, name='IDSPoll', daemon=True)
        self._worker_thread.start()
        self._poll_thread.start()

    def stop(self):
        self._stop_event.set()
        self._slot.stop()
        # Unblock a poll thread parked in WaitForFinishedBuffer so the join
        # returns promptly instead of waiting out the (exposure-scaled) timeout.
        # A long-exposure shutdown would otherwise hang for that whole window.
        try:
            self.data_stream.KillWait()
        except Exception as e:
            logger.debug(f'[CAM Class ] KillWait unavailable: {e}')
        for thread in (self._poll_thread, self._worker_thread):
            if thread is not None:
                thread.join(timeout=5)
        self._poll_thread = None
        self._worker_thread = None

    def _poll_loop(self):
        """Stage A: drain finished buffers and hand each to the unpack worker."""
        while not self._stop_event.is_set():
            try:
                buffer = self.data_stream.WaitForFinishedBuffer(self.timeout_ms)
            except Exception as e:
                if self._stop_event.is_set():
                    break
                if self._handle_wait_error(e):
                    break
                continue

            if self._stop_event.is_set():
                self._requeue(buffer)
                break

            if buffer.IsIncomplete():
                self._log_incomplete(buffer)
                self._requeue(buffer)
                if self._record_failure():
                    _cam_log.error('[CAM Class ] Too many grab failures; marking device as removed')
                    self._parent._mark_disconnected()
                    break
                continue

            # Hand the buffer to the worker. A buffer this displaces is re-queued
            # by the slot; the worker re-queues this one once it has unpacked it.
            self._slot.put(buffer)

    def _handle_wait_error(self, e: Exception) -> bool:
        """Classify a WaitForFinishedBuffer error; return True to stop the loop.

        An AbortedException is our own KillWait at teardown -- a clean stop, not
        a fault. A device-lost signal marks the camera removed. Anything else
        (timeout, transient SDK fault) counts toward the disconnect threshold.
        """
        if _exc_is(e, 'AbortedException', 'abort'):
            return True
        if _exc_is(e, 'DeviceLostException', 'removed', 'device'):
            _cam_log.warning(f'[CAM Class ] Device removal detected in grab loop: {e}')
            self._parent._mark_disconnected()
            return True
        # Log every wait error. Type + message vary (timeout vs malformed buffer
        # vs SDK-internal fault); throttling loses the cause distribution. !r so
        # any non-ASCII in the SDK message is escaped at format time.
        _cam_log.warning(f'[CAM Class ] ImageHandler poll exception: {type(e).__name__}: {e!r}')
        if self._record_failure():
            _cam_log.error('[CAM Class ] Too many grab exceptions; marking device as removed')
            self._parent._mark_disconnected()
            return True
        return False

    def _requeue(self, buffer):
        """Return a buffer to the SDK pool, serialized across threads.

        Both the poll thread (incomplete + newest-wins-displaced buffers) and
        the worker thread (unpacked buffers) return buffers, and the SDK does
        not promise QueueBuffer is concurrency-safe, so the lock makes it so.
        """
        with self._requeue_lock:
            try:
                self.data_stream.QueueBuffer(buffer)
            except Exception as e:
                logger.debug(f'[CAM Class ] QueueBuffer ignored: {e}')

    def _log_incomplete(self, buffer):
        # Log every incomplete buffer with fill info. Partial-fill extent is the
        # only signal we get for USB packet loss / bandwidth saturation;
        # throttling loses the cause distribution.
        try:
            bsize = buffer.SizeFilled() if hasattr(buffer, 'SizeFilled') else None
            bcap = buffer.Size() if hasattr(buffer, 'Size') else None
        except Exception as _bintrospect:
            bsize, bcap = None, f'<introspect failed: {_bintrospect!r}>'
        _cam_log.warning(
            f'[CAM Class ] IDS buffer.IsIncomplete()=True filled={bsize} capacity={bcap}'
        )

    def _worker_loop(self):
        """Stage B: unpack the freshest buffer, store it, and re-queue it."""
        while True:
            buffer = self._slot.get(self._WORKER_POLL_S)
            if buffer is _LatestBufferSlot._STOP:
                return
            if buffer is None:  # idle poll, no buffer waiting
                if self._stop_event.is_set():
                    return
                continue
            try:
                array, significant_bits = self._unpack(buffer)
                # Stamp the frame with the depth it was captured under so depth
                # and pixels stay paired across a later format switch.
                self._store_frame(array, datetime.datetime.now(), significant_bits=significant_bits)
            except Exception as e:
                # One bad frame must not kill the worker; the next stores
                # normally. Log so the cause stays visible without throttling.
                _cam_log.warning(f'[CAM Class ] IDS unpack failed: {type(e).__name__}: {e!r}')
            finally:
                # Re-queue exactly once, whether the unpack succeeded or threw --
                # ConvertTo has already copied the pixels out, so the SDK may
                # refill this buffer now.
                self._requeue(buffer)

    def _unpack(self, buffer):
        """Unpack one finished buffer to a right-aligned uint16 array + its depth.

        Uses the SDK's own BufferToImage + ConvertTo -- the exact conversion the
        original inline grab loop ran and the bench validated -- so there is no
        intermediate byte copy or image-from-buffer reconstruction to get wrong.
        ConvertTo produces an independent copy, so the buffer is safe to re-queue
        once this returns. ids_unpack.py is the bench cross-check. Worker-only.
        """
        wire = self._parent.get_pixel_format()
        target = _ids_ipl_target(wire)
        img = ids_peak_ipl_extension.BufferToImage(buffer)
        if img.PixelFormat() != target:
            img = img.ConvertTo(target)
        # get_numpy() is a view onto the converted image; copy before it leaves.
        array = img.get_numpy().copy()
        return array, ids_significant_bits(wire)
