# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import contextlib
import datetime
import os
import queue
import threading
import time

from pypylon import genicam, pylon

from drivers.camera import Camera, ImageHandlerBase
from drivers.exceptions import HardwareError
from drivers.registry import camera_registry
from lvp_logger import logger

try:
    from lib import profile_trace
except ImportError:
    profile_trace = None

try:
    from lvp_logger import camera_logger as _cam_log
except ImportError:
    _cam_log = None


# Pylon SDK error code returned by grabResult.GetErrorCode() when a
# buffer is cancelled by StopGrabbing in flight. Value 0xE2000102.
# USB3-Vision transport namespace (high byte 0xE2). Per Basler
# stream-grabber-parameters.html the transport split is "0xE1000014
# on GigE cameras and 0xE2000212 on USB 3.0 cameras"; on GigE the
# cancel code may differ. pypylon does not expose this as a named
# constant; if a future version adds pylon.GENERIC_BUFFER_CANCELED
# or similar, replace this.
_PYLON_ERR_BUFFER_CANCELED = 3791651074


@camera_registry.register('pylon', priority=100)
class PylonCamera(Camera):
    def __init__(self, **kwargs):

        if os.getenv('PYLON_CAMEMU', None) is not None:
            logger.info(
                '[CAM Class ] PylonCamera.connect() detected request to use camera emulation'
            )
            self._use_camera_emulation = True
        else:
            self._use_camera_emulation = False

        super().__init__()

    # _mark_disconnected() inherited from Camera base class

    def _query_dynamic_capabilities(self):
        """Query Pylon SDK for gain/exposure ranges and merge into profile."""
        if not self.active:
            return

        try:
            nm = self.active.GetNodeMap()

            # Gain ranges
            try:
                gain_node = nm.GetNode('Gain')
                if gain_node is not None:
                    self.profile.gain.total_min_db = gain_node.GetMin()
                    self.profile.gain.total_max_db = gain_node.GetMax()
                    logger.info(
                        f'[CAM Class ] Gain range: {self.profile.gain.total_min_db:.1f} - '
                        f'{self.profile.gain.total_max_db:.1f} dB'
                    )
            except Exception as e:
                logger.debug(f'[CAM Class ] Could not query gain range: {e}')

            # Exposure range
            try:
                exp_node = nm.GetNode('ExposureTime')
                if exp_node is not None:
                    self.profile.exposure_min_us = exp_node.GetMin()
                    self.profile.exposure_max_us = exp_node.GetMax()
                    logger.info(
                        f'[CAM Class ] Exposure range: {self.profile.exposure_min_us:.0f} - '
                        f'{self.profile.exposure_max_us:.0f} us'
                    )
            except Exception as e:
                logger.debug(f'[CAM Class ] Could not query exposure range: {e}')

        except Exception as e:
            logger.warning(f'[CAM Class ] _query_dynamic_capabilities failed: {e}')

    def disconnect(self) -> bool:
        try:
            if self.active is not None:
                try:
                    if self.is_grabbing():
                        self.stop_grabbing()
                except Exception as e:
                    logger.warning(
                        f'[CAM Class ] stop_grabbing during disconnect raised: {e}; '
                        f'continuing teardown'
                    )
                # Each teardown step is independently guarded so a failure on
                # one (e.g. Close on an already-removed device) does not
                # prevent the others from running. The behaviour the caller
                # expects after disconnect() returns is "self.active is None"
                # regardless of whether the SDK calls themselves succeeded.
                try:
                    self.active.Close()
                except Exception as e:
                    logger.warning(
                        f'[CAM Class ] Close() during disconnect raised: {e}; '
                        f'continuing teardown'
                    )
                # Explicit DetachDevice + DestroyDevice releases the SDK-side
                # device handle immediately rather than relying on CPython
                # refcount-driven cleanup. pypylon issues #547 and #792
                # document field cases where refcount cleanup left the SDK
                # handle held until the next CreateDevice for the same serial
                # failed with "device not reachable / controlled by another
                # application" (Err 0xE1020018). DetachDevice releases the
                # InstantCamera's ownership of the device pointer; DestroyDevice
                # explicitly destroys the pointer (vs. waiting for GC).
                try:
                    self.active.DetachDevice()
                except Exception as e:
                    logger.warning(
                        f'[CAM Class ] DetachDevice() during disconnect raised: {e}; '
                        f'continuing teardown'
                    )
                try:
                    self.active.DestroyDevice()
                except Exception as e:
                    logger.warning(
                        f'[CAM Class ] DestroyDevice() during disconnect raised: {e}; '
                        f'continuing teardown'
                    )
                self.active = None
                # Reset the connection-scoped self-validation flag so
                # the next connect re-runs the StreamGrabber NodeMap
                # walk against whatever camera attaches.
                self._pylon_self_validation_done = False
                logger.info('[CAM Class ] Disconnected from Pylon camera')
                return True
            else:
                logger.info('[CAM Class ] Pylon camera not connected')
        except Exception as e:
            logger.exception(f'[CAM Class ] Pylon camera disconnect failed: {e}')
        return False

    # __del__() inherited from Camera base class

    # Periodic Pylon SDK statistics + thread-count daemon poller.
    # No-op when profile_trace is disabled (env var LVP_PROFILE_TRACE
    # unset).
    _STATS_POLLER_INTERVAL_S = 5.0
    _UNDERRUN_NODE_NAME = 'Statistic_Buffer_Underrun_Count'
    _STATS_NODE_NAMES = (
        'Statistic_Total_Buffer_Count',
        'Statistic_Failed_Buffer_Count',
    )

    def _start_stats_poller(self):
        if profile_trace is None or not profile_trace.ENABLE_PROFILE_TRACE:
            return
        # Smoke 1 surfaced a 4.6-min gap in poller output that aligned with
        # rapid stop/start_grabbing cycles. Cause: prior code returned early
        # if existing.is_alive(); during the window between _stop_stats_poller
        # setting the event and the daemon thread actually exiting, a fresh
        # start_grabbing would skip starting a new poller. The old thread
        # then exits on its (already-set) event leaving NO poller running.
        # Fix: actively join the prior thread (with bounded timeout) before
        # starting a new one. Idempotent if no prior poller.
        existing = getattr(self, '_stats_poller_thread', None)
        existing_ev = getattr(self, '_stats_poller_stop', None)
        if existing is not None and existing.is_alive():
            if existing_ev is not None:
                existing_ev.set()
            existing.join(timeout=10.0)
            if existing.is_alive():
                logger.warning(
                    '[INSTR PYLON ] prior stats poller did not exit within 10s; '
                    'starting new one anyway (CSV may briefly contain rows from both)'
                )
        self._stats_poller_stop = threading.Event()
        t = threading.Thread(
            target=self._stats_poller_loop,
            name='PylonStatsPoller',
            daemon=True,
        )
        self._stats_poller_thread = t
        t.start()

    def _stop_stats_poller(self):
        # Capture the thread reference before signalling so we can join
        # it. Without the join, the thread may still be sleeping inside
        # ev.wait(_STATS_POLLER_INTERVAL_S) when a fresh _start_stats_poller
        # fires, briefly leaving two pollers writing to the same trace
        # CSV. _start_stats_poller also joins on entry; this is the
        # symmetric stop-side guard.
        t = getattr(self, '_stats_poller_thread', None)
        ev = getattr(self, '_stats_poller_stop', None)
        if ev is not None:
            ev.set()
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
            if t.is_alive():
                logger.warning(
                    '[INSTR PYLON ] stats poller did not exit within 2s; '
                    'releasing reference anyway'
                )
        self._stats_poller_thread = None

    def _stats_poller_loop(self):
        # Run the StreamGrabber NodeMap walk once per camera instance.
        # The walk costs several seconds when it fails on certain
        # SDK / transport combinations; each StartGrabbing spawns a new
        # poller, so without the per-instance gate the walk runs on
        # every restart and inflates startup time.
        if not getattr(self, '_pylon_self_validation_done', False):
            try:
                cam = self.active
                sg = cam.StreamGrabber if cam is not None else None
                if sg is not None:
                    # View 1: dir()
                    dir_nodes = sorted(n for n in dir(sg) if 'tatistic' in n.lower())
                    logger.info(f'[INSTR PYLON ] StreamGrabber dir() stat-like: {dir_nodes}')

                    # View 2: NodeMap walk (authoritative)
                    try:
                        nm = sg.GetNodeMap()
                        all_features = []
                        try:
                            for nd in nm.GetNodes():
                                try:
                                    nname = nd.GetNode().GetName()
                                except Exception:
                                    try:
                                        nname = nd.GetName()
                                    except Exception:
                                        continue
                                low = nname.lower()
                                if any(
                                    t in low
                                    for t in (
                                        'tatistic',
                                        'nderrun',
                                        'nderflow',
                                        'issing',
                                        'esync',
                                        'ailed',
                                        'otal_buf',
                                    )
                                ):
                                    all_features.append(nname)
                        except Exception as e2:
                            logger.debug(f'[INSTR PYLON ] NodeMap iteration error: {e2}')
                        logger.info(
                            f'[INSTR PYLON ] StreamGrabber NodeMap stat-like: '
                            f'{sorted(set(all_features))}'
                        )
                    except Exception as e:
                        logger.warning(f'[INSTR PYLON ] NodeMap walk failed: {e}')
                else:
                    logger.warning('[INSTR PYLON ] start: active camera is None, no stat dump')
            except Exception as e:
                logger.warning(f'[INSTR PYLON ] start: stat-node dump failed: {e}')
            finally:
                # Mark done regardless of success/failure — don't retry
                # the failing walk on every restart.
                self._pylon_self_validation_done = True

        ev = self._stats_poller_stop
        while not ev.wait(self._STATS_POLLER_INTERVAL_S):
            ts_ms = int(time.time() * 1000)
            # --- Pylon SDK statistics (N3) ---
            stats = {}
            rfr = None
            underrun_value = None
            underrun_name = ''
            try:
                cam = self.active
                sg = cam.StreamGrabber if cam is not None else None
                if sg is not None:
                    for name in self._STATS_NODE_NAMES:
                        try:
                            node = getattr(sg, name, None)
                            stats[name] = node.GetValue() if node is not None else None
                        except Exception:
                            stats[name] = None
                    # Read the canonical underrun counter directly. If
                    # absent on this SDK / transport, leave value None
                    # and emit the absence in the CSV row.
                    try:
                        node = getattr(sg, self._UNDERRUN_NODE_NAME, None)
                        if node is not None:
                            underrun_value = node.GetValue()
                            underrun_name = self._UNDERRUN_NODE_NAME
                    except Exception:
                        underrun_value = None
                if cam is not None:
                    # ace 2 / dart M/R expose BslResultingAcquisitionFrameRate
                    # as the canonical node; legacy ace exposes
                    # ResultingFrameRate. Try the Bsl variant first.
                    rfr = self._node_attr_get(
                        cam,
                        'BslResultingAcquisitionFrameRate',
                        'ResultingFrameRate',
                    )
            except Exception as e:
                logger.debug(f'[INSTR PYLON ] stats poll error: {e}')

            # Underrun is the load-bearing single bit per the experiment doc
            # — log on its own line with prominent marker, including which
            # GenICam node provided the value.
            if underrun_value is not None:
                logger.info(f'[INSTR UNDERRUN] {underrun_name}={underrun_value}')

            profile_trace.trace(
                'pylon_stats_trace.csv',
                'ts_ms,total_buffer_count,failed_buffer_count,'
                'underrun_node_name,underrun_value,resulting_fps',
                [
                    ts_ms,
                    stats.get('Statistic_Total_Buffer_Count'),
                    stats.get('Statistic_Failed_Buffer_Count'),
                    underrun_name,
                    underrun_value,
                    f'{rfr:.3f}' if rfr is not None else None,
                ],
            )

            # --- Thread counts (N4) ---
            try:
                threads = threading.enumerate()
                n_pylon_grab = sum(1 for t in threads if t.name.startswith('PylonImageGrab'))
                n_dummy = sum(1 for t in threads if t.name.startswith('Dummy'))
                n_total = len(threads)
                profile_trace.trace(
                    'pylon_threads_trace.csv',
                    'ts_ms,pylon_image_grab_count,dummy_count,total_thread_count',
                    [ts_ms, n_pylon_grab, n_dummy, n_total],
                )
            except Exception as e:
                logger.debug(f'[INSTR PYLON ] thread-count poll error: {e}')

    def stop_grabbing(self):
        # N3+N4: stop the stats poller before tearing down the grab loop.
        # No-op when poller wasn't started (LVP_PROFILE_TRACE unset).
        self._stop_stats_poller()
        camera = self.active
        if _cam_log is not None:
            _cam_log.info('pylon StopGrabbing()')
        try:
            camera.StopGrabbing()
        except Exception as e:
            if _cam_log is not None:
                _cam_log.warning(f'pylon StopGrabbing FAILED: {e}')
            logger.warning(f'[CAM Class ] stop_grabbing ignored error: {e}')

    def start_grabbing(self):
        camera = self.active
        try:
            # Cap the DMA buffer ring to 3. Pylon's default (10-25,
            # depending on SDK version) pins ~16 MB per buffer of Windows
            # kernel nonpaged pool at full-resolution Mono12, matching
            # the observed ~228 MB startup spike that never releases.
            # LatestImageOnly discards old frames anyway, so 3 buffers
            # is plenty — two active + one rotating.
            try:
                camera.MaxNumBuffer.SetValue(3)
                if _cam_log is not None:
                    _cam_log.info('pylon MaxNumBuffer.SetValue(3)')
            except Exception as e:
                if _cam_log is not None:
                    _cam_log.warning(f'pylon MaxNumBuffer cap FAILED: {e}')
                logger.warning(f'[CAM Class ] MaxNumBuffer cap failed: {e}')
            if _cam_log is not None:
                _cam_log.info('pylon StartGrabbing(LatestImageOnly, ProvidedByInstantCamera)')
            camera.StartGrabbing(
                pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByInstantCamera
            )
            # N3+N4 (STALL-1): start periodic Pylon stats + thread-count poller.
            # No-op when LVP_PROFILE_TRACE is unset.
            self._start_stats_poller()
        except Exception as e:
            if _cam_log is not None:
                _cam_log.warning(f'pylon StartGrabbing FAILED: {e}')
            logger.warning(f'[CAM Class ] start_grabbing ignored error: {e}')

    def is_grabbing(self):
        try:
            return self.active.IsGrabbing()
        except Exception:
            return False

    def connect(self) -> bool:
        """Try to connect to the first available basler camera"""
        try:
            p_device = pylon.TlFactory.GetInstance().CreateFirstDevice()
            self.active = pylon.InstantCamera(p_device)
            camera = self.active
            # Ensure previous removal flag does not persist across a new connection
            self._device_removed = False
            camera.RegisterConfiguration(
                pylon.AcquireContinuousConfiguration(),
                pylon.RegistrationMode_ReplaceAll,
                pylon.Cleanup_Delete,
            )
            # Register a minimal removal handler that only sets an internal flag
            try:
                camera.RegisterConfiguration(
                    _CameraRemovalHandler(self), pylon.RegistrationMode_Append, pylon.Cleanup_Delete
                )
            except Exception as e:
                logger.debug(f'[CAM Class ] Camera removal handler registration not supported: {e}')

            self.cam_image_handler = ImageHandler(self)
            # Cleanup_Delete: SDK takes ownership of the handler and
            # deletes it when the InstantCamera is destroyed. No
            # explicit DeregisterImageEventHandler / Deregister-
            # Configuration call is needed in disconnect() -- the
            # SDK lifecycle handles it. Same pattern for the two
            # RegisterConfiguration calls above.
            camera.RegisterImageEventHandler(
                self.cam_image_handler, pylon.RegistrationMode_Append, pylon.Cleanup_Delete
            )

            camera.Open()

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

                    try:
                        # Prefer dotted-string form (GetPylonVersionString)
                        # over the raw list GetPylonVersion returns —
                        # see lumaviewpro.py for the same rationale.
                        try:
                            _ver_str = pylon.GetPylonVersionString()
                        except Exception:
                            _v = pylon.GetPylonVersion()
                            _ver_str = '.'.join(str(x) for x in _v)
                        logger.info(f'[CAM Class ] Pylon SDK version: {_ver_str}')
                    except Exception as e:
                        logger.warning(f'[CAM Class ] Could not read Pylon SDK version: {e}')

                    # Transport + device class identify the kernel
                    # driver stack Pylon is routing through — useful
                    # when the runtime SDK says one thing but Device
                    # Manager shows a stale WinUSB/USB3Vision driver.
                    try:
                        logger.info(
                            f'[CAM Class ] Transport: {dev_info.GetTLType()} '
                            f'/ DeviceClass: {dev_info.GetDeviceClass()}'
                        )
                    except Exception as e:
                        logger.debug(f'[CAM Class ] TLType/DeviceClass unavailable: {e}')

                    device_serial = nm.GetNode('DeviceSerialNumber').ToString()
                    logger.info(f'[CAM Class ] Camera Serial Number: {device_serial}')

                    firmware = nm.GetNode('DeviceFirmwareVersion').ToString()
                    logger.info(f'[CAM Class ] Camera Firmware Version: {firmware}')

                    # Current pixel format + resolution + binning drive
                    # the DMA buffer footprint — critical context for
                    # memory / throughput analysis.
                    try:
                        pix = (
                            nm.GetNode('PixelFormat').ToString()
                            if nm.GetNode('PixelFormat') is not None
                            else '?'
                        )
                        w = camera.Width.GetValue() if hasattr(camera, 'Width') else '?'
                        h = camera.Height.GetValue() if hasattr(camera, 'Height') else '?'
                        bh = (
                            camera.BinningHorizontal.GetValue()
                            if hasattr(camera, 'BinningHorizontal')
                            else 1
                        )
                        bv = (
                            camera.BinningVertical.GetValue()
                            if hasattr(camera, 'BinningVertical')
                            else 1
                        )
                        logger.info(
                            f'[CAM Class ] Pixel format: {pix}, '
                            f'Resolution: {w}x{h}, Binning: {bh}x{bv}'
                        )
                    except Exception as e:
                        logger.debug(f'[CAM Class ] Pixel/resolution/binning unavailable: {e}')

                    temps = self.get_all_temperatures()
                    for name, temp in temps.items():
                        logger.info(f'[CAM Class ] Camera {name} Temperature : {temp:.2f} degC')

                except Exception as e:
                    logger.error(
                        f'[CAM Class ] Failed to read device info nodes: {e}', exc_info=True
                    )

            except Exception:
                self.model_name = None
                self._device_serial = None

            # Load camera profile and query dynamic capabilities
            self._load_profile()
            self._query_dynamic_capabilities()

            # Ensure no stale queued frames or state
            with contextlib.suppress(Exception):
                self.cam_image_handler.reset()

            self.init_camera_config()
            self.start_grabbing()

            self.error_report_count = 0
            logger.info('[CAM Class ] Connected to Pylon camera')
            return True

        except genicam.RuntimeException as ex:
            logger.error(
                f'[CAM Class ] Pylon camera connect failed (may be open in another application): {ex}'
            )
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

            selector = nodemap.GetNode('DeviceTemperatureSelector')
            temp = nodemap.GetNode('DeviceTemperature')

            if selector is None or temp is None:
                return {}

            temps: dict[str, float] = {}

            # Iterate all available selector entries
            for entry in selector.GetEntries():
                name = entry.GetSymbolic()  # e.g. "FpgaCore"
                value = entry.GetValue()  # enum integer value

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
                camera.UserSetSelector.SetValue('Default')
                camera.UserSetLoad.Execute()
                # Defense in depth -- 'Default' user set is documented to
                # leave AcquisitionMode=Continuous + TriggerMode=Off
                # (free-run is the Basler default). Re-assert explicitly
                # so a firmware bug, factory misconfig, or future user-set
                # change that leaks a different default surfaces here
                # rather than silently breaking free-run acquisition.
                # Per-write try/except: tolerate node-not-available on
                # camera models that don't expose the parameter.
                try:
                    camera.AcquisitionMode.SetValue('Continuous')
                except Exception as e:
                    logger.debug(f'[CAM Class ] AcquisitionMode set skipped: {e}')
                try:
                    camera.TriggerSelector.SetValue('FrameStart')
                    camera.TriggerMode.SetValue('Off')
                except Exception as e:
                    logger.debug(f'[CAM Class ] TriggerMode set skipped: {e}')
                # Enable per-frame chunks for gain/exposure/identity. Must
                # happen here -- ChunkModeActive is locked while grabbing
                # (genicam.AccessException). Settings persist across
                # subsequent stop/start cycles.
                self._enable_validity_chunks()
                self.set_pixel_format(pixel_format='Mono8')
                self.auto_gain(state=False)
                self.gain(0.0)  # Set explicit gain — camera default after UserSetLoad is undefined
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

    def _enable_validity_chunks(self) -> None:
        """Enable ChunkExposureTime / ChunkGain / ChunkFrameID for chunk-
        driven validity.

        MUST be called while the camera is NOT grabbing (ChunkModeActive
        is locked while grabbing). Canonical caller is init_camera_config()
        inside update_camera_config().

        Idempotent: safely re-asserts settings if chunks were already enabled.
        Per-chunk failures are logged but do not raise; the validity layer
        falls back to skip_frames for unenabled chunks.
        """
        camera = self.active
        if camera is None:
            return
        try:
            camera.ChunkModeActive.Value = True
        except Exception as e:
            logger.warning(
                f'[CAM Class ] could not enable ChunkModeActive: {e}; '
                f'frame_validity will fall back to skip_frames calibration'
            )
            return
        for sel in self._CHUNK_TARGETS_FOR_VALIDITY:
            try:
                camera.ChunkSelector.Value = sel
                camera.ChunkEnable.Value = True
            except Exception as e:
                logger.warning(
                    f'[CAM Class ] could not enable Chunk{sel}: {e}; '
                    f'frame_validity will fall back to skip_frames for that source'
                )

    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float = 1.0):
        try:
            self.active.AcquisitionFrameRateEnable.Value = enabled
            if enabled:
                self.active.AcquisitionFrameRate.Value = fps
        except genicam.RuntimeException as e:
            logger.error(
                f'[CAM Class ] Camera communication error in set_max_acquisition_frame_rate: {e}'
            )
            self._mark_disconnected()
        except Exception as e:
            logger.exception(
                f'[CAM Class ] Unexpected error in set_max_acquisition_frame_rate: {e}'
            )

    def set_device_link_throughput_limit(
        self,
        mode: str,
        value_bps: int | None = None,
    ) -> bool:
        """Set DeviceLinkThroughputLimitMode + DeviceLinkThroughputLimit.

        Both nodes are live-writable per the SDK lock-state table -- no
        StopGrabbing/StartGrabbing wrap is required. Per-camera defaults
        on our cameras (bench-witnessed): ace 2 a2A3536-31umBAS at
        360 MB/s -> 28.8 fps; dart daA3840-45um at 160 MB/s -> 18.7 fps.
        Setting Mode=Off lets the camera run at sensor-readout max
        (~31.2 fps ace 2; ~44.9 fps dart) -- but Basler warns "Corrupt
        or dropped frames may occur if the DeviceLinkThroughputLimit
        parameter is too high." Bench-test failure rate alongside fps
        before settling on a per-camera production default.

        Args:
            mode: ``'On'`` or ``'Off'``. Case-sensitive (matches Pylon
                enum entry symbolic names).
            value_bps: Throughput cap in bytes per second when
                ``mode='On'``. Ignored when ``mode='Off'``. If None
                while ``mode='On'``, only the mode is changed and the
                existing limit value is preserved.

        Returns:
            bool: True on success. False if the camera is inactive
                (caller-correctable guard) or the requested mode is
                rejected by the SDK.

        Raises:
            HardwareError: Underlying SDK call failed
                (RuntimeException). Camera is marked disconnected on
                RuntimeException.
        """
        if not self.active:
            return False
        if mode not in ('On', 'Off'):
            logger.error(
                f"[CAM Class ] set_device_link_throughput_limit: mode "
                f"must be 'On' or 'Off'; got {mode!r}"
            )
            return False
        try:
            if _cam_log is not None:
                _cam_log.info(
                    f'pylon DeviceLinkThroughputLimitMode.SetValue({mode!r}) '
                    f'value_bps={value_bps}'
                )
            self.active.DeviceLinkThroughputLimitMode.SetValue(mode)
            if mode == 'On' and value_bps is not None:
                self.active.DeviceLinkThroughputLimit.SetValue(int(value_bps))
            return True
        except genicam.RuntimeException as e:
            if _cam_log is not None:
                _cam_log.error(
                    f'pylon set_device_link_throughput_limit({mode}, '
                    f'{value_bps}) FAILED: {e}'
                )
            logger.error(
                f'[CAM Class ] Camera communication error in '
                f'set_device_link_throughput_limit: {e}'
            )
            self._mark_disconnected()
            raise HardwareError(
                f'set_device_link_throughput_limit({mode}, {value_bps}) '
                f'failed: {e}'
            ) from e
        except Exception as e:
            logger.exception(
                f'[CAM Class ] Unexpected error in '
                f'set_device_link_throughput_limit: {e}'
            )
            return False

    def set_pixel_format(self, pixel_format: str) -> bool:
        """Set the camera pixel format.

        Args:
            pixel_format: Pylon pixel format symbolic name (e.g. 'Mono8').

        Returns:
            bool: True on success. False only when the camera is inactive
                or the requested format is unsupported (caller-correctable
                guards). Hardware-level failure raises HardwareError.

        Raises:
            HardwareError: SDK call failed (RuntimeException, transient
                or persistent). Camera is marked disconnected on
                RuntimeException; transient timeouts let the caller decide
                whether to retry.
        """
        if not self.active:
            return False

        if pixel_format not in self.get_supported_pixel_formats():
            logger.error(f'[CAM Class ] Unsupported pixel format: {pixel_format}')
            return False

        try:
            if _cam_log is not None:
                _cam_log.info(f'pylon PixelFormat.SetValue({pixel_format!r}) (geometry-realloc)')
            with self.update_camera_config():
                self.active.PixelFormat.SetValue(pixel_format)
            return True
        except genicam.RuntimeException as e:
            if _cam_log is not None:
                _cam_log.error(f'pylon PixelFormat.SetValue({pixel_format!r}) FAILED: {e}')
            logger.error(
                f'[CAM Class ] Camera communication error during set_pixel_format({pixel_format}): {e}'
            )
            self._mark_disconnected()
            raise HardwareError(
                f'set_pixel_format({pixel_format}) failed: {type(e).__name__}: {e}'
            ) from e
        except Exception as e:
            if _cam_log is not None:
                _cam_log.error(f'pylon PixelFormat.SetValue({pixel_format!r}) FAILED: {e}')
            logger.exception(f'[CAM Class ] Unexpected error in set_pixel_format: {e}')
            raise HardwareError(
                f'set_pixel_format({pixel_format}) failed: {type(e).__name__}: {e}'
            ) from e

    def get_pixel_format(self) -> str:
        if not self.active:
            return ''

        try:
            return self.active.PixelFormat.GetValue()
        except genicam.RuntimeException as e:
            logger.error(
                f'[CAM Class ] Failed to read pixel format: Camera may be disconnected - {e}'
            )
            self._mark_disconnected()
            return ''
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error reading pixel format: {e}')
            return ''

    def get_supported_pixel_formats(self) -> tuple:
        try:
            return self.active.PixelFormat.GetSymbolics()
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Failed to read pixel formats: {e}')
            self._mark_disconnected()
            return ()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error reading pixel formats: {e}')
            return ()

    def set_binning_size(self, size: int) -> bool:
        """Set camera pixel binning size.

        Args:
            size: Binning factor (1 = no binning; up to 4 supported).

        Returns:
            bool: True on success. False only when the camera is inactive
                or size is out of range (caller-correctable guards).
                Hardware-level failure raises HardwareError.

        Raises:
            HardwareError: SDK call failed. TimeoutException is treated as
                transient (does NOT mark disconnected) so the caller can
                choose to retry; RuntimeException marks disconnected before
                raising.
        """
        if not self.active:
            return False

        if size < 1 or size > 4:
            logger.error(f'[CAM Class ] Unsupported bin size: {size}')
            return False

        try:
            logger.debug(
                f'[CAM Class ] Binning {self.get_binning_size()} -> {size}, frame {self.get_frame_size()}'
            )
            if _cam_log is not None:
                _cam_log.info(
                    f'pylon BinningVertical/Horizontal.SetValue({size}) Sum (geometry-realloc)'
                )
            with self.update_camera_config():
                self.active.BinningVertical.SetValue(size)
                self.active.BinningVerticalMode.SetValue('Sum')
                self.active.BinningHorizontal.SetValue(size)
                self.active.BinningHorizontalMode.SetValue('Sum')

            logger.debug(
                f'[CAM Class ] Binning set to {self.get_binning_size()}, frame now {self.get_frame_size()}'
            )

            return True
        except genicam.TimeoutException as e:
            # USB roundtrip timed out. Transient -- do NOT mark disconnected,
            # single timeouts can recover. Caller can choose to retry.
            logger.warning(f'[CAM Class ] set_binning_size({size}) timed out: {e}')
            raise HardwareError(
                f'set_binning_size({size}) timed out: {e}'
            ) from e
        except genicam.RuntimeException as e:
            logger.error(
                f'[CAM Class ] Camera communication error during set_binning_size({size}): {e}'
            )
            self._mark_disconnected()
            raise HardwareError(
                f'set_binning_size({size}) failed: {type(e).__name__}: {e}'
            ) from e
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in set_binning_size: {e}')
            raise HardwareError(
                f'set_binning_size({size}) failed: {type(e).__name__}: {e}'
            ) from e

    def get_binning_size(self) -> int:
        if not self.active:
            return 1

        try:
            vert_bin = self.active.BinningVertical.GetValue()
            horiz_bin = self.active.BinningHorizontal.GetValue()

            if horiz_bin != vert_bin:
                logger.warning(
                    f'[CAM Class ] Binning mismatch detected between horizontal ({horiz_bin}) and vertical ({vert_bin})'
                )

            return vert_bin
        except genicam.RuntimeException as e:
            logger.error(
                f'[CAM Class ] Failed to read binning size: Camera may be disconnected - {e}'
            )
            self._mark_disconnected()
            return 1
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error reading binning size: {e}')
            return 1

    def init_auto_gain_focus(
        self,
        auto_target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None,
    ):
        try:
            self.active.AutoFunctionROIWidth.SetValue(
                self.active.Width.Max - 2 * self.active.AutoFunctionROIOffsetX.GetValue()
            )
            self.active.AutoFunctionROIHeight.SetValue(
                self.active.Height.Max - 2 * self.active.AutoFunctionROIOffsetY.GetValue()
            )
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
            logger.error(
                f'[CAM Class ] Camera communication error during init_auto_gain_focus: {e}'
            )
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in init_auto_gain_focus: {e}')

    def update_auto_gain_target_brightness(self, auto_target_brightness: float):
        # Basler runtime-modifiable parameter -- AutoTargetBrightness can
        # be changed while StartGrabbing is active. Previous wrap in
        # update_camera_config() forced a stop_grabbing/start_grabbing
        # cycle on every call (same structural class as STALL-1's
        # per-step over-stop). docs/TODO.md item 24.
        try:
            if _cam_log is not None:
                _cam_log.info(f'pylon AutoTargetBrightness.SetValue({auto_target_brightness:.3f})')
            self.active.AutoTargetBrightness.SetValue(auto_target_brightness)
        except genicam.RuntimeException as e:
            if _cam_log is not None:
                _cam_log.error(
                    f'pylon AutoTargetBrightness.SetValue({auto_target_brightness}) FAILED: {e}'
                )
            logger.error(
                f'[CAM Class ] Camera communication error during update_auto_gain_target_brightness({auto_target_brightness}): {e}'
            )
            self._mark_disconnected()
        except Exception as e:
            if _cam_log is not None:
                _cam_log.error(
                    f'pylon AutoTargetBrightness.SetValue({auto_target_brightness}) FAILED: {e}'
                )
            logger.exception(
                f'[CAM Class ] Unexpected error in update_auto_gain_target_brightness: {e}'
            )

    def update_auto_gain_min_max(self, min_gain: float | None, max_gain: float | None):
        if not self.active:
            return

        # Basler runtime-modifiable parameters -- AutoGainLowerLimit /
        # AutoGainUpperLimit can be changed while StartGrabbing is
        # active. Previous wrap in update_camera_config() forced a
        # stop_grabbing/start_grabbing cycle on every call (same
        # structural class as STALL-1's per-step over-stop).
        # docs/TODO.md item 24. Note auto_gain() calls this AND
        # update_auto_gain_target_brightness, so the previous code
        # stop/started twice per auto_gain invocation; both wraps now
        # removed so the whole chain stays online.
        try:
            if min_gain is None:
                min_gain = self.active.AutoGainLowerLimit.Min

            if max_gain is None:
                max_gain = self.active.AutoGainUpperLimit.Max

            if _cam_log is not None:
                _cam_log.info(
                    f'pylon AutoGainLowerLimit.SetValue({min_gain}) AutoGainUpperLimit.SetValue({max_gain})'
                )
            self.active.AutoGainLowerLimit.SetValue(min_gain)
            self.active.AutoGainUpperLimit.SetValue(max_gain)
        except genicam.RuntimeException as e:
            logger.error(
                f'[CAM Class ] Camera communication error during update_auto_gain_min_max(min={min_gain}, max={max_gain}): {e}'
            )
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in update_auto_gain_min_max: {e}')

    # grab() inherited from Camera base class

    def grab_new_capture(self, timeout: float):
        """
        Drain any already-queued frames, then block up to `timeout`
        waiting for a genuinely new one. Saves the array into
        self.array when received. Returns (bool, ts).

        Previously dropped only one queued frame, which meant
        "force_new_capture" could still return a stale frame if the
        consumer had fallen behind — queue held backlog, we'd pop the
        oldest, then take the next-oldest. For AF / characterization
        timing measurements we want the freshest frame possible, so
        drain everything that's already captured before waiting.
        """
        # N2 (STALL-1 H1 vs H2 separator): per-grab duration trace.
        # See docs/STALL1_INSTRUMENTATION_EXPERIMENT.md (Firmware repo) §4 N2.
        _trace_enabled = profile_trace is not None and profile_trace.ENABLE_PROFILE_TRACE
        _t0 = time.perf_counter() if _trace_enabled else None
        _outcome = 'unknown'
        dropped = 0
        try:
            if not self.cam_image_handler:
                _outcome = 'no_handler'
                return False, None

            try:
                # Drain all frames captured before this call — we only want
                # the next one produced after we started waiting.
                while True:
                    try:
                        self.cam_image_handler._frame_queue.get_nowait()
                        dropped += 1
                    except queue.Empty:
                        break
                if dropped > 1:
                    logger.debug(f'[CAM Class ] grab_new_capture drained {dropped} stale frames')

                result, image, image_ts = self.cam_image_handler._frame_queue.get(
                    block=True, timeout=timeout
                )
                if result is False:
                    _outcome = 'result_false'
                    return False, None

                self.array = image
                _outcome = 'success'
                return True, image_ts

            except Exception as ex:
                # queue.Empty inherits from Exception — both timeout and other
                # errors are caught here, matching pre-N2 behavior. Outcome
                # classification distinguishes them in the trace row.
                _outcome = 'timeout' if isinstance(ex, queue.Empty) else 'exception'
                logger.exception(f'Failed to grab image: {ex}')
                return False, None
        finally:
            if _trace_enabled and _t0 is not None:
                _dt_ms = (time.perf_counter() - _t0) * 1000.0
                profile_trace.trace(
                    'pylon_grab_trace.csv',
                    'ts_ms,duration_ms,dropped_count,outcome,timeout_s',
                    [int(time.time() * 1000), f'{_dt_ms:.3f}', dropped, _outcome, f'{timeout:.3f}'],
                )

    def set_frame_size(self, w, h):
        """Set camera frame size to w by h and keep centered"""
        camera = self.active
        if camera is None:
            logger.warning(f'[CAM Class ] Cannot set frame size {w}x{h}: camera inactive')
            return

        try:
            width = int(min(int(w), camera.Width.Max) / 4) * 4
            height = int(min(int(h), camera.Height.Max) / 4) * 4

            if _cam_log is not None:
                _cam_log.info(
                    f'pylon Width.SetValue({width}) Height.SetValue({height}) BslCenterX/Y.Execute() (geometry-realloc)'
                )
            with self.update_camera_config():
                camera.Width.SetValue(width)
                camera.Height.SetValue(height)
                camera.BslCenterX.Execute()
                camera.BslCenterY.Execute()

            logger.info(f'[CAM Class ] Frame size set to {width}x{height}')
        except genicam.RuntimeException as e:
            logger.error(
                f'[CAM Class ] Camera communication error during set_frame_size({w}x{h}): {e}'
            )
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in set_frame_size: {e}')

    def get_min_frame_size(self) -> dict:
        camera = self.active
        if camera is None:
            return {}
        try:
            return {
                'width': camera.Width.GetMin(),
                'height': camera.Height.GetMin(),
            }
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Failed to read min frame size: {e}')
            self._mark_disconnected()
            return {}
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error reading min frame size: {e}')
            return {}

    def get_max_frame_size(self) -> dict:
        camera = self.active
        if camera is None:
            return {}
        try:
            return {
                'width': camera.Width.GetMax(),
                'height': camera.Height.GetMax(),
            }
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Failed to read max frame size: {e}')
            self._mark_disconnected()
            return {}
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error reading max frame size: {e}')
            return {}

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
            logger.error(
                f'[CAM Class ] Failed to read frame size: Camera may be disconnected - {e}'
            )
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
        except genicam.TimeoutException as e:
            # USB roundtrip timed out (transient). Don't mark disconnected; caller can retry.
            logger.warning(f'[CAM Class ] get_gain timed out: {e}')
            return -1
        except genicam.RuntimeException as e:
            logger.error(
                f'[CAM Class ] Failed to read gain value: Camera may be disconnected - {e}'
            )
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
        # Belt-and-suspenders -- the InstantCamera knows whether the SDK
        # received a removal notification before our handler fired or
        # before our removal-handler registration succeeded. Cheap query
        # (no transport enumeration); covers the case where the
        # _CameraRemovalHandler missed the event for any reason.
        try:
            if self.active.IsCameraDeviceRemoved():
                self._mark_disconnected()
                return False
        except Exception as e:
            # Native-side query failed -- treat as inconclusive; trust
            # the prior _device_removed / active checks.
            logger.debug(f'[CAM Class ] IsCameraDeviceRemoved query raised: {e}')
        return True

    def gain(self, value):
        """Set gain value in the camera hardware."""
        if self.active is None:
            if _cam_log is not None:
                _cam_log.warning(f'pylon Gain.SetValue({value}) SKIPPED: active=None')
            logger.warning(f'[CAM Class ] Cannot set gain {value}: camera inactive')
            return

        try:
            if _cam_log is not None:
                _cam_log.info(f'pylon Gain.SetValue({float(value):.3f})')
            self.active.Gain.SetValue(float(value))
            logger.info(f'[CAM Class ] Gain set to {value}')
        except genicam.RuntimeException as e:
            if _cam_log is not None:
                _cam_log.error(f'pylon Gain.SetValue({value}) FAILED: {e}')
            logger.error(f'[CAM Class ] Camera communication error during gain({value}): {e}')
            self._mark_disconnected()
        except Exception as e:
            if _cam_log is not None:
                _cam_log.error(f'pylon Gain.SetValue({value}) FAILED: {e}')
            logger.exception(f'[CAM Class ] Unexpected error in gain: {e}')

    def auto_gain(
        self,
        state=True,
        target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None,
    ):
        """Enable / Disable camera auto_gain with the value of 'state'
        It will be continueously updating based on the current image"""

        if self.active is None:
            logger.warning(f'[CAM Class ] Cannot set auto_gain({state}): camera inactive')
            return

        try:
            if _cam_log is not None:
                _cam_log.info(
                    f'pylon auto_gain(state={state}, target={target_brightness}, min={min_gain}, max={max_gain})'
                )
            if state:
                self.update_auto_gain_target_brightness(auto_target_brightness=target_brightness)
                self.update_auto_gain_min_max(min_gain=min_gain, max_gain=max_gain)
                self.active.GainAuto.SetValue('Continuous')  # 'Off' 'Once' 'Continuous'
                self.active.ExposureAuto.SetValue('Continuous')  # 'Off' 'Once' 'Continuous'
                if _cam_log is not None:
                    _cam_log.info(
                        'pylon GainAuto.SetValue(Continuous) ExposureAuto.SetValue(Continuous)'
                    )
            else:
                self.active.GainAuto.SetValue('Off')
                self.active.ExposureAuto.SetValue('Off')
                if _cam_log is not None:
                    _cam_log.info('pylon GainAuto.SetValue(Off) ExposureAuto.SetValue(Off)')
            logger.info(f'[CAM Class ] Auto gain {"enabled" if state else "disabled"}')
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Auto gain({state}) failed: {e}')
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in auto_gain: {e}')

    def auto_gain_once(
        self,
        state=True,
        target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None,
    ):
        """Enable / Disable camera auto_gain with the value of 'state'
        Auto Gain/Exposure executed one time"""

        if self.active is None:
            logger.warning(f'[CAM Class ] Cannot set auto_gain_once({state}): camera inactive')
            return

        try:
            if state:
                self.update_auto_gain_target_brightness(auto_target_brightness=target_brightness)
                self.update_auto_gain_min_max(min_gain=min_gain, max_gain=max_gain)
                self.active.GainAuto.SetValue('Once')  # 'Off' 'Once' 'Continuous'
                self.active.ExposureAuto.SetValue('Once')  # 'Off' 'Once' 'Continuous'
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
        """Set exposure time in the camera hardware t (msec)"""
        if self.active is None:
            if _cam_log is not None:
                _cam_log.warning(f'pylon ExposureTime.SetValue({t}ms) SKIPPED: active=None')
            logger.warning(f'[CAM Class ] Cannot set exposure {t}ms: camera inactive')
            return

        if t > self.max_exposure:
            if _cam_log is not None:
                _cam_log.warning(
                    f'pylon ExposureTime.SetValue({t}ms) SKIPPED: exceeds max {self.max_exposure}ms'
                )
            logger.warning(f'[CAM Class ] Exposure {t}ms exceeds max ({self.max_exposure}ms)')
            return

        # Pylon takes time in microseconds, so pass t*1000 to convert to us
        try:
            us_value = max(float(t) * 1000, self.active.ExposureTime.Min)
            if _cam_log is not None:
                _cam_log.info(f'pylon ExposureTime.SetValue({us_value:.0f}us) (={t}ms)')
            self.active.ExposureTime.SetValue(us_value)
            logger.info(f'[CAM Class ] Exposure set to {t}ms')
        except genicam.RuntimeException as e:
            if _cam_log is not None:
                _cam_log.error(f'pylon ExposureTime.SetValue({t}ms) FAILED: {e}')
            logger.error(f'[CAM Class ] Camera communication error during exposure_t({t}ms): {e}')
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in exposure_t: {e}')

    def get_exposure_t(self):
        """Get exposure time in the camera hardware
        Returns t (msec), or -1 if the camera is inactive"""

        if self.active is None:
            logger.warning('[CAM Class ] Cannot read exposure: camera inactive')
            return -1

        try:
            # ace 2, boost R, and dart R/M expose BslEffectiveExposureTime
            # as the read-only "value the camera actually used" (per
            # Basler doc exposure-time.html: "It takes factors like
            # internal offsets or clock speed requirements into account
            # that may cause the exposure time to differ from the
            # exposure time set."). Fall back to ExposureTime (the set
            # value) if the Bsl-prefixed effective node is absent
            # (legacy ace cameras don't expose it).
            microsec = self._node_attr_get(
                self.active,
                'BslEffectiveExposureTime',
                'ExposureTime',
            )
            if microsec is None:
                # Both nodes unreadable -- camera is unusable for any
                # acquisition; treat as disconnected.
                logger.error(
                    '[CAM Class ] Failed to read exposure time: both '
                    'BslEffectiveExposureTime and ExposureTime nodes '
                    'unavailable. Camera may be disconnected.'
                )
                self._mark_disconnected()
                return -1
            return microsec / 1000  # microseconds -> milliseconds
        except genicam.TimeoutException as e:
            # USB roundtrip timed out (transient). Don't mark disconnected; caller can retry.
            logger.warning(f'[CAM Class ] get_exposure_t timed out: {e}')
            return -1
        except genicam.RuntimeException as e:
            logger.error(
                f'[CAM Class ] Failed to read exposure time: Camera may be disconnected - {e}'
            )
            self._mark_disconnected()
            return -1
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error reading exposure time: {e}')
            return -1

    def auto_exposure_t(self, state=True):
        """Enable / Disable camera auto_exposure with the value of 'state'
        It will be continueously updating based on the current image"""

        if self.active is None:
            logger.warning(f'[CAM Class ] Cannot set auto_exposure({state}): camera inactive')
            return

        try:
            if state:
                self.active.ExposureAuto.SetValue('Continuous')  # 'Off' 'Once' 'Continuous'
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
            logger.error(
                f'[CAM Class ] Camera communication error during set_test_pattern({pattern}): {e}'
            )
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in set_test_pattern: {e}')

    # Chunks consumed by chunk-driven validity in modules.frame_validity.
    # LED has no chunk equivalent; motion is firmware-gated; these three
    # cover gain / exposure / per-frame identity.
    _CHUNK_TARGETS_FOR_VALIDITY = ('ExposureTime', 'Gain', 'FrameID')

    def probe_chunk_capabilities(self) -> dict:
        """Probe per-frame chunk-data feature support via static introspection.

        Enumerates the camera's ChunkSelector entries and attempts to
        enable each chunk frame_validity might use. Activation-acceptance
        is the sufficient test: if ChunkEnable accepts True for a given
        selector and reads back True, the camera supports that chunk.

        Pylon locks ChunkModeActive while grabbing -- the probe stops
        grabbing if needed and restarts after, restoring the prior chunk
        configuration. Does not grab a frame; live-value read is wired
        separately when chunks land in ImageHandler.

        Returns:
            dict with keys:
              'model': camera model name (None if unavailable)
              'firmware': camera firmware version (None if unavailable)
              'serial': camera serial number (None if unavailable)
              'advertised': sorted list of ChunkSelector entry symbols
              'enabled': dict of selector -> bool (True = activation succeeded)
              'errors': list of error strings encountered during probe
        """
        result = {
            'model': getattr(self, 'model_name', None),
            'firmware': None,
            'serial': None,
            'advertised': [],
            'enabled': {},
            'errors': [],
        }
        camera = self.active
        if camera is None:
            result['errors'].append('camera not connected')
            return result

        for genicam_attr, key in (('DeviceFirmwareVersion', 'firmware'),
                                  ('DeviceSerialNumber', 'serial')):
            try:
                node = camera.GetNodeMap().GetNode(genicam_attr)
                if node is not None:
                    result[key] = node.GetValue()
            except Exception:
                pass

        was_grabbing = False
        try:
            was_grabbing = bool(camera.IsGrabbing())
        except Exception:
            pass
        if was_grabbing:
            self.stop_grabbing()

        prior_chunk_mode = None
        prior_per_chunk: dict = {}
        try:
            try:
                nm = camera.GetNodeMap()
                selector_node = nm.GetNode('ChunkSelector')
                if selector_node is None:
                    result['errors'].append('ChunkSelector node missing')
                    return result
                for entry in selector_node.GetEntries():
                    try:
                        result['advertised'].append(entry.GetSymbolic())
                    except Exception:
                        pass
                result['advertised'].sort()
            except Exception as e:
                result['errors'].append(f'introspection failed: {e}')
                return result

            try:
                prior_chunk_mode = camera.ChunkModeActive.Value
                camera.ChunkModeActive.Value = True
            except Exception as e:
                result['errors'].append(f'could not enable ChunkModeActive: {e}')
                return result

            for sel in self._CHUNK_TARGETS_FOR_VALIDITY:
                if sel not in result['advertised']:
                    result['enabled'][sel] = False
                    continue
                try:
                    camera.ChunkSelector.Value = sel
                    prior_per_chunk[sel] = camera.ChunkEnable.Value
                    camera.ChunkEnable.Value = True
                    result['enabled'][sel] = bool(camera.ChunkEnable.Value)
                except Exception as e:
                    result['enabled'][sel] = False
                    result['errors'].append(f'could not enable Chunk{sel}: {e}')
        finally:
            for sel, prior in prior_per_chunk.items():
                try:
                    camera.ChunkSelector.Value = sel
                    camera.ChunkEnable.Value = prior
                except Exception:
                    pass
            if prior_chunk_mode is not None:
                try:
                    camera.ChunkModeActive.Value = prior_chunk_mode
                except Exception:
                    pass
            if was_grabbing:
                try:
                    self.start_grabbing()
                except Exception as e:
                    result['errors'].append(f'could not restart streaming: {e}')

        return result

    # Stream-grabber stat node names probed in read_diagnostic_snapshot.
    # All read defensively via _safe_node so missing nodes (e.g.
    # Statistic_Buffer_Underrun_Count is absent on USB3 in
    # pypylon 26.4.1 / pylon SDK 11.5.0) record '<missing>' rather
    # than raising. The full list is intentionally broader than what
    # the doc claims is USB3-available -- the probe is also a node-
    # availability survey across SDK versions and transports.
    _DIAG_STAT_NODES = (
        'Statistic_Total_Buffer_Count',
        'Statistic_Failed_Buffer_Count',
        'Statistic_Buffer_Underrun_Count',
        'Statistic_Missed_Frame_Count',
        'Statistic_Resynchronization_Count',
        'Statistic_Last_Failed_Buffer_Status',
        'Statistic_Last_Failed_Buffer_Status_Text',
    )
    _DIAG_STAT_COUNTERS = (
        'Statistic_Total_Buffer_Count',
        'Statistic_Failed_Buffer_Count',
        'Statistic_Buffer_Underrun_Count',
        'Statistic_Missed_Frame_Count',
        'Statistic_Resynchronization_Count',
    )

    @staticmethod
    def _safe_node(nodemap, *names: str):
        """Read a nodemap node defensively, trying each name in order.

        Camera nodemap returns None for missing nodes; stream-grabber
        nodemap raises ``_genicam.LogicalErrorException`` ("Node not
        existing"). This helper unifies both into a sentinel string so
        the probe never propagates an exception out of an optional
        read.

        Multiple names support the case where different camera families
        expose the same logical parameter under different canonical
        names. ace 2 / boost / dart M/R use ``BslResultingAcquisitionFrameRate``
        and ``BslEffectiveExposureTime`` (per Basler doc
        resulting-acquisition-frame-rate.html and exposure-time.html);
        legacy ace cameras use the unprefixed ``ResultingFrameRate``
        and ``ExposureTime``. Pass the Bsl-prefixed canonical first;
        the helper falls back to the unprefixed form if the camera
        doesn't expose the Bsl variant.
        """
        last_sentinel = '<missing>'
        for name in names:
            try:
                n = nodemap.GetNode(name)
                if n is None:
                    last_sentinel = '<not present>'
                    continue
                try:
                    return n.GetValue()
                except Exception as e:
                    last_sentinel = f'<read err: {type(e).__name__}>'
                    continue
            except Exception:
                continue
        return last_sentinel

    @staticmethod
    def _node_attr_get(camera, *names: str):
        """Read camera.<name>.GetValue() trying each name in order.

        Used at attribute-access call sites (live read paths) where
        going through GetNodeMap().GetNode() adds overhead. Companion
        to _safe_node which serves the diagnostic-snapshot probe path.
        Returns None if no name resolves to a readable value.
        """
        for name in names:
            node = getattr(camera, name, None)
            if node is None:
                continue
            try:
                return node.GetValue()
            except Exception:
                continue
        return None

    def read_diagnostic_snapshot(
        self,
        duration_s: float = 3.0,
        drain_camera_side_errors: bool = True,
    ) -> dict:
        """Capture a single diagnostic snapshot of camera + stream-grabber state.

        Reads camera identity, current configuration, and stream-grabber
        statistics counters at the start and end of a sampling window of
        ``duration_s`` seconds. Computes per-counter deltas and derived
        rates (observed_fps, fail_rate_pct, failures_per_second,
        resyncs_per_second). Optionally drains the camera-side error log
        via ``BslErrorPresent`` / ``BslErrorReportValue`` /
        ``BslErrorReportNext``.

        Does NOT change grab state. If the camera is not currently
        grabbing, the deltas will be near-zero (stats counters do not
        advance without an active grab loop) -- treated as a sentinel
        rather than an error, since the snapshot still captures
        configuration state useful for cross-host comparison.

        Reads are wrapped via ``_safe_node`` so missing nodes (e.g.
        ``Statistic_Buffer_Underrun_Count`` on USB3) record sentinel
        strings rather than raising. The counter delta computation
        only runs for entries that returned numeric values both pre
        and post.

        Args:
            duration_s: Sampling window in seconds. Default 3.0 matches
                the bench probe shape used to characterize dart vs ace 2.
            drain_camera_side_errors: When True, drain ``BslErrorPresent``
                queue (capped at 64 iterations as a defensive bound) and
                return the list of opaque error codes. Per Basler docs,
                these codes are "evaluated by Basler support" (no public
                translation table for ace 2 / dart R).

        Returns:
            dict with keys: connected, duration_s_requested,
                duration_s_actual, camera (model_name, firmware_version,
                serial), config (pixel_format, width/height,
                exposure_us, gain_db, black_level, dltl_mode/value,
                acquisition_frame_rate*, resulting_frame_rate,
                max_num_buffer/transfer_size/queued_urbs/buffer_size),
                stats_pre, stats_post, deltas, derived
                (observed_fps, fail_rate_pct, failures_per_second,
                resyncs_per_second), camera_side_errors, errors.
        """
        result: dict = {
            'connected': False,
            'duration_s_requested': float(duration_s),
            'duration_s_actual': 0.0,
            'camera': {},
            'config': {},
            'stats_pre': {},
            'stats_post': {},
            'deltas': {},
            'derived': {},
            'camera_side_errors': None,
            'errors': [],
        }

        cam = self.active
        if cam is None:
            result['errors'].append('camera not connected')
            return result
        result['connected'] = True

        try:
            nm = cam.GetNodeMap()
            sg = cam.GetStreamGrabberNodeMap()
        except Exception as e:
            result['errors'].append(f'nodemap fetch failed: {type(e).__name__}: {e}')
            return result

        # Camera identity
        for genicam_name, key in (
            ('DeviceModelName', 'model_name'),
            ('DeviceSerialNumber', 'serial'),
            ('DeviceFirmwareVersion', 'firmware_version'),
            ('DeviceVersion', 'device_version'),
        ):
            result['camera'][key] = self._safe_node(nm, genicam_name)

        # Camera-nodemap configuration. Tuple-of-names entries probe
        # each name in order until one resolves -- ace 2 / dart M/R
        # expose Bsl-prefixed canonical nodes for what legacy ace
        # exposes unprefixed (per Basler doc resulting-acquisition-
        # frame-rate.html and exposure-time.html).
        for names, key in (
            (('PixelFormat',), 'pixel_format'),
            (('Width',), 'width'),
            (('Height',), 'height'),
            (('SensorWidth',), 'sensor_width'),
            (('SensorHeight',), 'sensor_height'),
            (('BslEffectiveExposureTime', 'ExposureTime'), 'exposure_us'),
            (('Gain',), 'gain_db'),
            (('BlackLevel',), 'black_level'),
            (('DeviceLinkThroughputLimitMode',), 'dltl_mode'),
            (('DeviceLinkThroughputLimit',), 'dltl_value_bps'),
            (('AcquisitionFrameRateEnable',), 'acquisition_frame_rate_enable'),
            (('AcquisitionFrameRate',), 'acquisition_frame_rate'),
            (('BslResultingAcquisitionFrameRate', 'ResultingFrameRate'),
             'resulting_frame_rate'),
        ):
            result['config'][key] = self._safe_node(nm, *names)

        # Stream-grabber-nodemap configuration (defaults; transport-
        # specific availability varies)
        for name, key in (
            ('MaxNumBuffer', 'max_num_buffer'),
            ('MaxTransferSize', 'max_transfer_size'),
            ('NumMaxQueuedUrbs', 'num_max_queued_urbs'),
            ('MaxBufferSize', 'max_buffer_size'),
        ):
            result['config'][key] = self._safe_node(sg, name)

        # Stats pre
        for name in self._DIAG_STAT_NODES:
            result['stats_pre'][name] = self._safe_node(sg, name)

        # Sampling window
        t0 = time.monotonic()
        try:
            if duration_s > 0:
                time.sleep(duration_s)
        except Exception as e:
            result['errors'].append(f'sleep raised: {type(e).__name__}: {e}')
        dt = time.monotonic() - t0
        result['duration_s_actual'] = dt

        # Stats post
        for name in self._DIAG_STAT_NODES:
            result['stats_post'][name] = self._safe_node(sg, name)

        # Deltas (only for numeric counters where both pre and post
        # returned int/float; missing nodes record None)
        for name in self._DIAG_STAT_COUNTERS:
            pre = result['stats_pre'].get(name)
            post = result['stats_post'].get(name)
            if isinstance(pre, (int, float)) and isinstance(post, (int, float)):
                result['deltas'][name] = post - pre
            else:
                result['deltas'][name] = None

        # Derived rates
        total_d = result['deltas'].get('Statistic_Total_Buffer_Count')
        failed_d = result['deltas'].get('Statistic_Failed_Buffer_Count')
        resync_d = result['deltas'].get('Statistic_Resynchronization_Count')
        missed_d = result['deltas'].get('Statistic_Missed_Frame_Count')
        if isinstance(total_d, (int, float)) and dt > 0:
            result['derived']['observed_fps'] = total_d / dt
            if total_d > 0 and isinstance(failed_d, (int, float)):
                result['derived']['fail_rate_pct'] = 100.0 * failed_d / total_d
        if isinstance(failed_d, (int, float)) and dt > 0:
            result['derived']['failures_per_second'] = failed_d / dt
        if isinstance(resync_d, (int, float)) and dt > 0:
            result['derived']['resyncs_per_second'] = resync_d / dt
        if isinstance(missed_d, (int, float)) and dt > 0:
            result['derived']['misses_per_second'] = missed_d / dt

        # Camera-side error log drain (capped to prevent runaway loop
        # on a malformed firmware response)
        if drain_camera_side_errors:
            try:
                err_present = nm.GetNode('BslErrorPresent')
                code_node = nm.GetNode('BslErrorReportValue')
                next_cmd = nm.GetNode('BslErrorReportNext')
                if err_present is not None and code_node is not None and next_cmd is not None:
                    errs = []
                    for _ in range(64):
                        try:
                            present = err_present.GetValue()
                        except Exception:
                            break
                        if not present:
                            break
                        try:
                            code = code_node.GetValue()
                        except Exception:
                            break
                        if code == 0:
                            break
                        errs.append(int(code))
                        try:
                            next_cmd.Execute()
                        except Exception:
                            break
                    result['camera_side_errors'] = errs
            except Exception as e:
                result['errors'].append(
                    f'BslErrorPresent drain raised: {type(e).__name__}: {e}'
                )

        return result


class ImageHandler(pylon.ImageEventHandler):
    """Pylon camera image handler — receives frames via SDK callbacks.

    Uses ImageHandlerBase via composition (not inheritance) to avoid
    metaclass conflict with pylon.ImageEventHandler.
    """

    def __init__(self, parent_cam: PylonCamera):
        super().__init__()
        self._base = ImageHandlerBase()
        self._frame_queue = queue.Queue(maxsize=1)
        self._parent = parent_cam

    # Maps GrabResult chunk attribute names to the chunk_data dict keys
    # frame_validity expects (matches FrameValidity.CHUNK_KEY_FOR_SOURCE).
    _CHUNK_GRAB_RESULT_ATTRS = (
        ('ChunkExposureTime', 'ExposureTime'),
        ('ChunkGain', 'Gain'),
        ('ChunkFrameID', 'FrameID'),
    )

    @staticmethod
    def _read_validity_chunks(grabResult) -> dict | None:
        """Extract validity chunks from a successful GrabResult.

        Reads ChunkExposureTime / ChunkGain / ChunkFrameID via the
        GrabResult attribute interface (pypylon exposes chunks as
        attributes once ChunkModeActive + ChunkEnable are set on the
        camera). Per-chunk failures are silenced; missing chunks just
        don't appear in the returned dict.

        Returns:
            dict mapping chunk-key -> value (e.g. {'ExposureTime': 14530.0,
            'Gain': 1.0, 'FrameID': 12345}), or None if no chunks were
            readable (camera doesn't support chunks, or chunk_enable
            failed during init_camera_config -> _enable_validity_chunks).
        """
        chunks: dict = {}
        for chunk_attr, key in ImageHandler._CHUNK_GRAB_RESULT_ATTRS:
            try:
                node = getattr(grabResult, chunk_attr, None)
                if node is None:
                    continue
                if genicam.IsReadable(node):
                    chunks[key] = node.Value
            except Exception:
                pass
        return chunks if chunks else None

    def OnImageGrabbed(self, camera, grabResult):
        # N1 (STALL-1 H2): per-callback duration trace. Gated on
        # profile_trace.ENABLE_PROFILE_TRACE — zero overhead when disabled.
        # See docs/STALL1_INSTRUMENTATION_EXPERIMENT.md (Firmware repo) §4 N1.
        _trace_enabled = profile_trace is not None and profile_trace.ENABLE_PROFILE_TRACE
        _t0 = time.perf_counter() if _trace_enabled else None
        _outcome = 'unknown'
        _frame_bytes = 0
        try:
            # Rename SDK worker threads from Python's auto-assigned
            # "Dummy-N" to a stable label so the thread-count poller
            # (_stats_poller_loop) can count grab callbacks via
            # name-prefix match. Assumption: pypylon delivers
            # OnImageGrabbed on threads created outside Python (which
            # CPython assigns Dummy-N names) on pypylon 26.4.x. If a
            # future pypylon version uses pre-named threads, the rename
            # branch never fires and _stats_poller_loop's grab count
            # under-reports.
            if 'Dummy' in threading.current_thread().name:
                threading.current_thread().name = 'PylonImageGrab'

            # Check if parent camera was removed before processing
            if self._parent._device_removed:
                logger.debug(
                    '[CAM Class ] OnImageGrabbed called but device already marked as removed, ignoring'
                )
                _outcome = 'early_return_removed'
                return

            # Check if parent camera is still active
            if self._parent.active is None:
                logger.debug('[CAM Class ] OnImageGrabbed called but camera is inactive, ignoring')
                self._parent._mark_disconnected()
                _outcome = 'early_return_inactive'
                return

            if not self._frame_queue.empty():
                with contextlib.suppress(queue.Empty):
                    self._frame_queue.get_nowait()

            # Safely check grab result - this can throw native exceptions
            try:
                grab_succeeded = grabResult.GrabSucceeded()
            except Exception as e:
                logger.warning(f'[CAM Class ] GrabSucceeded() failed: {e}, assuming device removed')
                self._parent._mark_disconnected()
                _outcome = 'exception_grabsucceeded'
                return

            if grab_succeeded:
                try:
                    # GetArray() returns a view into the SDK buffer — copy immediately
                    # to decouple from buffer lifetime before it's requeued
                    img = grabResult.GetArray().copy()
                    ts = datetime.datetime.now()
                    _frame_bytes = img.nbytes
                    # Read per-frame chunk metadata if available.
                    # _read_validity_chunks is defensive (catches per-chunk
                    # exceptions); returns None if no chunks were readable
                    # (camera doesn't support chunks or chunk-enable failed
                    # during init).
                    chunks = self._read_validity_chunks(grabResult)
                    self._base._store_frame(img, ts, chunks=chunks)
                    self._frame_queue.put((True, img, ts))
                    _outcome = 'success_grabbed'
                except Exception as e:
                    logger.warning(
                        f'[CAM Class ] GetArray() failed: {e}, marking device as removed'
                    )
                    self._parent._mark_disconnected()
                    self._base._record_failure()
                    _outcome = 'exception_getarray'
            else:
                _outcome = 'success_no_grab'
                try:
                    err_code = grabResult.GetErrorCode()
                    err_desc = grabResult.GetErrorDescription()
                except Exception as _err_introspect:
                    err_code, err_desc = None, f'<introspect failed: {_err_introspect!r}>'
                if err_code == _PYLON_ERR_BUFFER_CANCELED or self._parent._device_removed:
                    # Pylon returns buffers with cancelled status when
                    # StopGrabbing fires while grabs are in flight. Not a
                    # hardware failure -- purely a lifecycle event from
                    # rapid stop/start cycles. Don't increment
                    # _failed_grabs (that path triggers auto-disconnect at
                    # MAX_CONSECUTIVE_FAILURES; cancel storms during
                    # config changes would falsely trip it).
                    # OR-with-removal-flag insurance: device-removal
                    # teardown delivers in-flight buffers as failures, but
                    # the precise err_code attached during removal is not
                    # documented by Basler. Treating any failure paired
                    # with _device_removed=True as expected teardown
                    # protects MAX_CONSECUTIVE_FAILURES from spurious
                    # auto-disconnect during removal storms even if a
                    # second cancel-flavoured code surfaces later.
                    logger.debug(
                        f'[CAM Class ] Grab cancelled (SDK lifecycle, '
                        f'not a failure) err_code={err_code} desc={err_desc!r} '
                        f'device_removed={self._parent._device_removed}'
                    )
                    _outcome = 'success_no_grab_cancelled'
                else:
                    # Real failure path. err_code + err_desc may differ
                    # between failures (USB CRC vs partial frame vs buffer
                    # underrun); logging every one preserves the cause
                    # distribution.
                    logger.warning(
                        f'[CAM Class ] grabResult.GrabSucceeded()=False '
                        f'err_code={err_code} desc={err_desc!r}'
                    )
                    # _record_failure returns True after
                    # ImageHandlerBase.MAX_CONSECUTIVE_FAILURES (128)
                    # consecutive failures. At 30 fps that is ~4.3s of
                    # back-to-back failures; at lower frame rates it
                    # can take longer.
                    should_stop = self._base._record_failure()
                    if should_stop:
                        try:
                            logger.error('[CAM Class ] Too many grab failures; stopping acquisition')
                            if self._parent.active and self._parent.is_grabbing():
                                self._parent.stop_grabbing()
                            self._parent._mark_disconnected()
                        except Exception:
                            pass
        except Exception as e:
            _outcome = 'exception_outer'
            logger.exception(f'[CAM Class ] OnImageGrabbed unexpected error: {e}')
        finally:
            if _trace_enabled and _t0 is not None:
                _dt_ms = (time.perf_counter() - _t0) * 1000.0
                profile_trace.trace(
                    'pylon_callback_trace.csv',
                    'ts_ms,duration_ms,thread_name,outcome,frame_bytes',
                    [
                        int(time.time() * 1000),
                        f'{_dt_ms:.3f}',
                        threading.current_thread().name,
                        _outcome,
                        _frame_bytes,
                    ],
                )

    def reset(self):
        """Clear frame buffer, queue, and failure counter."""
        try:
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
        except Exception:
            pass
        self._base.reset()

    def get_last_image(self):
        """Return (success, image_copy, timestamp) with parent-camera validity check."""
        try:
            if self._parent._device_removed:
                return False, None, None
            if self._parent.active is None:
                return False, None, None
        except Exception:
            return False, None, None

        return self._base.get_last_image()


# Handle camera removal events to flag device disconnect
class _CameraRemovalHandler(pylon.ConfigurationEventHandler):
    def __init__(self, parent_cam: PylonCamera):
        super().__init__()
        self._parent = parent_cam

    def OnCameraDeviceRemoved(self, camera):
        # Runs in a native Pylon SDK thread. _mark_disconnected acquires
        # _state_lock for microseconds and sets _device_removed +
        # _active=None atomically; safe from any thread including SDK
        # callbacks.
        self._parent._mark_disconnected()
        logger.error('[CAM Class ] Camera physically removed (Pylon SDK callback)')
