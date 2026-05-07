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
    #
    # 5.0s interval: long enough that CPU + trace-CSV row volume stay
    # negligible during a multi-hour bench run; short enough that
    # under-run / missed-frame / resync transitions land in the same
    # row their root cause did. Falsify: if an event class shows up
    # in the user log faster than the poller can catch the underlying
    # counter delta, drop the interval.
    _STATS_POLLER_INTERVAL_S = 5.0
    _UNDERRUN_NODE_NAME = 'Statistic_Buffer_Underrun_Count'
    _RESYNC_NODE_NAME = 'Statistic_Resynchronization_Count'
    _MISSED_FRAME_NODE_NAME = 'Statistic_Missed_Frame_Count'
    _STATS_NODE_NAMES = (
        'Statistic_Total_Buffer_Count',
        'Statistic_Failed_Buffer_Count',
        'Statistic_Resynchronization_Count',
        'Statistic_Missed_Frame_Count',
    )

    def _start_stats_poller(self):
        if profile_trace is None or not profile_trace.ENABLE_PROFILE_TRACE:
            return
        # Active poller cleanup before fresh start. Naive code that
        # returns early on existing.is_alive() loses the poller across
        # rapid stop/start_grabbing cycles: between _stop_stats_poller
        # setting the event and the daemon thread actually exiting, a
        # fresh start_grabbing skips starting a new poller, then the
        # old thread exits on its already-set event -- leaving no
        # poller running. Actively join the prior thread (with bounded
        # timeout) before starting a new one. Idempotent if no prior
        # poller.
        existing = getattr(self, '_stats_poller_thread', None)
        existing_ev = getattr(self, '_stats_poller_stop', None)
        if existing is not None and existing.is_alive():
            if existing_ev is not None:
                existing_ev.set()
            # 10.0s is far more than the ~5s _STATS_POLLER_INTERVAL_S
            # tick the daemon may be sleeping inside; bounded so a
            # stuck thread can't deadlock start_grabbing. Falsify: if
            # this timeout fires routinely, the thread is genuinely
            # stuck (not just slow-waking) and the start-side log
            # warning surfaces it.
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
            # 2.0s: stop is the fast path -- the daemon's longest
            # blocking call is ev.wait(_STATS_POLLER_INTERVAL_S=5.0)
            # but ev.set() above breaks it immediately; 2s covers any
            # in-progress node-read jitter. Symmetric with start-side
            # 10s join (which had to tolerate a fresh tick already in
            # flight). Falsify: if this fires, the daemon is wedged
            # in a long node-read or thread-rename block.
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
            # --- Pylon SDK statistics ---
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

            # Underrun: prominent marker, on its own log line.
            if underrun_value is not None:
                logger.info(f'[INSTR UNDERRUN] {underrun_name}={underrun_value}')

            # Resync: prominent marker on positive delta. Per Basler
            # doc stream-grabber-parameters.html, "A host
            # resynchronization is considered the most serious error
            # case in the USB 3.0 and USB3 Vision specification."
            # Logged at WARNING because a non-zero delta is operator-
            # actionable; the absolute count is just the running total.
            resync_value = stats.get(self._RESYNC_NODE_NAME)
            prev_resync = getattr(self, '_prev_resync_count', None)
            if isinstance(resync_value, (int, float)) and isinstance(prev_resync, (int, float)):
                delta = resync_value - prev_resync
                if delta > 0:
                    logger.warning(
                        f'[INSTR RESYNC] {self._RESYNC_NODE_NAME} delta={delta} '
                        f'(total={resync_value})'
                    )
            if isinstance(resync_value, (int, float)):
                self._prev_resync_count = resync_value

            # Missed frames: prominent marker on positive delta. Per
            # Basler doc stream-grabber-parameters.html, "A high Missed
            # Frame Count indicates that the host controller doesn't
            # support the bandwidth of the camera, i.e., the host
            # controller does not retrieve the acquired images in
            # time." Visible BEFORE Failed_Buffer_Count moves -- early
            # bandwidth-stress signal.
            missed_value = stats.get(self._MISSED_FRAME_NODE_NAME)
            prev_missed = getattr(self, '_prev_missed_frame_count', None)
            if isinstance(missed_value, (int, float)) and isinstance(prev_missed, (int, float)):
                delta = missed_value - prev_missed
                if delta > 0:
                    logger.warning(
                        f'[INSTR MISSED] {self._MISSED_FRAME_NODE_NAME} delta={delta} '
                        f'(total={missed_value})'
                    )
            if isinstance(missed_value, (int, float)):
                self._prev_missed_frame_count = missed_value

            # Temperature state: per Basler doc temperature-state.html,
            # ace 2 / boost / dart M/R cameras stop image acquisition
            # when over-temperature is reached and require cooldown
            # before restart -- presents identically to a frame-rate
            # stall in the user log without temperature attribution.
            # Warn on any non-Ok state so the cause is visible.
            temp_state = self._node_attr_get(cam, 'TemperatureState') if cam is not None else None
            prev_temp_state = getattr(self, '_prev_temp_state', None)
            if temp_state is not None and temp_state != prev_temp_state:
                if temp_state in ('Critical', 'Error'):
                    logger.warning(
                        f'[INSTR TEMP] TemperatureState={temp_state!r} '
                        f'(was {prev_temp_state!r})'
                    )
                else:
                    logger.info(
                        f'[INSTR TEMP] TemperatureState={temp_state!r} '
                        f'(was {prev_temp_state!r})'
                    )
            self._prev_temp_state = temp_state

            profile_trace.trace(
                'pylon_stats_trace.csv',
                'ts_ms,total_buffer_count,failed_buffer_count,'
                'resync_count,missed_frame_count,'
                'underrun_node_name,underrun_value,resulting_fps,'
                'temperature_state',
                [
                    ts_ms,
                    stats.get('Statistic_Total_Buffer_Count'),
                    stats.get('Statistic_Failed_Buffer_Count'),
                    resync_value,
                    missed_value,
                    underrun_name,
                    underrun_value,
                    f'{rfr:.3f}' if rfr is not None else None,
                    temp_state,
                ],
            )

            # --- Thread counts ---
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
        # Stop the stats poller before tearing down the grab loop so the
        # poller doesn't read a half-disposed StreamGrabber. No-op when
        # the poller wasn't started (LVP_PROFILE_TRACE unset).
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
            # Start the periodic Pylon stats + thread-count poller.
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

            logger.info('[CAM Class ] Connected to Pylon camera')
            return True

        except genicam.RuntimeException as ex:
            logger.error(
                '[CAM Class ] Pylon camera connect failed (may be open in another '
                f'application): {ex}'
            )
            self.active = None
        except Exception:
            logger.exception('[CAM Class ] Pylon camera connect failed')
            self.active = None

        return False

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
                # Per Basler doc user-sets.html: "Loading a user set is
                # only possible when the camera is idle, i.e., not
                # acquiring images." update_camera_config() stops the
                # grab loop, but on slow hosts SDK StopGrabbing may not
                # have fully settled by the time we arrive here. Bounded
                # poll surfaces the condition in logs (warning, not
                # silent failure inside the outer try/except).
                for _ in range(20):
                    if not self.is_grabbing():
                        break
                    time.sleep(0.05)
                else:
                    logger.warning(
                        '[CAM Class ] init_camera_config: camera still '
                        'grabbing 1 s after update_camera_config stop; '
                        'UserSetLoad may fail'
                    )
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
                    # Per Basler doc free-run-image-acquisition.html:
                    # "Repeat the steps above for all available trigger
                    # types." A camera that exposes AcquisitionStart /
                    # FrameBurstStart / etc. needs all of them off, not
                    # just FrameStart, or a stray non-Off type leaks
                    # through and blocks free-run.
                    for entry in camera.TriggerSelector.GetEntries():
                        try:
                            if not entry.IsAvailable():
                                continue
                            camera.TriggerSelector.SetValue(entry.GetSymbolic())
                            camera.TriggerMode.SetValue('Off')
                        except Exception as e_inner:
                            logger.debug(
                                f'[CAM Class ] TriggerMode=Off skipped for '
                                f'one trigger type: {e_inner}'
                            )
                except Exception as e:
                    logger.debug(f'[CAM Class ] TriggerMode set skipped: {e}')
                # Enable per-frame chunks for gain/exposure/identity. Must
                # happen here -- ChunkModeActive is locked while grabbing
                # (genicam.AccessException). Settings persist across
                # subsequent stop/start cycles.
                self._enable_validity_chunks()
                self.set_pixel_format(pixel_format='Mono8')
                self.auto_gain(state=False)
                # Set explicit gain — camera default after UserSetLoad is undefined
                self.gain(0.0)
                camera.ReverseX.SetValue(True)
                if not self._use_camera_emulation:
                    self.init_auto_gain_focus()
                self.exposure_t(t=10)
                # 1900x1900: bench-witnessed driver-init default. Both
                # production cameras (a2A3536 ace 2 and daA3840 dart)
                # support this size; the 2100x2100 production-max ROI
                # lives at the UI input layer (data/scopes.json driven)
                # not at the driver default. Falsify: if a future
                # camera body's sensor doesn't reach 1900 in either
                # axis, the SetValue will clamp + warn from set_frame_size.
                self.set_frame_size(w=1900, h=1900)
        except genicam.RuntimeException as e:
            logger.error(f'[CAM Class ] Camera communication error during init_camera_config: {e}')
            self._mark_disconnected()
        except Exception as e:
            logger.exception(f'[CAM Class ] Unexpected error in init_camera_config: {e}')

    # Frame-identity chunk candidates in preference order. Different
    # Basler models advertise different per-frame identity chunks
    # (data-chunks.html). We probe ChunkSelector and enable the first
    # candidate the camera advertises; the read-side alias maps both
    # back to the same 'FrameID' dict key for the trace.
    _FRAME_IDENTITY_CHUNK_CANDIDATES = ('FrameID', 'Framecounter')

    # Always-enable chunks (no per-camera fallback; both production
    # USB3 cameras and dart M GigE advertise these).
    _CHUNK_TARGETS_ALWAYS = ('ExposureTime', 'Gain')

    def _enable_validity_chunks(self) -> None:
        """Enable ChunkExposureTime / ChunkGain + a per-frame identity
        chunk (ChunkFrameID or ChunkFramecounter) for chunk-driven
        validity.

        MUST be called while the camera is NOT grabbing (ChunkModeActive
        is locked while grabbing). Canonical caller is
        init_camera_config() inside update_camera_config().

        Idempotent: safely re-asserts settings if chunks were already
        enabled. Per-chunk failures are logged but do not raise; the
        validity layer falls back to skip_frames for unenabled chunks.
        Frame-identity is trace-only (frame_validity validates gain
        and exposure); skipping it does not break validity.
        """
        camera = self.active
        if camera is None:
            return
        advertised = self._probe_advertised_chunks(camera)
        if advertised is None:
            return  # _probe_advertised_chunks already logged
        try:
            camera.ChunkModeActive.Value = True
        except Exception as e:
            logger.warning(
                f'[CAM Class ] could not enable ChunkModeActive: {e}; '
                f'frame_validity will fall back to skip_frames calibration'
            )
            return

        targets = list(self._CHUNK_TARGETS_ALWAYS)
        frame_id_chunk = next(
            (c for c in self._FRAME_IDENTITY_CHUNK_CANDIDATES if c in advertised),
            None,
        )
        if frame_id_chunk is not None:
            targets.append(frame_id_chunk)
        else:
            logger.info(
                f'[CAM Class ] no frame-identity chunk advertised '
                f'(tried {self._FRAME_IDENTITY_CHUNK_CANDIDATES}); '
                f'enabling Gain + ExposureTime only'
            )

        for sel in targets:
            if sel not in advertised:
                logger.warning(
                    f'[CAM Class ] Chunk{sel} not advertised by this '
                    f'camera; frame_validity will fall back to skip_frames '
                    f'for that source'
                )
                continue
            try:
                camera.ChunkSelector.Value = sel
                camera.ChunkEnable.Value = True
            except Exception as e:
                logger.warning(
                    f'[CAM Class ] could not enable Chunk{sel}: {e}; '
                    f'frame_validity will fall back to skip_frames for that source'
                )

    @staticmethod
    def _probe_advertised_chunks(camera) -> set | None:
        """Return the set of ChunkSelector entry names advertised by
        the camera, or None if the ChunkSelector node is missing /
        unreadable. Shared by _enable_validity_chunks and
        probe_chunk_capabilities.
        """
        try:
            nm = camera.GetNodeMap()
            selector_node = nm.GetNode('ChunkSelector')
            if selector_node is None:
                logger.warning(
                    '[CAM Class ] ChunkSelector node missing; '
                    'frame_validity will fall back to skip_frames calibration'
                )
                return None
            advertised = set()
            for entry in selector_node.GetEntries():
                try:
                    advertised.add(entry.GetSymbolic())
                except Exception as e:
                    logger.debug(
                        f'[CAM Class ] ChunkSelector entry GetSymbolic() failed: {e}'
                    )
            return advertised
        except Exception as e:
            logger.warning(
                f'[CAM Class ] ChunkSelector introspection failed: {e}; '
                f'frame_validity will fall back to skip_frames calibration'
            )
            return None

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

        DLTL throttles the camera by inserting pauses between network
        packets. Per Basler doc network-bandwidth-control-(blaze).md
        the mechanism is identical across transports (USB3 ace 2 /
        dart R, GigE dart M).

        Per-camera defaults bench-witnessed (USB3):
          ace 2 a2A3536-31umBAS    On / 360 MB/s -> 28.8 fps
          dart   daA3840-45um      On / 160 MB/s -> 18.7 fps

        Setting Mode=Off lets the camera run at sensor-readout max
        (~31.2 fps ace 2; ~44.9 fps dart on USB3). Two failure modes
        per per-camera spec pages (a2a3536-31umbas.html,
        daa3840-45um.html):

          - Too high: "Corrupt or dropped frames may occur if the
            DeviceLinkThroughputLimit parameter is too high."
          - Too low (rolling shutter cameras): "image distortion
            (increased rolling shutter effect) may occur if the
            DeviceLinkThroughputLimit parameter is too low."

        Both production USB3 cameras are rolling-shutter so both
        warnings apply. Bench-test failure rate AND image quality
        alongside fps before settling on a per-camera production
        default.

        **Transport caveat (GigE):** on GigE cameras (e.g.
        dmA3536-9gm), DLTL is bounded above by the GigE wire limit
        (~110 MB/s usable on 1 Gbps Ethernet, ~109 MB/s on this
        camera at 9.3 fps Mono8). Setting DLTL above the wire limit
        is a no-op; setting it below caps fps proportionally.
        Materially different from USB3 where DLTL sits well below
        the 5 Gbps wire limit and the knob has full bandwidth
        headroom. For GigE bandwidth control across multiple
        cameras on one link, use ``set_gev_inter_packet_delay`` and
        ``set_bandwidth_reserve_mode`` instead -- those are the
        GigE-side tools.

        ``value_bps`` outside the camera's supported range is clamped
        to ``DeviceLinkThroughputLimit.GetMin()`` /
        ``GetMax()`` with a warning log; the SDK would otherwise raise
        OutOfRangeException.

        Args:
            mode: ``'On'`` or ``'Off'``. Case-sensitive (matches Pylon
                enum entry symbolic names).
            value_bps: Throughput cap in bytes per second when
                ``mode='On'``. Ignored when ``mode='Off'``. If None
                while ``mode='On'``, only the mode is changed and the
                existing limit value is preserved. Out-of-range values
                are clamped to the camera's supported range with a
                warning.

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
                value_bps = self._clamp_dltl_value_bps(int(value_bps))
                self.active.DeviceLinkThroughputLimit.SetValue(value_bps)
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

    def _clamp_dltl_value_bps(self, value_bps: int) -> int:
        """Clamp a DLTL value to the camera's supported range.

        Returns the value unchanged if min/max query fails (best
        effort -- the SDK will reject out-of-range values with
        OutOfRangeException; the caller's RuntimeException branch
        catches that).
        """
        try:
            node = self.active.DeviceLinkThroughputLimit
            lo = int(node.GetMin())
            hi = int(node.GetMax())
        except Exception as e:
            logger.debug(
                f'[CAM Class ] DLTL min/max query failed: {e}; not clamping'
            )
            return value_bps
        if value_bps < lo:
            logger.warning(
                f'[CAM Class ] DLTL value {value_bps} below camera minimum '
                f'{lo}; clamping. Per Basler doc, very low DLTL on rolling-'
                f'shutter cameras can introduce image distortion.'
            )
            return lo
        if value_bps > hi:
            logger.warning(
                f'[CAM Class ] DLTL value {value_bps} above camera maximum '
                f'{hi}; clamping. Per Basler doc, very high DLTL can cause '
                f'corrupt or dropped frames.'
            )
            return hi
        return value_bps

    def set_max_transfer_size(self, value_bytes: int) -> bool:
        """Set StreamGrabber MaxTransferSize (USB3 only).

        Per Basler `stream-grabber-parameters.html`, MaxTransferSize
        is the bytes-per-USB-transfer the SDK requests from the kernel.
        It is the doc's named lever for the symptom "fails to receive
        image stream" -- decreasing the value works around kernel /
        driver USB-transfer-size constraints on some Windows hosts.
        Default is camera/SDK-version dependent.

        USB3-only. The node is absent on the StreamGrabber NodeMap of
        GigE cameras; in that case the write fails with a SDK
        exception, which we surface as HardwareError.

        Bench knob exposed parallel to the DLTL setter for
        ``tools/pylon_probe_sweep.py`` characterization. No production
        default change.

        Args:
            value_bytes: New MaxTransferSize in bytes. SDK clamps to
                the node's supported range and raises OutOfRangeException
                on invalid values; the RuntimeException branch surfaces
                that as HardwareError.

        Returns:
            bool: True on success. False if the camera is inactive
                (caller-correctable guard).

        Raises:
            HardwareError: Underlying SDK call failed (RuntimeException
                or node-missing). Camera is marked disconnected on
                RuntimeException.
        """
        return self._set_stream_grabber_int_node(
            node_name='MaxTransferSize',
            value=int(value_bytes),
            method_label='set_max_transfer_size',
        )

    def set_num_max_queued_urbs(self, value: int) -> bool:
        """Set StreamGrabber NumMaxQueuedUrbs (USB3 only).

        Per Basler `stream-grabber-parameters.html`, NumMaxQueuedUrbs
        controls how many USB Request Blocks (URBs) the SDK keeps in
        flight to the kernel at once. It is the doc's named lever for
        the symptom "insufficient system memory" with status codes
        0xe2010130 / 0xe2100001 -- decreasing the value reduces kernel
        URB allocation pressure on memory-constrained hosts.

        USB3-only. The node is absent on the StreamGrabber NodeMap of
        GigE cameras; in that case the write fails with a SDK
        exception, which we surface as HardwareError.

        Bench knob exposed parallel to the DLTL setter for
        ``tools/pylon_probe_sweep.py`` characterization. No production
        default change.

        Args:
            value: New NumMaxQueuedUrbs (count). SDK clamps to the
                node's supported range and raises OutOfRangeException
                on invalid values; the RuntimeException branch surfaces
                that as HardwareError.

        Returns:
            bool: True on success. False if the camera is inactive
                (caller-correctable guard).

        Raises:
            HardwareError: Underlying SDK call failed (RuntimeException
                or node-missing). Camera is marked disconnected on
                RuntimeException.
        """
        return self._set_stream_grabber_int_node(
            node_name='NumMaxQueuedUrbs',
            value=int(value),
            method_label='set_num_max_queued_urbs',
        )

    def _set_stream_grabber_int_node(
        self,
        node_name: str,
        value: int,
        method_label: str,
    ) -> bool:
        """Shared helper for StreamGrabber integer-node setters.

        Used by ``set_max_transfer_size`` and ``set_num_max_queued_urbs``.
        Both write a single integer node on the StreamGrabber NodeMap
        with identical error / log shape; this helper is the canonical
        path so the two public setters stay one-liners (Rule 35).
        """
        if not self.active:
            return False
        try:
            sg = self.active.GetStreamGrabberNodeMap()
            node = sg.GetNode(node_name)
            if node is None:
                logger.error(
                    f'[CAM Class ] {method_label}: StreamGrabber node '
                    f'{node_name!r} not present (likely GigE camera; '
                    f'this knob is USB3-only)'
                )
                raise HardwareError(
                    f'{method_label}: StreamGrabber node {node_name!r} '
                    f'not present on this camera'
                )
            if _cam_log is not None:
                _cam_log.info(
                    f'pylon StreamGrabber.{node_name}.SetValue({value})'
                )
            node.SetValue(value)
            return True
        except HardwareError:
            raise
        except genicam.RuntimeException as e:
            if _cam_log is not None:
                _cam_log.error(
                    f'pylon {method_label}({value}) FAILED: {e}'
                )
            logger.error(
                f'[CAM Class ] Camera communication error in '
                f'{method_label}: {e}'
            )
            self._mark_disconnected()
            raise HardwareError(
                f'{method_label}({value}) failed: {e}'
            ) from e
        except Exception as e:
            logger.exception(
                f'[CAM Class ] Unexpected error in {method_label}: {e}'
            )
            return False

    _ACQ_STOP_MODES = ('Complete', 'CancelExposure', 'AbortExposure')

    def set_acquisition_stop_mode(self, mode: str) -> bool:
        """Set BslAcquisitionStopMode -- behavior when StopGrabbing fires
        during an in-flight exposure.

        Per Basler doc acquisition-start-stop-and-abort.html:

          - ``'Complete'`` (default): waits for the current exposure to
            finish before stopping. On a 10 s fluorescence exposure,
            StopGrabbing waits up to 10 s -- presents identically to a
            multi-second app-side stall when the user toggles modes.
          - ``'CancelExposure'``: stops cleanly; partial frame discarded.
          - ``'AbortExposure'``: aborts immediately; partial frame
            discarded.

        Specifics table confirms ace 2 + dart M/R + boost R support
        the parameter. Default is unchanged in init_camera_config;
        this setter exists for bench characterization. Per Eric
        direction: setter-only first, default unchanged, bench-
        validate, flip default if validated.

        Args:
            mode: One of ``'Complete'``, ``'CancelExposure'``,
                ``'AbortExposure'``.

        Returns:
            bool: True on success. False if the camera is inactive,
                the mode argument is invalid, or
                BslAcquisitionStopMode is not exposed by this camera /
                firmware.

        Raises:
            HardwareError: Underlying SDK call failed
                (RuntimeException). Camera is marked disconnected.
        """
        if not self.active:
            return False
        if mode not in self._ACQ_STOP_MODES:
            logger.error(
                f"[CAM Class ] set_acquisition_stop_mode: mode must be one "
                f"of {self._ACQ_STOP_MODES}; got {mode!r}"
            )
            return False
        try:
            if _cam_log is not None:
                _cam_log.info(
                    f'pylon BslAcquisitionStopMode.SetValue({mode!r})'
                )
            node = self.active.GetNodeMap().GetNode('BslAcquisitionStopMode')
            if node is None:
                logger.warning(
                    f'[CAM Class ] BslAcquisitionStopMode node not exposed; '
                    f'set_acquisition_stop_mode({mode!r}) is a no-op on '
                    f'this camera'
                )
                return False
            node.SetValue(mode)
            return True
        except genicam.RuntimeException as e:
            if _cam_log is not None:
                _cam_log.error(
                    f'pylon set_acquisition_stop_mode({mode!r}) FAILED: {e}'
                )
            logger.error(
                f'[CAM Class ] Camera communication error in '
                f'set_acquisition_stop_mode: {e}'
            )
            self._mark_disconnected()
            raise HardwareError(
                f'set_acquisition_stop_mode({mode!r}) failed: {e}'
            ) from e
        except Exception as e:
            logger.exception(
                f'[CAM Class ] Unexpected error in '
                f'set_acquisition_stop_mode: {e}'
            )
            return False

    _BANDWIDTH_RESERVE_MODES = ('Default', 'Performance')

    def set_bandwidth_reserve_mode(self, mode: str) -> bool:
        """Set BandwidthReserveMode -- GigE only.

        Per Basler doc network-related-parameters.md:

          - ``'Default'``: camera reserves a portion of bandwidth for
            packet retransmits.
          - ``'Performance'``: all bandwidth dedicated to image
            transmit; minimal retransmit reserve.

        dmA3536-9gm spec footnote: "9.5 fps (with Bandwidth Reserve
        mode set to Performance)" -- vs the default 9.3 fps. The
        knob is load-bearing for actual fps on GigE cameras.

        USB3 cameras do not expose the node; returns False without
        warning so the bench-probe sweep can call this method
        unconditionally per cell.

        Args:
            mode: ``'Default'`` or ``'Performance'``.

        Returns:
            bool: True on success. False if the camera is inactive,
                mode is invalid, or the BandwidthReserveMode node is
                not exposed (USB3 camera).

        Raises:
            HardwareError: SDK RuntimeException during the write.
        """
        if not self.active:
            return False
        if mode not in self._BANDWIDTH_RESERVE_MODES:
            logger.error(
                f"[CAM Class ] set_bandwidth_reserve_mode: mode must be one "
                f"of {self._BANDWIDTH_RESERVE_MODES}; got {mode!r}"
            )
            return False
        try:
            node = self.active.GetNodeMap().GetNode('BandwidthReserveMode')
            if node is None:
                return False
            if _cam_log is not None:
                _cam_log.info(f'pylon BandwidthReserveMode.SetValue({mode!r})')
            node.SetValue(mode)
            return True
        except genicam.RuntimeException as e:
            if _cam_log is not None:
                _cam_log.error(
                    f'pylon set_bandwidth_reserve_mode({mode!r}) FAILED: {e}'
                )
            logger.error(
                f'[CAM Class ] Camera communication error in '
                f'set_bandwidth_reserve_mode: {e}'
            )
            self._mark_disconnected()
            raise HardwareError(
                f'set_bandwidth_reserve_mode({mode!r}) failed: {e}'
            ) from e
        except Exception as e:
            logger.exception(
                f'[CAM Class ] Unexpected error in '
                f'set_bandwidth_reserve_mode: {e}'
            )
            return False

    def set_gev_packet_size(self, size_bytes: int) -> bool:
        """Set GevSCPSPacketSize -- GigE only.

        Per Basler doc network-related-parameters.md, the GigE
        packet size governs jumbo-frame negotiation. 1500 is the
        standard Ethernet MTU; 9000 is the typical jumbo-frame
        size. Loss probability scales with packet count, so larger
        packets reduce host CPU and packet rate -- but jumbo frames
        require OS-level configuration on the host (per the
        bundled network-configuration-(gige-cameras).md).

        Per dmA3536-9gm spec at 9.3 fps Mono8 (~109 MB/s) the camera
        is at the GigE wire limit; packet size dominates per-frame
        packet count (1500 MTU -> ~78k pkts/s; 9000 MTU -> ~13k
        pkts/s).

        USB3 cameras do not expose the node; returns False without
        warning so the bench-probe sweep can call this method
        unconditionally per cell.

        Args:
            size_bytes: Packet size in bytes. Typical values:
                1500 (standard MTU) or 9000 (jumbo).

        Returns:
            bool: True on success. False if the camera is inactive,
                size_bytes is non-positive, or the GevSCPSPacketSize
                node is not exposed (USB3 camera).

        Raises:
            HardwareError: SDK RuntimeException during the write.
        """
        if not self.active:
            return False
        if not isinstance(size_bytes, int) or size_bytes <= 0:
            logger.error(
                f"[CAM Class ] set_gev_packet_size: size_bytes must be a "
                f"positive int; got {size_bytes!r}"
            )
            return False
        try:
            node = self.active.GetNodeMap().GetNode('GevSCPSPacketSize')
            if node is None:
                return False
            if _cam_log is not None:
                _cam_log.info(f'pylon GevSCPSPacketSize.SetValue({size_bytes})')
            node.SetValue(int(size_bytes))
            return True
        except genicam.RuntimeException as e:
            if _cam_log is not None:
                _cam_log.error(
                    f'pylon set_gev_packet_size({size_bytes}) FAILED: {e}'
                )
            logger.error(
                f'[CAM Class ] Camera communication error in '
                f'set_gev_packet_size: {e}'
            )
            self._mark_disconnected()
            raise HardwareError(
                f'set_gev_packet_size({size_bytes}) failed: {e}'
            ) from e
        except Exception as e:
            logger.exception(
                f'[CAM Class ] Unexpected error in set_gev_packet_size: {e}'
            )
            return False

    def set_gev_inter_packet_delay(self, delay_ticks: int) -> bool:
        """Set GevSCPD (inter-packet delay in clock ticks) -- GigE only.

        Per Basler doc network-related-parameters.md, GevSCPD inserts
        a wait between successive packets. Used to throttle a camera
        when multiple cameras share a single GigE link or when the
        host CPU can't keep up at full bandwidth. Larger values =
        slower transmit = lower per-camera throughput; combine across
        multiple cameras to share a single link.

        USB3 cameras do not expose the node; returns False without
        warning so the bench-probe sweep can call this method
        unconditionally per cell.

        Args:
            delay_ticks: Inter-packet delay in GigE clock ticks
                (camera-specific tick rate). 0 = no delay.

        Returns:
            bool: True on success. False if the camera is inactive,
                delay_ticks is negative, or the GevSCPD node is
                not exposed (USB3 camera).

        Raises:
            HardwareError: SDK RuntimeException during the write.
        """
        if not self.active:
            return False
        if not isinstance(delay_ticks, int) or delay_ticks < 0:
            logger.error(
                f"[CAM Class ] set_gev_inter_packet_delay: delay_ticks "
                f"must be a non-negative int; got {delay_ticks!r}"
            )
            return False
        try:
            node = self.active.GetNodeMap().GetNode('GevSCPD')
            if node is None:
                return False
            if _cam_log is not None:
                _cam_log.info(f'pylon GevSCPD.SetValue({delay_ticks})')
            node.SetValue(int(delay_ticks))
            return True
        except genicam.RuntimeException as e:
            if _cam_log is not None:
                _cam_log.error(
                    f'pylon set_gev_inter_packet_delay({delay_ticks}) '
                    f'FAILED: {e}'
                )
            logger.error(
                f'[CAM Class ] Camera communication error in '
                f'set_gev_inter_packet_delay: {e}'
            )
            self._mark_disconnected()
            raise HardwareError(
                f'set_gev_inter_packet_delay({delay_ticks}) failed: {e}'
            ) from e
        except Exception as e:
            logger.exception(
                f'[CAM Class ] Unexpected error in '
                f'set_gev_inter_packet_delay: {e}'
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
                '[CAM Class ] Camera communication error during '
                f'set_pixel_format({pixel_format}): {e}'
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
                f'[CAM Class ] Binning {self.get_binning_size()} -> {size}, '
                f'frame {self.get_frame_size()}'
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
                f'[CAM Class ] Binning set to {self.get_binning_size()}, '
                f'frame now {self.get_frame_size()}'
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
                    '[CAM Class ] Binning mismatch detected between '
                    f'horizontal ({horiz_bin}) and vertical ({vert_bin})'
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
        # be changed while StartGrabbing is active. The previous wrap
        # in update_camera_config() forced a stop_grabbing /
        # start_grabbing cycle on every call -- a needless over-stop
        # of the same structural class as wrapping any other
        # live-writable parameter.
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
                '[CAM Class ] Camera communication error during '
                f'update_auto_gain_target_brightness({auto_target_brightness}): {e}'
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
        # active. The previous wrap in update_camera_config() forced a
        # stop_grabbing / start_grabbing cycle on every call -- a
        # needless over-stop of the same structural class as wrapping
        # any other live-writable parameter. auto_gain() calls this
        # method AND update_auto_gain_target_brightness, so the prior
        # wrapped form stop/started twice per auto_gain invocation;
        # both wraps removed so the whole chain stays online.
        try:
            if min_gain is None:
                min_gain = self.active.AutoGainLowerLimit.Min

            if max_gain is None:
                max_gain = self.active.AutoGainUpperLimit.Max

            if _cam_log is not None:
                _cam_log.info(
                    f'pylon AutoGainLowerLimit.SetValue({min_gain}) '
                    f'AutoGainUpperLimit.SetValue({max_gain})'
                )
            self.active.AutoGainLowerLimit.SetValue(min_gain)
            self.active.AutoGainUpperLimit.SetValue(max_gain)
        except genicam.RuntimeException as e:
            logger.error(
                '[CAM Class ] Camera communication error during '
                f'update_auto_gain_min_max(min={min_gain}, max={max_gain}): {e}'
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
        # Per-grab duration trace; zero overhead when
        # ENABLE_PROFILE_TRACE is unset (production builds).
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
                # queue.Empty inherits from Exception -- both timeout and
                # other errors are caught here. Outcome classification
                # distinguishes them in the trace row.
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
                    f'pylon Width.SetValue({width}) Height.SetValue({height}) '
                    'BslCenterX/Y.Execute() (geometry-realloc)'
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
            # Per Basler doc gain.html three-step recipe: GainAuto Off
            # (caller's responsibility) -> GainSelector All -> Gain
            # SetValue. Asserting GainSelector='All' here is defensive
            # against upstream code that may have set GainSelector to
            # a per-channel selector. Per-write try/except: tolerate
            # cameras that don't expose the selector.
            try:
                self.active.GainSelector.SetValue('All')
            except Exception as e_sel:
                logger.debug(
                    f'[CAM Class ] GainSelector.SetValue(All) skipped: {e_sel}'
                )
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
                    f'pylon auto_gain(state={state}, target={target_brightness}, '
                    f'min={min_gain}, max={max_gain})'
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
            except Exception as e:
                logger.debug(
                    f'[CAM Class ] chunk_data_probe could not read {genicam_attr}: {e}'
                )

        was_grabbing = False
        try:
            was_grabbing = bool(camera.IsGrabbing())
        except Exception as e:
            logger.debug(
                f'[CAM Class ] chunk_data_probe IsGrabbing() check failed: {e}'
            )
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
                    except Exception as e:
                        logger.debug(
                            f'[CAM Class ] chunk_data_probe entry GetSymbolic() failed: {e}'
                        )
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
                except Exception as e:
                    logger.warning(
                        f'[CAM Class ] chunk_data_probe could not restore '
                        f'Chunk{sel} prior state ({prior}): {e}'
                    )
            if prior_chunk_mode is not None:
                try:
                    camera.ChunkModeActive.Value = prior_chunk_mode
                except Exception as e:
                    logger.warning(
                        f'[CAM Class ] chunk_data_probe could not restore '
                        f'ChunkModeActive prior state ({prior_chunk_mode}): {e}'
                    )
            if was_grabbing:
                try:
                    self.start_grabbing()
                except Exception as e:
                    result['errors'].append(f'could not restart streaming: {e}')

        return result

    # Stream-grabber stat node names probed in read_diagnostic_snapshot.
    # All read defensively via _safe_node so missing nodes record
    # '<missing>' rather than raising. The list spans both transports;
    # the GigE-only resend counters return '<missing>' on USB3, and
    # the USB3 underrun counter returns '<missing>' on GigE on some
    # SDK versions. Per Basler doc stream-grabber-parameters.html.
    _DIAG_STAT_NODES = (
        'Statistic_Total_Buffer_Count',
        'Statistic_Failed_Buffer_Count',
        'Statistic_Buffer_Underrun_Count',
        'Statistic_Missed_Frame_Count',
        'Statistic_Resynchronization_Count',
        'Statistic_Last_Failed_Buffer_Status',
        'Statistic_Last_Failed_Buffer_Status_Text',
        'Statistic_Resend_Packet_Count',
        'Statistic_Resend_Request_Count',
        'Statistic_Failed_Packet_Count',
    )
    _DIAG_STAT_COUNTERS = (
        'Statistic_Total_Buffer_Count',
        'Statistic_Failed_Buffer_Count',
        'Statistic_Buffer_Underrun_Count',
        'Statistic_Missed_Frame_Count',
        'Statistic_Resynchronization_Count',
        'Statistic_Resend_Packet_Count',
        'Statistic_Resend_Request_Count',
        'Statistic_Failed_Packet_Count',
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
        # 3.0s default: matches the bench probe shape used to
        # characterize dart vs ace 2 on Mac (session 65 / 68).
        # Long enough for cumulative counters to advance visibly at
        # 18-30 fps without being so long the operator gets bored.
        # Falsify: if running this at 3s misses a class of error that
        # only shows up over longer windows, callers raise the value
        # explicitly per-call.
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
            (('BslDeviceLinkCurrentThroughput',),
             'current_throughput_bps'),
            (('AcquisitionFrameRateEnable',), 'acquisition_frame_rate_enable'),
            (('AcquisitionFrameRate',), 'acquisition_frame_rate'),
            (('BslResultingAcquisitionFrameRate', 'ResultingFrameRate'),
             'resulting_frame_rate'),
            # GigE network-related parameters per Basler doc
            # network-related-parameters.md. Read-only entries
            # (assigned bandwidth, max throughput, frame jitter max,
            # payload size) characterise the network state; settable
            # entries (heartbeat, packet size, inter-packet delay,
            # bandwidth reserve) are tunable knobs. All return
            # `<missing>` on USB3 cameras (a2A3536-31umBAS,
            # daA3840-45um) per the doc's per-camera matrix.
            (('GevHeartbeatTimeout',), 'gev_heartbeat_timeout_ms'),
            (('GevSCPSPacketSize',), 'gev_packet_size_bytes'),
            (('GevSCPD',), 'gev_inter_packet_delay_ticks'),
            (('GevSCBWR',), 'gev_bandwidth_reserve_pct'),
            (('GevSCBWRA',), 'gev_bandwidth_reserve_accumulation'),
            (('GevSCBWA',), 'gev_bandwidth_assigned_bps'),
            (('GevSCDMT',), 'gev_device_max_throughput_bps'),
            (('GevSCFJM',), 'gev_frame_jitter_max_ticks'),
            (('GevSCFTD',), 'gev_frame_transmission_delay_ticks'),
            (('BandwidthReserveMode',), 'bandwidth_reserve_mode'),
            (('PayloadSize',), 'payload_size_bytes'),
            # Thermal state. TemperatureState is the enum that drives
            # the over-temperature acquisition halt described in
            # temperature-state.html. BslTemperatureMax records the
            # peak observed; BslTemperatureStatusErrorCount counts
            # over-temp events. All read defensively.
            (('TemperatureState',), 'temperature_state'),
            (('BslTemperatureMax',), 'temperature_max_degC'),
            (('BslTemperatureStatusErrorCount',),
             'temperature_status_error_count'),
        ):
            result['config'][key] = self._safe_node(nm, *names)

        # Stream-grabber-nodemap configuration. The first four are
        # USB3-only per stream-grabber-parameters.html (URBs are USB
        # request blocks). The remaining group covers the GigE Vision
        # Packet Resend Mechanism per the same doc -- they apply on
        # GigE only. All read defensively: USB3-only entries return
        # `<missing>` on GigE and vice versa.
        for name, key in (
            ('MaxNumBuffer', 'max_num_buffer'),
            ('MaxTransferSize', 'max_transfer_size'),
            ('NumMaxQueuedUrbs', 'num_max_queued_urbs'),
            ('MaxBufferSize', 'max_buffer_size'),
            ('EnableResend', 'enable_resend'),
            ('PacketTimeout', 'packet_timeout_ms'),
            ('FrameRetention', 'frame_retention_ms'),
            ('MaximumNumberResendRequests', 'max_resend_requests'),
            ('FirewallTraversalInterval', 'firewall_traversal_interval_ms'),
            ('AutoPacketSize', 'auto_packet_size'),
            ('Type', 'tl_provider_type'),
            ('SocketBufferSize', 'socket_buffer_size_kb'),
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
                    # 64-iteration cap: bound the drain so a wedged
                    # firmware that reports BslErrorPresent=True
                    # forever can't infinite-loop the snapshot. 64 is
                    # far more than any expected error queue depth on
                    # ace 2 / dart M/R (typical: 0-3 entries per
                    # session). Falsify: if a real run actually drains
                    # 64 errors, the camera is in a catastrophic state
                    # AND the snapshot logs the truncation.
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
    # ChunkFrameID and ChunkFramecounter both map to 'FrameID' -- the
    # camera advertised one of them, _enable_validity_chunks enabled
    # whichever is present, the read side tries both and the active
    # one returns a value.
    _CHUNK_GRAB_RESULT_ATTRS = (
        ('ChunkExposureTime', 'ExposureTime'),
        ('ChunkGain', 'Gain'),
        ('ChunkFrameID', 'FrameID'),
        ('ChunkFramecounter', 'FrameID'),
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
            except Exception as e:
                logger.debug(
                    f'[CAM Class ] _extract_chunk_data could not read {chunk_attr}: {e}'
                )
        return chunks if chunks else None

    def OnImageGrabbed(self, camera, grabResult):
        # Per-callback duration trace; zero overhead when
        # ENABLE_PROFILE_TRACE is unset (production builds).
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
                    '[CAM Class ] OnImageGrabbed called but device '
                    'already marked as removed, ignoring'
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
                            logger.error(
                                '[CAM Class ] Too many grab failures; '
                                'stopping acquisition'
                            )
                            if self._parent.active and self._parent.is_grabbing():
                                self._parent.stop_grabbing()
                            self._parent._mark_disconnected()
                        except Exception as e:
                            logger.warning(
                                f'[CAM Class ] OnImageGrabbed could not stop grabbing '
                                f'after max failures: {e}'
                            )
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
        except Exception as e:
            logger.warning(f'[CAM Class ] handler reset queue-drain failed: {e}')
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
