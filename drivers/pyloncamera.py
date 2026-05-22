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


def _log_safely(message: str) -> None:
    """Best-effort logger for native-callback contexts.

    Pylon's OnImageGrabbed / OnCameraDeviceRemoved callbacks run on
    SDK-owned native threads. A Python exception escaping these
    callbacks crosses the Python/C++ boundary as a C++ exception in
    a native worker thread, which Windows resolves to std::terminate
    -- a silent process abort with no Python traceback. This helper
    is the marker for "exception swallowed because re-raising would
    crash the process." Logging is best-effort: if logging itself
    raises, the swallow stands.
    """
    try:
        if _cam_log is not None:
            _cam_log.warning(f'[CAM Class ] {message}')
    except BaseException:
        # If the logger itself fails there is no safer fallback from
        # a native callback context. The swallow stands.
        _ = None  # noqa -- intentional no-op marker for the AST scan


def _log_cam(level: str, message: str) -> None:
    """Mirror a camera-relevant log entry to BOTH the main log AND camera.log.

    Use for events that future debuggers will want to see in either file:
    identity readback (serial, firmware, profile), connect / disconnect,
    configuration changes (gain, exposure, frame size), TlFactory device
    enumeration, MaxNumBuffer + StreamGrabber settings, GetArray failures
    on the worker. High-volume per-frame SDK trace lines (Gain.SetValue,
    ExposureTime.SetValue) stay on _cam_log directly -- those are
    deliberately kept out of the main log to control noise.

    Args:
        level: 'info' / 'warning' / 'error' / 'debug' / 'exception'.
        message: The full log message. Convention: include '[CAM Class ]'
            prefix for grep-ability (matches the prefix in the main log).
    """
    getattr(logger, level)(message)
    if _cam_log is not None:
        try:
            getattr(_cam_log, level)(message)
        except Exception:
            # If the camera-log handler is wedged for any reason, the
            # main-log call above still landed. Swallow so the caller's
            # control flow isn't broken by a logging-side fault.
            logger.debug(f'[CAM Class ] _log_cam: camera_logger.{level}() raised')


# Pylon SDK error code returned by grabResult.GetErrorCode() when a
# buffer is cancelled by StopGrabbing in flight. Value 0xE2000102.
# USB3-Vision transport namespace (high byte 0xE2). Per Basler
# stream-grabber-parameters.html the transport split is "0xE1000014
# on GigE cameras and 0xE2000212 on USB 3.0 cameras"; on GigE the
# cancel code may differ. pypylon does not expose this as a named
# constant; if a future version adds pylon.GENERIC_BUFFER_CANCELED
# or similar, replace this.
_PYLON_ERR_BUFFER_CANCELED = 3791651074

# Camera-side FIFO overflow: payload dropped before transmission because
# the host stalled long enough that the camera's internal buffer filled.
# In AF-heavy protocols, SetValue for gain/exposure inside an AF cycle
# stalls the host briefly; with MaxNumBuffer at its default the camera-
# side FIFO can overflow during the stall. The next frame is invalidated
# via frame_validity, so consumers already wait for a clean frame -- the
# dropped frame is one the AF runner would have rejected anyway.
_PYLON_ERR_PAYLOAD_DISCARDED = 0xE2050012

# Device-not-found: USB-Vision transport returns this when the device
# handle no longer resolves on the bus (cable unplug, USB hub power
# loss, OS-level device removal). Bench-witnessed value 433 (decimal,
# from the LVP_Logbumped.wire bench session): "A device which does not
# exist was specified". Cascade rate observed at ~100+ events in <2s
# after disconnect, which would take ~4.3s to trip the slow-path
# MAX_CONSECUTIVE_FAILURES auto-disconnect (128 frames at 30fps). Fast
# classification short-circuits the cascade so the disconnect surfaces
# in 1 frame instead of 128, and the user notification (driven by
# _mark_disconnected -> API layer per Rule 14) fires immediately
# instead of 4 seconds late behind a wall of WARNING log lines.
_PYLON_ERR_DEVICE_NOT_FOUND = 433


# Build marker -- bumped whenever the OnImageGrabbed / disconnect
# defensive layer changes. Grep this in lumaviewpro.log to verify
# which Pylon-defense generation a bench build is running.
# Generation 3: Stage A / Stage B split (OnImageGrabbed runs only the
# native-thread fast path; heavy work moves to _PylonImageGrabWorker).
_PYLON_DEFENSE_BUILD = 'pylon-defense-3'


@camera_registry.register('pylon', priority=100)
class PylonCamera(Camera):
    def __init__(self, **kwargs):
        _log_cam('info',
            f'[CAM Class ] PylonCamera defense generation: '
            f'{_PYLON_DEFENSE_BUILD}'
        )

        if os.getenv('PYLON_CAMEMU', None) is not None:
            logger.info(
                '[CAM Class ] PylonCamera.connect() detected request to use camera emulation'
            )
            self._use_camera_emulation = True
        else:
            self._use_camera_emulation = False

        super().__init__()

    # _mark_disconnected() inherited from Camera base class

    def _schedule_async_teardown(self) -> None:
        """Spawn a daemon thread that runs disconnect() in a safe context.

        Used from OnImageGrabbed (and any other Pylon-callback path)
        when we detect device removal. The SDK callback thread MUST
        NOT call StopGrabbing on itself -- it deadlocks or triggers
        a native abort (pypylon issue #225). Spawning a daemon thread
        lets the callback return to the SDK immediately while teardown
        runs in a Python-owned context where Close() / DestroyDevice()
        are safe to call.

        Idempotent: a second call while a teardown thread is already
        running is a no-op (the in-flight thread does the work).
        Re-entrant safe via _async_teardown_started flag under
        _state_lock.
        """
        with self._state_lock:
            if getattr(self, '_async_teardown_started', False):
                return
            self._async_teardown_started = True

        def _run_teardown():
            try:
                # Small delay so the in-flight OnImageGrabbed callback
                # that scheduled us has time to return to the SDK
                # before we touch the camera handle from outside.
                time.sleep(0.05)
                _cam_log.info(
                    '[CAM Class ] async teardown after device removal: '
                    'calling disconnect() from daemon thread'
                )
                # disconnect() does the full safe sequence:
                # stop_grabbing -> wait_for_acquisition_idle -> Close
                # -> DetachDevice -> DestroyDevice, each independently
                # guarded.
                self.disconnect()
            except BaseException as e:
                # Best-effort log of the teardown failure. If logging
                # itself raises, suppress -- daemon thread death must
                # not leak. Done daemon, so process exit is fine if
                # everything below also fails.
                _log_safely(f'async teardown raised {type(e).__name__}: {e}')
            finally:
                with self._state_lock:
                    self._async_teardown_started = False

        t = threading.Thread(
            target=_run_teardown,
            name='PylonAsyncTeardown',
            daemon=True,
        )
        t.start()

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
            _cam_log.warning(f'[CAM Class ] _query_dynamic_capabilities failed: {e}')

    def disconnect(self) -> bool:
        """Tear down the active Pylon camera and release the SDK device.

        Each teardown step is independently guarded so one failure doesn't
        block the rest. After return, ``self.active is None`` regardless of
        SDK call success. Returns False if not connected or outer aborted.

        When the device is already known to be removed (cable unplug,
        callback err_code=433 already observed), the SDK-touching
        teardown steps (StopGrabbing, wait_for_acquisition_idle, Close)
        are SKIPPED. Calling these on a dead device handle is the
        pypylon #225 abort hazard: StopGrabbing can hang or trigger a
        native abort, and Close on an already-removed handle has been
        observed to crash. Only DetachDevice + DestroyDevice are kept,
        since they release Python-side ownership of the handle and do
        not communicate with the device.
        """
        try:
            if self.active is not None:
                device_removed = False
                try:
                    device_removed = self.is_device_removed()
                except BaseException:
                    device_removed = False
                if not device_removed:
                    try:
                        if self.is_grabbing():
                            self.stop_grabbing()
                    except Exception as e:
                        _cam_log.warning(
                            f'[CAM Class ] stop_grabbing during disconnect raised: {e}; '
                            f'continuing teardown'
                        )
                    # Drain in-flight frames after StopGrabbing before releasing
                    # the device handle (Basler `acquisition-status.html`). Bounded.
                    self._wait_for_acquisition_idle(timeout_s=2.0)
                    # Stage B worker stop AFTER stop_grabbing + idle-wait
                    # (so SDK has stopped firing callbacks and in-flight
                    # frames have drained from the SDK side) and BEFORE
                    # Close() / DetachDevice() / DestroyDevice() (so the
                    # worker has released its grabResult refs and the SDK
                    # input queue won't underrun on teardown). Inverting
                    # this order silently drops grabResults without release
                    # per `class_pylon_1_1_c_grab_result_ptr.html`.
                    self._stop_image_grab_worker()
                    # Each teardown step is independently guarded so a failure on
                    # one (e.g. Close on an already-removed device) does not
                    # prevent the others from running. The behaviour the caller
                    # expects after disconnect() returns is "self.active is None"
                    # regardless of whether the SDK calls themselves succeeded.
                    try:
                        self.active.Close()
                    except Exception as e:
                        _cam_log.warning(
                            f'[CAM Class ] Close() during disconnect raised: {e}; '
                            f'continuing teardown'
                        )
                else:
                    _log_safely(
                        'disconnect: device already removed -- skipping '
                        'StopGrabbing/wait_idle/Close (pypylon #225 hazard); '
                        'releasing Python-side handle only'
                    )
                    # Worker stop is safe regardless of device state: it
                    # touches only its own queue + thread, not the SDK
                    # handle. Run it before DetachDevice/DestroyDevice so
                    # any in-flight grabResults are released cleanly.
                    self._stop_image_grab_worker()
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
                    _cam_log.warning(
                        f'[CAM Class ] DetachDevice() during disconnect raised: {e}; '
                        f'continuing teardown'
                    )
                try:
                    self.active.DestroyDevice()
                except Exception as e:
                    _cam_log.warning(
                        f'[CAM Class ] DestroyDevice() during disconnect raised: {e}; '
                        f'continuing teardown'
                    )
                self.active = None
                # Reset the connection-scoped self-validation flag so
                # the next connect re-runs the StreamGrabber NodeMap
                # walk against whatever camera attaches.
                self._pylon_self_validation_done = False
                _log_cam('info', '[CAM Class ] Disconnected from Pylon camera')
                return True
            else:
                _log_cam('info', '[CAM Class ] Pylon camera not connected')
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Pylon camera disconnect failed: {e}')
        return False

    # __del__() inherited from Camera base class

    # Periodic Pylon SDK statistics + thread-count daemon poller.
    # No-op when profile_trace_enabled is false in settings.json.
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
                _cam_log.warning(
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
                _cam_log.warning(
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
                        _cam_log.warning(f'[INSTR PYLON ] NodeMap walk failed: {e}')
                else:
                    _cam_log.warning('[INSTR PYLON ] start: active camera is None, no stat dump')
            except Exception as e:
                _cam_log.warning(f'[INSTR PYLON ] start: stat-node dump failed: {e}')
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
                    _cam_log.warning(
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
                    _cam_log.warning(
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
                    _cam_log.warning(
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
            # Post-split: PylonImageGrab is the SDK native grab thread
            # (renamed by OnImageGrabbed Stage A); PylonImageGrabWorker
            # is the daemon that drains Stage B. Counted separately so
            # the bench trace can distinguish a leaked native thread
            # (SDK didn't tear down cleanly) from a leaked worker (our
            # disconnect path missed worker.stop). The earlier prefix-
            # match conflated both into a single column and silently
            # absorbed worker leakage.
            try:
                threads = threading.enumerate()
                n_pylon_native = sum(1 for t in threads if t.name == 'PylonImageGrab')
                n_pylon_worker = sum(
                    1 for t in threads if t.name == 'PylonImageGrabWorker'
                )
                n_dummy = sum(1 for t in threads if t.name.startswith('Dummy'))
                n_total = len(threads)
                profile_trace.trace(
                    'pylon_threads_trace.csv',
                    'ts_ms,pylon_native_grab_count,pylon_worker_count,'
                    'dummy_count,total_thread_count',
                    [ts_ms, n_pylon_native, n_pylon_worker, n_dummy, n_total],
                )
            except Exception as e:
                logger.debug(f'[INSTR PYLON ] thread-count poll error: {e}')

    def stop_grabbing(self) -> None:
        """Stop the camera's grab loop.

        Idempotent and exception-tolerant: if the SDK is already
        stopped or in an error state the call is logged and ignored
        rather than propagated, so callers (notably ``disconnect``)
        can safely sequence stop -> idle-wait -> teardown.
        """
        # Stop the stats poller before tearing down the grab loop so the
        # poller doesn't read a half-disposed StreamGrabber. No-op when
        # the poller wasn't started (profile_trace_enabled false).
        self._stop_stats_poller()
        camera = self.active
        if _cam_log is not None:
            _cam_log.info('pylon StopGrabbing()')
        try:
            camera.StopGrabbing()
        except Exception as e:
            if _cam_log is not None:
                _cam_log.warning(f'pylon StopGrabbing FAILED: {e}')
            _cam_log.warning(f'[CAM Class ] stop_grabbing ignored error: {e}')

    _ACQ_IDLE_POLL_INTERVAL_S = 0.020

    def _wait_for_acquisition_idle(self, timeout_s: float = 2.0) -> bool:
        """Poll AcquisitionActive (and ExposureActive when available)
        until the camera reports idle, or until ``timeout_s`` elapses.

        Per Basler ``acquisition-status.html`` both nodes are exposed
        on ace 2 / dart M / dart R cameras. Used by ``disconnect()``
        between ``stop_grabbing()`` and ``Close()`` so in-flight frames
        have time to drain before the device handle is released.

        Bounded by ``timeout_s`` so a stuck-active camera cannot block
        disconnect indefinitely; on timeout the disconnect path
        proceeds and a warning is logged for diagnosis.

        Args:
            timeout_s: Wall-clock timeout in seconds. Caller picks per
                use-case (disconnect uses 2.0).

        Returns:
            bool: True if the camera reached idle within timeout. False
                if timeout expired, the nodes are not present (older
                firmware / non-Basler), or a poll error fired.
        """
        if self.active is None:
            return True
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                nm = self.active.GetNodeMap()
                acq_node = nm.GetNode('AcquisitionActive')
                if acq_node is None:
                    # Older firmware or non-Basler -- can't poll, bail
                    # quickly without warning (this is expected).
                    return False
                acq_active = bool(acq_node.GetValue())
                exp_node = nm.GetNode('ExposureActive')
                exp_active = (
                    bool(exp_node.GetValue())
                    if exp_node is not None else False
                )
                if not acq_active and not exp_active:
                    return True
            except Exception as e:
                logger.debug(
                    f'[CAM Class ] _wait_for_acquisition_idle: node poll '
                    f'failed: {e}; bailing on idle wait'
                )
                return False
            time.sleep(self._ACQ_IDLE_POLL_INTERVAL_S)
        _cam_log.warning(
            f'[CAM Class ] _wait_for_acquisition_idle: timed out after '
            f'{timeout_s}s waiting for AcquisitionActive=False; '
            f'proceeding with teardown'
        )
        return False

    def _log_stream_grabber_status(self, label: str) -> None:
        """Snapshot StreamGrabber Status into the camera trace log.

        Per Basler ``stream-grabber-parameters.html`` the read-only
        ``Status`` node reflects grabber lifecycle state (Closed,
        Open, Grabbing, OutOfMemory, BufferAllocFailed,
        BufferRequestFailed, OutOfResources, ...). Logging the value
        at lifecycle transitions (e.g. before StartGrabbing) gives
        post-mortem analysis a concrete entry-state for
        STALL-1-class investigations.

        No-op when ``_cam_log`` is unset (profile_trace_enabled false
        in production builds) or when the camera is inactive. Older
        firmware / non-Basler cameras that do not expose the node
        log a missing-node line and continue (no failure).

        Args:
            label: Short tag identifying the lifecycle point (e.g.
                ``'pre-StartGrabbing'``, ``'post-StopGrabbing'``).
        """
        if _cam_log is None or self.active is None:
            return
        try:
            sg = self.active.GetStreamGrabberNodeMap()
            node = sg.GetNode('Status')
            if node is None:
                _cam_log.info(
                    f'pylon StreamGrabber.Status [{label}]: <node missing>'
                )
                return
            status = node.GetValue()
            _cam_log.info(
                f'pylon StreamGrabber.Status [{label}]: {status}'
            )
        except Exception as e:
            _cam_log.warning(
                f'pylon StreamGrabber.Status [{label}] read failed: {e}'
            )

    def start_grabbing(self) -> None:
        """Start the camera's grab loop with `LatestImageOnly` strategy.

        Idempotent: if grabbing already started (pypylon 26.4.x's
        ``AcquireContinuousConfiguration + Open()`` triggers an implicit
        StartGrabbing), this method returns early. Snapshots StreamGrabber
        Status into the trace log per B23. Starts the optional stats
        poller (no-op when profile_trace_enabled is false).

        Exception-tolerant: SDK failures are logged but not raised so
        UI handlers can call this without wrapping.

        ``LVP_PYLON_MAX_NUM_BUFFER`` / ``LVP_PYLON_MAX_TRANSFER_SIZE`` /
        ``LVP_PYLON_NUM_QUEUED_URBS`` / ``LVP_PYLON_GRAB_STRATEGY`` env
        vars override defaults for bench characterization.
        """
        camera = self.active
        if camera is None:
            return
        # Idempotent guard (B35). pypylon 26.4.x: AcquireContinuousConfiguration
        # + Open() triggers an implicit StartGrabbing before connect()'s
        # explicit start_grabbing() runs. Bench evidence 2026-05-08 (Mac +
        # Windows): a second StartGrabbing raises RuntimeException
        # "Grabbing has already been started" -- caught but logs spurious
        # WARNING on every connect.
        try:
            if camera.IsGrabbing():
                if _cam_log is not None:
                    _cam_log.info('start_grabbing: already grabbing, skipping')
                return
        except Exception as e:
            if _cam_log is not None:
                _cam_log.debug(f'start_grabbing IsGrabbing() check raised: {e}')
        try:
            # MaxNumBuffer cap retired 2026-05-08 (B34). The previous cap
            # of 3 was for Windows non-paged-pool pressure at full-res
            # Mono12 (originally observed ~228 MB startup spike). pypylon
            # 26.4.x makes MaxNumBuffer RO once grabbing has begun, AND
            # AcquireContinuousConfiguration auto-starts on Open() in
            # 26.4.x -- the cap window no longer exists. Bench data
            # (Mac dart M, 2026-05-08) shows the cap was also
            # counterproductive: ring-of-3 starves the buffer pool under
            # USB transfer hiccups (28% fail vs 13% at default 10).
            # Production now runs at SDK default. Override via
            # LVP_PYLON_MAX_NUM_BUFFER if a future Windows memory regression
            # surfaces.
            _mnb_env = os.environ.get('LVP_PYLON_MAX_NUM_BUFFER')
            if _mnb_env:
                try:
                    camera.MaxNumBuffer.SetValue(int(_mnb_env))
                    if _cam_log is not None:
                        _cam_log.info(f'pylon MaxNumBuffer.SetValue({_mnb_env}) [env]')
                except Exception as e:
                    if _cam_log is not None:
                        _cam_log.warning(f'pylon MaxNumBuffer override FAILED: {e}')
                    logger.debug(f'[CAM Class ] MaxNumBuffer override failed: {e}')
            # USB3 StreamGrabber tuning. Production default = SDK default
            # (MaxTransferSize=256KB, NumMaxQueuedUrbs=64). Bench overrides
            # via LVP_PYLON_MAX_TRANSFER_SIZE / LVP_PYLON_NUM_QUEUED_URBS to
            # characterize bandwidth-discard rate at sensor-max throughput.
            _mts = os.environ.get('LVP_PYLON_MAX_TRANSFER_SIZE')
            if _mts:
                try:
                    self.set_max_transfer_size(int(_mts))
                except Exception as e:
                    _cam_log.warning(
                        f'[CAM Class ] set_max_transfer_size({_mts}) failed: {e}'
                    )
            _nqu = os.environ.get('LVP_PYLON_NUM_QUEUED_URBS')
            if _nqu:
                try:
                    self.set_num_max_queued_urbs(int(_nqu))
                except Exception as e:
                    _cam_log.warning(
                        f'[CAM Class ] set_num_max_queued_urbs({_nqu}) failed: {e}'
                    )
            # B20 / B23: snapshot StreamGrabber.Status into the trace
            # log before StartGrabbing so post-mortem analysis can
            # correlate "weird StartGrabbing behavior" with the entry
            # state. Per Basler stream-grabber-parameters.html the node
            # is read-only and reflects grabber lifecycle state
            # (Closed / Open / Grabbing / OutOfMemory / etc.).
            self._log_stream_grabber_status('pre-StartGrabbing')
            # LVP_PYLON_GRAB_STRATEGY env var overrides the strategy for bench
            # characterization. Default LatestImageOnly is the production
            # contract -- frame_validity, capture_and_wait, and the
            # auto-discard skip_frames floor all depend on this strategy
            # (see DAILY_LOG 2026-05-04). OneByOne for apples-to-apples vs
            # Pylon Viewer's bandwidth test (which delivers every frame).
            _strategy_name = os.environ.get(
                'LVP_PYLON_GRAB_STRATEGY', 'LatestImageOnly'
            )
            _strategy_map = {
                'LatestImageOnly': pylon.GrabStrategy_LatestImageOnly,
                'OneByOne': pylon.GrabStrategy_OneByOne,
            }
            _strategy = _strategy_map.get(
                _strategy_name, pylon.GrabStrategy_LatestImageOnly
            )
            if _cam_log is not None:
                _cam_log.info(f'pylon StartGrabbing({_strategy_name}, ProvidedByInstantCamera)')
            camera.StartGrabbing(
                _strategy, pylon.GrabLoop_ProvidedByInstantCamera
            )
            # Start the periodic Pylon stats + thread-count poller.
            # No-op when profile_trace_enabled is false.
            self._start_stats_poller()
        except Exception as e:
            if _cam_log is not None:
                _cam_log.warning(f'pylon StartGrabbing FAILED: {e}')
            _cam_log.warning(f'[CAM Class ] start_grabbing ignored error: {e}')

    def is_grabbing(self) -> bool:
        """Return True if the camera is currently in a grab loop.

        Returns False on any SDK error so callers can use this as a
        defensive precondition (e.g. B28 ChunkModeActive guard) without
        a try/except.
        """
        try:
            return self.active.IsGrabbing()
        except Exception:
            return False

    def connect(self) -> bool:
        """Try to connect to the first available basler camera"""
        try:
            # Enumerate ALL Basler devices visible to pylon BEFORE
            # CreateFirstDevice() so the "no device available" failure
            # mode can be distinguished from "found but Open() failed".
            # The list also surfaces multi-camera-bench cases (wrong
            # serial selected) and transient enumeration races (device
            # visible but Pylon hasn't claimed it yet).
            try:
                _devs = pylon.TlFactory.GetInstance().EnumerateDevices()
                _log_cam('info',
                    f'[CAM Class ] pylon TlFactory.EnumerateDevices() returned '
                    f'{len(_devs)} device(s)'
                )
                for _i, _d in enumerate(_devs):
                    try:
                        _log_cam('info',
                            f'[CAM Class ]   device[{_i}]: '
                            f'model={_d.GetModelName()!r} '
                            f'serial={_d.GetSerialNumber()!r} '
                            f'tl={_d.GetTLType()!r} '
                            f'device_class={_d.GetDeviceClass()!r}'
                        )
                    except Exception as _e_acc:
                        _log_cam('debug',
                            f'[CAM Class ]   device[{_i}]: '
                            f'enumeration accessor failed: {_e_acc}'
                        )
            except Exception as _e_enum:
                _log_cam('warning',
                    f'[CAM Class ] pylon TlFactory.EnumerateDevices() '
                    f'failed: {_e_enum}'
                )

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

            # Start the Stage B worker BEFORE camera.Open(). pypylon 26.4.x's
            # AcquireContinuousConfiguration auto-StartGrabbing fires from
            # inside Open(); the worker queue must be drained-capable before
            # the first OnImageGrabbed can enqueue, or Stage A would lose
            # frames with no consumer scheduled.
            self.cam_image_handler._worker.start()

            camera.Open()

            # MaxNumBuffer cap (default 3). Applied post-Open() -- earliest
            # window before pypylon 26.4.x's AcquireContinuousConfiguration
            # auto-starts grabbing and locks the node. Bench-validated
            # 2026-05-08 (Windows dart M, sensor-max Mono8): cap=3 and
            # SDK-default both run at 0% fail rate / 0 resyncs/sec, so the
            # cap restores the original Windows non-paged-pool bound at
            # zero observed perf cost. Override via LVP_PYLON_MAX_NUM_BUFFER.
            _mnb_env = os.environ.get('LVP_PYLON_MAX_NUM_BUFFER', '3')
            try:
                camera.MaxNumBuffer.SetValue(int(_mnb_env))
                actual = camera.MaxNumBuffer.GetValue()
                _log_cam('info',
                    f'[CAM Class ] MaxNumBuffer cap applied post-Open: '
                    f'requested={_mnb_env} actual={actual}'
                )
            except Exception as e:
                _log_cam('warning',
                    f'[CAM Class ] MaxNumBuffer cap post-Open failed '
                    f'(window may have closed): {e}'
                )

            # Pre-allocate chunk node-map pool to match the buffer pool.
            # Per Basler grabchunkimage sample: node maps are otherwise
            # created lazily on StartGrabbing(), stalling first-frame
            # delivery while the SDK parses XML. Sized to MaxNumBuffer so
            # every in-flight grab buffer has a paired pre-built node map.
            try:
                _mnb_actual = camera.MaxNumBuffer.GetValue()
                camera.StaticChunkNodeMapPoolSize.Value = _mnb_actual
                _log_cam('info',
                    f'[CAM Class ] StaticChunkNodeMapPoolSize set to '
                    f'{_mnb_actual}'
                )
            except Exception as e:
                _log_cam('warning',
                    f'[CAM Class ] StaticChunkNodeMapPoolSize set failed '
                    f'(node may be unavailable on this transport): {e}'
                )

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
                        _log_cam('info', f'[CAM Class ] Pylon SDK version: {_ver_str}')
                    except Exception as e:
                        _log_cam('warning', f'[CAM Class ] Could not read Pylon SDK version: {e}')

                    # Transport + device class identify the kernel
                    # driver stack Pylon is routing through — useful
                    # when the runtime SDK says one thing but Device
                    # Manager shows a stale WinUSB/USB3Vision driver.
                    try:
                        _log_cam('info',
                            f'[CAM Class ] Transport: {dev_info.GetTLType()} '
                            f'/ DeviceClass: {dev_info.GetDeviceClass()}'
                        )
                    except Exception as e:
                        _log_cam('debug', f'[CAM Class ] TLType/DeviceClass unavailable: {e}')

                    device_serial = nm.GetNode('DeviceSerialNumber').ToString()
                    _log_cam('info', f'[CAM Class ] Camera Serial Number: {device_serial}')

                    firmware = nm.GetNode('DeviceFirmwareVersion').ToString()
                    _log_cam('info', f'[CAM Class ] Camera Firmware Version: {firmware}')

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
                        _log_cam('info',
                            f'[CAM Class ] Pixel format: {pix}, '
                            f'Resolution: {w}x{h}, Binning: {bh}x{bv}'
                        )
                    except Exception as e:
                        _log_cam('debug', f'[CAM Class ] Pixel/resolution/binning unavailable: {e}')

                    temps = self.get_all_temperatures()
                    for name, temp in temps.items():
                        _log_cam(
                            'info',
                            f'[CAM Class ] Camera {name} Temperature : {temp:.2f} degC',
                        )

                except Exception as e:
                    _cam_log.error(
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

            _log_cam('info', '[CAM Class ] Connected to Pylon camera')
            return True

        except genicam.RuntimeException as ex:
            _cam_log.error(
                '[CAM Class ] Pylon camera connect failed (may be open in another '
                f'application): {ex}'
            )
            self.active = None
            self._stop_image_grab_worker()
        except Exception:
            _cam_log.exception('[CAM Class ] Pylon camera connect failed')
            self.active = None
            self._stop_image_grab_worker()

        return False

    def _stop_image_grab_worker(self) -> None:
        """Stop the Stage B worker if it was started during connect().

        Tolerant of partial-construct states: a connect() failure before
        `cam_image_handler` is set, or before its `_worker` was attached,
        is a no-op. Idempotent on the success path -- `disconnect()`
        calls this explicitly before SDK teardown.
        """
        handler = getattr(self, 'cam_image_handler', None)
        if handler is None:
            return
        worker = getattr(handler, '_worker', None)
        if worker is None:
            return
        try:
            worker.stop(timeout=1.0)
        except Exception as e:
            _cam_log.warning(
                f'[CAM Class ] _stop_image_grab_worker raised: {e}; '
                f'continuing teardown'
            )

    def get_all_temperatures(self) -> dict:
        """Return {selector: degC, ...} per DeviceTemperatureSelector entry; {} if unreadable."""
        if not self.active:
            _cam_log.warning('[CAM Class ] get_all_temperatures(): inactive camera')
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
            _cam_log.error(f'[CAM Class ] Failed to read camera temperatures: {e}')
            self._mark_disconnected()
            return {}
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error reading temperatures: {e}')
            return {}

    def init_camera_config(self) -> None:
        """Apply Etaluma's canonical camera configuration once on connect.

        Loads the `Default` user set, sets pixel format to Mono8, sets
        explicit gain (0 dB), enables `ReverseX`, enables validity
        chunks (B28-guarded), and runs the auto-gain init defaults.
        Wrapped in `update_camera_config()` so the caller does not
        need a separate stop/start cycle.

        Exception-tolerant: SDK failures are logged but not raised --
        the camera remains connected with whatever subset of settings
        applied successfully.
        """
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
                    _cam_log.warning(
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
                # Enable per-frame chunks for gain/exposure/identity/timestamp.
                # Must happen here -- ChunkModeActive is locked while grabbing
                # (genicam.AccessException). Settings persist across
                # subsequent stop/start cycles.
                self._enable_validity_chunks()
                self._read_timestamp_tick_frequency()
                self.set_pixel_format(pixel_format='Mono8')
                self.auto_gain(state=False)
                # Set explicit gain — camera default after UserSetLoad is undefined
                self.gain(0.0)
                camera.ReverseX.SetValue(True)
                if not self._use_camera_emulation:
                    self.init_auto_gain_focus()
                self.exposure_t(exposure_ms=10)
                # 1900x1900: bench-witnessed driver-init default. Both
                # production cameras (a2A3536 ace 2 and daA3840 dart)
                # support this size; the 2100x2100 production-max ROI
                # lives at the UI input layer (data/scopes.json driven)
                # not at the driver default. Falsify: if a future
                # camera body's sensor doesn't reach 1900 in either
                # axis, the SetValue will clamp + warn from set_frame_size.
                self.set_frame_size(w=1900, h=1900)
        except genicam.RuntimeException as e:
            _cam_log.error(
                f'[CAM Class ] Camera communication error during init_camera_config: {e}'
            )
            self._mark_disconnected()
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error in init_camera_config: {e}')

    # Frame-identity chunk candidates in preference order. Different
    # Basler models advertise different per-frame identity chunks
    # (data-chunks.html). We probe ChunkSelector and enable the first
    # candidate the camera advertises; the read-side alias maps both
    # back to the same 'FrameID' dict key for the trace.
    _FRAME_IDENTITY_CHUNK_CANDIDATES = ('FrameID', 'Framecounter')

    # Always-enable chunks (no per-camera fallback; both production
    # USB3 cameras and dart M GigE advertise these). Timestamp is the
    # camera-side capture-time tick value; surfaced through the chunks
    # dict for downstream metadata writes (TIFF Plane fields, session
    # manifest). frame_validity does not validate against it -- it's
    # provenance, not state-equality.
    _CHUNK_TARGETS_ALWAYS = ('ExposureTime', 'Gain', 'Timestamp')

    def _enable_validity_chunks(self) -> None:
        """Enable ChunkExposureTime / ChunkGain + a per-frame identity
        chunk (ChunkFrameID or ChunkFramecounter) for chunk-driven
        validity.

        MUST be called while the camera is NOT grabbing (ChunkModeActive
        is locked while grabbing). Canonical caller is
        init_camera_config() inside update_camera_config(); B28 adds a
        runtime guard below so a future caller violating the contract
        gets a logged warning instead of a silent SDK lock error.

        Idempotent: safely re-asserts settings if chunks were already
        enabled. Per-chunk failures are logged but do not raise; the
        validity layer falls back to skip_frames for unenabled chunks.
        Frame-identity is trace-only (frame_validity validates gain
        and exposure); skipping it does not break validity.
        """
        camera = self.active
        if camera is None:
            return
        # B28: ChunkModeActive is locked while grabbing per Basler
        # data-chunks.html. The caller contract (docstring above)
        # requires "not grabbing", but a future caller could violate
        # it. Guard rather than silently corrupt -- frame_validity
        # falls back to skip_frames calibration when chunks aren't
        # enabled, so refusing the write is the safe default.
        if self.is_grabbing():
            _cam_log.warning(
                '[CAM Class ] _enable_validity_chunks called while '
                'grabbing; ChunkModeActive is locked. Skipping write. '
                'frame_validity will fall back to skip_frames '
                'calibration. Stop grabbing before enabling chunks.'
            )
            return
        advertised = self._probe_advertised_chunks(camera)
        if advertised is None:
            return  # _probe_advertised_chunks already logged
        try:
            camera.ChunkModeActive.Value = True
        except Exception as e:
            _cam_log.warning(
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
                _cam_log.warning(
                    f'[CAM Class ] Chunk{sel} not advertised by this '
                    f'camera; frame_validity will fall back to skip_frames '
                    f'for that source'
                )
                continue
            try:
                camera.ChunkSelector.Value = sel
                camera.ChunkEnable.Value = True
            except Exception as e:
                _cam_log.warning(
                    f'[CAM Class ] could not enable Chunk{sel}: {e}; '
                    f'frame_validity will fall back to skip_frames for that source'
                )

    # Basler USB3 ace 2 / dart M / dart R: documented fixed 1 GHz timestamp
    # tick rate (no GenAPI node). Basler GigE: GevTimestampTickFrequency or
    # TimestampTickFrequency feature. For unknown future cameras: fall back
    # to None so the metadata writer skips camera-tick fields rather than
    # writing wrong values (1 GHz default would be a silent lie if the
    # camera's actual rate differs).
    _BASLER_USB3_DEFAULT_TICK_HZ = 1_000_000_000

    def _read_timestamp_tick_frequency(self) -> None:
        """Resolve and cache the camera's Timestamp chunk tick frequency.

        Sets ``self.timestamp_tick_frequency_hz``. Tries the GigE node names
        first; falls back to the documented USB3 default (1 GHz) when no
        node is available; sets None if the device-info transport is
        unrecognised so downstream metadata is honest about the unknown.
        """
        camera = self.active
        if camera is None:
            return
        for node_name in ('GevTimestampTickFrequency', 'TimestampTickFrequency'):
            try:
                nm = camera.GetNodeMap()
                node = nm.GetNode(node_name)
                if node is not None and genicam.IsReadable(node):
                    self.timestamp_tick_frequency_hz = int(node.GetValue())
                    logger.info(
                        f'[CAM Class ] Timestamp tick frequency '
                        f'{self.timestamp_tick_frequency_hz} Hz ({node_name})'
                    )
                    return
            except Exception as e:
                logger.debug(
                    f'[CAM Class ] {node_name} read skipped: {e}'
                )
        # No GigE node -- assume Basler USB3 fixed rate per data-chunks doc.
        # Probe the device-info transport key to confirm we're actually on
        # USB3 before applying the default; if introspection fails, leave
        # tick frequency as None.
        try:
            di = camera.GetDeviceInfo()
            transport = di.GetDeviceClass() if hasattr(di, 'GetDeviceClass') else ''
        except Exception:
            transport = ''
        if 'Usb' in transport or 'USB' in transport:
            self.timestamp_tick_frequency_hz = self._BASLER_USB3_DEFAULT_TICK_HZ
            logger.info(
                f'[CAM Class ] Timestamp tick frequency assumed '
                f'{self._BASLER_USB3_DEFAULT_TICK_HZ} Hz (Basler USB3 default; '
                f'no TickFrequency node)'
            )
        else:
            _cam_log.warning(
                f'[CAM Class ] Could not determine Timestamp tick frequency '
                f'(transport={transport!r}, no TickFrequency node); '
                f'ChunkTimestamp values will be unconvertible to seconds'
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
                _cam_log.warning(
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
            _cam_log.warning(
                f'[CAM Class ] ChunkSelector introspection failed: {e}; '
                f'frame_validity will fall back to skip_frames calibration'
            )
            return None

    def set_max_acquisition_frame_rate(
        self,
        enabled: bool,
        fps: float = 1.0,
    ) -> None:
        """Enable or disable the camera's `AcquisitionFrameRateEnable` cap.

        When enabled, the camera will not produce frames faster than
        `fps` regardless of sensor-readout capability. Used by IDS char-
        tool crash protection (also relevant to Pylon for stability
        soaks). Setting `enabled=False` returns the camera to its
        sensor-readout-rate ceiling.

        Args:
            enabled: True to cap frame rate, False to remove the cap.
            fps: Target frame rate in fps when ``enabled=True``.
                Ignored when ``enabled=False``.
        """
        try:
            self.active.AcquisitionFrameRateEnable.Value = enabled
            if enabled:
                self.active.AcquisitionFrameRate.Value = fps
        except genicam.RuntimeException as e:
            _cam_log.error(
                f'[CAM Class ] Camera communication error in set_max_acquisition_frame_rate: {e}'
            )
            self._mark_disconnected()
        except Exception as e:
            _cam_log.exception(
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
            _cam_log.error(
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
            _cam_log.error(
                f'[CAM Class ] Camera communication error in '
                f'set_device_link_throughput_limit: {e}'
            )
            self._mark_disconnected()
            raise HardwareError(
                f'set_device_link_throughput_limit({mode}, {value_bps}) '
                f'failed: {e}'
            ) from e
        except Exception as e:
            _cam_log.exception(
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
            _cam_log.warning(
                f'[CAM Class ] DLTL value {value_bps} below camera minimum '
                f'{lo}; clamping. Per Basler doc, very low DLTL on rolling-'
                f'shutter cameras can introduce image distortion.'
            )
            return lo
        if value_bps > hi:
            _cam_log.warning(
                f'[CAM Class ] DLTL value {value_bps} above camera maximum '
                f'{hi}; clamping. Per Basler doc, very high DLTL can cause '
                f'corrupt or dropped frames.'
            )
            return hi
        return value_bps

    def set_max_transfer_size(self, value_bytes: int) -> bool:
        """Set StreamGrabber MaxTransferSize (USB3-only bench knob).

        Per Basler `stream-grabber-parameters.html`: bytes-per-USB-transfer
        the SDK requests. Doc's named lever for "fails to receive image
        stream"; decreasing works around kernel USB-transfer-size limits
        on some Windows hosts. SDK clamps to node range. Returns False if
        camera inactive; raises HardwareError on SDK failure.
        """
        return self._set_stream_grabber_int_node(
            node_name='MaxTransferSize',
            value=int(value_bytes),
            method_label='set_max_transfer_size',
        )

    def set_num_max_queued_urbs(self, value: int) -> bool:
        """Set StreamGrabber NumMaxQueuedUrbs (USB3-only bench knob).

        Per Basler `stream-grabber-parameters.html`: count of in-flight USB
        Request Blocks. Doc's named lever for "insufficient system memory"
        (status 0xe2010130 / 0xe2100001); decreasing reduces kernel URB
        allocation pressure on memory-constrained hosts. SDK clamps to node
        range. Returns False if camera inactive; raises HardwareError on
        SDK failure.
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
                _cam_log.error(
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
            _cam_log.error(
                f'[CAM Class ] Camera communication error in '
                f'{method_label}: {e}'
            )
            self._mark_disconnected()
            raise HardwareError(
                f'{method_label}({value}) failed: {e}'
            ) from e
        except Exception as e:
            _cam_log.exception(
                f'[CAM Class ] Unexpected error in {method_label}: {e}'
            )
            return False

    _ACQ_STOP_MODES = ('Complete', 'CancelExposure', 'AbortExposure')

    def set_acquisition_stop_mode(self, mode: str) -> bool:
        """Set BslAcquisitionStopMode -- StopGrabbing behavior during in-flight exposure.

        Per Basler `acquisition-start-stop-and-abort.html`:
          - ``'Complete'`` (default): waits for current exposure (up to N s on long FL).
          - ``'CancelExposure'``: stops cleanly, partial frame discarded.
          - ``'AbortExposure'``: aborts immediately, partial frame discarded.

        Supported on ace 2 / dart M/R / boost R. Default unchanged; setter
        exists for bench characterization. Returns False if camera inactive,
        mode invalid, or node absent. Raises HardwareError on SDK failure.
        """
        if not self.active:
            return False
        if mode not in self._ACQ_STOP_MODES:
            _cam_log.error(
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
                _cam_log.warning(
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
            _cam_log.error(
                f'[CAM Class ] Camera communication error in '
                f'set_acquisition_stop_mode: {e}'
            )
            self._mark_disconnected()
            raise HardwareError(
                f'set_acquisition_stop_mode({mode!r}) failed: {e}'
            ) from e
        except Exception as e:
            _cam_log.exception(
                f'[CAM Class ] Unexpected error in '
                f'set_acquisition_stop_mode: {e}'
            )
            return False

    _BANDWIDTH_RESERVE_MODES = ('Default', 'Performance')

    def set_bandwidth_reserve_mode(self, mode: str) -> bool:
        """Set BandwidthReserveMode (GigE only).

        Per Basler `network-related-parameters.md`:
          - ``'Default'``: reserves bandwidth for packet retransmits.
          - ``'Performance'``: all bandwidth to transmit; minimal retransmit.

        Load-bearing for fps on GigE; dmA3536-9gm goes 9.3 -> 9.5 fps with
        ``Performance``. USB3 cameras silently return False so the bench
        sweep can call unconditionally. Raises HardwareError on SDK failure.
        """
        if not self.active:
            return False
        if mode not in self._BANDWIDTH_RESERVE_MODES:
            _cam_log.error(
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
            _cam_log.error(
                f'[CAM Class ] Camera communication error in '
                f'set_bandwidth_reserve_mode: {e}'
            )
            self._mark_disconnected()
            raise HardwareError(
                f'set_bandwidth_reserve_mode({mode!r}) failed: {e}'
            ) from e
        except Exception as e:
            _cam_log.exception(
                f'[CAM Class ] Unexpected error in '
                f'set_bandwidth_reserve_mode: {e}'
            )
            return False

    def set_gev_packet_size(self, size_bytes: int) -> bool:
        """Set GevSCPSPacketSize (GigE only).

        Per Basler `network-related-parameters.md`: 1500 = standard MTU,
        9000 = jumbo. Larger packets cut host CPU + packet rate but require
        OS-level jumbo-frame config. USB3 cameras silently return False so
        the bench sweep can call unconditionally. Raises HardwareError on
        SDK failure.
        """
        if not self.active:
            return False
        if not isinstance(size_bytes, int) or size_bytes <= 0:
            _cam_log.error(
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
            _cam_log.error(
                f'[CAM Class ] Camera communication error in '
                f'set_gev_packet_size: {e}'
            )
            self._mark_disconnected()
            raise HardwareError(
                f'set_gev_packet_size({size_bytes}) failed: {e}'
            ) from e
        except Exception as e:
            _cam_log.exception(
                f'[CAM Class ] Unexpected error in set_gev_packet_size: {e}'
            )
            return False

    def set_gev_inter_packet_delay(self, delay_ticks: int) -> bool:
        """Set GevSCPD inter-packet delay in clock ticks (GigE only).

        Per Basler `network-related-parameters.md`: throttles per-camera
        throughput so multiple cameras can share a GigE link. 0 = no delay.
        USB3 cameras silently return False so the bench sweep can call
        unconditionally. Raises HardwareError on SDK failure.
        """
        if not self.active:
            return False
        if not isinstance(delay_ticks, int) or delay_ticks < 0:
            _cam_log.error(
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
            _cam_log.error(
                f'[CAM Class ] Camera communication error in '
                f'set_gev_inter_packet_delay: {e}'
            )
            self._mark_disconnected()
            raise HardwareError(
                f'set_gev_inter_packet_delay({delay_ticks}) failed: {e}'
            ) from e
        except Exception as e:
            _cam_log.exception(
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
            _cam_log.error(f'[CAM Class ] Unsupported pixel format: {pixel_format}')
            return False

        # Short-circuit redundant SetValue: update_camera_config() bounces
        # the grab loop. init_camera_config + channel-select hit this
        # setter multiple times with the same target across threads.
        try:
            if self.active.PixelFormat.GetValue() == pixel_format:
                if _cam_log is not None:
                    _cam_log.info(
                        f'pylon PixelFormat.SetValue({pixel_format!r}) '
                        'short-circuited (already active)'
                    )
                return True
        except (genicam.RuntimeException, genicam.TimeoutException) as e:
            logger.debug(
                f'[CAM Class ] PixelFormat short-circuit read failed; '
                f'falling through to SetValue path: {e}'
            )

        try:
            if _cam_log is not None:
                _cam_log.info(f'pylon PixelFormat.SetValue({pixel_format!r}) (geometry-realloc)')
            with self.update_camera_config():
                self.active.PixelFormat.SetValue(pixel_format)
            return True
        except genicam.RuntimeException as e:
            if _cam_log is not None:
                _cam_log.error(f'pylon PixelFormat.SetValue({pixel_format!r}) FAILED: {e}')
            _cam_log.error(
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
            _cam_log.exception(f'[CAM Class ] Unexpected error in set_pixel_format: {e}')
            raise HardwareError(
                f'set_pixel_format({pixel_format}) failed: {type(e).__name__}: {e}'
            ) from e

    def get_pixel_format(self) -> str:
        """Return active PixelFormat (e.g. 'Mono8'); '' on inactive / read failure."""
        if not self.active:
            return ''

        try:
            return self.active.PixelFormat.GetValue()
        except genicam.RuntimeException as e:
            _cam_log.error(
                f'[CAM Class ] Failed to read pixel format: Camera may be disconnected - {e}'
            )
            self._mark_disconnected()
            return ''
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error reading pixel format: {e}')
            return ''

    def get_supported_pixel_formats(self) -> tuple:
        """Return every PixelFormat the camera advertises; () on inactive / read failure."""
        try:
            return self.active.PixelFormat.GetSymbolics()
        except genicam.RuntimeException as e:
            _cam_log.error(f'[CAM Class ] Failed to read pixel formats: {e}')
            self._mark_disconnected()
            return ()
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error reading pixel formats: {e}')
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
            _cam_log.error(f'[CAM Class ] Unsupported bin size: {size}')
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
            _cam_log.warning(f'[CAM Class ] set_binning_size({size}) timed out: {e}')
            raise HardwareError(
                f'set_binning_size({size}) timed out: {e}'
            ) from e
        except genicam.RuntimeException as e:
            _cam_log.error(
                f'[CAM Class ] Camera communication error during set_binning_size({size}): {e}'
            )
            self._mark_disconnected()
            raise HardwareError(
                f'set_binning_size({size}) failed: {type(e).__name__}: {e}'
            ) from e
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error in set_binning_size: {e}')
            raise HardwareError(
                f'set_binning_size({size}) failed: {type(e).__name__}: {e}'
            ) from e

    def get_binning_size(self) -> int:
        """Return current binning factor; 1 on inactive / read failure.

        Vertical wins on asymmetric mismatch (operator misconfig).
        """
        if not self.active:
            return 1

        try:
            vert_bin = self.active.BinningVertical.GetValue()
            horiz_bin = self.active.BinningHorizontal.GetValue()

            if horiz_bin != vert_bin:
                _cam_log.warning(
                    '[CAM Class ] Binning mismatch detected between '
                    f'horizontal ({horiz_bin}) and vertical ({vert_bin})'
                )

            return vert_bin
        except genicam.RuntimeException as e:
            _cam_log.error(
                f'[CAM Class ] Failed to read binning size: Camera may be disconnected - {e}'
            )
            self._mark_disconnected()
            return 1
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error reading binning size: {e}')
            return 1

    def init_auto_gain_focus(
        self,
        auto_target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None,
    ) -> None:
        """Configure the AutoFunctionROI + auto-gain limits for AF use.

        Sets ROI to the full sensor minus the existing offset, picks
        the `MinimizeExposureTime` profile (autofocus prefers shorter
        exposures), and applies caller-supplied gain bounds (or the
        camera's reported min/max when `None`).

        Args:
            auto_target_brightness: Target brightness in 0..1, fed to
                `AutoTargetBrightness`.
            min_gain: Lower bound for the auto-gain controller in dB,
                or ``None`` to use the camera's reported minimum.
            max_gain: Upper bound in dB, or ``None`` to use the
                camera's reported maximum.
        """
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
            _cam_log.error(
                f'[CAM Class ] Camera communication error during init_auto_gain_focus: {e}'
            )
            self._mark_disconnected()
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error in init_auto_gain_focus: {e}')

    def update_auto_gain_target_brightness(
        self,
        auto_target_brightness: float,
    ) -> None:
        """Update `AutoTargetBrightness` without an over-stop cycle.

        Live-writable per Basler -- no `update_camera_config()` wrap
        needed. The previous wrap forced a stop/start cycle on every
        call (the same anti-pattern as STALL-1).

        Args:
            auto_target_brightness: Target brightness in 0..1.
        """
        # Basler runtime-modifiable parameter -- AutoTargetBrightness can
        # be changed while StartGrabbing is active. The previous wrap
        # in update_camera_config() forced a stop_grabbing /
        # start_grabbing cycle on every call -- a needless over-stop
        # of the same structural class as wrapping any other
        # live-writable parameter.
        # Short-circuit when already at target. Matches the e042c7f pattern
        # on gain / exposure_t -- avoids redundant SetValue serialization
        # against the grab thread during LED-toggle storms.
        try:
            current = float(self.active.AutoTargetBrightness.GetValue())
            if abs(current - float(auto_target_brightness)) < 1e-3:
                if _cam_log is not None:
                    _cam_log.info(
                        f'pylon AutoTargetBrightness.SetValue'
                        f'({auto_target_brightness:.3f}) short-circuited'
                    )
                return
        except (genicam.RuntimeException, genicam.TimeoutException) as e:
            logger.debug(
                f'[CAM Class ] AutoTargetBrightness short-circuit read failed; '
                f'falling through to SetValue path: {e}'
            )

        try:
            if _cam_log is not None:
                _cam_log.info(f'pylon AutoTargetBrightness.SetValue({auto_target_brightness:.3f})')
            self.active.AutoTargetBrightness.SetValue(auto_target_brightness)
        except genicam.RuntimeException as e:
            if _cam_log is not None:
                _cam_log.error(
                    f'pylon AutoTargetBrightness.SetValue({auto_target_brightness}) FAILED: {e}'
                )
            _cam_log.error(
                '[CAM Class ] Camera communication error during '
                f'update_auto_gain_target_brightness({auto_target_brightness}): {e}'
            )
            self._mark_disconnected()
        except Exception as e:
            if _cam_log is not None:
                _cam_log.error(
                    f'pylon AutoTargetBrightness.SetValue({auto_target_brightness}) FAILED: {e}'
                )
            _cam_log.exception(
                f'[CAM Class ] Unexpected error in update_auto_gain_target_brightness: {e}'
            )

    def update_auto_gain_min_max(
        self,
        min_gain_db: float | None,
        max_gain_db: float | None,
    ) -> None:
        """Update auto-gain min/max bounds without an over-stop cycle.

        Live-writable per Basler. ``None`` for either bound is treated
        as "leave that side at its current value"; only the explicit
        ones are written.

        Args:
            min_gain_db: Lower bound in dB, or ``None`` to leave unchanged.
            max_gain_db: Upper bound in dB, or ``None`` to leave unchanged.
        """
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
            if min_gain_db is None:
                min_gain_db = self.active.AutoGainLowerLimit.Min

            if max_gain_db is None:
                max_gain_db = self.active.AutoGainUpperLimit.Max

            # Short-circuit when both bounds already at target. Matches
            # the e042c7f pattern on gain / exposure_t.
            try:
                cur_min = float(self.active.AutoGainLowerLimit.GetValue())
                cur_max = float(self.active.AutoGainUpperLimit.GetValue())
                if (abs(cur_min - float(min_gain_db)) < 1e-3
                        and abs(cur_max - float(max_gain_db)) < 1e-3):
                    if _cam_log is not None:
                        _cam_log.info(
                            f'pylon AutoGainLowerLimit/UpperLimit.SetValue'
                            f'({min_gain_db}, {max_gain_db}) short-circuited'
                        )
                    return
            except (genicam.RuntimeException, genicam.TimeoutException) as e:
                logger.debug(
                    f'[CAM Class ] AutoGain min/max short-circuit read failed; '
                    f'falling through to SetValue path: {e}'
                )

            if _cam_log is not None:
                _cam_log.info(
                    f'pylon AutoGainLowerLimit.SetValue({min_gain_db}) '
                    f'AutoGainUpperLimit.SetValue({max_gain_db})'
                )
            self.active.AutoGainLowerLimit.SetValue(min_gain_db)
            self.active.AutoGainUpperLimit.SetValue(max_gain_db)
        except genicam.RuntimeException as e:
            _cam_log.error(
                '[CAM Class ] Camera communication error during '
                f'update_auto_gain_min_max(min_db={min_gain_db}, max_db={max_gain_db}): {e}'
            )
            self._mark_disconnected()
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error in update_auto_gain_min_max: {e}')

    # grab() inherited from Camera base class

    def grab_new_capture(self, timeout_s: float) -> tuple:
        """Drain queued frames, then block for a genuinely new one.

        Drops every frame already queued in the image handler, then
        blocks up to ``timeout_s`` seconds waiting for the next callback.
        Used by AF / characterization paths that need the freshest
        possible frame -- previously dropped only one queued frame,
        which could return a stale frame when the consumer had fallen
        behind.

        Args:
            timeout_s: Wall-clock seconds to wait for a new frame after
                draining queued frames.

        Returns:
            tuple: ``(success: bool, timestamp: float | None)``.
                ``success=False`` if the camera is inactive, the handler
                is missing, or no frame arrived within ``timeout_s``.
                ``timestamp`` is the host-side capture timestamp on
                success, ``None`` otherwise.
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
                # Drain all frames captured before this call -- we only want
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
                    block=True, timeout=timeout_s
                )
                if result is False:
                    _outcome = 'result_false'
                    return False, None

                self.array = image
                _outcome = 'success'
                return True, image_ts

            except queue.Empty:
                # Expected outcome when no frame arrives within `timeout_s`.
                # WARNING (not ERROR), no traceback -- the wait simply
                # expired, no SDK fault to attribute.
                _outcome = 'timeout'
                _cam_log.warning(
                    f'[CAM Class ] grab_new_capture timed out after '
                    f'{timeout_s:.1f}s (no frame queued; dropped {dropped} stale)'
                )
                return False, None
            except Exception as ex:
                _outcome = 'exception'
                _cam_log.exception(
                    f'[CAM Class ] grab_new_capture raised '
                    f'{type(ex).__name__}: {ex}'
                )
                return False, None
        finally:
            if _trace_enabled and _t0 is not None:
                _dt_ms = (time.perf_counter() - _t0) * 1000.0
                profile_trace.trace(
                    'pylon_grab_trace.csv',
                    'ts_ms,duration_ms,dropped_count,outcome,timeout_s',
                    [
                        int(time.time() * 1000),
                        f'{_dt_ms:.3f}',
                        dropped,
                        _outcome,
                        f'{timeout_s:.3f}',
                    ],
                )

    def set_frame_size(self, w, h) -> None:
        """Set camera frame size to ``w`` x ``h`` and recenter the ROI.

        Width and height are clamped to the camera's reported maxima
        and rounded down to the nearest multiple of 4 (Pylon
        constraint on most current models). The
        ``BslCenterX`` / ``BslCenterY`` execute calls keep the ROI
        centered on the sensor after the size change. Wrapped in
        ``update_camera_config()`` because Width/Height require a
        buffer realloc.

        Args:
            w: Requested frame width in pixels.
            h: Requested frame height in pixels.
        """
        camera = self.active
        if camera is None:
            _cam_log.warning(f'[CAM Class ] Cannot set frame size {w}x{h}: camera inactive')
            return

        try:
            width = int(min(int(w), camera.Width.Max) / 4) * 4
            height = int(min(int(h), camera.Height.Max) / 4) * 4

            # Short-circuit when geometry already matches: Width/Height SetValue
            # requires update_camera_config() buffer realloc + grab-loop bounce.
            # init + bring-up call this multiple times with the same clamped
            # dims. BslCenterX/Y.Execute is also skipped because we don't write
            # OffsetX/OffsetY directly elsewhere, so the ROI stays centered.
            try:
                if (camera.Width.GetValue() == width
                        and camera.Height.GetValue() == height):
                    if _cam_log is not None:
                        _cam_log.info(
                            f'pylon Width.SetValue({width}) Height.SetValue({height}) '
                            'short-circuited'
                        )
                    _log_cam('info', f'[CAM Class ] Frame size already at {width}x{height}')
                    return
            except (genicam.RuntimeException, genicam.TimeoutException) as e:
                logger.debug(
                    f'[CAM Class ] Frame-size short-circuit read failed; '
                    f'falling through to SetValue path: {e}'
                )

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

            _log_cam('info', f'[CAM Class ] Frame size set to {width}x{height}')
        except genicam.RuntimeException as e:
            _cam_log.error(
                f'[CAM Class ] Camera communication error during set_frame_size({w}x{h}): {e}'
            )
            self._mark_disconnected()
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error in set_frame_size: {e}')

    def get_min_frame_size(self) -> dict:
        """Return min frame dims as {'width': int, 'height': int}; {} on inactive / read failure."""
        camera = self.active
        if camera is None:
            return {}
        try:
            return {
                'width': camera.Width.GetMin(),
                'height': camera.Height.GetMin(),
            }
        except genicam.RuntimeException as e:
            _cam_log.error(f'[CAM Class ] Failed to read min frame size: {e}')
            self._mark_disconnected()
            return {}
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error reading min frame size: {e}')
            return {}

    def get_max_frame_size(self) -> dict:
        """Return sensor-driven max frame dims; {} on inactive / read failure.

        Lens-driven ceiling (typically tighter) lives at the API layer in
        `data/scopes.json` ``max_usable_roi``.
        """
        camera = self.active
        if camera is None:
            return {}
        try:
            return {
                'width': camera.Width.GetMax(),
                'height': camera.Height.GetMax(),
            }
        except genicam.RuntimeException as e:
            _cam_log.error(f'[CAM Class ] Failed to read max frame size: {e}')
            self._mark_disconnected()
            return {}
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error reading max frame size: {e}')
            return {}

    def get_frame_size(self) -> dict | None:
        """Return active dims as {'width': int, 'height': int}; None on inactive / read failure."""
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
            _cam_log.error(
                f'[CAM Class ] Failed to read frame size: Camera may be disconnected - {e}'
            )
            self._mark_disconnected()
            return None
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error reading frame size: {e}')
            return None

    def get_gain(self) -> float:
        """Return current Gain (dB); -1.0 on any error so callers can treat < 0 as failure."""
        if self.active is None:
            _cam_log.warning('[CAM Class ] Cannot read gain: camera inactive')
            return -1

        try:
            return float(self.active.Gain.GetValue())
        except genicam.TimeoutException as e:
            # USB roundtrip timed out (transient). Don't mark disconnected; caller can retry.
            _cam_log.warning(f'[CAM Class ] get_gain timed out: {e}')
            return -1
        except genicam.RuntimeException as e:
            _cam_log.error(
                f'[CAM Class ] Failed to read gain value: Camera may be disconnected - {e}'
            )
            self._mark_disconnected()
            return -1
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error reading gain: {e}')
            return -1

    def is_connected(self) -> bool:
        """Return True if connected; uses removal flag + SDK device-removed query."""
        if self._device_removed:
            self._mark_disconnected()
            return False
        if self.active is None:
            self._mark_disconnected()
            return False
        # Cheap SDK-side query covers cases where _CameraRemovalHandler missed the event.
        try:
            if self.active.IsCameraDeviceRemoved():
                self._mark_disconnected()
                return False
        except Exception as e:
            logger.debug(f'[CAM Class ] IsCameraDeviceRemoved query raised: {e}')
        return True

    def gain(self, value) -> None:
        """Set Gain in dB. Asserts GainSelector='All' first (Basler gain.html three-step).

        Caller is responsible for ``GainAuto=Off``. GainSelector write
        failures are tolerated (cameras without the selector).
        """
        if self.active is None:
            if _cam_log is not None:
                _cam_log.warning(f'pylon Gain.SetValue({value}) SKIPPED: active=None')
            _cam_log.warning(f'[CAM Class ] Cannot set gain {value}: camera inactive')
            return

        try:
            try:
                self.active.GainSelector.SetValue('All')
            except Exception as e_sel:
                logger.debug(
                    f'[CAM Class ] GainSelector.SetValue(All) skipped: {e_sel}'
                )
            # Short-circuit when already at target. Selector is 'All' (asserted
            # above) so the read matches the requested write. Tolerance 1e-3 dB
            # is below GenICam Gain increment on ace 2 / dart.
            try:
                if abs(float(self.active.Gain.GetValue()) - float(value)) < 1e-3:
                    if _cam_log is not None:
                        _cam_log.info(
                            f'pylon Gain.SetValue({float(value):.3f}) short-circuited'
                        )
                    _log_cam('info', f'[CAM Class ] Gain already at {value}')
                    return
            except (genicam.RuntimeException, genicam.TimeoutException) as e:
                logger.debug(
                    f'[CAM Class ] Gain short-circuit read failed; '
                    f'falling through to SetValue path: {e}'
                )
            if _cam_log is not None:
                _cam_log.info(f'pylon Gain.SetValue({float(value):.3f})')
            self.active.Gain.SetValue(float(value))
            _log_cam('info', f'[CAM Class ] Gain set to {value}')
        except genicam.RuntimeException as e:
            if _cam_log is not None:
                _cam_log.error(f'pylon Gain.SetValue({value}) FAILED: {e}')
            _cam_log.error(f'[CAM Class ] Camera communication error during gain({value}): {e}')
            self._mark_disconnected()
        except Exception as e:
            if _cam_log is not None:
                _cam_log.error(f'pylon Gain.SetValue({value}) FAILED: {e}')
            _cam_log.exception(f'[CAM Class ] Unexpected error in gain: {e}')

    def auto_gain(
        self,
        state=True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
    ) -> None:
        """Enable or disable continuous auto-gain + auto-exposure.

        When enabled, ``GainAuto`` and ``ExposureAuto`` are set to
        ``Continuous`` -- the camera continuously adjusts based on the
        live image's brightness. When disabled, both are set to
        ``Off``. Caller-supplied ``target_brightness`` / ``min_gain_db``
        / ``max_gain_db`` are applied via ``update_auto_gain_*`` helpers
        before enabling.

        Args:
            state: ``True`` to enable Continuous mode, ``False`` to
                disable.
            target_brightness: Target brightness in 0..1.
            min_gain_db: Lower bound in dB, or ``None`` to leave
                unchanged.
            max_gain_db: Upper bound in dB, or ``None`` to leave
                unchanged.
        """

        if self.active is None:
            _cam_log.warning(f'[CAM Class ] Cannot set auto_gain({state}): camera inactive')
            return

        try:
            if _cam_log is not None:
                _cam_log.info(
                    f'pylon auto_gain(state={state}, target={target_brightness}, '
                    f'min_db={min_gain_db}, max_db={max_gain_db})'
                )
            if state:
                self.update_auto_gain_target_brightness(auto_target_brightness=target_brightness)
                self.update_auto_gain_min_max(min_gain_db=min_gain_db, max_gain_db=max_gain_db)
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
            _log_cam('info', f'[CAM Class ] Auto gain {"enabled" if state else "disabled"}')
        except genicam.RuntimeException as e:
            _cam_log.error(f'[CAM Class ] Auto gain({state}) failed: {e}')
            self._mark_disconnected()
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error in auto_gain: {e}')

    def auto_gain_once(
        self,
        state=True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
    ) -> None:
        """Run a single-shot auto-gain + auto-exposure pass.

        ``GainAuto`` and ``ExposureAuto`` are set to ``Once`` -- the
        camera adjusts once and then transitions back to ``Off`` on
        its own. When ``state=False``, both are set to ``Off``
        explicitly.

        Args:
            state: ``True`` to fire a single Once-mode adjustment,
                ``False`` to disable.
            target_brightness: Target brightness in 0..1.
            min_gain_db: Lower bound in dB, or ``None`` to leave
                unchanged.
            max_gain_db: Upper bound in dB, or ``None`` to leave
                unchanged.
        """

        if self.active is None:
            _cam_log.warning(f'[CAM Class ] Cannot set auto_gain_once({state}): camera inactive')
            return

        try:
            if state:
                self.update_auto_gain_target_brightness(auto_target_brightness=target_brightness)
                self.update_auto_gain_min_max(min_gain_db=min_gain_db, max_gain_db=max_gain_db)
                self.active.GainAuto.SetValue('Once')  # 'Off' 'Once' 'Continuous'
                self.active.ExposureAuto.SetValue('Once')  # 'Off' 'Once' 'Continuous'
            else:
                self.active.GainAuto.SetValue('Off')
                self.active.ExposureAuto.SetValue('Off')
            _log_cam('info', f'[CAM Class ] Auto gain once {"enabled" if state else "disabled"}')
        except genicam.RuntimeException as e:
            _cam_log.error(f'[CAM Class ] Auto gain once({state}) failed: {e}')
            self._mark_disconnected()
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error in auto_gain_once: {e}')

    def exposure_t(self, exposure_ms) -> None:
        """Set the camera's exposure time in milliseconds.

        Pylon's ``ExposureTime`` node uses microseconds; this method
        accepts milliseconds and converts. Values exceeding
        ``self.max_exposure`` are rejected with a warning. Sub-minimum
        values are clamped to ``ExposureTime.Min``.

        Args:
            exposure_ms: Exposure time in milliseconds.
        """
        if self.active is None:
            if _cam_log is not None:
                _cam_log.warning(
                    f'pylon ExposureTime.SetValue({exposure_ms}ms) SKIPPED: active=None'
                )
            _cam_log.warning(
                f'[CAM Class ] Cannot set exposure {exposure_ms}ms: camera inactive'
            )
            return

        if exposure_ms > self.max_exposure:
            if _cam_log is not None:
                _cam_log.warning(
                    f'pylon ExposureTime.SetValue({exposure_ms}ms) SKIPPED: '
                    f'exceeds max {self.max_exposure}ms'
                )
            _cam_log.warning(
                f'[CAM Class ] Exposure {exposure_ms}ms exceeds max '
                f'({self.max_exposure}ms)'
            )
            return

        # Pylon takes time in microseconds, so multiply by 1000 to convert
        try:
            us_value = max(float(exposure_ms) * 1000, self.active.ExposureTime.Min)
            # Short-circuit when already at target us. SDK rounds to its clock
            # grid so same target ms maps to same us count; 1 us tolerance is
            # below ExposureTime increment on ace 2 / dart.
            try:
                if abs(float(self.active.ExposureTime.GetValue()) - us_value) < 1.0:
                    if _cam_log is not None:
                        _cam_log.info(
                            f'pylon ExposureTime.SetValue({us_value:.0f}us) short-circuited'
                        )
                    _log_cam('info', f'[CAM Class ] Exposure already at {exposure_ms}ms')
                    return
            except (genicam.RuntimeException, genicam.TimeoutException) as e:
                logger.debug(
                    f'[CAM Class ] ExposureTime short-circuit read failed; '
                    f'falling through to SetValue path: {e}'
                )
            if _cam_log is not None:
                _cam_log.info(f'pylon ExposureTime.SetValue({us_value:.0f}us) (={exposure_ms}ms)')
            self.active.ExposureTime.SetValue(us_value)
            _log_cam('info', f'[CAM Class ] Exposure set to {exposure_ms}ms')
        except genicam.RuntimeException as e:
            if _cam_log is not None:
                _cam_log.error(f'pylon ExposureTime.SetValue({exposure_ms}ms) FAILED: {e}')
            _cam_log.error(
                f'[CAM Class ] Camera communication error during '
                f'exposure_t({exposure_ms}ms): {e}'
            )
            self._mark_disconnected()
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error in exposure_t: {e}')

    def get_exposure_t(self) -> float:
        """Read the camera's currently-active exposure time in ms.

        Prefers ``BslEffectiveExposureTime`` (the value the camera
        actually used, accounting for internal clock-rate rounding)
        when the node is exposed; falls back to ``ExposureTime`` (the
        set value) on legacy ace cameras that do not expose the
        Bsl-prefixed node.

        Returns:
            float: Exposure time in milliseconds. Returns ``-1.0`` on
                any error path (inactive camera, both nodes
                unreadable) so callers can treat ``< 0`` as "read
                failed" without needing a try/except.
        """

        if self.active is None:
            _cam_log.warning('[CAM Class ] Cannot read exposure: camera inactive')
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
                _cam_log.error(
                    '[CAM Class ] Failed to read exposure time: both '
                    'BslEffectiveExposureTime and ExposureTime nodes '
                    'unavailable. Camera may be disconnected.'
                )
                self._mark_disconnected()
                return -1
            return microsec / 1000  # microseconds -> milliseconds
        except genicam.TimeoutException as e:
            # USB roundtrip timed out (transient). Don't mark disconnected; caller can retry.
            _cam_log.warning(f'[CAM Class ] get_exposure_t timed out: {e}')
            return -1
        except genicam.RuntimeException as e:
            _cam_log.error(
                f'[CAM Class ] Failed to read exposure time: Camera may be disconnected - {e}'
            )
            self._mark_disconnected()
            return -1
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error reading exposure time: {e}')
            return -1

    def auto_exposure_t(self, state=True) -> None:
        """Enable or disable continuous auto-exposure.

        When ``state=True``, ``ExposureAuto`` is set to ``Continuous``;
        the camera continuously adjusts exposure based on the live
        image. When ``state=False``, ``ExposureAuto`` is set to
        ``Off``.

        Args:
            state: ``True`` to enable Continuous mode, ``False`` to
                disable.
        """

        if self.active is None:
            _cam_log.warning(f'[CAM Class ] Cannot set auto_exposure({state}): camera inactive')
            return

        try:
            if state:
                self.active.ExposureAuto.SetValue('Continuous')  # 'Off' 'Once' 'Continuous'
            else:
                self.active.ExposureAuto.SetValue('Off')
            _log_cam('info', f'[CAM Class ] Auto exposure {"enabled" if state else "disabled"}')
        except genicam.RuntimeException as e:
            _cam_log.error(f'[CAM Class ] Auto exposure({state}) failed: {e}')
            self._mark_disconnected()
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error in auto_exposure_t: {e}')

    def set_test_pattern(
        self,
        enabled: bool = False,
        pattern: str = 'Black',
    ) -> None:
        """Apply a Pylon ``TestPattern`` and immediately grab one frame.

        Per-camera supported patterns include ``'Black'``, ``'White'``,
        ``'GreyHorizontalRamp'``, ``'GreyVerticalRamp'``,
        ``'ColorDiagonalSawtooth8'``, etc. (per Basler
        ``test-images.html``). ``enabled`` is currently a no-op marker
        kept for API compatibility -- writes are unconditional when
        called.

        Args:
            enabled: Reserved for future use.
            pattern: Pylon symbolic test-pattern name.
        """
        if self.active is None:
            return

        try:
            self.active.TestPattern.SetValue(pattern)
            self.grab()
        except genicam.RuntimeException as e:
            _cam_log.error(
                f'[CAM Class ] Camera communication error during set_test_pattern({pattern}): {e}'
            )
            self._mark_disconnected()
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Unexpected error in set_test_pattern: {e}')

    # Chunks consumed by chunk-driven validity in modules.frame_validity.
    # LED has no chunk equivalent; motion is firmware-gated; these three
    # cover gain / exposure / per-frame identity.
    _CHUNK_TARGETS_FOR_VALIDITY = ('ExposureTime', 'Gain', 'FrameID')

    def probe_chunk_capabilities(self) -> dict:
        """Probe per-frame chunk support via activation-acceptance.

        For each ChunkSelector entry, sets ChunkEnable=True and reads back
        to confirm support. ChunkModeActive is locked while grabbing, so
        the probe stops grabbing if needed and restores prior config.

        Returns dict with keys: 'model', 'firmware', 'serial', 'advertised'
        (sorted list of selector symbols), 'enabled' (selector -> bool),
        'errors' (list).
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
                    _cam_log.warning(
                        f'[CAM Class ] chunk_data_probe could not restore '
                        f'Chunk{sel} prior state ({prior}): {e}'
                    )
            if prior_chunk_mode is not None:
                try:
                    camera.ChunkModeActive.Value = prior_chunk_mode
                except Exception as e:
                    _cam_log.warning(
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
            try:
                node = getattr(camera, name, None)
            except Exception:
                # pypylon's InstantCamera.__getattr__ raises
                # genicam.LogicalErrorException for missing nodes instead
                # of letting Python's default-arg fallback fire. Treat as
                # "not present" and try the next name. Bench-verified
                # 2026-05-08: 1540 tracebacks per protocol run before fix.
                continue
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


# Maps GrabResult chunk attrs to keys in FrameValidity.CHUNK_KEY_FOR_SOURCE
# plus the Timestamp provenance chunk (not validated; surfaced for metadata).
# ChunkFrameID + ChunkFramecounter both map to 'FrameID' (camera advertises one;
# read side tries both, the active one returns a value).
#
# Module-level so both the SDK callback (ImageHandler.OnImageGrabbed) and
# the Stage B worker (_PylonImageGrabWorker._process_frame) can call into
# the same code. Defining it on the ImageHandler class would put it
# behind a MagicMock-replaced class shim under conftest's mocked pypylon
# (Python's class statement with a MagicMock base discards class-body
# attributes), making the worker unreachable from unit tests.
_CHUNK_GRAB_RESULT_ATTRS = (
    ('ChunkExposureTime', 'ExposureTime'),
    ('ChunkGain', 'Gain'),
    ('ChunkFrameID', 'FrameID'),
    ('ChunkFramecounter', 'FrameID'),
    ('ChunkTimestamp', 'Timestamp'),
)


def _read_validity_chunks(grabResult) -> dict | None:
    """Extract validity chunks from a successful GrabResult.

    Returns dict like {'ExposureTime': float, 'Gain': float, 'FrameID': int},
    or None if no chunks readable (chunks unsupported or not enabled).
    """
    chunks: dict = {}
    for chunk_attr, key in _CHUNK_GRAB_RESULT_ATTRS:
        try:
            node = getattr(grabResult, chunk_attr, None)
            if node is None:
                continue
            if genicam.IsReadable(node):
                chunks[key] = node.Value
        except Exception as e:
            logger.debug(
                f'[CAM Class ] _read_validity_chunks could not read {chunk_attr}: {e}'
            )
    return chunks if chunks else None


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
        # Stage B worker for the OnImageGrabbed two-stage split. Created
        # here so it shares lifetime with the handler; start() / stop()
        # are driven from PylonCamera.connect / disconnect at the ordering
        # the SDK contract requires.
        self._worker = _PylonImageGrabWorker(
            parent_cam, self._base, self._frame_queue
        )

    def OnImagesSkipped(self, camera, countOfSkippedImages) -> None:
        """Pylon SDK callback fired when the grab strategy drops frames.

        Per `class_pylon_1_1_c_image_event_handler.html`: fires when
        GrabStrategy_LatestImageOnly (or LatestImages) discards an older
        frame in favor of a newer one. The SDK contract: "Exceptions
        from this call will propagate through" -- same hazard as
        OnImageGrabbed, so wrap accordingly.

        Logged at info so the skip distribution is visible in camera.log
        post-R12 (correlate with worker-queue-full count if applicable).
        Today (pre-R12) skips fire when the consumer reads
        ImageHandlerBase.last_img slower than the SDK grabs; cause
        distribution stays in the log without raising the noise floor.
        """
        try:
            if countOfSkippedImages > 0:
                _cam_log.info(
                    f'[CAM Class ] OnImagesSkipped: '
                    f'{countOfSkippedImages} frame(s) discarded by SDK '
                    f'(grab strategy = LatestImageOnly)'
                )
        except Exception as e:
            _log_safely(f'OnImagesSkipped logging raised: {e}')
        except BaseException as e:
            _log_safely(
                f'OnImagesSkipped BaseException guard '
                f'caught {type(e).__name__}: {e}'
            )

    def OnImageGrabbed(self, camera, grabResult) -> None:
        """Stage A native-thread fast-path: state checks + enqueue to worker.

        Heavy work (16 MB GetArray copy, chunk reads, _store_frame,
        per-frame callback fanout, legacy frame_queue publish, generic
        failure classification + counter) runs on Stage B's
        _PylonImageGrabWorker. The native-thread exposure surface here
        targets ~100us p99; the BaseException outer guard catches
        anything escaping into Pylon's grab thread per
        `class_pylon_1_1_c_image_event_handler.html` ("exceptions will
        propagate through").

        The err_code=DEVICE_NOT_FOUND (433) disconnect fast-path stays
        INLINE here -- the user-facing disconnect notification can't
        wait behind Stage B's queue (cascade rate observed at 100+
        events in under 2 seconds during USB unplug).

        Single-thread invariant on CGrabResultData: Stage A is the SOLE
        reader of GrabSucceeded() (and GetErrorCode on failure) before
        the handoff; Stage B is the SOLE reader of GetArray + chunk
        node-map accessors after. The put_nowait / get pair provides
        happens-before without explicit locking.
        """
        _trace_enabled = profile_trace is not None and profile_trace.ENABLE_PROFILE_TRACE
        _t0 = time.perf_counter() if _trace_enabled else None
        _outcome = 'unknown'
        try:
            # Rename Pylon's Dummy-N worker thread to a stable label so
            # _stats_poller_loop can count grab callbacks by name. If a
            # future pypylon pre-names threads, this branch no-ops AND
            # _stats_poller_loop's grab count under-reports.
            if 'Dummy' in threading.current_thread().name:
                threading.current_thread().name = 'PylonImageGrab'

            if self._parent._device_removed:
                logger.debug(
                    '[CAM Class ] OnImageGrabbed called but device '
                    'already marked as removed, ignoring'
                )
                _outcome = 'early_return_removed'
                return

            if self._parent.active is None:
                logger.debug(
                    '[CAM Class ] OnImageGrabbed called but camera is inactive, ignoring'
                )
                self._parent._mark_disconnected()
                _outcome = 'early_return_inactive'
                return

            # GrabSucceeded() can leak native exceptions in cancel /
            # teardown paths; trapping here classifies as disconnect
            # rather than letting the exception escape into Pylon's grab
            # thread.
            try:
                grab_succeeded = grabResult.GrabSucceeded()
            except Exception as e:
                _cam_log.warning(
                    f'[CAM Class ] GrabSucceeded() failed: {e}, '
                    f'assuming device removed'
                )
                self._parent._mark_disconnected()
                _outcome = 'exception_grabsucceeded'
                return

            ts = datetime.datetime.now()

            # H2/H3 diagnostic: Stage A is otherwise silent on the success
            # enqueue path. Log the GrabSucceeded value + grabResult identity
            # for the first 5 frames so a "GetArray() failed on worker"
            # warning in Stage B (see _process_frame re-check) can be
            # correlated against what Stage A saw at handoff. Capped to 5
            # frames to keep the noise floor manageable on long runs; the
            # cap is exposed via _stage_a_frame_count for future tuning.
            try:
                if getattr(self, '_stage_a_frame_count', 0) < 5:
                    self._stage_a_frame_count = getattr(self, '_stage_a_frame_count', 0) + 1
                    try:
                        _bid = grabResult.GetBlockID()
                    except Exception:
                        _bid = None
                    try:
                        _gid = grabResult.GetID()
                    except Exception:
                        _gid = None
                    _log_cam('info',
                        f'[CAM Class ] Stage A frame #{self._stage_a_frame_count}: '
                        f'GrabSucceeded={grab_succeeded} BlockID={_bid} GrabID={_gid}'
                    )
            except Exception as _e_diag:
                # Diagnostic logging must never break the callback path.
                # _log_safely is the bench-safe logger for native-thread
                # contexts (best-effort, swallows its own failures).
                _log_safely(f'Stage A first-frame diagnostic raised: {_e_diag}')

            if grab_succeeded:
                try:
                    # SWIG director hands OnImageGrabbed a non-owning wrapper
                    # around a CGrabResultPtr that lives on the SDK callback's
                    # stack. Without an explicit copy-ctor invocation here,
                    # the wrapper goes dangling the instant this function
                    # returns, even when our Python queue still references
                    # it. pylon.GrabResult(rhs) is the binding's surface for
                    # the documented C++ copy ctor (which bumps the
                    # CGrabResultPtrImpl refcount); the result is an owning
                    # wrapper that survives cross-thread handoff.
                    owned = pylon.GrabResult(grabResult)
                    self._worker.enqueue('frame', owned, ts)
                    _outcome = 'success_enqueued_frame'
                except queue.Full:
                    # Stage B is wedged; drop this frame rather than
                    # block the native thread. Bench-monitored via the
                    # pylon_stage_a_worker_queue_full_per_min budget.
                    _outcome = 'queue_full_dropped_frame'
                    _log_safely(
                        'Stage A: worker queue full -- dropping frame'
                    )
            else:
                try:
                    err_code = grabResult.GetErrorCode()
                except Exception:
                    err_code = None
                if err_code == _PYLON_ERR_DEVICE_NOT_FOUND:
                    # USB-Vision removal fast-path. Cascade rate observed
                    # at 100+ events in <2s, so the disconnect
                    # notification must fire IMMEDIATELY rather than wait
                    # behind Stage B's queue. _schedule_async_teardown
                    # runs the heavy SDK teardown on a daemon thread per
                    # pypylon issue #225 ("don't call StopGrabbing from
                    # the SDK callback thread").
                    _cam_log.error(
                        f'[CAM Class ] Camera device not found '
                        f'(USB disconnect / device removed) err_code={err_code}'
                    )
                    self._parent._mark_disconnected()
                    self._parent._schedule_async_teardown()
                    _outcome = 'success_no_grab_device_not_found'
                else:
                    # buffer-canceled / payload-discarded / generic
                    # transport failures hand off to Stage B for the
                    # full classification + per-failure logging + cascade
                    # counter handling. Same owning-wrapper requirement
                    # as the success path; Stage B reads GetErrorCode /
                    # GetErrorDescription / GetBlockID across threads.
                    try:
                        owned = pylon.GrabResult(grabResult)
                        self._worker.enqueue('fail', owned, ts)
                        _outcome = 'failure_enqueued'
                    except queue.Full:
                        _outcome = 'queue_full_dropped_failure'
                        _log_safely(
                            'Stage A: worker queue full on failure '
                            '-- dropping'
                        )
        except Exception as e:
            _outcome = 'exception_outer'
            _log_safely(f'OnImageGrabbed unexpected error: {e}')
        except BaseException as e:
            # Outer guard: anything that's NOT a regular Exception
            # subclass (SystemExit, KeyboardInterrupt, any non-
            # Exception BaseException) MUST be swallowed before this
            # callback returns to Pylon's grab thread. A C++ exception
            # escaping a native worker thread on Windows resolves to
            # std::terminate -- a silent process abort with no Python
            # traceback. Log best-effort and swallow.
            _outcome = 'exception_outer_baseexc'
            _log_safely(
                f'OnImageGrabbed BaseException guard '
                f'caught {type(e).__name__}: {e}'
            )
        finally:
            # finally block itself is wrapped because any raise here would
            # also escape to the native grab thread. Trace + handle-tick
            # are diagnostic; their failure must not crash the process.
            try:
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
                            # Stage A doesn't copy; bytes are read by
                            # Stage B if needed. Column kept for CSV
                            # schema stability.
                            0,
                        ],
                    )
            except BaseException as e:
                _log_safely(f'profile_trace.trace raised {type(e).__name__}: {e}')
            try:
                # Env-gated handle-leak tracking; zero overhead when disabled.
                # Enable with LVP_HANDLE_TRACE=1.
                from lib.handle_trace import tick as _h_tick
                _h_tick('OnImageGrabbed')
            except BaseException as e:
                _log_safely(f'handle_trace tick raised {type(e).__name__}: {e}')

    def reset(self) -> None:
        """Clear frame buffer, drain the queue, and reset failure counter."""
        try:
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
        except Exception as e:
            _cam_log.warning(f'[CAM Class ] handler reset queue-drain failed: {e}')
        self._base.reset()

    def get_last_image(self) -> tuple:
        """Return ``(success, image_copy, timestamp)`` with validity guard.

        Wraps the base ``ImageHandlerBase.get_last_image`` with a
        parent-camera validity check: if the camera has been marked
        removed or ``self._parent.active`` has been cleared, returns
        ``(False, None, None)`` immediately rather than handing back
        a frame from a no-longer-attached device.

        Returns:
            tuple: ``(success: bool, image: ndarray | None,
                timestamp: float | None)``.
        """
        try:
            if self._parent._device_removed:
                return False, None, None
            if self._parent.active is None:
                return False, None, None
        except Exception:
            return False, None, None

        return self._base.get_last_image()

    def get_last_image_with_chunks(self) -> tuple:
        """Return ``(success, image, timestamp, chunks)`` with validity guard.

        Atomic snapshot of frame + chunks under one lock acquisition in
        the base handler; see ``ImageHandlerBase.get_last_image_with_chunks``
        for the rationale. This wrapper applies the same parent-camera
        validity check as ``get_last_image``: a frame from a no-longer-
        attached device is suppressed rather than returned.

        Used by the manual-record path (``drivers/camera.py``
        ``grab_latest_with_chunks``) so per-frame TIFF metadata pairs
        with the correct image.

        Returns:
            tuple: ``(success: bool, image: ndarray | None,
                timestamp: float | None, chunks: dict | None)``.
        """
        try:
            if self._parent._device_removed:
                return False, None, None, None
            if self._parent.active is None:
                return False, None, None, None
        except Exception:
            return False, None, None, None

        return self._base.get_last_image_with_chunks()

    def register_frame_callback(self, cb) -> None:
        """Composition delegate to ``ImageHandlerBase.register_frame_callback``."""
        self._base.register_frame_callback(cb)

    def unregister_frame_callback(self, cb) -> None:
        """Composition delegate to ``ImageHandlerBase.unregister_frame_callback``."""
        self._base.unregister_frame_callback(cb)


class _PylonImageGrabWorker:
    """Stage B worker for the OnImageGrabbed two-stage split.

    Stage A is the Pylon SDK callback `OnImageGrabbed` running on a
    native grab thread; it does only the enqueue + fail-fast work and
    hands the grab-result smart pointer to Stage B (this worker) for
    the heavy memcpy + chunk parsing + callback fanout. Handoff is via
    the bounded `_worker_queue`.

    Single-thread invariant on the SDK's `CGrabResultData`: Stage A is
    the SOLE reader of `GrabSucceeded()` (and `GetErrorCode()` /
    `GetErrorDescription()` on the failure branch). After `put_nowait`
    Stage B becomes the SOLE reader of `GetArray()` + the chunk
    node-map accessors. The put / get pair provides happens-before; no
    lock required.

    Per `class_pylon_1_1_c_grab_result_ptr.html`: cross-thread smart-
    pointer retention is officially supported provided the result is
    released promptly. The `del grabResult` in the per-item `finally`
    satisfies this -- a grabResult must never linger in the worker
    indefinitely, or the SDK input queue underruns.

    NOT a SequentialIOExecutor lane. Driver-internal reactive thread,
    closest in shape to `_stats_poller_thread` (daemon, owned by the
    PylonCamera instance, started/stopped at connect / disconnect).
    """

    # Bounded queue depth. 8 gives ~264 ms of headroom at 30 FPS, large
    # enough that brief Stage B stalls (16 MB memcpy + chunk reads +
    # callback fanout) don't trip Stage A's queue.Full drop path, small
    # enough that a stalled consumer can't OOM the host. Bench-tunable
    # via LVP_PYLON_WORKER_QUEUE_DEPTH.
    _DEFAULT_QUEUE_DEPTH = 8

    # Poll interval inside _run's get(). Bounds the wakeup latency for
    # a stop() that loses the sentinel-enqueue race (queue.Full at stop
    # time). At 0.5 s the worst-case stop latency is ~0.5 s + per-item
    # processing time, well under the 1.0 s default join timeout.
    _GET_POLL_TIMEOUT_S = 0.5

    def __init__(self, parent, base, frame_queue) -> None:
        self._parent = parent  # PylonCamera instance
        self._base = base      # ImageHandlerBase instance
        self._frame_queue = frame_queue  # legacy maxsize=1 consumer queue
        _depth_env = os.environ.get('LVP_PYLON_WORKER_QUEUE_DEPTH')
        try:
            depth = int(_depth_env) if _depth_env else self._DEFAULT_QUEUE_DEPTH
        except ValueError:
            depth = self._DEFAULT_QUEUE_DEPTH
        self._worker_queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the worker thread (daemon=True).

        Idempotent: a second call while a thread is already alive is a
        no-op. After `stop()` returns, `start()` may be called again to
        respawn for a new connect cycle.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name='PylonImageGrabWorker',
            daemon=True,
        )
        self._thread.start()

    def enqueue(self, kind: str, grabResult, ts) -> None:
        """Hand a Stage A item off to Stage B.

        Raises `queue.Full` if the worker queue is at capacity -- Stage A
        catches this and drops the frame rather than blocking the native
        grab thread. Returns silently if the worker is shutting down
        (stop_event already set); the caller's `grabResult` local will
        release the SDK buffer when Stage A returns.

        Args:
            kind: 'frame' for a successful grab, 'fail' for a failure
                that needs Stage B classification.
            grabResult: The SDK CGrabResultPtr to hand off. Stage A MUST
                NOT access it after this returns (single-thread invariant
                on CGrabResultData).
            ts: Capture timestamp from Stage A's `datetime.now()`.
        """
        if self._stop_event.is_set():
            return
        self._worker_queue.put_nowait((kind, grabResult, ts))

    def stop(self, timeout: float = 1.0) -> None:
        """Signal stop, post sentinel, join with bounded timeout.

        Ordering contract: the caller MUST invoke `camera.StopGrabbing()`
        BEFORE this so the SDK stops firing callbacks, and MUST invoke
        SDK teardown (Close / DetachDevice / DestroyDevice) AFTER this
        so the worker has released its grabResult refs. Inverting the
        order silently drops grabResults without release and reintroduces
        the input-queue-underrun the worker was designed to avoid.

        Idempotent: calling stop on an already-stopped worker is safe.

        Args:
            timeout: Maximum seconds to wait for the worker thread to
                exit. Bounded so a wedged worker can't deadlock
                disconnect.
        """
        self._stop_event.set()
        try:
            self._worker_queue.put_nowait(('stop', None, None))
        except queue.Full:
            # Sentinel doesn't fit; the event flag plus the
            # _GET_POLL_TIMEOUT_S inside _run ensures exit within one
            # poll interval after the next item is processed.
            logger.debug(
                '[CAM Class ] worker.stop: queue full, sentinel dropped; '
                'exit will run via stop_event fallback'
            )
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        # Loop structure: always drain a pending item before checking
        # the stop event. The event check at the *top* of the while loop
        # would race with the producer -- if stop() set the event AFTER
        # Stage A enqueued the last batch but BEFORE the worker came
        # back from processing the previous item, the next iteration
        # would exit without draining the rest. The SDK input queue
        # would then underrun on the next connect cycle per
        # `class_pylon_1_1_c_grab_result_ptr.html`: "grabbing will stop
        # with an input queue underrun, when the grab results are never
        # released." Only consult the event when get() comes back empty.
        while True:
            try:
                kind, grabResult, ts = self._worker_queue.get(
                    timeout=self._GET_POLL_TIMEOUT_S
                )
            except queue.Empty:
                if self._stop_event.is_set():
                    # Sentinel was dropped (queue.Full at stop time) or
                    # never enqueued. Nothing left to drain.
                    return
                continue
            if kind == 'stop':
                # Drain remaining items + release their grabResults so
                # the SDK input queue doesn't underrun on shutdown.
                self._drain_and_release()
                return
            try:
                if self._parent._device_removed:
                    # Drop quietly; the finally below releases the
                    # smart pointer so the SDK can reclaim the buffer.
                    continue
                if kind == 'frame':
                    self._process_frame(grabResult, ts)
                elif kind == 'fail':
                    self._process_failure(grabResult, ts)
            except Exception as e:
                if _cam_log is not None:
                    _cam_log.exception(
                        f'[CAM Class ] worker handling raised: {e}'
                    )
            finally:
                # Explicit release semantics. Per CGrabResultPtr docs:
                # "grabbing will stop with an input queue underrun,
                # when the grab results are never released."
                del grabResult

    def _process_frame(self, grabResult, ts) -> None:
        """Stage B success path: 16 MB memcpy + chunks + store + legacy queue.

        Preserves the defensive behavior of the pre-cutover OnImageGrabbed
        for GetArray() failure: mark the device removed (so future Stage A
        callbacks early-return) and record a grab failure (so the cascade
        counter triggers if multiple frames fail in a row). Cross-thread
        smart-pointer access is officially supported per
        `class_pylon_1_1_c_grab_result_ptr.html` provided GetArray succeeds;
        a raised exception here means the underlying buffer is gone.
        """
        try:
            img = grabResult.GetArray().copy()
        except Exception as e:
            # H2/H3 diagnostic: Stage A enqueued this with GrabSucceeded=True.
            # Re-read GrabSucceeded + GetErrorCode/Description on Stage B's
            # side. The re-check distinguishes (a) race conditions where SDK
            # state flipped between threads (GrabSucceeded was True, now False
            # or unreadable) from (b) Stage A misclassification (GrabSucceeded
            # was wrong from the start). Also reads BlockID / GrabID if
            # available so the failure can be matched to the Stage A
            # first-frames diagnostic by frame identity.
            try:
                _gs_now = grabResult.GrabSucceeded()
            except Exception as _ee:
                _gs_now = f'<re-read failed: {_ee}>'
            try:
                _err_code = grabResult.GetErrorCode()
                _err_desc = grabResult.GetErrorDescription()
            except Exception:
                _err_code, _err_desc = None, None
            try:
                _bid = grabResult.GetBlockID()
            except Exception:
                _bid = None
            try:
                _gid = grabResult.GetID()
            except Exception:
                _gid = None
            _log_cam('warning',
                f'[CAM Class ] GetArray() failed on worker: {e} '
                f'(Stage B re-check: GrabSucceeded={_gs_now} '
                f'err_code={_err_code} err_desc={_err_desc!r} '
                f'BlockID={_bid} GrabID={_gid}), '
                f'marking device as removed'
            )
            self._parent._mark_disconnected()
            self._base._record_failure()
            return
        chunks = _read_validity_chunks(grabResult)
        self._base._store_frame(img, ts, chunks=chunks)
        try:
            if not self._frame_queue.empty():
                with contextlib.suppress(queue.Empty):
                    self._frame_queue.get_nowait()
            self._frame_queue.put_nowait((True, img, ts))
        except queue.Full:
            # latest-wins; older drop is intended (legacy consumer can
            # only hold one frame). Log at debug so the cause stays in
            # the post-mortem trace without raising the noise floor.
            logger.debug(
                '[CAM Class ] worker frame_queue full after drain attempt; '
                'dropping latest frame (legacy maxsize=1 consumer race)'
            )

    def _process_failure(self, grabResult, ts) -> None:
        """Stage B failure path: classify + record + auto-stop on cascade.

        Stage A handles the err_code=DEVICE_NOT_FOUND fast-path inline so
        the disconnect notification doesn't wait behind Stage B. Everything
        else (buffer-canceled, payload-discarded, generic transport
        failures) is classified here.
        """
        try:
            err_code = grabResult.GetErrorCode()
            err_desc = grabResult.GetErrorDescription()
        except Exception as e:
            err_code, err_desc = None, repr(e)

        if err_code == _PYLON_ERR_BUFFER_CANCELED or self._parent._device_removed:
            # Cancelled buffers (StopGrabbing mid-flight) and any failure
            # paired with the removal flag are SDK lifecycle events, not
            # real failures. The OR insurance guards a race where the
            # removal-forwarding SDK thread flips _device_removed between
            # Stage A's early-return check and this classification: an
            # undocumented removal-time err_code would otherwise count
            # toward MAX_CONSECUTIVE_FAILURES.
            logger.debug(
                f'[CAM Class ] Grab cancelled (SDK lifecycle, '
                f'not a failure) err_code={err_code} desc={err_desc!r} '
                f'device_removed={self._parent._device_removed}'
            )
        elif err_code == _PYLON_ERR_PAYLOAD_DISCARDED:
            # Camera-side FIFO overflow during host stalls. The dropped
            # frame is one frame_validity would have rejected anyway
            # (invalidate runs after each SetValue). Logged at info so
            # the cause distribution stays visible in camera.log without
            # raising the noise floor; NOT counted toward
            # MAX_CONSECUTIVE_FAILURES because acquisition is healthy.
            if _cam_log is not None:
                _cam_log.info(
                    f'[CAM Class ] payload discarded (camera-side FIFO '
                    f'overflow during host stall) err_code={err_code} '
                    f'desc={err_desc!r}'
                )
        else:
            # err_code/desc varies (USB CRC, partial frame, underrun);
            # log each to preserve cause distribution + count toward the
            # consecutive-failure cascade so MAX_CONSECUTIVE_FAILURES
            # eventually trips auto-disconnect on a wedged transport.
            if _cam_log is not None:
                _cam_log.warning(
                    f'[CAM Class ] grabResult.GrabSucceeded()=False '
                    f'err_code={err_code} desc={err_desc!r}'
                )
            # Returns True after MAX_CONSECUTIVE_FAILURES (128 frames at
            # 30 fps ~= 4.3s) consecutive failures.
            if self._base._record_failure():
                try:
                    if _cam_log is not None:
                        _cam_log.error(
                            '[CAM Class ] Too many grab failures; '
                            'stopping acquisition'
                        )
                    if self._parent.active and self._parent.is_grabbing():
                        self._parent.stop_grabbing()
                    self._parent._mark_disconnected()
                except Exception as e:
                    if _cam_log is not None:
                        _cam_log.warning(
                            f'[CAM Class ] worker could not stop grabbing '
                            f'after max failures: {e}'
                        )

    def _drain_and_release(self) -> None:
        """Drain the worker queue and release each grabResult.

        Called only from the stop-sentinel branch in `_run`. Per
        CGrabResultPtr docs the SDK input queue underruns if results
        are never released; this loop satisfies the release contract
        on shutdown.
        """
        while True:
            try:
                _, gr, _ = self._worker_queue.get_nowait()
                del gr
            except queue.Empty:
                return


# Handle camera removal events to flag device disconnect
class _CameraRemovalHandler(pylon.ConfigurationEventHandler):
    def __init__(self, parent_cam: PylonCamera):
        super().__init__()
        self._parent = parent_cam

    def OnCameraDeviceRemoved(self, camera) -> None:
        """Pylon SDK callback fired when the device disappears.

        Runs in a native Pylon SDK thread under the camera lock per
        DoxyPylon.i. Per the pypylon contract any exception raised
        here is swallowed by the SDK; we still wrap the body in
        BaseException to keep our logs clean. Schedules async
        teardown so the heavy Close/DestroyDevice work runs from a
        Python-owned thread rather than the SDK callback context.

        Args:
            camera: SDK reference (unused; pylon contract).
        """
        try:
            self._parent._mark_disconnected()
            _log_safely('Camera physically removed (Pylon SDK callback)')
            self._parent._schedule_async_teardown()
        except BaseException as e:
            _log_safely(
                f'OnCameraDeviceRemoved guard caught '
                f'{type(e).__name__}: {e}'
            )

    def OnGrabError(self, camera, errorMessage) -> None:
        """Pylon SDK callback fired when an exception happens in the grab thread.

        Per `class_pylon_1_1_c_configuration_event_handler.html`:
        "This method is called when an exception has been triggered
        during grabbing... An exception has been triggered by a grab
        thread. The grab will be stopped after this event call."
        Runs inside the camera lock from a separate SDK thread; SDK
        catches and ignores exceptions per the contract.

        This is the LAST diagnostic signal before the SDK tears down
        its grab loop. If the silent-crash mechanism per
        PYLON_DISCONNECT_DEFENSE.md ("Pylon's grab thread's OWN
        internal cleanup after our callback returns") fires, this is
        the log line we expect to see right before the abort. Worth
        logging at ERROR so it pops in post-mortem.

        Args:
            camera: SDK reference (unused; pylon contract).
            errorMessage: SDK-supplied error description (C++ char*).
        """
        try:
            _cam_log.error(
                f'[CAM Class ] OnGrabError fired (SDK grab thread caught '
                f'exception; grab will stop): {errorMessage}'
            )
            self._parent._mark_disconnected()
        except BaseException as e:
            _log_safely(
                f'OnGrabError guard caught {type(e).__name__}: {e}'
            )
