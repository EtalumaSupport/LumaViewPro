# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import contextlib
import logging
import math
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any, ClassVar

from pypylon import genicam, pylon

from _lumascope.drivers.camera import (
    Camera,
    CameraNodeHasNoValue,
    CameraNodeValue,
    FrameGrid,
    HandOver,
    ImageHandlerBase,
    NOTHING_OWED,
    copy_oriented,
    declared_speed,
    link_info,
    node_snapshot_value,
    orientation_owed,
)
from _lumascope.drivers.exceptions import (
    CameraNoAnswerError,
    CameraReadFailedError,
    CameraSetupFailedError,
    CameraStreamStartFailedError,
    CameraUsbNotFoundError,
    CameraUsbOpenFailedError,
    CameraWriteFailedError,
    HardwareError,
    PartAbsentError,
)
from _lumascope.drivers.registry import camera_registry
from _lumascope.lib.log_helpers import camera_logger as _cam_log
from _lumascope.lib.log_helpers import log_to, logger
from _lumascope.lib.log_helpers import VERBOSE

from _lumascope.lib import profile_trace
from _lumascope.lib.real_numbers import is_finite_number, is_integer_number


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
        _cam_log.warning(f'[CAM Class ] {message}')
    except BaseException:
        # If the logger itself fails there is no safer fallback from
        # a native callback context. The swallow stands.
        _ = None  # intentional no-op marker for the AST scan


def _log_cam(level: str, message: str) -> None:
    """Mirror a camera-relevant log entry to BOTH the main log AND camera.log.

    Use for events that future debuggers will want to see in either file:
    identity readback (serial, firmware, profile), connect / disconnect,
    configuration changes (gain, exposure, frame size), TlFactory device
    enumeration, MaxNumBuffer + StreamGrabber settings, GetArray failures
    on the worker. High-volume per-frame SDK trace lines (Gain.SetValue,
    ExposureTime.SetValue) stay on _cam_log directly -- those are
    deliberately kept out of the main log to control noise.

    Thin wrapper over ``_lumascope.lib.log_helpers.log_to`` so the same dual-write
    pattern can be reused by motorboard / ledboard / idscamera once
    they grow per-subsystem dedicated log files (motor.log / led.log /
    ids.log). When that lands, drop this wrapper in favor of direct
    ``log_to(logger, <subsystem>_log, ...)`` calls at the use sites.

    Args:
        level: 'info' / 'warning' / 'error' / 'debug' / 'exception'.
        message: The full log message. Convention: include '[CAM Class ]'
            prefix for grep-ability (matches the prefix in the main log).
    """
    log_to(logger, _cam_log, level=level, message=message)


def pylon_versions() -> dict:
    """The Basler pylon SDK's version and the pypylon binding's, as dotted strings.

    The one reader of both: the driver's ``get_sdk_info``, its connect log,
    the environment banner and the support report all state these. Read
    from ``pylon``, not the ``pypylon`` package, which carries no
    ``__version__``; ``GetPylonVersion()`` answers a list, its string form
    ``GetPylonVersionString()`` the dotted version.

    Returns:
        dict: ``{'version': '12.5.2.1565', 'pypylon_version': '26.10'}``.
    """
    return {'version': pylon.GetPylonVersionString(), 'pypylon_version': pylon.__version__}


def describe_device_info(dev_info: Any) -> str:
    """Render every property the SDK reports for an enumerated device.

    The property list comes from the SDK, not from a set of getters
    here. A CDeviceInfo carries roughly three dozen properties and
    which of them are populated depends on the transport, so any
    hand-picked set both misses the fields that identify what sits
    UNDER the SDK -- the USB driver kind, its Windows driver key, the
    port and the transfer mode -- and silently stops growing when a
    pylon release adds one. Those transport fields are what attributes
    a stall inside an SDK call to a machine, and a support bundle is
    the only place they can ever be read from.

    Degradation is per-property so one bad accessor cannot cost the
    whole description: a property the SDK marks unavailable is simply
    not applicable to this transport and is skipped, while one whose
    read raises is named with its reason, so a gap is visible rather
    than silent.

    Args:
        dev_info: An enumerated device-info object exposing the SDK's
            generic property interface (``GetPropertyNames`` returning
            ``(count, names)``, ``GetPropertyValue`` returning
            ``(available, value)``).

    Returns:
        str: Space-separated ``Name=value`` pairs, sorted by name so two
            bundles from one machine diff cleanly, or a marker naming
            why no description could be built.
    """
    try:
        _, _names = dev_info.GetPropertyNames()
    except Exception as e:
        return f'<device info unreadable: {type(e).__name__}: {e}>'

    _parts = []
    for _name in sorted(_names):
        try:
            _available, _value = dev_info.GetPropertyValue(_name)
        except Exception as e:
            _parts.append(f'{_name}=<unreadable: {type(e).__name__}>')
            continue
        if _available:
            _parts.append(f'{_name}={_value!r}')
    return ' '.join(_parts)


def _accepted_range(node, name: str) -> tuple[float, float]:
    """Return a node's (min, max) narrowed to values it will actually accept.

    A GenICam float node reports its range in continuous units but accepts
    only steps of its own increment, and the reported maximum can carry a
    float tail past the last real step -- a 48 dB gain node reports
    48.00000004350822. Published raw, that tail becomes the slider cap and
    is persisted to the settings store, so a user who types the advertised
    maximum has it written back as a long decimal, and the camera rejects
    the very value it advertised.

    The maximum is snapped DOWN to the last accepted step. It is never
    snapped up: a bound must be a value the camera takes, and rounding to
    nearest can land past the ceiling.

    A node whose increment is not a single fixed step is published exactly
    as reported. `listIncrement` means the legal values are an arbitrary
    list and `noIncrement` means the range really is continuous; in both
    cases there is no step to snap to, and inventing one would narrow a
    range the camera did not narrow. `GetInc()` is only meaningful under
    `fixedIncrement`, and an `IInteger` node has no `HasInc` at all, so
    the increment MODE is the thing to ask.
    """
    low, high = node.GetMin(), node.GetMax()
    try:
        if node.GetIncMode() != genicam.fixedIncrement:
            return low, high
        increment = node.GetInc()
    except Exception as e:
        logger.debug(f'[CAM Class ] {name}: no usable increment ({e}); range as reported')
        return low, high

    if not increment or increment <= 0 or high < low:
        return low, high

    steps = math.floor((high - low) / increment)
    decimals = max(0, -math.floor(math.log10(increment)))
    snapped = round(low + steps * increment, decimals)

    if snapped != high:
        logger.debug(
            f'[CAM Class ] {name}: max {high} is not on the {increment} step grid; '
            f'publishing {snapped}'
        )
    return low, snapped


# The device class of Basler's camera emulator (``PYLON_CAMEMU``), as
# ``CDeviceInfo.GetDeviceClass()`` reports it.
_EMULATOR_DEVICE_CLASS = 'BaslerCamEmu'

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
# Value 0xE2000212 (decimal 3791651346) -- the USB3-Vision payload-discard
# code per Basler's stream-grabber-parameters.html, bench-confirmed on a
# daA3840 dart. (Earlier this read 0xE2050012, a value no hardware emits,
# so the benign branch below never matched and every camera-side discard
# fell through to the generic path: logged as a WARNING and counted toward
# MAX_CONSECUTIVE_FAILURES.)
_PYLON_ERR_PAYLOAD_DISCARDED = 0xE2000212

# Device-not-found: USB-Vision transport returns this when the device
# handle no longer resolves on the bus (cable unplug, USB hub power
# loss, OS-level device removal). Bench-witnessed value 433 (decimal,
# from the LVP_Logbumped.wire bench session): "A device which does not
# exist was specified". Cascade rate observed at ~100+ events in <2s
# after disconnect, which would take ~4.3s to trip the slow-path
# MAX_CONSECUTIVE_FAILURES auto-disconnect (128 frames at 30fps). Fast
# classification short-circuits the cascade so the disconnect surfaces
# in 1 frame instead of 128, and the user notification (driven by
# _mark_disconnected -> API layer) fires immediately
# instead of 4 seconds late behind a wall of WARNING log lines.
_PYLON_ERR_DEVICE_NOT_FOUND = 433


# MaxNumBuffer applied post-Open() in connect(). 10 is the pylon SDK /
# Pylon Viewer default. A 3-buffer cap was tried to cut RAM, but on a slow
# host the small pool let the grab worker fall behind under CPU load: the
# camera discarded payloads ("bandwidth insufficient") and, once the
# matching grab-result pool filled, dropped the connection entirely. More
# buffers absorb the host stall for a few MB of RAM; revisit under a RAM /
# buffer-reuse audit. connect() is the node's one writer: the InstantCamera
# node goes read-only once grabbing has begun, so it is set before the
# implicit AcquireContinuousConfiguration auto-start.
_DEFAULT_MAX_NUM_BUFFER = 10

# MaxTransferSize applied post-Open() in connect(), pinned regardless of
# platform. It is the per-USB-transfer byte budget; bigger transfers mean
# fewer kernel transitions and less CPU per frame. Per Basler's stream-
# grabber doc the default "depends on the operating system" -- the OSX /
# Pylon Viewer default is 256 KB, which is unstable here, whereas 1 MB
# holds. That doc also requires this be set before grabbing starts, so the
# connect-time requested-vs-actual readback is the tell that the post-Open
# window applied it.
_DEFAULT_MAX_TRANSFER_SIZE = 1024 * 1024  # 1 MB

# Production grab strategy is LatestImageOnly -- frame_validity,
# capture_still, and the auto-discard skip_frames floor all depend
# on this strategy (see DAILY_LOG 2026-05-04). OneByOne delivers every
# frame for apples-to-apples bench comparisons against Pylon Viewer.
_DEFAULT_GRAB_STRATEGY = 'LatestImageOnly'


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

        # The grab strategy start_grabbing() uses, written by the imaging
        # sub-API's lever. connect() writes MaxNumBuffer and MaxTransferSize
        # from their constants; NumMaxQueuedUrbs stays at the SDK default
        # (set only via its lever).
        self._grab_strategy_name = _DEFAULT_GRAB_STRATEGY

        # Cache of the active PixelFormat. PixelFormat only changes through
        # set_pixel_format(), so the cache is refreshed there and cleared on
        # disconnect; get_pixel_format() serves from it to avoid a live
        # GenICam node read on the per-frame image-metadata path.
        self._pixel_format_cache = None

        super().__init__()

    # _mark_disconnected() inherited from Camera base class

    # _schedule_async_teardown() inherited from Camera: the Pylon callback
    # thread must never call StopGrabbing on itself -- it deadlocks or aborts
    # natively (pypylon issue #225) -- so OnImageGrabbed and the removal
    # callback hand the teardown to the base's thread. disconnect() then runs
    # stop_grabbing -> wait_for_acquisition_idle -> Close -> DetachDevice ->
    # DestroyDevice, each independently guarded.

    def _read_node_range(self, name: str) -> tuple[float, float] | None:
        """The (min, max) a float node accepts now, read from the camera.

        Narrowed to the values the node takes (``_accepted_range``). None for
        a node this camera does not offer.

        Raises:
            CameraReadFailedError: The node or its range could not be read.
        """
        try:
            node = self._node(name)
            if node is None:
                return None
            return _accepted_range(node, name)
        except Exception as e:
            raise CameraReadFailedError(f'{name} range read failed: {type(e).__name__}: {e}') from e

    @property
    def min_exposure(self) -> float | None:
        """The shortest exposure the camera takes now, in milliseconds.

        Read from the ExposureTime node each call: its minimum moves with
        the pixel format (Mono8 to Mono12 raises it on the a2A3536 and the
        daA3840) and with the camera's firmware. A camera without the node
        answers the profile's documented floor.

        Raises:
            CameraReadFailedError: The node's range could not be read.
        """
        if not self.active:
            return super().min_exposure
        span = self._read_node_range('ExposureTime')
        return super().min_exposure if span is None else span[0] / 1000.0

    @property
    def max_exposure(self) -> float:
        """The longest exposure the camera takes now, in milliseconds,
        read as ``min_exposure`` is.

        Raises:
            CameraReadFailedError: The node's range could not be read.
        """
        if not self.active:
            return super().max_exposure
        span = self._read_node_range('ExposureTime')
        return super().max_exposure if span is None else span[1] / 1000.0

    @property
    def min_gain(self) -> float | None:
        """The least gain the camera takes now, in dB, read from the Gain node.

        A camera without the node answers the profile's documented floor.

        Raises:
            CameraReadFailedError: The node's range could not be read.
        """
        if not self.active:
            return super().min_gain
        span = self._read_node_range('Gain')
        return super().min_gain if span is None else span[0]

    @property
    def max_gain(self) -> float:
        """The most gain the camera takes now, in dB, read as ``min_gain`` is.

        Raises:
            CameraReadFailedError: The node's range could not be read.
        """
        if not self.active:
            return super().max_gain
        span = self._read_node_range('Gain')
        return super().max_gain if span is None else span[1]

    def _query_dynamic_capabilities(self):
        """Query the Pylon SDK for the camera's formats and pixel pitch, and log its ranges."""
        if not self.active:
            return

        try:
            # The gain and exposure ranges are not stored: they move with the
            # readout (Mono8 to Mono12 raises the exposure floor), so they are
            # read from the camera when asked (min_gain, max_gain,
            # min_exposure, max_exposure). The range the camera came
            # up with is logged, the record of the camera as it connected.
            try:
                gain_range = self._read_node_range('Gain')
                if gain_range is None:
                    logger.info('[CAM Class ] Gain not offered by this camera')
                else:
                    logger.info(
                        f'[CAM Class ] Gain range: {gain_range[0]:.1f} - {gain_range[1]:.1f} dB'
                    )
            except CameraReadFailedError as e:
                _cam_log.warning(f'[CAM Class ] Could not query gain range: {e}')
            try:
                exposure_range = self._read_node_range('ExposureTime')
                if exposure_range is None:
                    logger.info('[CAM Class ] ExposureTime not offered by this camera')
                else:
                    logger.info(
                        f'[CAM Class ] Exposure range: {exposure_range[0]:.0f} - '
                        f'{exposure_range[1]:.0f} us'
                    )
            except CameraReadFailedError as e:
                _cam_log.warning(f'[CAM Class ] Could not query exposure range: {e}')

            # The formats this body offers, as the IDS driver records them: the
            # profile's list is what capabilities publish, so it is the camera's
            # answer, not the static entry's. An unreadable list keeps the
            # documented one, said here, and the pixel pitch is still read.
            try:
                supported = self.get_supported_pixel_formats()
            except HardwareError as e:
                _cam_log.warning(
                    f'[CAM Class ] PixelFormat entries unreadable ({e}); the profile keeps '
                    f'its documented formats {self.profile.pixel_formats}'
                )
            else:
                self.profile.pixel_formats = list(supported)
                logger.info(f'[CAM Class ] Supported PixelFormat entries: {list(supported)}')

            # Sensor pixel pitch (micrometers). Read live so the micron scale
            # bar and click-to-center distance are correct for ANY Basler body,
            # not only ones with a curated profile. Fill only when the profile
            # did not supply one (the generic fallback carries 0.0) -- never
            # override a curated, bench-validated value -- but always log the
            # hardware value so a mismatch is visible. A body that does not
            # advertise the node leaves sensor_pitch_um unset; the scale then
            # degrades honestly rather than inheriting a guessed default.
            try:
                px_node = self._node('SensorPixelWidth')
                if px_node is None:
                    logger.info(
                        '[CAM Class ] SensorPixelWidth not offered by this camera; '
                        'pixel pitch left unset'
                    )
                else:
                    sensor_px_um = px_node.GetValue()
                    logger.info(f'[CAM Class ] SensorPixelWidth: {sensor_px_um:.4f} um')
                    if sensor_px_um and sensor_px_um > 0 and not self.profile.sensor_pitch_um:
                        self.profile.sensor_pitch_um = float(sensor_px_um)
            except Exception as e:
                _cam_log.warning(f'[CAM Class ] Could not query SensorPixelWidth: {e}')

        except Exception as e:
            _cam_log.warning(f'[CAM Class ] _query_dynamic_capabilities failed: {e}')

    def _disconnect(self) -> bool:
        """Tear down the active Pylon camera and release the SDK device.

        Each teardown step is independently guarded so one failure doesn't
        block the rest. After return, ``self.active is None`` regardless of
        SDK call success. Returns False if not connected or outer aborted.

        When the device is already known to be removed (cable unplug,
        callback err_code=433 already observed), the SDK-touching
        teardown steps (StopGrabbing, wait_for_acquisition_idle, Close)
        are SKIPPED; a removal marked while the teardown runs is asked
        again before Close, which is then skipped too. Calling these on a
        dead device handle is the pypylon #225 abort hazard: StopGrabbing
        can hang or trigger a native abort, and Close on an already-removed
        handle has been observed to crash. Only DetachDevice + DestroyDevice are kept,
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
                    # A removal latched since the check at entry -- during the
                    # StopGrabbing or the idle wait -- is asked again here, so
                    # Close() never meets a handle marked removed.
                    if self.is_device_removed():
                        _log_safely(
                            'disconnect: device removed during the teardown -- '
                            'skipping Close (pypylon #225 hazard)'
                        )
                    else:
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
                # Invalidate the PixelFormat cache; the next connect
                # repopulates it via init_camera_config -> set_pixel_format.
                self._pixel_format_cache = None
                # Clear the start gate + last-frame buffer so a same-instance
                # reconnect starts clean (re-grabs, no stale image).
                self._reset_lifecycle_state()
                _log_cam('info', '[CAM Class ] Disconnected from Pylon camera')
                return True
            else:
                _log_cam('info', '[CAM Class ] Pylon camera not connected')
        except Exception as e:
            _cam_log.exception(f'[CAM Class ] Pylon camera disconnect failed: {e}')
        # Not-connected / error paths also clear the gate + last-frame buffer: a
        # FAILED recovery can null active while the gate is still OPEN, so a
        # same-instance reconnect must not inherit a stranded-open gate (blank
        # live view) or a stale image.
        self._reset_lifecycle_state()
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
        if not profile_trace.ENABLE_PROFILE_TRACE:
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
                    '[INSTR PYLON ] stats poller did not exit within 2s; releasing reference anyway'
                )
        self._stats_poller_thread = None

    def _stats_poller_loop(self):
        ev = self._stats_poller_stop
        while not ev.wait(self._STATS_POLLER_INTERVAL_S):
            ts_ms = int(time.time() * 1000)
            # --- Pylon SDK statistics ---
            stats = {}
            rfr = None
            underrun_value = None
            underrun_name = ''
            temp_state = None
            # A node this camera or transport lacks reads None and the CSV
            # row states the absence; a failed read ends this cycle's reads
            # and is told.
            try:
                if self.active is not None:
                    for name in self._STATS_NODE_NAMES:
                        stats[name] = self._read_first_offered((name,), 'stream')
                    # Read the canonical underrun counter directly.
                    underrun_value = self._read_first_offered((self._UNDERRUN_NODE_NAME,), 'stream')
                    if underrun_value is not None:
                        underrun_name = self._UNDERRUN_NODE_NAME
                    rfr = self._read_first_offered(self._RESULTING_FRAME_RATE_NODES)
                    temp_state = self._read_first_offered(('TemperatureState',))
            except Exception as e:
                _cam_log.warning(f'[INSTR PYLON ] stats poll error: {e}')

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
            if is_finite_number(resync_value) and is_finite_number(prev_resync):
                delta = resync_value - prev_resync
                if delta > 0:
                    _cam_log.warning(
                        f'[INSTR RESYNC] {self._RESYNC_NODE_NAME} delta={delta} '
                        f'(total={resync_value})'
                    )
            if is_finite_number(resync_value):
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
            if is_finite_number(missed_value) and is_finite_number(prev_missed):
                delta = missed_value - prev_missed
                if delta > 0:
                    _cam_log.warning(
                        f'[INSTR MISSED] {self._MISSED_FRAME_NODE_NAME} delta={delta} '
                        f'(total={missed_value})'
                    )
            if is_finite_number(missed_value):
                self._prev_missed_frame_count = missed_value

            # Temperature state: per Basler doc temperature-state.html,
            # ace 2 / boost / dart M/R cameras stop image acquisition
            # when over-temperature is reached and require cooldown
            # before restart -- presents identically to a frame-rate
            # stall in the user log without temperature attribution.
            # Warn on any non-Ok state so the cause is visible.
            prev_temp_state = getattr(self, '_prev_temp_state', None)
            if temp_state is not None and temp_state != prev_temp_state:
                if temp_state in ('Critical', 'Error'):
                    _cam_log.warning(
                        f'[INSTR TEMP] TemperatureState={temp_state!r} (was {prev_temp_state!r})'
                    )
                else:
                    logger.info(
                        f'[INSTR TEMP] TemperatureState={temp_state!r} (was {prev_temp_state!r})'
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
                recording_id=profile_trace.NO_RECORDING,
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
                n_pylon_worker = sum(1 for t in threads if t.name == 'PylonImageGrabWorker')
                n_dummy = sum(1 for t in threads if t.name.startswith('Dummy'))
                n_total = len(threads)
                profile_trace.trace(
                    'pylon_threads_trace.csv',
                    'ts_ms,pylon_native_grab_count,pylon_worker_count,'
                    'dummy_count,total_thread_count',
                    [ts_ms, n_pylon_native, n_pylon_worker, n_dummy, n_total],
                    recording_id=profile_trace.NO_RECORDING,
                )
            except Exception as e:
                logger.debug(f'[INSTR PYLON ] thread-count poll error: {e}')

    def _stop_grabbing(self) -> None:
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
        try:
            self._timed_stop_grabbing(self.active, 'stop_grabbing')
        except Exception as e:
            _cam_log.warning(f'[CAM Class ] stop_grabbing ignored error: {e}')

    # A stop at or past this is logged as a WARNING. The fast stops measured
    # on a daA3840-45um took 14-71 ms; its stalled ones a near-constant 11 s.
    _SLOW_GRAB_STOP_S = 1.0

    # The cap's share of the link (``_cap_the_frame_rate_under_the_link``).
    # A 1.1.0 daA3840 held at 0.90-0.95 of its link-bound rate stopped
    # without stalling, and stalled from the second stop at 0.99. The camera
    # reaches 0.958-0.976 of limit / payload, so 0.92 of it is 0.94-0.96 of
    # the rate it reaches, inside the clean band.
    _LINK_RATE_SHARE = 0.92

    # A camera reset at connect (``_resets_at_connect``) leaves the bus and
    # comes back within this many seconds of its reset, both on one clock;
    # the daA3840s on the bench came back in 2.2-3.3 s. The settle is the
    # wait after it is listed again before it is opened: every bench reset
    # of a daA3840 reopened after it.
    _RESET_RETURN_S = 90.0
    _RESET_SETTLE_S = 1.0
    _RESET_POLL_S = 0.2

    # The device transport layer's counters of timed-out and failed accesses
    # (Basler doc transport-layer-parameters.html). The stop's stalled wait is
    # inside a register write, so a timeout counter that rises across a slow
    # stop says a register access timed out during it.
    _TL_COUNTER_NODES = (
        'StatisticReadWriteTimeoutCount',
        'Statistic_Write_Operations_Failed_Count',
        'Statistic_Write_Pipe_Reset_Count',
        'Statistic_Read_Operations_Failed_Count',
        'Statistic_Last_Error_Status_Text',
    )

    def _timed_stop_grabbing(self, camera, caller: str) -> None:
        """Call ``camera.StopGrabbing()``, logging the state it stopped from and how long it took.

        One line before the stop records what was in flight, one after
        records the elapsed time (WARNING at ``_SLOW_GRAB_STOP_S`` or more)
        and the transport layer's access counters again, so a bundle tells a
        slow stop from a fast one, and shows what differed between them,
        without anyone timing log lines by hand. A failure is logged with its
        elapsed time and raised to the caller.
        """
        _cam_log.info(f'pylon StopGrabbing() [{caller}]: {self._grab_stop_state(camera)}')
        t0 = time.perf_counter()
        try:
            camera.StopGrabbing()
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            _cam_log.warning(
                f'pylon StopGrabbing [{caller}] FAILED after {elapsed_ms:.1f} ms: {e}; '
                f'{self._transport_layer_state(camera, self._TL_COUNTER_NODES)}'
            )
            raise
        elapsed_s = time.perf_counter() - t0
        level = logging.WARNING if elapsed_s >= self._SLOW_GRAB_STOP_S else logging.INFO
        _cam_log.log(
            level,
            f'pylon StopGrabbing [{caller}] returned after {elapsed_s * 1000.0:.1f} ms: '
            f'{self._transport_layer_state(camera, self._TL_COUNTER_NODES)}',
        )

    def _transport_layer_nodes(self, camera, errors: list) -> dict:
        """Every node of the device transport layer's nodemap, name to value.

        Which nodes a transport layer has differs by camera and pylon
        release (a daA3840-45um under pylon 11.5 exposes no register-access
        timeout or retry count under any name), so the snapshot lists what
        this camera has rather than asking for names it may not. A node with no
        value reads as ``_safe_node``'s sentinel; a nodemap that cannot be
        fetched or walked is named in ``errors`` and lists nothing.
        """
        try:
            tl = camera.GetTLNodeMap()
            names = []
            for node in tl.GetNodes():
                # pypylon hands back either the node or a wrapper holding it.
                try:
                    names.append(node.GetName())
                except AttributeError:
                    names.append(node.GetNode().GetName())
        except Exception as e:
            errors.append(f'transport layer nodemap failed: {type(e).__name__}: {e}')
            return {}
        return {name: self._safe_node(tl, name) for name in names}

    def _transport_layer_state(self, camera, names) -> str:
        """Read the named device transport-layer nodes for a stop's log line; never raises."""
        if camera is None:
            return 'no camera'
        try:
            tl = camera.GetTLNodeMap()
        except Exception as e:
            return f'tl_nodemap=<{type(e).__name__}: {e}>'
        return ' '.join(f'{name}={self._safe_node(tl, name)}' for name in names)

    def _grab_stop_state(self, camera) -> str:
        """Describe the grab a stop is about to end, for the pre-stop log line.

        Host-side reads first (frames stored, the stream grabber's status and
        statistics, the instant camera's buffer counts, the transport layer's
        access counters), then the stop mode,
        then a timed live read of ``AcquisitionActive`` / ``ExposureActive``:
        the values say whether an exposure or transfer was in flight, and the
        read's time says whether the camera's control channel was already
        slow before the stop. A node that is absent or unreadable is named
        in the line by ``_safe_node``'s sentinel; a nodemap that cannot be
        fetched is named by its error.
        """
        if camera is None:
            return 'no camera'
        parts = [f'frames_delivered={self.frames_delivered}']
        try:
            sg = camera.GetStreamGrabberNodeMap()
            parts.append(f'grabber_status={self._safe_node(sg, "Status")}')
            parts.extend(f'{name}={self._safe_node(sg, name)}' for name in self._DIAG_STAT_NODES)
        except Exception as e:
            parts.append(f'stream_nodemap=<{type(e).__name__}: {e}>')
        try:
            icm = camera.GetInstantCameraNodeMap()
            parts.extend(
                f'{name}={self._safe_node(icm, name)}'
                for name in ('NumQueuedBuffers', 'NumReadyBuffers', 'NumEmptyBuffers')
            )
        except Exception as e:
            parts.append(f'instant_camera_nodemap=<{type(e).__name__}: {e}>')
        parts.append(self._transport_layer_state(camera, self._TL_COUNTER_NODES))
        try:
            nm = camera.GetNodeMap()
            parts.append(f'BslAcquisitionStopMode={self._safe_node(nm, "BslAcquisitionStopMode")}')
            t0 = time.perf_counter()
            acquisition_active = self._safe_node(nm, 'AcquisitionActive')
            exposure_active = self._safe_node(nm, 'ExposureActive')
            live_read_ms = (time.perf_counter() - t0) * 1000.0
            parts.append(
                f'AcquisitionActive={acquisition_active} ExposureActive={exposure_active} '
                f'live_read_ms={live_read_ms:.1f}'
            )
        except Exception as e:
            parts.append(f'device_nodemap=<{type(e).__name__}: {e}>')
        return ' '.join(parts)

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
                acq_node = self._node('AcquisitionActive')
                if acq_node is None:
                    # Older firmware or non-Basler -- can't poll, bail
                    # quickly without warning (this is expected).
                    return False
                acq_active = bool(acq_node.GetValue())
                # A camera without ExposureActive is polled on
                # AcquisitionActive alone.
                exp_node = self._node('ExposureActive')
                exp_active = bool(exp_node.GetValue()) if exp_node is not None else False
                if not acq_active and not exp_active:
                    return True
            except Exception as e:
                _cam_log.warning(
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
        if self.active is None:
            return
        try:
            node = self._node('Status', 'stream')
            if node is None:
                _cam_log.info(f'pylon StreamGrabber.Status [{label}]: <node missing>')
                return
            status = node.GetValue()
            _cam_log.info(f'pylon StreamGrabber.Status [{label}]: {status}')
        except Exception as e:
            _cam_log.warning(f'pylon StreamGrabber.Status [{label}] read failed: {e}')

    def _start_grabbing(self) -> None:
        """Start the camera's grab loop with the configured grab strategy.

        Idempotent: if grabbing already started (pypylon 26.4.x's
        ``AcquireContinuousConfiguration + Open()`` triggers an implicit
        StartGrabbing), this method returns early. Snapshots StreamGrabber
        Status into the trace log per B23. Caps the frame rate under the
        USB link before every start (``_cap_the_frame_rate_under_the_link``);
        a cap not written fails the start. Starts the optional stats
        poller (no-op when profile_trace_enabled is false).

        A start that does not take raises: the grab-state read, the cap and
        ``StartGrabbing`` itself each fail the start with the library's
        words; a defect of ours leaves as itself.

        Grab strategy is read from ``self._grab_strategy_name`` (default
        ``LatestImageOnly``; bench overrides via
        ``imaging._set_grab_strategy``). connect() writes MaxNumBuffer and
        MaxTransferSize; NumMaxQueuedUrbs goes through
        ``imaging._set_num_max_queued_urbs`` (no env-var path).
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
            # A failed read is not "not grabbing": it fails the start.
            if self._is_grabbing():
                _cam_log.info('start_grabbing: already grabbing, skipping')
                return
            # B20 / B23: snapshot StreamGrabber.Status into the trace
            # log before StartGrabbing so post-mortem analysis can
            # correlate "weird StartGrabbing behavior" with the entry
            # state. Per Basler stream-grabber-parameters.html the node
            # is read-only and reflects grabber lifecycle state
            # (Closed / Open / Grabbing / OutOfMemory / etc.).
            self._log_stream_grabber_status('pre-StartGrabbing')
            _strategy_name = self._grab_strategy_name
            _strategy_map = {
                'LatestImageOnly': pylon.GrabStrategy_LatestImageOnly,
                'OneByOne': pylon.GrabStrategy_OneByOne,
            }
            _strategy = _strategy_map.get(_strategy_name, pylon.GrabStrategy_LatestImageOnly)
            # A cap not written fails the start: a camera never runs uncapped,
            # owing a stop that stalls.
            self._cap_the_frame_rate_under_the_link(camera)
            # The counts at the start, against the pre-stop line's, give how
            # long the camera ran before its next stop and what it delivered.
            try:
                _total_buffers = self._safe_node(
                    camera.GetStreamGrabberNodeMap(), 'Statistic_Total_Buffer_Count'
                )
            except Exception as e:
                _total_buffers = f'<{type(e).__name__}: {e}>'
            _cam_log.info(
                f'pylon StartGrabbing({_strategy_name}, ProvidedByInstantCamera): '
                f'frames_delivered={self.frames_delivered} '
                f'Statistic_Total_Buffer_Count={_total_buffers}'
            )
            camera.StartGrabbing(_strategy, pylon.GrabLoop_ProvidedByInstantCamera)
        except (genicam.GenericException, HardwareError) as e:
            _cam_log.warning(f'pylon StartGrabbing FAILED: {e}')
            raise CameraStreamStartFailedError(f'The camera stream did not start: {e}') from e
        else:
            # Mirror the IDS free-run delivered-depth line so an 8-bit run on
            # either camera prints the same "delivers=" fact: Pylon Mono8 is
            # delivered as uint8, Mono10/Mono12 as uint16 (the GetArray dtype).
            # A bundle then states the delivered dtype, not just the wire format.
            # The cache is what get_pixel_format answers after any good read or
            # set; a log line asks the camera nothing, so a format read cannot
            # fail a grab that started.
            _fmt = self._pixel_format_cache
            if _fmt is None:
                _cam_log.info(
                    '[CAM Class ] Grabbing: pixel format unknown; delivered depth unknown'
                )
            else:
                if _fmt.startswith('Mono8'):
                    _bits = 8
                elif 'Mono12' in _fmt:
                    _bits = 12
                elif 'Mono10' in _fmt:
                    _bits = 10
                else:
                    _bits = 8
                _dtype = 'uint8' if _bits <= 8 else 'uint16'
                _cam_log.info(
                    f'[CAM Class ] Grabbing: pixel_format={_fmt} delivers={_bits}-bit ({_dtype})'
                )
            # Start the periodic Pylon stats + thread-count poller.
            # No-op when profile_trace_enabled is false.
            self._start_stats_poller()

    def _cap_the_frame_rate_under_the_link(self, camera) -> None:
        """Enable ``AcquisitionFrameRate`` just under what the USB link carries, before a start.

        A daA3840 on firmware 1.1.0, stopped under ``CompleteExposure`` while
        it streams at its link limit, stalls the stop 11 s, and every stop
        after it until a power cycle; held under the link for the whole grab,
        it does not. So every start caps the rate at ``_LINK_RATE_SHARE`` x
        ``DeviceLinkThroughputLimit`` / ``PayloadSize``, clamped to the
        node's range. Where the sensor or the exposure binds, the cap sits
        above the camera's own rate and changes nothing; where the link
        binds, it costs about 5% of the frames. A start follows every payload
        change (frame size, binning, pixel format), so the cap follows the
        payload.

        The cap is left on at a disconnect: the next connect loads the
        ``Default`` user set, which replaces it.

        A camera not on USB, or without ``AcquisitionFrameRate``, is told at
        INFO and left as it is. A limit switched off is told at WARNING, and
        the cap is computed from the limit's maximum, the most the camera
        would send.

        Raises:
            HardwareError: A read the cap needs, or one of its writes, failed
                or was not answered.
        """
        try:
            transport = camera.GetDeviceInfo().GetDeviceClass()
        except genicam.GenericException as e:
            raise CameraReadFailedError(f'The camera did not give its device class: {e}') from e
        if 'Usb' not in transport:
            _cam_log.info(
                f'[CAM Class ] AcquisitionFrameRate not capped: the camera is not on USB '
                f'({transport})'
            )
            return
        if not self._offers('AcquisitionFrameRate'):
            _cam_log.info(
                '[CAM Class ] AcquisitionFrameRate not offered by this camera; its frame rate '
                'is not capped under the link'
            )
            return
        payload = self._read_first_offered(('PayloadSize',))
        if not payload:
            raise CameraReadFailedError(f'PayloadSize read {payload!r}, not a frame size in bytes')
        mode = self._read_first_offered(('DeviceLinkThroughputLimitMode',))
        if mode == 'On':
            limit = self._read_first_offered(('DeviceLinkThroughputLimit',))
        else:
            limit = self._bounds('DeviceLinkThroughputLimit')[1]
            _cam_log.warning(
                f'[CAM Class ] DeviceLinkThroughputLimitMode is {mode}: the frame rate is '
                f'capped under the limit maximum, {limit} B/s'
            )
        low, high = self._bounds('AcquisitionFrameRate')
        rate = min(max(self._LINK_RATE_SHARE * limit / payload, low), high)
        _cam_log.info(
            f'pylon AcquisitionFrameRate capped at {rate:.2f} fps under the link '
            f'({self._LINK_RATE_SHARE} x {limit} B/s / {payload} B)'
        )
        self.write_node('AcquisitionFrameRate', rate)
        self.write_node('AcquisitionFrameRateEnable', True)

    def _is_grabbing(self) -> bool:
        """Return True if the camera is currently in a grab loop; False when not connected.

        Raises:
            CameraReadFailedError: The SDK did not answer the grab state.
                The camera is kept.
        """
        camera = self.active
        if not camera:
            return False
        try:
            return camera.IsGrabbing()
        except genicam.GenericException as e:
            raise CameraReadFailedError(f'The camera did not answer its grab state: {e}') from e

    def connect(self) -> bool:
        """Try to connect to the first available basler camera"""
        try:
            # Enumerate ALL Basler devices visible to pylon BEFORE
            # CreateFirstDevice() so the "no device available" failure
            # mode can be distinguished from "found but Open() failed".
            # The list also surfaces multi-camera-bench cases (wrong
            # serial selected) and transient enumeration races (device
            # visible but Pylon hasn't claimed it yet). The connect-failure
            # handler below reads this list and the device CreateFirstDevice
            # returned: with neither there is no Basler (False, so the registry
            # tries the next driver); with either, the failure is the found
            # camera's, named by ``step`` so a post-open configuration failure
            # is never misreported as "not openable."
            _devs = ()
            _created = False
            # The failure's type is set with ``step``, from how far the
            # connect got: the camera's open, its set-up once open, or its
            # return to the bus after a reset.
            fault = CameraUsbOpenFailedError
            step = 'the device lookup'
            try:
                _devs = pylon.TlFactory.GetInstance().EnumerateDevices()
                _log_cam(
                    'info',
                    f'[CAM Class ] pylon TlFactory.EnumerateDevices() returned '
                    f'{len(_devs)} device(s)',
                )
                for _i, _d in enumerate(_devs):
                    _log_cam('info', f'[CAM Class ]   device[{_i}]: {describe_device_info(_d)}')
            except Exception as _e_enum:
                _log_cam(
                    'warning', f'[CAM Class ] pylon TlFactory.EnumerateDevices() failed: {_e_enum}'
                )

            if self._use_camera_emulation:
                # PYLON_CAMEMU adds the emulator beside any real Basler on the
                # bus, and a real one enumerates first: the emulator is
                # selected by its device class, so a real camera is never
                # opened and configured as the emulator.
                emulator = pylon.DeviceInfo()
                emulator.SetDeviceClass(_EMULATOR_DEVICE_CLASS)
                p_device = pylon.TlFactory.GetInstance().CreateFirstDevice(emulator)
            else:
                p_device = pylon.TlFactory.GetInstance().CreateFirstDevice()
            _created = True
            camera = pylon.InstantCamera(p_device)
            device_info = camera.GetDeviceInfo()
            if self._resets_at_connect(device_info):
                # The device reset is the one created above, so which camera
                # is connected does not change. A camera told to reset is
                # never configured as it was: it may reboot under the session.
                serial = device_info.GetSerialNumber()
                step = 'the open before its reset'
                camera.Open()
                fault = CameraSetupFailedError
                step = 'its reset'
                # Active for the reset's one write alone: nothing is registered
                # on it, and it is never active again once destroyed.
                self.active = camera
                if not self.write_node('DeviceReset', self.RUN, if_present=True):
                    camera.Close()
                else:
                    self.active = None
                    camera.DestroyDevice()
                    reset_at = time.monotonic()
                    fault = CameraUsbNotFoundError
                    step = 'its return after its reset'
                    returned = self._wait_for_return_after_reset(serial, reset_at)
                    fault = CameraUsbOpenFailedError
                    step = 'the device lookup after its reset'
                    camera = pylon.InstantCamera(
                        pylon.TlFactory.GetInstance().CreateDevice(returned)
                    )
                    _cam_log.info(
                        f'[CAM Class ] camera {serial} reset at connect; back after '
                        f'{time.monotonic() - reset_at:.1f} s'
                    )
                fault = CameraUsbOpenFailedError
            step = 'its registration'
            self.active = camera
            # Ensure previous removal flag does not persist across a new connection
            self._device_removed = False
            camera.RegisterConfiguration(
                pylon.AcquireContinuousConfiguration(),
                pylon.RegistrationMode_ReplaceAll,
                pylon.Cleanup_Delete,
            )
            # The removal handler is how an unplugged camera is noticed while
            # it is idle: is_connected() reads the flag it sets and asks the
            # SDK nothing. A camera connected without it would read as
            # connected after it is unplugged, so the connect is refused.
            try:
                camera.RegisterConfiguration(
                    _CameraRemovalHandler(self), pylon.RegistrationMode_Append, pylon.Cleanup_Delete
                )
            except Exception as e:
                step = 'its removal handler'
                raise RuntimeError(
                    'the camera-removal handler could not be registered, so an '
                    f'unplugged camera would still read as connected: {e}'
                ) from e

            self.cam_image_handler = ImageHandler(self)
            # Fresh handler starts with an empty dispatch list; re-push any
            # durable listeners so a reconnect on the same instance does not
            # silently stop delivering frames to recording / plugins.
            self._reapply_frame_callbacks()
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

            step = 'the open'
            camera.Open()
            fault = CameraSetupFailedError

            # Start gate, SDK-side enforcement: pypylon 26.4.x's
            # AcquireContinuousConfiguration can auto-StartGrabbing from
            # inside Open(). connect() must return NOT grabbing, so stop it
            # immediately -- one rule, every SDK. Doing it here (before the
            # MaxNumBuffer / MaxTransferSize block below) also means those
            # StreamGrabber node writes land while stopped, which is where
            # the SDK wants them. IsGrabbing/StopGrabbing are core
            # InstantCamera methods present on every transport (unlike the
            # optional nodes below), so no node-availability guard is needed;
            # a genuine SDK failure propagates to connect()'s own handler.
            step = 'its post-open configuration'
            if camera.IsGrabbing():
                self._timed_stop_grabbing(camera, 'post-Open')
                _log_cam('info', '[CAM Class ] post-Open: stopped SDK auto-start grab (start gate)')

            # MaxNumBuffer cap. Applied post-Open() -- earliest window
            # before pypylon 26.4.x's AcquireContinuousConfiguration
            # auto-starts grabbing and locks the node. Bench-validated
            # 2026-05-08 (Windows dart M, sensor-max Mono8): cap=3 and
            # SDK-default both run at 0% fail rate / 0 resyncs/sec, so
            # the cap restores the original Windows non-paged-pool bound
            # at zero observed perf cost.
            try:
                self.write_node('MaxNumBuffer', _DEFAULT_MAX_NUM_BUFFER, 'instant_camera')
                actual = self._read_first_offered(('MaxNumBuffer',), 'instant_camera')
                _log_cam(
                    'info',
                    f'[CAM Class ] MaxNumBuffer cap applied post-Open: '
                    f'requested={_DEFAULT_MAX_NUM_BUFFER} actual={actual}',
                )
            except HardwareError as e:
                _log_cam(
                    'warning',
                    f'[CAM Class ] MaxNumBuffer cap post-Open failed: {e}',
                )

            # Pre-allocate chunk node-map pool to match the buffer pool.
            # Per Basler grabchunkimage sample: node maps are otherwise
            # created lazily on StartGrabbing(), stalling first-frame
            # delivery while the SDK parses XML. Sized to MaxNumBuffer so
            # every in-flight grab buffer has a paired pre-built node map.
            try:
                _mnb_actual = self._read_first_offered(('MaxNumBuffer',), 'instant_camera')
                self.write_node('StaticChunkNodeMapPoolSize', _mnb_actual, 'instant_camera')
                _log_cam('info', f'[CAM Class ] StaticChunkNodeMapPoolSize set to {_mnb_actual}')
            except HardwareError as e:
                _log_cam(
                    'warning',
                    f'[CAM Class ] StaticChunkNodeMapPoolSize set failed '
                    f'(node may be unavailable on this transport): {e}',
                )

            # MaxTransferSize. Pin the per-USB-transfer byte budget so it does
            # not fall back to a platform default that is unstable here (OSX /
            # Pylon Viewer uses 256 KB). Set in the same post-Open window as
            # MaxNumBuffer; the requested-vs-actual readback is the tell that
            # the window was early enough, since the StreamGrabber doc requires
            # this before grab. USB3-only -- the node is absent on GigE.
            try:
                if not self.write_node(
                    'MaxTransferSize', _DEFAULT_MAX_TRANSFER_SIZE, 'stream', if_present=True
                ):
                    _log_cam(
                        'debug',
                        '[CAM Class ] MaxTransferSize node absent post-Open '
                        '(non-USB transport or grabber not open); left at default',
                    )
                else:
                    _mts_actual = self._read_first_offered(('MaxTransferSize',), 'stream')
                    _log_cam(
                        'info',
                        f'[CAM Class ] MaxTransferSize applied post-Open: '
                        f'requested={_DEFAULT_MAX_TRANSFER_SIZE} actual={_mts_actual}',
                    )
            except HardwareError as e:
                _log_cam(
                    'warning',
                    f'[CAM Class ] MaxTransferSize set failed: {e}',
                )

            # Store device identity if possible
            self._device_firmware = None
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

                    _log_cam(
                        'info', f'[CAM Class ] Pylon SDK version: {pylon_versions()["version"]}'
                    )

                    # Transport + device class identify the kernel
                    # driver stack Pylon is routing through -- useful
                    # when the runtime SDK says one thing but Device
                    # Manager shows a stale WinUSB/USB3Vision driver.
                    try:
                        _log_cam(
                            'info',
                            f'[CAM Class ] Transport: {dev_info.GetTLType()} '
                            f'/ DeviceClass: {dev_info.GetDeviceClass()}',
                        )
                    except Exception as e:
                        _log_cam('info', f'[CAM Class ] TLType/DeviceClass unavailable: {e}')

                    _log_cam('info', f'[CAM Class ] Camera Serial Number: {self._device_serial}')

                    firmware_node = self._node_in(nm, 'DeviceFirmwareVersion')
                    if firmware_node is None:
                        _log_cam(
                            'info', '[CAM Class ] DeviceFirmwareVersion not offered by this camera'
                        )
                    else:
                        self._device_firmware = firmware_node.ToString()
                        _log_cam(
                            'info', f'[CAM Class ] Camera Firmware Version: {self._device_firmware}'
                        )

                    # Current pixel format + resolution + binning drive
                    # the DMA buffer footprint -- critical context for
                    # memory / throughput analysis.
                    try:

                        def _node_text(name, absent):
                            node = self._node_in(nm, name)
                            return absent if node is None else node.ToString()

                        pix = _node_text('PixelFormat', '?')
                        w = _node_text('Width', '?')
                        h = _node_text('Height', '?')
                        bh = _node_text('BinningHorizontal', 1)
                        bv = _node_text('BinningVertical', 1)
                        _log_cam(
                            'info',
                            f'[CAM Class ] Pixel format: {pix}, '
                            f'Resolution: {w}x{h}, Binning: {bh}x{bv}',
                        )
                    except Exception as e:
                        _log_cam('info', f'[CAM Class ] Pixel/resolution/binning unavailable: {e}')

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
            # connect() returns CONFIGURED but NOT grabbing; the single
            # start fires later via open_and_start() (the start gate).

            _log_cam('info', '[CAM Class ] Connected to Pylon camera')
            return True

        except Exception as ex:
            self.active = None
            self._stop_image_grab_worker()
            if not (_created or _devs):
                # No Basler is not a fault: the registry tries the next driver
                # and owns the absence verdict, and the enumeration line above
                # is this driver's one report of it. What pylon said is kept
                # here, below it.
                _cam_log.debug(
                    f'[CAM Class ] Pylon camera connect failed (no device enumerated): {ex}'
                )
                return False
            # A Basler IS on the bus: an open that fails is typically one held
            # by another application (e.g. pylon Viewer) or a claim race; a
            # failure after the open is the camera's set-up, which no cable or
            # other program explains. Either must survive an error-level log
            # sweep and reach the bring-up as the found camera's failure.
            failure = f'Basler camera found, {step} failed: {type(ex).__name__}: {ex}'
            _cam_log.error(
                f'[CAM Class ] {failure}', exc_info=not isinstance(ex, genicam.RuntimeException)
            )
            raise fault(failure) from ex

    def _resets_at_connect(self, device_info) -> bool:
        """Whether ``connect`` resets this camera before configuring it.

        Every Basler USB camera is reset: a 1.1.0 daA3840 left stalling its
        stops 11 s by an earlier session stays so through a reconnect, and
        a reset clears it for a while, as a power cycle does. The emulator
        (``BaslerCamEmu``) has no stall to clear, and no GigE body's reset
        has been measured. The one place the choice is made, so turning the
        reset off, or limiting it to some cameras, is a change here alone.

        Raises:
            genicam.GenericException: The device did not give its class.
        """
        return 'Usb' in device_info.GetDeviceClass()

    def _wait_for_return_after_reset(self, serial: str, reset_at: float):
        """Wait for the camera ``serial``, reset at ``reset_at``, to leave the
        bus and come back; return its device info once it has settled.

        Raises:
            TimeoutError: It had not left and come back ``_RESET_RETURN_S``
                after its reset.
        """
        factory = pylon.TlFactory.GetInstance()
        deadline = reset_at + self._RESET_RETURN_S
        left = False
        while time.monotonic() < deadline:
            listed = next(
                (d for d in factory.EnumerateDevices() if d.GetSerialNumber() == serial), None
            )
            if listed is None:
                left = True
            elif left:
                time.sleep(self._RESET_SETTLE_S)
                return listed
            time.sleep(self._RESET_POLL_S)
        how = 'left the bus and did not come back' if left else 'never left the bus'
        raise TimeoutError(
            f'the camera {serial} did not come back after LumaViewPro reset it: it {how} '
            f'in the {self._RESET_RETURN_S:.0f} s waited'
        )

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
                f'[CAM Class ] _stop_image_grab_worker raised: {e}; continuing teardown'
            )

    def get_all_temperatures(self) -> dict:
        """Return {selector: degC, ...} per DeviceTemperatureSelector entry.

        {} for a camera without the temperature nodes. See
        ``Camera.get_all_temperatures``.
        """
        if not self.active:
            raise PartAbsentError('Camera temperature read: no camera is active')

        try:
            if not self._has_temperature_nodes():
                return {}

            temp = self._node('DeviceTemperature')

            temps: dict[str, float] = {}

            # Iterate all available selector entries
            for name in self._entries('DeviceTemperatureSelector'):  # e.g. "FpgaCore"
                # Select this temperature source
                self.write_node('DeviceTemperatureSelector', name)

                # Read temperature
                if genicam.IsReadable(temp):
                    temps[name] = temp.GetValue()

            # The camera has the nodes, so it has a sensor: an empty answer
            # would read as "no sensor" to every caller.
            if not temps:
                raise CameraReadFailedError('no listed temperature sensor was readable')
            return temps
        except Exception as e:
            # Intentionally NO disconnect teardown here: this getter used to
            # latch the device-removed flag (no reset short of a full
            # reconnect) on a single possibly-transient node read, which
            # silently killed the live stream. Removal detection is owned by
            # the SDK removal callback and the grab loop's definitive
            # DEVICE_NOT_FOUND / consecutive-failure paths, which still fire
            # on a real unplug; a read failure here only means this VALUE is
            # unavailable right now.
            raise CameraReadFailedError(
                f'Camera temperature read failed: {type(e).__name__}: {e}'
            ) from e

    def supports_temperature(self) -> bool:
        """True if the camera exposes the temperature selector and reading nodes."""
        if not self.active:
            return False
        try:
            return self._has_temperature_nodes()
        except Exception as e:
            # No disconnect teardown, as in the read above.
            raise CameraReadFailedError(f'Temperature probe failed: {type(e).__name__}: {e}') from e

    def _has_temperature_nodes(self) -> bool:
        return (
            self._node('DeviceTemperatureSelector') is not None
            and self._node('DeviceTemperature') is not None
        )

    def supports_clock_latch(self) -> bool:
        """True if the camera exposes the timestamp latch command and its value."""
        if not self.active:
            return False
        try:
            return (
                self._node('TimestampLatch') is not None
                and self._node('TimestampLatchValue') is not None
            )
        except Exception as e:
            # No disconnect teardown, as in the probes above.
            raise CameraReadFailedError(f'Clock latch probe failed: {type(e).__name__}: {e}') from e

    def latch_clock(self) -> None:
        """Execute ``TimestampLatch``. See ``Camera.latch_clock``.

        The latch is the first half of a clock read, so its failure is the
        read's. A failure raises and does not tear the camera down, as a
        failed read does not: a transient one must not latch a removal.

        Raises:
            CameraReadFailedError: The camera did not take or answer the
                latch, or no camera is open.
        """
        try:
            self.write_node('TimestampLatch', self.RUN)
        except (CameraWriteFailedError, CameraNoAnswerError) as e:
            raise CameraReadFailedError(f'TimestampLatch failed: {e}') from e

    def latched_clock_ticks(self) -> int:
        """Read ``TimestampLatchValue``. See ``Camera.latched_clock_ticks``."""
        try:
            return int(self._node('TimestampLatchValue').GetValue())
        except Exception as e:
            raise CameraReadFailedError(
                f'TimestampLatchValue read failed: {type(e).__name__}: {e}'
            ) from e

    def get_sdk_info(self) -> dict:
        """Basler pylon SDK provenance (name + versions) for diagnostics.

        Returns:
            dict: ``{'name': 'Basler pylon', 'version': <dotted SDK version>,
            'pypylon_version': <binding version>}``, from ``pylon_versions``.
        """
        return {'name': 'Basler pylon', **pylon_versions()}

    def init_camera_config(self) -> None:
        """Apply Etaluma's canonical camera configuration once on connect.

        Loads the `Default` user set, sets pixel format to Mono8, sets
        explicit gain (0 dB), sets the image orientation (`ReverseY` and
        `ReverseX`, ``orientation_owed``), enables validity
        chunks (B28-guarded), runs the auto-gain init defaults, and sets
        the stop mode that finishes the exposure in flight.
        Wrapped in `update_camera_config()` so the caller does not
        need a separate stop/start cycle.

        Each step runs whatever an earlier one did: a step the camera did
        not take is logged and named in ``config_not_applied``, which
        bring-up reports, and the camera stays connected with the rest
        applied. A RuntimeException is one such step: one failed node access
        is not evidence the camera is gone, and a camera really removed is
        marked so by the SDK's removal callback. The orientation is not such
        a step: a camera whose frame could arrive either way up does not
        come up.

        Raises:
            CameraReadFailedError: The config guard's grab-state read failed,
                before any step ran; or the orientation could not be read.
            CameraWriteFailedError: The orientation was not taken.
        """
        if not self.active:
            return
        self.config_not_applied = ()
        self.owed_reverse = NOTHING_OWED

        def load_default_user_set():
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
            self.write_node('UserSetSelector', 'Default')
            self.write_node('UserSetLoad', self.RUN)

        def free_run():
            # Defense in depth -- 'Default' user set is documented to
            # leave AcquisitionMode=Continuous + TriggerMode=Off
            # (free-run is the Basler default). Re-assert explicitly
            # so a firmware bug, factory misconfig, or future user-set
            # change that leaks a different default surfaces here
            # rather than silently breaking free-run acquisition.
            # A camera model that does not expose a node skips it.
            self.write_node('AcquisitionMode', 'Continuous', if_present=True)
            if self._node('TriggerSelector') is None:
                return

            # Per Basler doc free-run-image-acquisition.html:
            # "Repeat the steps above for all available trigger
            # types." A camera that exposes AcquisitionStart /
            # FrameBurstStart / etc. needs all of them off, not
            # just FrameStart, or a stray non-Off type leaks
            # through and blocks free-run.
            def trigger_off(entry):
                self.write_node('TriggerSelector', entry)
                self.write_node('TriggerMode', 'Off', if_present=True)

            self._write_each('TriggerMode Off', self._entries('TriggerSelector'), trigger_off)

        steps = [
            ('the Default user set', load_default_user_set),
            ('free-run acquisition', free_run),
            # Enable per-frame chunks for gain/exposure/identity/timestamp.
            # Must happen here -- ChunkModeActive is locked while grabbing
            # (genicam.AccessException). Settings persist across
            # subsequent stop/start cycles.
            # The timestamp tick frequency is read with them, for a camera
            # whose Timestamp chunk turned on.
            ('the validity chunks', self._enable_validity_chunks),
            ('the Mono8 pixel format', lambda: self.set_pixel_format(pixel_format='Mono8')),
            ('auto-gain off', lambda: self.auto_gain(state=False)),
            # Set explicit gain -- camera default after UserSetLoad is undefined
            ('gain 0 dB', lambda: self.gain(0.0)),
        ]
        steps_after = []
        if not self._use_camera_emulation:
            steps_after.append(('the auto-gain and focus defaults', self.init_auto_gain_focus))
        steps_after += [
            ('exposure 10 ms', lambda: self.exposure_t(exposure_ms=10)),
            # 1900x1900: bench-witnessed driver-init default. Both
            # production cameras (a2A3536 ace 2 and daA3840 dart)
            # support this size. It is the default, not a ceiling:
            # any size up to the full sensor can be set after
            # bring-up. Falsify: if a future
            # camera body's sensor doesn't reach 1900 in either
            # axis, the SetValue will clamp + warn from set_frame_size.
            ('the 1900x1900 frame', lambda: self.set_frame_size(w=1900, h=1900)),
            ('the stop mode', self._stop_after_the_exposure),
        ]
        # Each step's own failure is _apply_config_steps'; the guard's own
        # entry and exit (the grab-state read, the stop, the restart) are not
        # a step, and a failure there reaches connect()'s handler, which
        # reports a found camera whose configuration failed. So does the
        # orientation's, which runs between the steps, in its place.
        with self.update_camera_config():
            not_applied = self._apply_config_steps(steps)
            self.owed_reverse = orientation_owed(
                lambda name: self.read_node(name, 'device'),
                self._reverse_is_writable,
                self._write_reverse,
            )
            self.config_not_applied = not_applied + self._apply_config_steps(steps_after)

    def _stop_after_the_exposure(self) -> None:
        """Have every later stop finish the exposure in flight (``BslAcquisitionStopMode``).

        The ``Default`` user set leaves the mode the body shipped with, and on
        a daA3840 at firmware 1.1.0 that is ``AbortExposure``: there a stop
        can hang 11 s in the camera's own ``AcquisitionStop``, and once one
        has, nearly every later stop does until power-off. Under
        ``CompleteExposure`` the same body stopped 321 times without a
        hang. A stop then waits at most the exposure in flight.

        A camera without the node, or one that does not offer
        ``CompleteExposure`` or will not take a write of it, keeps its own
        mode: that is told at INFO, naming the mode kept, and is not a step
        left undone, so it returns, never raises.

        Raises:
            HardwareError: The camera did not answer the node's access, or
                refused the write of an offered value; the step is then
                named in ``config_not_applied``.
        """
        node = self._node('BslAcquisitionStopMode')
        if node is None:
            _cam_log.info(
                '[CAM Class ] BslAcquisitionStopMode not offered by this camera; '
                'its stops run in its own mode'
            )
            return
        offered = self._entries('BslAcquisitionStopMode')
        if 'CompleteExposure' not in offered or not genicam.IsWritable(node):
            kept = self._safe_node(self._nodemap('device'), 'BslAcquisitionStopMode')
            _cam_log.info(
                f'[CAM Class ] BslAcquisitionStopMode kept at {kept}: this camera does not '
                f'take CompleteExposure (offers {list(offered)})'
            )
            return
        _cam_log.info("pylon BslAcquisitionStopMode.SetValue('CompleteExposure')")
        self.write_node('BslAcquisitionStopMode', 'CompleteExposure')

    def _reverse_is_writable(self, name: str) -> bool:
        """Whether the camera takes a write of the reverse node ``name`` (``orientation_owed``).

        Raises:
            CameraReadFailedError: The camera did not answer the node's access.
        """
        try:
            return genicam.IsWritable(self._node(name))
        except genicam.GenericException as e:
            raise CameraReadFailedError(
                f'The camera did not answer the access of {name}: {e}'
            ) from e

    def _write_reverse(self, name: str, value: bool) -> None:
        """Write the reverse node ``name`` (``orientation_owed``).

        Raises:
            CameraWriteFailedError: The camera refused the write.
            CameraNoAnswerError: The camera did not answer it.
        """
        self.write_node(name, value)

    # Frame-identity chunk candidates in preference order. Different
    # Basler models advertise different per-frame identity chunks
    # (data-chunks.html). We probe ChunkSelector and enable the first
    # candidate the camera advertises; the read-side alias maps both
    # back to the same 'FrameID' dict key for the trace.
    _FRAME_IDENTITY_CHUNK_CANDIDATES = ('FrameID', 'Framecounter')

    # Always-enable chunks (no per-camera fallback; both production
    # USB3 cameras and dart M GigE advertise these). Timestamp is the
    # camera-side tick at which the frame started exposing; surfaced
    # through the chunks dict for downstream metadata writes (TIFF Plane
    # fields, session manifest), and a recording's frame times.
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
        enabled. A chunk the camera does not advertise, or a chunk node it
        does not offer, is logged and skipped; the validity layer falls
        back to skip_frames for each chunk not enabled. Frame-identity is
        trace-only (frame_validity validates gain and exposure); skipping
        it does not break validity.

        Sets ``self.timestamp_tick_frequency_hz`` only when the Timestamp
        chunk turned on, None otherwise: a tick rate is the clock frames
        carry their start on, and the camera's clock latch is published
        only with one (``ScopeCapabilities.camera_has_clock_latch``), so a
        camera whose frames carry no start never has a frame judged by one.

        Raises:
            HardwareError: The camera did not answer, or refused, a chunk
                node's read or write; each chunk is still tried, and the
                step is then named in ``config_not_applied``.
        """
        # B28: ChunkModeActive is locked while grabbing per Basler
        # data-chunks.html. The caller contract (docstring above)
        # requires "not grabbing", but a future caller could violate
        # it. Guard rather than silently corrupt -- frame_validity
        # falls back to skip_frames calibration when chunks aren't
        # enabled, so refusing the write is the safe default.
        if self.is_grabbing():
            # The chunks, and so the tick frequency, are left as they were.
            _cam_log.warning(
                '[CAM Class ] _enable_validity_chunks called while '
                'grabbing; ChunkModeActive is locked. Skipping write. '
                'frame_validity will fall back to skip_frames '
                'calibration. Stop grabbing before enabling chunks.'
            )
            return
        self.timestamp_tick_frequency_hz = None
        advertised = self._advertised_chunks()
        if advertised is None:
            return  # _advertised_chunks already logged
        if not self.write_node('ChunkModeActive', True, if_present=True):
            _cam_log.warning(
                '[CAM Class ] ChunkModeActive not offered; '
                'frame_validity will fall back to skip_frames calibration'
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

        def enable(sel):
            if sel not in advertised:
                _cam_log.warning(
                    f'[CAM Class ] Chunk{sel} not advertised by this '
                    f'camera; frame_validity will fall back to skip_frames '
                    f'for that source'
                )
                return
            self.write_node('ChunkSelector', sel)
            if not self.write_node('ChunkEnable', True, if_present=True):
                _cam_log.warning(
                    f'[CAM Class ] ChunkEnable not offered, so Chunk{sel} is not enabled; '
                    f'frame_validity will fall back to skip_frames for that source'
                )
                return
            if sel == 'Timestamp':
                self._read_timestamp_tick_frequency()

        self._write_each('Chunk enable', targets, enable)

    @staticmethod
    def _write_each(what: str, items, write) -> None:
        """``write(item)`` for every item, each tried whatever the one before met.

        One item the camera refuses or does not answer must not cost the
        items after it (a refused ChunkGain leaving Timestamp off, and with
        it the clock latch); the step still fails, naming every item not
        written, so it is named in ``config_not_applied``.

        Raises:
            CameraWriteFailedError: An item's write failed; the words name
                each item and its failure.
        """
        failed = []
        for item in items:
            try:
                write(item)
            except HardwareError as e:
                failed.append(f'{item}: {e}')
        if failed:
            raise CameraWriteFailedError(f'{what} not written for ' + '; '.join(failed))

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
        unrecognized so downstream metadata is honest about the unknown.
        """
        camera = self.active
        if camera is None:
            return
        for node_name in ('GevTimestampTickFrequency', 'TimestampTickFrequency'):
            try:
                node = self._node(node_name)
                if node is not None and genicam.IsReadable(node):
                    self.timestamp_tick_frequency_hz = int(node.GetValue())
                    logger.info(
                        f'[CAM Class ] Timestamp tick frequency '
                        f'{self.timestamp_tick_frequency_hz} Hz ({node_name})'
                    )
                    return
            except Exception as e:
                # What a failed tick read publishes is decided with every
                # other range field's failed read; until then it is told.
                _cam_log.warning(f'[CAM Class ] {node_name} read failed: {e}')
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

    def _advertised_chunks(self) -> set | None:
        """The ChunkSelector entries the camera advertises, or None when it
        has no ChunkSelector. Read by _enable_validity_chunks.

        Raises:
            CameraReadFailedError: The camera did not answer the read.
            CameraNoAnswerError: Nor in time.
        """
        if self._node('ChunkSelector') is None:
            _cam_log.warning(
                '[CAM Class ] ChunkSelector node missing; '
                'frame_validity will fall back to skip_frames calibration'
            )
            return None
        return set(self._entries('ChunkSelector'))

    def set_device_link_throughput_limit(
        self,
        mode: str,
        value_bps: int | None = None,
    ) -> None:
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
        to its range read live, with a warning log; the SDK would
        otherwise raise OutOfRangeException.

        Args:
            mode: ``'On'`` or ``'Off'``. Case-sensitive (matches Pylon
                enum entry symbolic names).
            value_bps: Throughput cap in bytes per second when
                ``mode='On'``. Ignored when ``mode='Off'``. If None
                while ``mode='On'``, only the mode is changed and the
                existing limit value is preserved.

        Raises:
            CameraWriteFailedError: The mode is not 'On' or 'Off', the
                camera does not offer a node, or it refused or failed the
                write. The camera is not marked removed: one failed write is
                not evidence it is gone.
            CameraReadFailedError: The limit's range could not be read;
                nothing was written.
            CameraNoAnswerError: The camera did not answer in time.
        """
        if mode not in ('On', 'Off'):
            raise CameraWriteFailedError(
                f"set_device_link_throughput_limit: mode must be 'On' or 'Off', not {mode!r}; "
                f'nothing was written'
            )
        limit = None
        if mode == 'On' and value_bps is not None:
            limit = self._clamp_dltl_value_bps(int(value_bps))
        _cam_log.info(
            f'pylon DeviceLinkThroughputLimitMode={mode} DeviceLinkThroughputLimit={limit}'
        )
        self.write_node('DeviceLinkThroughputLimitMode', mode)
        if limit is not None:
            self.write_node('DeviceLinkThroughputLimit', limit)

    def _clamp_dltl_value_bps(self, value_bps: int) -> int:
        """Clamp a DLTL value to the camera's supported range, read live.

        Raises:
            CameraReadFailedError: The range could not be read, or the
                camera does not offer the node.
            CameraNoAnswerError: The camera did not answer the read.
        """
        lo, hi = (int(bound) for bound in self._bounds('DeviceLinkThroughputLimit'))
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

    def set_num_max_queued_urbs(self, value: int) -> None:
        """Set StreamGrabber NumMaxQueuedUrbs (USB3-only bench knob).

        Per Basler `stream-grabber-parameters.html`: count of in-flight USB
        Request Blocks. Doc's named lever for "insufficient system memory"
        (status 0xe2010130 / 0xe2100001); decreasing reduces kernel URB
        allocation pressure on memory-constrained hosts. SDK clamps to node
        range.

        Raises:
            HardwareError: As ``write_node``; a GigE camera does not offer
                the node.
        """
        _cam_log.info(f'pylon NumMaxQueuedUrbs.SetValue({int(value)})')
        self.write_node('NumMaxQueuedUrbs', int(value), 'stream')

    _GRAB_STRATEGY_NAMES = ('LatestImageOnly', 'OneByOne')

    def set_grab_strategy(self, name: str) -> bool:
        """Set the GrabStrategy used by the next start_grabbing() call.

        Pure attribute write: takes effect on the next start_grabbing()
        call, not on the active grab loop. To switch strategies on a
        live camera: set_grab_strategy(new) -> stop_grabbing() ->
        start_grabbing().

        Returns True on accepted name. Raises ValueError on an unknown
        strategy name.
        """
        if name not in self._GRAB_STRATEGY_NAMES:
            raise ValueError(
                f'set_grab_strategy: expected one of {self._GRAB_STRATEGY_NAMES}, got {name!r}'
            )
        self._grab_strategy_name = name
        _cam_log.info(f'pylon grab strategy set to {name!r} (applies on next start_grabbing())')
        return True

    _BANDWIDTH_RESERVE_MODES = ('Default', 'Performance')

    def set_bandwidth_reserve_mode(self, mode: str) -> None:
        """Set BandwidthReserveMode (GigE only).

        Per Basler `network-related-parameters.md`:
          - ``'Default'``: reserves bandwidth for packet retransmits.
          - ``'Performance'``: all bandwidth to transmit; minimal retransmit.

        Load-bearing for fps on GigE; dmA3536-9gm goes 9.3 -> 9.5 fps with
        ``Performance``.

        Raises:
            CameraWriteFailedError: The mode is not one of
                ``_BANDWIDTH_RESERVE_MODES``, the camera does not offer the
                node (a USB3 camera), or it refused the write.
            CameraNoAnswerError: The camera did not answer in time.
        """
        if mode not in self._BANDWIDTH_RESERVE_MODES:
            raise CameraWriteFailedError(
                f'set_bandwidth_reserve_mode: mode must be one of '
                f'{self._BANDWIDTH_RESERVE_MODES}, not {mode!r}; nothing was written'
            )
        _cam_log.info(f'pylon BandwidthReserveMode.SetValue({mode!r})')
        self.write_node('BandwidthReserveMode', mode)

    def set_gev_packet_size(self, size_bytes: int) -> None:
        """Set GevSCPSPacketSize (GigE only).

        Per Basler `network-related-parameters.md`: 1500 = standard MTU,
        9000 = jumbo. Larger packets cut host CPU + packet rate but require
        OS-level jumbo-frame config.

        Raises:
            CameraWriteFailedError: ``size_bytes`` is not a positive int,
                the camera does not offer the node (a USB3 camera), or it
                refused the write.
            CameraNoAnswerError: The camera did not answer in time.
        """
        if not is_integer_number(size_bytes) or size_bytes <= 0:
            raise CameraWriteFailedError(
                f'set_gev_packet_size: size_bytes must be a positive int, not {size_bytes!r}; '
                f'nothing was written'
            )
        _cam_log.info(f'pylon GevSCPSPacketSize.SetValue({size_bytes})')
        self.write_node('GevSCPSPacketSize', size_bytes)

    def set_gev_inter_packet_delay(self, delay_ticks: int) -> None:
        """Set GevSCPD inter-packet delay in clock ticks (GigE only).

        Per Basler `network-related-parameters.md`: throttles per-camera
        bandwidth on a shared link.

        Raises:
            CameraWriteFailedError: ``delay_ticks`` is not a non-negative
                int, the camera does not offer the node (a USB3 camera), or
                it refused the write.
            CameraNoAnswerError: The camera did not answer in time.
        """
        if not is_integer_number(delay_ticks) or delay_ticks < 0:
            raise CameraWriteFailedError(
                f'set_gev_inter_packet_delay: delay_ticks must be a non-negative int, '
                f'not {delay_ticks!r}; nothing was written'
            )
        _cam_log.info(f'pylon GevSCPD.SetValue({delay_ticks})')
        self.write_node('GevSCPD', delay_ticks)

    def set_pixel_format(self, pixel_format: str) -> None:
        """Set the camera pixel format (``Camera.set_pixel_format``).

        Args:
            pixel_format: Pylon pixel format symbolic name (e.g. 'Mono8').

        Raises:
            CameraWriteFailedError: The camera does not offer the format, or
                did not take it. The camera is not marked removed: one
                failed write is not evidence it is gone.
            CameraReadFailedError: The read of the supported formats, or the
                grab state the write needs, failed; nothing was written.
            CameraNoAnswerError: The read of the supported formats or the
                PixelFormat write timed out; the camera is not marked
                disconnected, and the caller may send it again.
        """
        if pixel_format not in self._entries('PixelFormat'):
            raise CameraWriteFailedError(
                f'The camera does not offer the pixel format {pixel_format}; nothing was written'
            )
        # Short-circuit redundant SetValue: update_camera_config() bounces
        # the grab loop. init_camera_config + channel-select hit this
        # setter multiple times with the same target across threads.
        if self._already('PixelFormat', pixel_format):
            _cam_log.info(
                f'pylon PixelFormat.SetValue({pixel_format!r}) short-circuited (already active)'
            )
            self._pixel_format_cache = pixel_format
            return
        _cam_log.info(f'pylon PixelFormat.SetValue({pixel_format!r}) (geometry-realloc)')
        with self.update_camera_config():
            self.write_node('PixelFormat', pixel_format)
            # Inside the config guard, while the grab is stopped: its
            # __exit__ restarts grabbing and logs the delivered format
            # from this cache, which would otherwise still name the old one.
            self._pixel_format_cache = pixel_format

    def get_pixel_format(self) -> str | None:
        """Return active PixelFormat (e.g. 'Mono8'); None when not connected.

        Served from the cache populated on first read and on every
        set_pixel_format(); PixelFormat only changes through that setter, so
        the cache stays valid. The format is read once per saved frame, so a
        live GenICam node read here would touch the SDK on every capture.

        Raises:
            CameraReadFailedError: The camera did not answer the read, a
                timeout included. The camera is kept.
        """
        if not self.active:
            return None

        if self._pixel_format_cache is not None:
            return self._pixel_format_cache

        try:
            value = self._node('PixelFormat').GetValue()
        except genicam.GenericException as e:
            # Intentionally NO disconnect teardown: latching device-removed
            # from a possibly-transient node read silently killed the live
            # stream until a manual reconnect. Removal stays owned by the SDK
            # removal callback + the grab loop's definitive paths (still fire
            # on a real unplug); this failure only means the format is
            # unreadable right now.
            raise CameraReadFailedError(
                f'The camera did not answer a read of PixelFormat: {e}'
            ) from e
        self._pixel_format_cache = value
        return value

    def get_supported_pixel_formats(self) -> tuple:
        """Return every PixelFormat the camera advertises; () when not connected.

        Raises:
            CameraReadFailedError: The camera did not answer the read, a
                timeout included. A write that needs the list is refused
                for it, nothing sent.
        """
        if not self.active:
            return ()
        try:
            return self._node('PixelFormat').GetSymbolics()
        except genicam.GenericException as e:
            # Intentionally NO disconnect teardown (latching device-removed on
            # a transient node read killed the stream); removal is owned by
            # the SDK removal callback + the grab loop's definitive paths.
            raise CameraReadFailedError(
                f'The camera did not answer a read of PixelFormat entries: {e}'
            ) from e

    def _set_hardware_binning(self, size: int) -> None:
        """Set camera pixel binning size, summing the binned pixels.

        Args:
            size: Binning factor (1 = no binning; up to 4 supported).

        Raises:
            CameraWriteFailedError: ``size`` is outside 1-4, or the camera
                did not take a binning write. The camera is not marked
                removed: one failed write is not evidence it is gone.
            CameraReadFailedError: The grab-state read the config guard needs
                failed; nothing was written.
            CameraNoAnswerError: A binning write timed out. Transient: the
                camera is not marked disconnected, and the caller may send
                it again.
        """
        if size < 1 or size > 4:
            raise CameraWriteFailedError(
                f'Binning {size} is not offered (1 to 4); nothing was written'
            )
        logger.debug(f'[CAM Class ] Binning -> {size}')
        _cam_log.info(f'pylon BinningVertical/Horizontal.SetValue({size}) Sum (geometry-realloc)')
        with self.update_camera_config():
            self.write_node('BinningVertical', size)
            self.write_node('BinningVerticalMode', 'Sum')
            self.write_node('BinningHorizontal', size)
            self.write_node('BinningHorizontalMode', 'Sum')
        logger.debug(f'[CAM Class ] Binning set to {size}')

    def get_binning_size(self) -> int:
        """Return current binning factor; 1 when not connected.

        No camera means no binning, and callers treat not connected as the
        1x default. Vertical wins on asymmetric mismatch (operator misconfig).

        Raises:
            CameraReadFailedError: The camera did not answer the read, a
                timeout included. The camera is kept.
        """
        if not self.active:
            return 1

        try:
            vert_bin = self._node('BinningVertical').GetValue()
            horiz_bin = self._node('BinningHorizontal').GetValue()
        except genicam.GenericException as e:
            # Intentionally NO disconnect teardown (latching device-removed on
            # a transient node read killed the stream); removal is owned by
            # the SDK removal callback + the grab loop's definitive paths.
            raise CameraReadFailedError(
                f'The camera did not answer a read of BinningVertical/BinningHorizontal: {e}'
            ) from e

        if horiz_bin != vert_bin:
            _cam_log.warning(
                '[CAM Class ] Binning mismatch detected between '
                f'horizontal ({horiz_bin}) and vertical ({vert_bin})'
            )

        return vert_bin

    def set_conversion_gain_mode(self, mode: str) -> None:
        """Set the sensor conversion gain mode.

        High conversion gain raises the pixel charge-to-voltage gain so
        the sensor read-noise floor drops (better signal-to-noise in low
        light) at the cost of a smaller full well and lower dynamic
        range. Low is the standard mode for normal-to-bright light. This
        is a sensor-level setting, distinct from the line-noise filter.

        Args:
            mode: 'High' (low noise) or 'Low' (wide dynamic range).

        Raises:
            CameraWriteFailedError: The mode is not 'High' or 'Low', the
                camera does not offer the node, or it did not take the
                write. The camera is not marked removed.
            CameraReadFailedError: The grab-state read the config guard needs
                failed; nothing was written.
            CameraNoAnswerError: The write timed out; the camera is not
                marked disconnected, and the caller may send it again.
        """
        if mode not in ('Low', 'High'):
            raise CameraWriteFailedError(
                f'Conversion gain mode {mode!r} is not offered (Low or High); nothing was written'
            )
        _cam_log.info(f'pylon BslConversionGainMode.SetValue({mode!r})')
        with self.update_camera_config():
            self.write_node('BslConversionGainMode', mode)

    def set_line_noise_reduction(self, enabled: bool, strength: float = 0.5) -> None:
        """Enable or disable the camera line-noise reduction filter.

        A camera-side digital filter that smooths horizontal stripe
        ("line noise") artifacts in the sensor readout. Strength runs
        0.0 (none) to 1.0 (max); higher trades fine detail for fewer
        stripes. 0.5 is the Basler-documented starting value.

        Args:
            enabled: True turns the filter on at `strength`; False off.
            strength: Reduction amount 0.0-1.0, applied when enabled.

        Raises:
            CameraWriteFailedError: The camera does not offer a node, or did
                not take a write. The camera is not marked removed.
            CameraReadFailedError: The grab-state read the config guard needs
                failed; nothing was written.
            CameraNoAnswerError: A write timed out; the camera is not marked
                disconnected, and the caller may send it again.
        """
        _cam_log.info(f'pylon BslLineNoiseReductionEnable.SetValue({enabled}) strength={strength}')
        with self.update_camera_config():
            self.write_node('BslLineNoiseReductionEnable', enabled)
            if enabled:
                self.write_node('BslLineNoiseReduction', strength)

    # The three feature probes. A failed probe raises and does not tear the
    # camera down: latching device-removed on a transient probe read killed
    # the stream; removal is owned by the SDK removal callback + the grab
    # loop's definitive paths.

    def supports_conversion_gain_mode(self) -> bool:
        """True if the camera exposes the conversion-gain-mode node.
        See ``Camera.supports_conversion_gain_mode``."""
        if not self.active:
            return False
        try:
            return self._node('BslConversionGainMode') is not None
        except Exception as e:
            raise CameraReadFailedError(
                f'BslConversionGainMode probe failed: {type(e).__name__}: {e}'
            ) from e

    def supports_line_noise_reduction(self) -> bool:
        """True if the camera exposes the line-noise-reduction nodes.
        See ``Camera.supports_line_noise_reduction``."""
        if not self.active:
            return False
        try:
            return (
                self._node('BslLineNoiseReductionEnable') is not None
                and self._node('BslLineNoiseReduction') is not None
            )
        except Exception as e:
            raise CameraReadFailedError(
                f'BslLineNoiseReduction probe failed: {type(e).__name__}: {e}'
            ) from e

    def supports_black_level(self) -> bool:
        """True if the camera exposes the BlackLevel node."""
        if not self.active:
            return False
        try:
            return self._node('BlackLevel') is not None
        except Exception as e:
            raise CameraReadFailedError(f'BlackLevel probe failed: {type(e).__name__}: {e}') from e

    def get_black_level(self) -> float | None:
        """Read BlackLevel live. See ``Camera.get_black_level``.

        A failed read raises and does not tear the camera down: a transient
        read must not latch a removal (the probes above say why).
        """
        if not self.active:
            return None
        try:
            node = self._node('BlackLevel')
            if node is None:
                return None
            return float(node.GetValue())
        except Exception as e:
            raise CameraReadFailedError(f'BlackLevel read failed: {type(e).__name__}: {e}') from e

    def read_black_level_range(self) -> tuple[float, float] | None:
        """BlackLevel's minimum and maximum, live. See ``Camera.read_black_level_range``."""
        if not self.active:
            return None
        try:
            node = self._node('BlackLevel')
            if node is None:
                return None
            return float(node.GetMin()), float(node.GetMax())
        except Exception as e:
            raise CameraReadFailedError(
                f'BlackLevel range read failed: {type(e).__name__}: {e}'
            ) from e

    def set_black_level(self, value: float) -> float | None:
        """Set BlackLevel, with BlackLevelSelector 'All' where the body has
        the selector (Basler black-level.html). Returns as ``gain`` does:
        the level read back, the node may snap to its increment.

        Raises:
            CameraWriteFailedError: The camera does not offer BlackLevel, or
                did not take the selector or the level.
            CameraNoAnswerError: The selector or the BlackLevel write timed
                out; the camera is not marked disconnected.
        """
        self.write_node('BlackLevelSelector', 'All', if_present=True)
        _cam_log.info(f'pylon BlackLevel.SetValue({float(value):g})')
        self.write_node('BlackLevel', float(value))
        return self._read_back('BlackLevel', value)

    def get_resulting_frame_rate(self) -> float | None:
        """Read the resulting frame rate live. See ``Camera.get_resulting_frame_rate``.

        ace 2 / boost / dart name it ``BslResultingAcquisitionFrameRate``,
        legacy ace ``ResultingFrameRate`` (Basler
        resulting-acquisition-frame-rate.html). A failed read raises and does
        not tear the camera down, as ``get_black_level``.
        """
        if not self.active:
            return None
        rate = self._read_first_offered(self._RESULTING_FRAME_RATE_NODES)
        return None if rate is None else float(rate)

    def get_link_info(self) -> dict | None:
        """The device class, DeviceLinkSpeed and, on GigE, the stream packet
        size and inter-packet delay, live. See ``Camera.get_link_info``."""
        if not self.active:
            return None
        try:

            def _read(name):
                node = self._node(name)
                return None if node is None else node.GetValue()

            speed = None
            node = self._node('DeviceLinkSpeed')
            if node is not None:
                speed = declared_speed(node.GetValue(), node.GetUnit())
            return link_info(
                transport=self.active.GetDeviceInfo().GetDeviceClass(),
                link_speed=speed,
                packet_size_bytes=_read('GevSCPSPacketSize'),
                inter_packet_delay=_read('GevSCPD'),
            )
        except Exception as e:
            raise CameraReadFailedError(f'Link info read failed: {type(e).__name__}: {e}') from e

    def init_auto_gain_focus(
        self,
        auto_target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None,
    ) -> None:
        """Configure the AutoFunctionROI + auto-gain limits for AF use.

        Sets ROI to a 50%x50% centered crop (avoids plate-edge + dust +
        uneven-illumination contributions that caused BF AG/AE
        oscillation), picks the `MinimizeGain` profile (stable
        brightness on BF where light is bright + consistent; longer
        exposure / lower noise on fluorescence is also the right
        tradeoff), and applies caller-supplied gain bounds (or the
        camera's reported min/max when `None`) through the setters that
        write them at any other time.

        Args:
            auto_target_brightness: Target brightness in 0..1, fed to
                `AutoTargetBrightness`.
            min_gain: Lower bound for the auto-gain controller in dB,
                or ``None`` to use the camera's reported minimum.
            max_gain: Upper bound in dB, or ``None`` to use the
                camera's reported maximum.

        Raises:
            HardwareError: A write the camera did not take, or a read it
                needs, so the connect-time step that runs this is named as
                not applied. The camera is not marked removed: one failed
                write is not evidence it is gone.
        """

        # AG/AE BF stability fix (#551): shrink AutoFunction ROI to
        # ~50%x50% centered. Full-frame ROI samples plate edges + dust
        # + uneven illumination on BF, which drives the auto-gain
        # controller into oscillation. A 50% centered crop keeps the
        # controller focused on the well center where illumination is
        # uniform. Width / Height steps on Basler ace 2 / dart are
        # multiples of 16 -- _align_down enforces.
        def _align_down(value: int, granularity: int = 16) -> int:
            return (max(int(value), granularity) // granularity) * granularity

        self.write_node('AutoFunctionROISelector', 'ROI1')

        # Pylon node interdependency: each AutoFunctionROIOffset*.Max
        # equals (sensor bound) - (current AutoFunctionROIWidth/Height).
        # An existing non-zero offset caps the achievable Width/Height
        # below their nominal sensor Max, AND the centered-offset
        # setpoint we want (~ Width.Max / 4) is rejected if computed
        # against sensor Width.Max while OffsetX.Max is still
        # constrained by the previous Width. The dart daA3840-45um
        # reports AutoFunctionROIOffsetX.Max = 20 by default; ace 2
        # reports the full sensor extent. Zero the offsets first to
        # unlock the full Width / Height range, then re-center.
        self.write_node('AutoFunctionROIOffsetX', 0)
        self.write_node('AutoFunctionROIOffsetY', 0)

        # Clamp each setpoint against the AutoFunctionROI* node's own
        # Max -- some cameras (the dart family) report tighter bounds
        # on these nodes than on Width / Height proper.
        width_max = int(self._bounds('Width')[1])
        height_max = int(self._bounds('Height')[1])
        roi_width = _align_down(min(width_max // 2, int(self._bounds('AutoFunctionROIWidth')[1])))
        roi_height = _align_down(
            min(height_max // 2, int(self._bounds('AutoFunctionROIHeight')[1]))
        )
        self.write_node('AutoFunctionROIWidth', roi_width)
        self.write_node('AutoFunctionROIHeight', roi_height)

        roi_offset_x = _align_down(
            min((width_max - roi_width) // 2, int(self._bounds('AutoFunctionROIOffsetX')[1]))
        )
        roi_offset_y = _align_down(
            min((height_max - roi_height) // 2, int(self._bounds('AutoFunctionROIOffsetY')[1]))
        )
        self.write_node('AutoFunctionROIOffsetX', roi_offset_x)
        self.write_node('AutoFunctionROIOffsetY', roi_offset_y)
        self.write_node('AutoFunctionROIUseBrightness', True)
        self.update_auto_gain_target_brightness(auto_target_brightness)
        self.update_auto_gain_min_max(min_gain, max_gain)

        # AG/AE BF stability fix (#551): switch from
        # 'MinimizeExposureTime' to 'MinimizeGain'. MinimizeGain pins
        # gain at the lower limit and adjusts exposure first,
        # yielding stable brightness on BF where light is bright and
        # consistent. For fluorescence channels this trades higher
        # exposure time for lower read noise, which is also the right
        # tradeoff. The previous MinimizeExposureTime preference
        # caused gain to track noise and bounce, especially on Red BF
        # where the customer reported the instability.
        self.write_node('AutoFunctionProfile', 'MinimizeGain')

    def update_auto_gain_target_brightness(
        self,
        auto_target_brightness: float,
    ) -> None:
        """Update `AutoTargetBrightness` without an over-stop cycle.

        Live-writable per Basler -- no `update_camera_config()` wrap
        needed. The previous wrap forced a stop/start cycle on every
        call (the same anti-pattern as STALL-1). A write that would change
        nothing is skipped, which avoids redundant writes serialized
        against the grab thread during LED-toggle storms.

        Args:
            auto_target_brightness: Target brightness in 0..1.

        Raises:
            CameraWriteFailedError: The camera did not take the write.
            CameraNoAnswerError: The write timed out; the camera is not
                marked disconnected.
        """
        if self._already('AutoTargetBrightness', float(auto_target_brightness), within=1e-3):
            _cam_log.info(
                f'pylon AutoTargetBrightness.SetValue({auto_target_brightness:.3f}) short-circuited'
            )
            return
        _cam_log.info(f'pylon AutoTargetBrightness.SetValue({auto_target_brightness:.3f})')
        self.write_node('AutoTargetBrightness', auto_target_brightness)

    def update_auto_gain_min_max(
        self,
        min_gain_db: float | None,
        max_gain_db: float | None,
    ) -> None:
        """Update auto-gain min/max bounds without an over-stop cycle.

        Live-writable per Basler: AutoGainLowerLimit / AutoGainUpperLimit
        can be changed while grabbing, so no ``update_camera_config()``
        wrap (auto_gain() calls this and the target setter, and a wrap here
        stop/started the grab twice per arm). A bound that is ``None`` is
        the camera's own end of the limit's range, read live. A write that
        would change neither bound is skipped.

        Args:
            min_gain_db: Lower bound in dB, or ``None`` for the camera's
                minimum.
            max_gain_db: Upper bound in dB, or ``None`` for the camera's
                maximum.

        Raises:
            CameraWriteFailedError: The camera did not take a limit.
            CameraReadFailedError: The read of a limit's range failed;
                nothing was written.
            CameraNoAnswerError: A limit write, or the read of a limit's
                range it needs, timed out; the camera is not marked
                disconnected.
        """
        if min_gain_db is None:
            min_gain_db = self._bounds('AutoGainLowerLimit')[0]
        if max_gain_db is None:
            max_gain_db = self._bounds('AutoGainUpperLimit')[1]
        if self._already('AutoGainLowerLimit', float(min_gain_db), within=1e-3) and self._already(
            'AutoGainUpperLimit', float(max_gain_db), within=1e-3
        ):
            _cam_log.info(
                f'pylon AutoGainLowerLimit/UpperLimit.SetValue'
                f'({min_gain_db}, {max_gain_db}) short-circuited'
            )
            return
        _cam_log.info(
            f'pylon AutoGainLowerLimit.SetValue({min_gain_db}) '
            f'AutoGainUpperLimit.SetValue({max_gain_db})'
        )
        self.write_node('AutoGainLowerLimit', min_gain_db)
        self.write_node('AutoGainUpperLimit', max_gain_db)

    def _set_auto_exposure_time_bounds(self, max_exposure_ms: float | None = None) -> None:
        """Set AutoExposureTime bounds for the auto-exposure loop.

        Lower bound is opened to the sensor minimum so AG can drop
        exposure on bright scenes. The UPPER bound is the per-channel
        AG/AE ceiling (`max_exposure_ms`, converted to microseconds; the
        API resolves it against the camera's range before it arrives),
        NOT the sensor max.
        Capping the upper bound keeps AG from driving exposure to the
        sensor maximum on dim scenes -- which washes out brightfield
        and makes the live auto loop hunt (the flicker / "both go to
        1000 ms" reports). When `max_exposure_ms` is None the upper
        bound opens to the node max (legacy behavior for callers that
        do not supply a ceiling).

        Pylon AutoExposureTime nodes are in microseconds. Live-writable
        per Basler; no stop_grabbing wrap needed. A sensor without these
        nodes has no exposure bound to set: AG works on the gain axis.

        Raises:
            CameraWriteFailedError: The camera did not take a bound.
            CameraReadFailedError: The node probe or the read of a bound's
                range failed; nothing was written.
            CameraNoAnswerError: A bound write, the node probe or the read
                of a bound's range timed out; the camera is not marked
                disconnected.
        """
        if not self._offers('AutoExposureTimeUpperLimit'):
            _cam_log.info('[CAM Class ] No AutoExposureTime limits on this sensor; none set')
            return
        exp_min = self._bounds('AutoExposureTimeLowerLimit')[0]
        if max_exposure_ms is not None:
            exp_max = max_exposure_ms * 1000.0
        else:
            exp_max = self._bounds('AutoExposureTimeUpperLimit')[1]
        _cam_log.info(
            f'pylon AutoExposureTimeLowerLimit.SetValue({exp_min}) '
            f'AutoExposureTimeUpperLimit.SetValue({exp_max}) '
            f'(cap={max_exposure_ms}ms)'
        )
        self.write_node('AutoExposureTimeLowerLimit', exp_min)
        self.write_node('AutoExposureTimeUpperLimit', exp_max)

    # grab() and grab_new_capture() inherited from Camera base class

    def _frame_grid(self) -> FrameGrid | None:
        """The Width / Height nodes' grid at the current binning.

        None when the camera is not connected.

        Raises:
            CameraReadFailedError: The camera did not answer a read, a
                timeout included; the frame size cannot be planned without
                the grid, so no write is sent.
        """
        if not self.active:
            return None
        try:
            width, height = self._node('Width'), self._node('Height')
            return FrameGrid(
                step=(width.GetInc(), height.GetInc()),
                max_size=(width.GetMax(), height.GetMax()),
                size_min=(width.GetMin(), height.GetMin()),
            )
        except genicam.GenericException as e:
            # Intentionally NO disconnect teardown, as for the other node
            # reads: removal is owned by the SDK removal callback and the grab
            # loop's definitive paths.
            raise CameraReadFailedError(
                f'The camera did not answer a read of the Width/Height grid: {e}'
            ) from e

    def _set_hardware_window(self, plan) -> None:
        """Set Width and Height to the planned acquisition and recenter the ROI.

        The ``BslCenterX`` / ``BslCenterY`` execute calls keep the ROI
        centered on the sensor after the size change. Wrapped in
        ``update_camera_config()`` because Width/Height require a
        buffer realloc.

        Raises:
            CameraWriteFailedError: The camera did not take a Width / Height
                write or a recenter, or no camera is open.
            CameraReadFailedError: The grab-state read the config guard needs
                failed; nothing was written.
            CameraNoAnswerError: A Width / Height write or a recenter timed
                out; the camera is not marked disconnected.
        """
        width, height = plan.acq_width, plan.acq_height
        # Short-circuit when geometry already matches: Width/Height SetValue
        # requires update_camera_config() buffer realloc + grab-loop bounce.
        # init + bring-up call this multiple times with the same clamped
        # dims. BslCenterX/Y.Execute is also skipped because we don't write
        # OffsetX/OffsetY directly elsewhere, so the ROI stays centered.
        if self._already('Width', width) and self._already('Height', height):
            _cam_log.log(
                VERBOSE,
                f'pylon Width.SetValue({width}) Height.SetValue({height}) short-circuited',
            )
            return
        _cam_log.info(
            f'pylon Width.SetValue({width}) Height.SetValue({height}) '
            'BslCenterX/Y.Execute() (geometry-realloc)'
        )
        with self.update_camera_config():
            self.write_node('Width', width)
            self.write_node('Height', height)
            self.write_node('BslCenterX', self.RUN)
            self.write_node('BslCenterY', self.RUN)
        _log_cam('info', f'[CAM Class ] Frame size set to {width}x{height}')

    def get_max_frame_size(self) -> dict:
        """Return sensor-driven max frame dims; {} when not connected.

        A model whose lens images less than the sensor declares its own
        ceiling in `data/scopes.json` ``MaxFrame``; the API takes the smaller.

        Raises:
            CameraReadFailedError: The camera did not answer the read, a
                timeout included. The camera is kept.
        """
        if not self.active:
            return {}
        try:
            return {
                'width': self._node('Width').GetMax(),
                'height': self._node('Height').GetMax(),
            }
        except genicam.GenericException as e:
            # Intentionally NO disconnect teardown (latching device-removed on
            # a transient node read killed the stream); removal is owned by
            # the SDK removal callback + the grab loop's definitive paths.
            raise CameraReadFailedError(
                f'The camera did not answer a read of the Width/Height maximum: {e}'
            ) from e

    def _hardware_frame_size(self) -> dict | None:
        """Return active dims as {'width': int, 'height': int}; None when not connected.

        Raises:
            CameraReadFailedError: The camera did not answer the read, a
                timeout included. The camera is kept.
        """
        if not self.active:
            return None

        try:
            width = self._node('Width').GetValue()
            height = self._node('Height').GetValue()
        except genicam.GenericException as e:
            # Driver-wide policy (every pure getter): NO disconnect teardown
            # from a caught node-read error. An SDK exception here is
            # ambiguous -- the teardown helper has no signal to tell transient
            # from fatal, so the distinction lives with the callers that DO
            # have definitive evidence (the SDK removal callback, the
            # DEVICE_NOT_FOUND grab path, the consecutive-failure cascade),
            # which keep sole disconnect authority and still fire on a real
            # unplug. Tearing down from here latched device-removed (no reset
            # short of reconnect) and killed the live stream on one flaky read.
            raise CameraReadFailedError(
                f'The camera did not answer a read of Width/Height: {e}'
            ) from e
        return {
            'width': width,
            'height': height,
        }

    def get_gain(self) -> float:
        """Return current Gain (dB); -1 when not connected.

        Raises:
            CameraReadFailedError: The camera did not answer the read, a
                timeout included. The camera is kept.
        """
        if not self.active:
            _cam_log.warning('[CAM Class ] Cannot read gain: camera inactive')
            return -1

        try:
            return float(self._node('Gain').GetValue())
        except genicam.GenericException as e:
            # Intentionally NO disconnect teardown (latching device-removed on
            # a transient node read killed the stream); removal is owned by
            # the SDK removal callback + the grab loop's definitive paths.
            raise CameraReadFailedError(f'The camera did not answer a read of Gain: {e}') from e

    def get_auto_gain_mode(self) -> str:
        """Read ``GainAuto`` from the camera, past GenApi's node cache.

        A one-shot's end is the camera's own change, which a cached read
        would never see: it would answer ``Once`` for good.

        Raises:
            CameraReadFailedError: The node's lookup or its read failed;
                the camera is kept, as ``get_black_level`` keeps it.
        """
        return self._read_auto_mode('GainAuto')

    def get_auto_exposure_mode(self) -> str:
        """Read ``ExposureAuto`` from the camera, past GenApi's node cache,
        as ``get_auto_gain_mode`` reads ``GainAuto``."""
        return self._read_auto_mode('ExposureAuto')

    def _read_auto_mode(self, name: str) -> str:
        """The mode node ``name`` holds on the camera now (``IgnoreCache``).

        The lookup is inside the ``try``: an SDK failure there, or a node
        the camera does not offer, is a failed read.
        """
        try:
            node = self._node(name)
            return str(node.GetCurrentEntry(False, True).GetSymbolic())
        except Exception as e:
            raise CameraReadFailedError(f'{name} read failed: {type(e).__name__}: {e}') from e

    def gain(self, value: float) -> float | None:
        """Set Gain in dB, with GainSelector 'All' first where the body has
        the selector (Basler gain.html three-step).

        Caller is responsible for ``GainAuto=Off``. A write that would
        change nothing is skipped, answering the gain already in effect.
        Otherwise the gain is read back rather than the request echoed:
        bodies with a Gain increment snap an off-increment value, and the
        caller records this answer as the gain in effect.

        Returns:
            float | None: See ``Camera.gain``.

        Raises:
            CameraWriteFailedError: The camera did not take the selector or
                the gain. The camera is not marked removed: one failed node
                access is not evidence the camera is gone.
            CameraNoAnswerError: The selector or the Gain write timed out;
                the camera is not marked disconnected.
        """
        self.write_node('GainSelector', 'All', if_present=True)
        # Selector is 'All' (written above), so the read matches the
        # requested write. 1e-3 dB is below the Gain increment on ace 2 / dart.
        if self._already('Gain', float(value), within=1e-3):
            _cam_log.log(VERBOSE, f'pylon Gain.SetValue({float(value):.3f}) short-circuited')
            return float(value)
        _cam_log.log(VERBOSE, f'pylon Gain.SetValue({float(value):.3f})')
        self.write_node('Gain', float(value))
        applied = self._read_back('Gain', value)
        if applied is not None:
            _log_cam('debug', f'[CAM Class ] Gain set to {applied} (requested {value})')
        return applied

    def _write_arm_limits(
        self,
        target_brightness: float,
        min_gain_db: float | None,
        max_gain_db: float | None,
        ae_max_exposure_ms: float | None,
    ) -> None:
        """Write an arm's target and limits, each through its one setter.

        Raises:
            HardwareError: One of the writes, or a read it needs, failed; the
                arm is then not made.
        """
        self.update_auto_gain_target_brightness(auto_target_brightness=target_brightness)
        self.update_auto_gain_min_max(min_gain_db=min_gain_db, max_gain_db=max_gain_db)
        # Bound AutoExposureTime before enabling AG. Upper bound is the
        # per-class AG/AE ceiling, not the sensor max -- an uncapped bound
        # lets AG drive exposure to the sensor maximum on dim scenes, washing
        # out brightfield and making the live auto loop hunt (issue #655).
        # Lower bound opened so AG can still drop exposure on saturating
        # scenes.
        self._set_auto_exposure_time_bounds(max_exposure_ms=ae_max_exposure_ms)

    def _write_auto_modes(self, gain_mode: str | None, exposure_mode: str | None) -> None:
        """Write ``GainAuto``, then ``ExposureAuto``: the one writer of both modes.

        ``None`` leaves that mode as it is. Each mode is ``'Off'``,
        ``'Once'`` or ``'Continuous'``.

        Raises:
            CameraWriteFailedError: The camera did not take a mode.
            CameraNoAnswerError: A mode write timed out; the camera is not
                marked disconnected.
        """
        if gain_mode is not None:
            self.write_node('GainAuto', gain_mode)
        if exposure_mode is not None:
            self.write_node('ExposureAuto', exposure_mode)
        _cam_log.info(f'pylon GainAuto={gain_mode} ExposureAuto={exposure_mode}')

    def auto_gain(
        self,
        state: bool = True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
        ae_max_exposure_ms: float | None = None,
    ) -> None:
        """Enable or disable continuous auto-gain + auto-exposure.

        When enabled, ``GainAuto`` and ``ExposureAuto`` are set to
        ``Continuous`` -- the camera continuously adjusts based on the
        live image's brightness -- after the caller's target and limits.
        When disabled, both are set to ``Off``.

        Args:
            state: ``True`` to enable Continuous mode, ``False`` to
                disable.
            target_brightness: Target brightness in 0..1.
            min_gain_db: Lower bound in dB, or ``None`` for the camera's
                minimum.
            max_gain_db: Upper bound in dB, or ``None`` for the camera's
                maximum.
            ae_max_exposure_ms: Per-channel-class upper bound (ms) on the
                exposure AG/AE may drive to, or ``None`` to open the
                bound to the sensor max. Caller (which knows the layer)
                supplies the class cap.

        Raises:
            HardwareError: A mode, target or limit write, or a read one
                needs, failed (``Camera.auto_gain``). The arm is what its
                target and limits make it: one the camera did not take
                leaves an arm it would not have chosen, so the modes are
                then not written.
        """
        _cam_log.info(
            f'pylon auto_gain(state={state}, target={target_brightness}, '
            f'min_db={min_gain_db}, max_db={max_gain_db}, '
            f'ae_max_exp_ms={ae_max_exposure_ms})'
        )
        if state:
            self._write_arm_limits(target_brightness, min_gain_db, max_gain_db, ae_max_exposure_ms)
            self._write_auto_modes('Continuous', 'Continuous')
        else:
            self._write_auto_modes('Off', 'Off')
        _log_cam('info', f'[CAM Class ] Auto gain {"enabled" if state else "disabled"}')

    def auto_gain_once(
        self,
        state: bool = True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
        ae_max_exposure_ms: float | None = None,
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
            min_gain_db: Lower bound in dB, or ``None`` for the camera's
                minimum.
            max_gain_db: Upper bound in dB, or ``None`` for the camera's
                maximum.
            ae_max_exposure_ms: Per-channel-class upper bound (ms) on the
                exposure AG/AE may drive to, or ``None`` for the sensor
                max.

        Raises:
            HardwareError: As ``auto_gain``.
        """
        if state:
            self._write_arm_limits(target_brightness, min_gain_db, max_gain_db, ae_max_exposure_ms)
            self._write_auto_modes('Once', 'Once')
        else:
            self._write_auto_modes('Off', 'Off')
        _log_cam('info', f'[CAM Class ] Auto gain once {"enabled" if state else "disabled"}')

    def exposure_t(self, exposure_ms: float) -> float:
        """Set the camera's exposure time in milliseconds.

        Pylon's ``ExposureTime`` node uses microseconds; this method
        accepts milliseconds and converts. The range is the API's to judge
        (``CameraAPI.set_exposure_ms``); a value the camera will not take
        is refused by the SDK, which raises.

        The microseconds written are RETURNED -- the caller stamps them as
        the frame-validity chunk target.

        Args:
            exposure_ms: Exposure time in milliseconds.

        Returns:
            float: See ``Camera.exposure_t``. Microseconds in effect on both
                the write and short-circuit paths.

        Raises:
            CameraWriteFailedError: The camera did not take the write.
            CameraNoAnswerError: The ExposureTime write timed out; the
                camera is not marked disconnected.
        """
        us_value = float(exposure_ms) * 1000
        # Short-circuit when already at target us. SDK rounds to its clock
        # grid so same target ms maps to same us count; 1 us tolerance is
        # below ExposureTime increment on ace 2 / dart.
        if self._already('ExposureTime', us_value, within=1.0):
            _cam_log.log(VERBOSE, f'pylon ExposureTime.SetValue({us_value:.0f}us) short-circuited')
            return us_value
        _cam_log.log(VERBOSE, f'pylon ExposureTime.SetValue({us_value:.0f}us) (={exposure_ms}ms)')
        self.write_node('ExposureTime', us_value)
        _log_cam('debug', f'[CAM Class ] Exposure set to {exposure_ms}ms')
        return us_value

    def get_exposure_t(self) -> float:
        """Read the camera's currently-active exposure time in ms.

        Prefers ``BslEffectiveExposureTime`` (the value the camera
        actually used, accounting for internal clock-rate rounding)
        when the node is exposed; falls back to ``ExposureTime`` (the
        set value) on legacy ace cameras that do not expose the
        Bsl-prefixed node.

        Pure getter: a failed read never mutates connection state.
        Disconnect authority lives solely with the callers holding
        definitive evidence (the SDK removal callback, the
        DEVICE_NOT_FOUND grab path, the consecutive-failure cascade),
        which still fire on a real unplug; tearing down from a read
        latched device-removed (no reset short of reconnect) and killed
        the live stream on one flaky node read.

        Returns:
            float: Exposure time in milliseconds; ``-1`` when the camera
                is not connected.

        Raises:
            CameraReadFailedError: The read failed, a timeout included, or
                the camera offers neither node. Every camera LVP drives
                has ``ExposureTime``.
        """

        if not self.active:
            _cam_log.warning('[CAM Class ] Cannot read exposure: camera inactive')
            return -1

        microsec = self._read_first_offered(self._EFFECTIVE_EXPOSURE_NODES)
        if microsec is None:
            raise CameraReadFailedError(
                'The camera offers neither BslEffectiveExposureTime nor ExposureTime.'
            )
        return microsec / 1000  # microseconds -> milliseconds

    def auto_exposure_t(self, state: bool = True) -> None:
        """Enable or disable continuous auto-exposure.

        When ``state=True``, ``ExposureAuto`` is set to ``Continuous``;
        the camera continuously adjusts exposure based on the live
        image. When ``state=False``, ``ExposureAuto`` is set to
        ``Off``.

        Args:
            state: ``True`` to enable Continuous mode, ``False`` to
                disable.

        Raises:
            CameraWriteFailedError: The camera did not take the mode.
            CameraNoAnswerError: The ExposureAuto write timed out; the camera
                is not marked disconnected.
        """
        self._write_auto_modes(None, 'Continuous' if state else 'Off')
        _log_cam('info', f'[CAM Class ] Auto exposure {"enabled" if state else "disabled"}')

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

    # The instant camera's nodemaps, by the name a caller asks for: the
    # camera's own, the device transport layer's, the stream grabber's, the
    # SDK's view of the camera (its buffer counts), the event grabber's. A
    # node is on one map or several (``MaxNumBuffer`` is on the stream's and
    # the instant camera's), never found by searching them.
    _NODE_MAPS: ClassVar[dict[str, str]] = {
        'device': 'GetNodeMap',
        'transport': 'GetTLNodeMap',
        'stream': 'GetStreamGrabberNodeMap',
        'instant_camera': 'GetInstantCameraNodeMap',
        'event': 'GetEventGrabberNodeMap',
    }

    def node_maps(self) -> tuple[str, ...]:
        """The instant camera's five nodemaps (``Camera.node_maps``)."""
        return tuple(self._NODE_MAPS)

    def read_node(self, name: str, node_map: str) -> CameraNodeValue | CameraNodeHasNoValue | None:
        """Read one node from one of the instant camera's nodemaps (``Camera.read_node``)."""
        try:
            nodemap = self._nodemap(node_map)
        except genicam.GenericException as e:
            raise CameraReadFailedError(
                f'The camera did not give its {node_map} node map: {e}'
            ) from e
        return self._read_node_in(nodemap, name)

    def _nodemap(self, node_map: str):
        """The active instant camera's nodemap of this name (``_NODE_MAPS``)."""
        return getattr(self.active, self._NODE_MAPS[node_map])()

    def _node(self, name: str, node_map: str = 'device'):
        """The node of this name on one of the active camera's nodemaps (``Camera._node``)."""
        return self._node_in(self._nodemap(node_map), name)

    @staticmethod
    def _node_in(nodemap, name: str):
        """``_node`` on a nodemap already in hand: the node, or None when the map has none.

        A stop's or a snapshot's reads take the nodemaps of the camera they
        were handed, which teardown may already have taken off ``active``.

        pypylon's GetNode answers a placeholder for a name the map lacks,
        which fails only at its first read or write, so whether the map
        holds the node is ``Contains``'s answer, never GetNode's. Any error
        is the camera's or the SDK's failure and propagates.
        """
        if not nodemap.Contains(name):
            return None
        return nodemap.GetNode(name)

    # ``write_node``'s and ``_bounds``'s driver parts (``Camera``).
    _SDK_ERRORS: ClassVar[tuple[type[BaseException], ...]] = (genicam.GenericException,)

    def _put(self, node, value: object) -> None:
        """Set ``node`` to ``value`` (an enumeration by its symbolic name), or run it."""
        if value is self.RUN:
            node.Execute()
        else:
            node.SetValue(value)

    def _is_timeout(self, error: BaseException) -> bool:
        """pypylon raises a timeout as its own class."""
        return isinstance(error, genicam.TimeoutException)

    def _limits_of(self, node) -> tuple[float, float]:
        return node.Min, node.Max

    def _entries_of(self, node) -> tuple[str, ...]:
        return tuple(node.GetSymbolics())

    # Node types that hold no value, by what they are. Every other node is
    # one of genicam's value types (IInteger, IFloat, IBoolean, IString,
    # IEnumeration); an enumeration's GetValue answers its symbolic name.
    _NO_VALUE_TYPES = (
        (genicam.ICommand, CameraNodeHasNoValue.COMMAND),
        (genicam.ICategory, CameraNodeHasNoValue.CATEGORY),
        (genicam.IRegister, CameraNodeHasNoValue.REGISTER),
        (genicam.IPort, CameraNodeHasNoValue.PORT),
        (genicam.IEnumEntry, CameraNodeHasNoValue.ENUM_ENTRY),
    )

    @classmethod
    def _read_node_in(cls, nodemap, name: str) -> CameraNodeValue | CameraNodeHasNoValue | None:
        """``read_node`` on a nodemap already in hand."""
        try:
            node = cls._node_in(nodemap, name)
            if node is None:
                return None
            for node_type, kind in cls._NO_VALUE_TYPES:
                if isinstance(node, node_type):
                    return kind
            if not genicam.IsReadable(node):
                return CameraNodeHasNoValue.NOT_READABLE_NOW
            unit = node.GetUnit() if isinstance(node, (genicam.IFloat, genicam.IInteger)) else ''
            return CameraNodeValue(node.GetValue(), unit)
        except genicam.GenericException as e:
            raise CameraReadFailedError(f'The camera did not answer a read of {name}: {e}') from e

    @classmethod
    def _safe_node(cls, nodemap, *names: str):
        """A diagnostic snapshot's or log line's entry for one setting (``node_snapshot_value``).

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
        return node_snapshot_value(lambda name: cls._read_node_in(nodemap, name), *names)

    # One setting under the names camera families give it: ace 2 / boost /
    # dart M/R name it Bsl-prefixed, legacy ace unprefixed (Basler
    # resulting-acquisition-frame-rate.html, exposure-time.html). The
    # effective exposure is the value the camera used, after its clock
    # rounding; ExposureTime is the value set.
    _RESULTING_FRAME_RATE_NODES = ('BslResultingAcquisitionFrameRate', 'ResultingFrameRate')
    _EFFECTIVE_EXPOSURE_NODES = ('BslEffectiveExposureTime', 'ExposureTime')

    def _read_first_offered(self, names: tuple[str, ...], node_map: str = 'device'):
        """The value of the first of ``names`` the camera offers, read live.

        A live reader's contract, where the diagnostic snapshot's
        (``_safe_node``) reports every answer: a name the camera does not
        offer tries the next, and none offered answers None.

        Raises:
            CameraReadFailedError: The read failed, or the node offered
                holds no value now (``CameraNodeHasNoValue``), which is not
                the value asked for.
        """
        for name in names:
            answer = self.read_node(name, node_map)
            if answer is None:
                continue
            if isinstance(answer, CameraNodeHasNoValue):
                raise CameraReadFailedError(f'{name} holds no value to read ({answer})')
            return answer.value
        return None

    def read_diagnostic_snapshot(
        self,
        # 3.0s default: matches the bench probe shape used to
        # characterize dart vs ace 2 on Mac. Long enough for
        # cumulative counters to advance visibly at 18-30 fps without
        # being so long the operator gets bored. Falsify: if running
        # this at 3s misses a class of error that only shows up over
        # longer windows, callers raise the value explicitly per-call.
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
                transport_layer (every node of the device transport
                layer, name to value), stats_pre, stats_post, deltas, derived
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
            'transport_layer': {},
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
            (self._EFFECTIVE_EXPOSURE_NODES, 'exposure_us'),
            (('Gain',), 'gain_db'),
            (('BlackLevel',), 'black_level'),
            (('DeviceLinkThroughputLimitMode',), 'dltl_mode'),
            (('DeviceLinkThroughputLimit',), 'dltl_value_bps'),
            (('BslDeviceLinkCurrentThroughput',), 'current_throughput_bps'),
            (('AcquisitionFrameRateEnable',), 'acquisition_frame_rate_enable'),
            (('AcquisitionFrameRate',), 'acquisition_frame_rate'),
            (self._RESULTING_FRAME_RATE_NODES, 'resulting_frame_rate'),
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
            (('BslTemperatureStatusErrorCount',), 'temperature_status_error_count'),
            # Which instant the Timestamp chunk stamps (ace 2 and dart,
            # per Basler doc data-chunks.html); the sentinel on a camera
            # whose chunk is the legacy one.
            (('BslChunkTimestampSelector',), 'chunk_timestamp_selector'),
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

        result['transport_layer'] = self._transport_layer_nodes(cam, result['errors'])

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
            if is_finite_number(pre) and is_finite_number(post):
                result['deltas'][name] = post - pre
            else:
                result['deltas'][name] = None

        # Derived rates
        total_d = result['deltas'].get('Statistic_Total_Buffer_Count')
        failed_d = result['deltas'].get('Statistic_Failed_Buffer_Count')
        resync_d = result['deltas'].get('Statistic_Resynchronization_Count')
        missed_d = result['deltas'].get('Statistic_Missed_Frame_Count')
        if is_finite_number(total_d) and dt > 0:
            result['derived']['observed_fps'] = total_d / dt
            if total_d > 0 and is_finite_number(failed_d):
                result['derived']['fail_rate_pct'] = 100.0 * failed_d / total_d
        if is_finite_number(failed_d) and dt > 0:
            result['derived']['failures_per_second'] = failed_d / dt
        if is_finite_number(resync_d) and dt > 0:
            result['derived']['resyncs_per_second'] = resync_d / dt
        if is_finite_number(missed_d) and dt > 0:
            result['derived']['misses_per_second'] = missed_d / dt

        # Camera-side error log drain (capped to prevent runaway loop
        # on a malformed firmware response)
        if drain_camera_side_errors:
            try:
                err_present = self._node_in(nm, 'BslErrorPresent')
                code_node = self._node_in(nm, 'BslErrorReportValue')
                next_cmd = self._node_in(nm, 'BslErrorReportNext')
                if err_present is not None and code_node is not None and next_cmd is not None:
                    # The codes drained before a failed read are kept,
                    # and the failure is recorded beside them.
                    errs = result['camera_side_errors'] = []
                    # 64-iteration cap: bound the drain so a wedged
                    # firmware that reports BslErrorPresent=True
                    # forever can't infinite-loop the snapshot. 64 is
                    # far more than any expected error queue depth on
                    # ace 2 / dart M/R (typical: 0-3 entries per
                    # session). Falsify: if a real run actually drains
                    # 64 errors, the camera is in a catastrophic state
                    # AND the snapshot logs the truncation.
                    for _ in range(64):
                        if not err_present.GetValue():
                            break
                        code = code_node.GetValue()
                        if code == 0:
                            break
                        errs.append(int(code))
                        # The map in hand, not the active camera's: the
                        # snapshot's, which teardown may have taken off it.
                        self._put(next_cmd, self.RUN)
            except Exception as e:
                result['errors'].append(f'BslErrorPresent drain raised: {type(e).__name__}: {e}')

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


def _read_validity_chunks(grabResult) -> dict[str, int | float] | None:
    """Extract validity chunks from a successful GrabResult.

    Returns dict like {'ExposureTime': float, 'Gain': float, 'FrameID': int},
    or None if no chunks readable (chunks unsupported or not enabled).
    """
    chunks: dict[str, int | float] = {}
    for chunk_attr, key in _CHUNK_GRAB_RESULT_ATTRS:
        try:
            node = getattr(grabResult, chunk_attr, None)
            if node is None:
                continue
            if genicam.IsReadable(node):
                chunks[key] = node.Value
        except Exception as e:
            logger.debug(f'[CAM Class ] _read_validity_chunks could not read {chunk_attr}: {e}')
    return chunks if chunks else None


class ImageHandler(pylon.ImageEventHandler, ImageHandlerBase):
    """Pylon camera image handler -- receives frames via SDK callbacks.

    Inherits ``ImageHandlerBase`` beside the SDK event handler so the whole
    base surface is present here. ``Camera`` reads its handler through that
    surface, and a hand-maintained subset of it went stale every time the
    base grew. The two bases coexist: the SWIG proxy's metaclass is ``type``
    and it defines none of the base's names (verified on pypylon 26.10).
    The SDK ``__init__`` does not forward along the MRO, so both bases are
    initialized explicitly.
    """

    def __init__(self, parent_cam: PylonCamera):
        pylon.ImageEventHandler.__init__(self)
        ImageHandlerBase.__init__(self)
        self._parent = parent_cam
        # Stage B worker for the OnImageGrabbed two-stage split. Created
        # here so it shares lifetime with the handler; start() / stop()
        # are driven from PylonCamera.connect / disconnect at the ordering
        # the SDK contract requires.
        self._worker = _PylonImageGrabWorker(parent_cam, self)

    def OnImagesSkipped(self, camera: pylon.InstantCamera, countOfSkippedImages: int) -> None:
        """Pylon SDK callback fired when the grab strategy drops frames.

        Per `class_pylon_1_1_c_image_event_handler.html`: fires when
        GrabStrategy_LatestImageOnly (or LatestImages) discards an older
        frame in favor of a newer one. The SDK contract: "Exceptions
        from this call will propagate through" -- same hazard as
        OnImageGrabbed, so wrap accordingly.

        Logged at info so the skip distribution is visible in camera.log
        post-R12 (correlate with worker-queue-full count if applicable).
        Today (pre-R12) skips fire when the consumer reads
        the handler's last stored frame slower than the SDK grabs; cause
        distribution stays in the log without raising the noise floor.
        """
        try:
            # Once the device is gone, the SDK fires a burst of skip
            # callbacks during teardown -- those frames were dropped by the
            # removal, not by LatestImageOnly pressure, so logging them is
            # misleading noise. Mirror OnImageGrabbed's removal guard.
            if self._parent._device_removed:
                return
            if countOfSkippedImages > 0:
                _cam_log.info(
                    f'[CAM Class ] OnImagesSkipped: '
                    f'{countOfSkippedImages} frame(s) discarded by SDK '
                    f'(grab strategy = LatestImageOnly)'
                )
        except Exception as e:
            _log_safely(f'OnImagesSkipped logging raised: {e}')
        except BaseException as e:
            _log_safely(f'OnImagesSkipped BaseException guard caught {type(e).__name__}: {e}')

    def OnImageGrabbed(self, camera: pylon.InstantCamera, grabResult: pylon.GrabResult) -> None:
        """Stage A native-thread fast-path: state checks + enqueue to worker.

        Heavy work (16 MB GetArray copy, chunk reads, _store_frame,
        per-frame callback fanout, generic failure classification +
        counter) runs on Stage B's _PylonImageGrabWorker. The native-thread exposure surface here
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
        _trace_enabled = profile_trace.ENABLE_PROFILE_TRACE
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
                logger.debug('[CAM Class ] OnImageGrabbed called but camera is inactive, ignoring')
                self._parent._mark_disconnected()
                _outcome = 'early_return_inactive'
                return

            # GrabSucceeded() can leak native exceptions in cancel /
            # teardown paths. Trapped here so nothing escapes into Pylon's
            # grab thread, and handed to Stage B as a failed grab: one
            # unreadable result is not evidence the camera is gone, and the
            # failed-grab cascade, counted on Stage B's thread alone, is
            # what decides that. A result the copy cannot own is handed on
            # as no result, which Stage B classifies as a generic failure.
            try:
                grab_succeeded = grabResult.GrabSucceeded()
            except Exception as e:
                _cam_log.warning(f'[CAM Class ] GrabSucceeded() failed: {e}; a failed grab')
                try:
                    owned = pylon.GrabResult(grabResult)
                except Exception:
                    owned = None
                try:
                    self._worker.enqueue('fail', owned, None, error_code='exception')
                    _outcome = 'exception_grabsucceeded'
                except queue.Full:
                    # Counted in the worker's drop burst, which logs it.
                    _outcome = 'queue_full_dropped_failure'
                return

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
                    # Numbered here, where the driver first has the frame, so
                    # a frame still in the worker queue when a write goes out
                    # is placed before it.
                    self._worker.enqueue('frame', owned, self.hand_over())
                    _outcome = 'success_enqueued_frame'
                except queue.Full:
                    # Stage B is wedged and the queue holds only failed
                    # grabs, which a frame never displaces; drop this frame
                    # rather than block the native thread. Bench-monitored via the
                    # pylon_stage_a_worker_queue_full_per_min budget, read
                    # from the worker's drop-burst summaries.
                    _outcome = 'queue_full_dropped_frame'
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
                        self._worker.enqueue('fail', owned, None, error_code=err_code)
                        _outcome = 'failure_enqueued'
                    except queue.Full:
                        # Counted in the worker's drop burst, which logs it.
                        _outcome = 'queue_full_dropped_failure'
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
            _log_safely(f'OnImageGrabbed BaseException guard caught {type(e).__name__}: {e}')
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
                        recording_id=profile_trace.NO_RECORDING,
                    )
            except BaseException as e:
                _log_safely(f'profile_trace.trace raised {type(e).__name__}: {e}')
            try:
                # Handle-leak tracking; zero overhead when disabled. Enable
                # via the profiling.handle_trace_enabled setting.
                from _lumascope.lib.handle_trace import tick as _h_tick

                _h_tick('OnImageGrabbed')
            except BaseException as e:
                _log_safely(f'handle_trace tick raised {type(e).__name__}: {e}')


def _error_code_words(code: int | str | None) -> str:
    """A failed grab's error code as Basler documents it: hex, or what stood in for one."""
    if is_integer_number(code):
        return f'0x{code:08X}'
    return 'unknown' if code is None else code


class _DropBurst:
    """The grabs Stage A dropped in one stall of the worker queue.

    A field bundle held 568 per-drop warnings in four bursts, and the failed
    grabs among them named no error code, so neither the bursts' bounds nor
    whether the failures were buffer underruns could be read. A burst is said
    at its start and summarized at its end instead. Written only on Stage A's
    thread; ``stop`` reads it after StopGrabbing has returned.

    A stall drops grabs and takes some between them, so the burst stays open
    until the queue has gone ``QUIET_S`` without a drop: closed at the first
    grab taken after a drop, a field box's one 1 s stall logged as 5 - 24
    bursts of 0.0 - 0.2 s each, each understating the stall it was part of.
    """

    QUIET_S = 1.0

    def __init__(self, now: float) -> None:
        self.started = now
        self.last_drop = now
        self.frames = 0
        self.failures: dict[str, int] = {}

    def add(self, kind: str, error_code: int | str | None, now: float) -> None:
        """Count one dropped grab: a frame, or a failed grab under its error code."""
        self.last_drop = now
        if kind == 'frame':
            self.frames += 1
        else:
            words = _error_code_words(error_code)
            self.failures[words] = self.failures.get(words, 0) + 1

    def is_over(self, now: float) -> bool:
        """Whether the queue has gone ``QUIET_S`` since the burst's last drop."""
        return now - self.last_drop >= self.QUIET_S

    def summary(self, ended: str) -> str:
        """The burst's one closing line, ``ended`` saying how it ended.

        Its length runs from the first drop to the last, the stall itself.
        """
        failed = sum(self.failures.values())
        codes = ', '.join(f'{words} x{n}' for words, n in sorted(self.failures.items()))
        return (
            f'Stage A: {ended}; the burst lasted {self.last_drop - self.started:.1f} s and '
            f'dropped {self.frames} frame(s) and {failed} failed grab(s)'
            + (f' (error codes: {codes})' if codes else '')
        )


class _PylonImageGrabWorker:
    """Stage B worker for the OnImageGrabbed two-stage split.

    Stage A is the Pylon SDK callback `OnImageGrabbed` running on a
    native grab thread; it does only the enqueue + fail-fast work and
    hands the grab-result smart pointer to Stage B (this worker) for
    the heavy memcpy + chunk parsing + callback fanout. Handoff is via
    the bounded `_items`, under `_cond`.

    A full handoff drops its oldest queued frame for the new item, so
    the frames Stage B delivers after a stall are the newest; a failed
    grab is never dropped for a frame, since the failed-grab cascade
    counts it, and a handoff holding only failed grabs drops the new
    item.

    Single-thread invariant on the SDK's `CGrabResultData`: Stage A is
    the SOLE reader of `GrabSucceeded()` (and `GetErrorCode()` /
    `GetErrorDescription()` on the failure branch). After `enqueue`
    Stage B becomes the SOLE reader of `GetArray()` + the chunk
    node-map accessors. The condition's lock, taken by the append and
    the pop, provides happens-before.

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
    # callback fanout) don't drop a frame, small enough that a stalled
    # consumer can't OOM the host. Bench-tunable
    # via the queue_depth kwarg to __init__.
    _DEFAULT_QUEUE_DEPTH = 8

    def __init__(
        self,
        parent,
        base,
        *,
        queue_depth: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._parent = parent  # PylonCamera instance
        self._base = base  # ImageHandlerBase instance
        self._depth = self._DEFAULT_QUEUE_DEPTH if queue_depth is None else int(queue_depth)
        # (kind, grabResult, handed_over), oldest first; read and written
        # under _cond alone.
        self._items: deque[tuple[str, pylon.GrabResult, HandOver | None]] = deque()
        self._cond = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # The drops of the stall in progress; None while the queue takes every grab.
        self._drop_burst: _DropBurst | None = None
        # The burst's time source, a test's own in place of the monotonic clock.
        self._clock = clock

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

    def enqueue(
        self,
        kind: str,
        grabResult: pylon.GrabResult,
        handed_over: HandOver | None,
        *,
        error_code: int | str | None = None,
    ) -> None:
        """Hand a Stage A item off to Stage B.

        At capacity, the oldest queued frame is dropped and the new item
        taken, and this returns normally. Raises `queue.Full` when the
        queue is at capacity with failed grabs alone -- the new item is
        dropped, and Stage A records that rather than blocking the native
        grab thread. Returns silently if the worker is shutting down
        (stop_event already set); the caller's `grabResult` local will
        release the SDK buffer when Stage A returns.

        The drops are logged as bursts, each drop counted as the item
        dropped (the evicted frame, or the new item): a burst's first drop
        is said once, later drops are counted, and the first grab the
        queue takes without a drop once it has gone ``_DropBurst.QUIET_S``
        without one closes the burst with one summary (its frames, its
        failed grabs by error code, its length from first drop to last).

        Args:
            kind: 'frame' for a successful grab, 'fail' for a failure
                that needs Stage B classification.
            grabResult: The SDK CGrabResultPtr to hand off. Stage A MUST
                NOT access it after this returns (single-thread invariant
                on CGrabResultData).
            handed_over: A frame's ``hand_over()``, taken by Stage A where
                the driver first had it; None for a failure, which is no
                frame.
            error_code: A failure's ``GetErrorCode()``, or ``'exception'``
                when ``GrabSucceeded()`` raised; counted if the failure is
                dropped. None for a frame.
        """
        if self._stop_event.is_set():
            return
        # The item dropped to make room, as (kind, error_code); None when
        # the queue had room.
        dropped = None
        evicted = None
        with self._cond:
            if len(self._items) >= self._depth:
                oldest_frame = next(
                    (i for i, item in enumerate(self._items) if item[0] == 'frame'), None
                )
                if oldest_frame is None:
                    dropped = (kind, error_code)
                else:
                    evicted = self._items[oldest_frame]
                    del self._items[oldest_frame]
                    dropped = ('frame', None)
            if dropped is None or evicted is not None:
                self._items.append((kind, grabResult, handed_over))
                self._cond.notify()
        if dropped is not None:
            self._count_drop(*dropped)
            if evicted is None:
                raise queue.Full
            # The evicted frame's grab result is released here, on Stage
            # A's thread and outside the lock, as a dropped new item's is
            # when Stage A returns.
            del evicted
            return
        if self._drop_burst is not None and self._drop_burst.is_over(self._clock()):
            _log_safely(
                self._drop_burst.summary(
                    f'the worker queue took every grab for {_DropBurst.QUIET_S:.1f} s'
                )
            )
            self._drop_burst = None

    def _count_drop(self, kind: str, error_code: int | str | None) -> None:
        """Count one dropped item in the burst, opening the burst with its one line."""
        now = self._clock()
        if self._drop_burst is None:
            self._drop_burst = _DropBurst(now)
            _log_safely(
                f'Stage A: worker queue full -- dropping a '
                f'{"frame" if kind == "frame" else "failed grab"}; further drops '
                f'are counted until the queue goes {_DropBurst.QUIET_S:.1f} s without one'
            )
        self._drop_burst.add(kind, error_code, now)

    def stop(self, timeout: float = 1.0) -> None:
        """Signal stop, wake the worker, join with bounded timeout.

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
        # StopGrabbing has returned (the contract above), so Stage A no
        # longer writes the burst: one still open is summarized here.
        if self._drop_burst is not None:
            _log_safely(self._drop_burst.summary('the worker stopped with a burst still open'))
            self._drop_burst = None
        with self._cond:
            self._stop_event.set()
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        # Loop structure: always take a pending item before checking the
        # stop event. Exiting on the event while items remain -- stop()
        # setting it AFTER Stage A enqueued the last batch but BEFORE the
        # worker came back from processing the previous item -- would
        # leave the rest unreleased, and the SDK input queue would then
        # underrun on the next connect cycle per
        # `class_pylon_1_1_c_grab_result_ptr.html`: "grabbing will stop
        # with an input queue underrun, when the grab results are never
        # released." The event is consulted only once the queue is empty.
        while True:
            with self._cond:
                while not self._items and not self._stop_event.is_set():
                    self._cond.wait()
                if not self._items:
                    return
                kind, grabResult, handed_over = self._items.popleft()
            try:
                if self._parent._device_removed:
                    # Drop quietly; the finally below releases the
                    # smart pointer so the SDK can reclaim the buffer.
                    continue
                if kind == 'frame':
                    self._process_frame(grabResult, handed_over)
                elif kind == 'fail':
                    self._process_failure(grabResult)
            except Exception as e:
                _cam_log.exception(f'[CAM Class ] worker handling raised: {e}')
            finally:
                # Explicit release semantics. Per CGrabResultPtr docs:
                # "grabbing will stop with an input queue underrun,
                # when the grab results are never released."
                del grabResult

    def _process_frame(self, grabResult, handed_over: HandOver) -> None:
        """Stage B success path: 16 MB memcpy + chunks + store + legacy queue.

        A GetArray() failure is one failed grab, counted toward the
        consecutive-failure cascade: one unreadable buffer is not evidence the
        camera is gone, and the cascade is what decides that. Cross-thread
        smart-pointer access is officially supported per
        `class_pylon_1_1_c_grab_result_ptr.html` provided GetArray succeeds;
        a raised exception here means the underlying buffer is gone.
        """
        try:
            # GetArray() already returns a numpy array backed by its own copy of
            # the SDK buffer -- decoupled from the reused grab-result pool, so it
            # stays valid after grabResult is released downstream. A trailing
            # .copy() would be a redundant second full-frame memmove per frame.
            img = grabResult.GetArray()
        except Exception as e:
            # Stage A enqueued this with GrabSucceeded=True. Re-read
            # GrabSucceeded + GetErrorCode/Description on Stage B's side. The
            # re-check distinguishes (a) race conditions where SDK state
            # flipped between threads (GrabSucceeded was True, now False or
            # unreadable) from (b) Stage A misclassification (GrabSucceeded
            # was wrong from the start). BlockID / GrabID name the frame.
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
            _log_cam(
                'warning',
                f'[CAM Class ] GetArray() failed on worker: {e} '
                f'(Stage B re-check: GrabSucceeded={_gs_now} '
                f'err_code={_err_code} err_desc={_err_desc!r} '
                f'BlockID={_bid} GrabID={_gid}), '
                f'a failed grab',
            )
            self._count_failed_grab()
            return
        # A reverse the sensor did not take is turned in a second copy, paid
        # only on a body that owes one.
        owed = self._parent.owed_reverse
        if owed != NOTHING_OWED:
            img = copy_oriented(img, owed)
        chunks = _read_validity_chunks(grabResult)
        # Read the depth from the frame's OWN pixel type -- the format the SDK
        # delivered this buffer in -- so the depth stays bound to the frame.
        # pylon.BitDepth(PixelType_Mono12) is 12, Mono8 is 8. A later pixel-format
        # switch cannot make this buffered frame downconvert at the wrong depth,
        # because the depth came from the frame, not the camera's current state.
        significant_bits = pylon.BitDepth(grabResult.GetPixelType())
        self._base._store_frame(
            img,
            chunks=chunks,
            handed_over=handed_over,
            significant_bits=significant_bits,
            wire_bytes=grabResult.GetPayloadSize(),
        )

    def _process_failure(self, grabResult) -> None:
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
                f'[CAM Class ] Grab canceled (SDK lifecycle, '
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
            _cam_log.warning(
                f'[CAM Class ] grabResult.GrabSucceeded()=False '
                f'err_code={err_code} desc={err_desc!r}'
            )
            self._count_failed_grab()

    def _count_failed_grab(self) -> None:
        """Count one failed grab; at the cascade's limit the camera is gone.

        MAX_CONSECUTIVE_FAILURES (128 frames at 30 fps ~= 4.3 s) failed grabs
        with no delivered frame between them are the grab path's definite
        evidence of a removal: the camera is marked removed and torn down off
        this thread, as every definite removal is. The worker sends the
        camera nothing itself: StopGrabbing on a removed handle is the
        pypylon #225 hazard the teardown skips for a removed camera, and
        from here it would also run outside the teardown, beside a close's
        own. One owner, run on this worker's thread alone, because the count
        it keeps is not locked.
        """
        if not self._base._record_failure():
            return
        _cam_log.error('[CAM Class ] Too many grab failures; the camera is removed')
        self._parent._mark_disconnected()
        self._parent._schedule_async_teardown()


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
            _log_safely(f'OnCameraDeviceRemoved guard caught {type(e).__name__}: {e}')

    def OnGrabError(self, camera: pylon.InstantCamera, errorMessage: str) -> None:
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
            # Diagnostic only -- do NOT mark the camera disconnected here. A
            # grab-thread exception (e.g. a GEV chunk-parse error right after a
            # SetValue) is not evidence the device was removed, and latching on
            # the first one dropped a working camera. Definitive removal is
            # owned elsewhere: OnCameraDeviceRemoved, the DEVICE_NOT_FOUND grab
            # path, and the consecutive-failure cascade. The grab the SDK
            # stops after this callback is still wanted (grab_wanted), so the
            # imaging API's stream check reports it as a stall.
        except BaseException as e:
            _log_safely(f'OnGrabError guard caught {type(e).__name__}: {e}')
