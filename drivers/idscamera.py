# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
import atexit
import datetime
import math
import re
import threading
import time

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

# modules.aoi_geometry (plan_aoi) and modules.image_utils (center_crop) are
# imported function-locally where used: the driver layer must not import from
# modules/ at top level (enforced by tests/test_architecture_fixes.py). Both are
# pure helpers, so the lazy import carries no cycle risk; after first use the
# import is a sys.modules dict hit, negligible even on the per-frame unpack path.

# IDS Library.Close() shuts down the entire SDK (not per-device).
# Defer to atexit so it only runs once at process exit.
_ids_library_initialized = False

# The DeviceManager enumeration timeout the presence probe RESTORES after its
# own tightened per-probe value, so a later connect() Update() on the
# process-wide singleton never inherits the probe's short window (which would
# risk missing a present-but-slow-to-enumerate camera).
_DEVICE_UPDATE_TIMEOUT_MS = 5000

# OpenDevice(Control) retry on a transient access denial. Control access is
# exclusive on the camera: it stays claimed for a short window after a prior
# holder lets go -- our own just-released handle on a rapid reconnect, a
# just-closed vendor tool (Cockpit / firmware updater), or the re-enumeration
# right after a DeviceReset. A single-shot open turns that whole window into a
# hard connect failure, so retry briefly with a growing backoff before giving
# up (a device genuinely held by another application still fails, just ~2s
# slower and with the holder named in the log).
_OPEN_CONTROL_RETRIES = 3
_OPEN_CONTROL_BACKOFF_S = 0.4

# Active device-presence probe on a SUSTAINED poll stall. DeviceLost never
# fires on a USB unplug for the U3-34L, so a real removal otherwise looks
# identical to a wedged-but-present stall -- both just time out forever. After
# the stream has been stalled this long, actively check the device is still
# enumerated (rate-limited so a wedged-but-present camera isn't hammered with
# DeviceManager.Update calls). A short update timeout keeps the probe from
# blocking the poll thread (and thus teardown) -- a present USB3 device
# re-enumerates well within it; an absent one never appears regardless.
_PRESENCE_PROBE_STALL_S = 6.0
_PRESENCE_PROBE_INTERVAL_S = 5.0
_PRESENCE_PROBE_UPDATE_TIMEOUT_MS = 500
# Require this many CONSECUTIVE absent probes (each a separate Update, spaced by
# _PRESENCE_PROBE_INTERVAL_S) before concluding removal. A single truncated
# Update on a busy/slow-to-enumerate but still-present USB3 stack can omit the
# device; confirming across probes prevents a transient enumeration hiccup from
# permanently tearing down a connected camera. A present (or inconclusive)
# reading resets the count.
_PRESENCE_PROBE_CONFIRMATIONS = 2
# FLOOR of the hard ceiling on a sustained stall, independent of what the
# presence probe reads: the producer can keep a PHANTOM entry enumerated after
# a physical unplug (observed on the bench -- the probe read the ghost as
# present for the whole stall, so removal was never latched, teardown took the
# live-device path, and app close hung on remote writes to dead hardware). A
# started free-run stream that delivers NOTHING past the ceiling is dead
# regardless of what enumeration claims, so escalate to the uniform removal
# drop. The effective ceiling scales with the exposure-scaled wait window
# (several consecutive empty windows, never below this floor), so a
# long-exposure inter-frame gap can never reach it.
_STALL_ESCALATION_CEILING_S = 30.0
# FLOOR of the stall age past which teardown assumes the device is DEAD: the
# remote-device writes in stop_grabbing (AcquisitionStop, TLParamsLocked)
# each block the calling thread ~10 s against unplugged hardware before
# failing, so a stop past this age skips them and stops host-side only.
# Shorter than the escalation ceiling above -- a user can close the app
# mid-stall, before the ceiling latches removal -- but past the presence
# probe's window, so a live camera with a brief hiccup never has its remote
# stop skipped. Scaled with the wait window like the ceiling.
_TEARDOWN_ASSUME_DEAD_STALL_S = 15.0

# A single frame that arrives but yields no usable image -- incomplete (USB
# packet loss / bandwidth saturation) OR its handle raises a non-removal fault --
# is an absorbed blip: the poll loop logs it and keeps streaming. But a stream
# delivering ONLY such bad frames keeps ARRIVING, so it never times out and the
# sustained-stall presence probe never fires; without a ceiling it freezes
# forever on garbage. On this many bad frames since the last COMPLETE one the
# stream is treated as wedged and escalated to the Category-2 drop (the same
# uniform teardown an unplug takes). Bench-informed: high enough that a transient
# saturation burst is absorbed, low enough that a genuine wedge does not hang the
# live view indefinitely (~2-4 s of only-bad frames at typical frame rates). A
# complete frame resets the count.
_MAX_BAD_FRAMES_BEFORE_WEDGE = 120


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
    """Native payload depth of the given wire PixelFormat (the oracle counterpart
    of _ids_ipl_target).

    Derived from the GenICam format name so the depth tracks the sensor's
    real format rather than the (16-bit) container the unpacked frame rides
    in: Mono12* -> 12, Mono10* -> 10, Mono8* -> 8. Falls back to the leading
    bit count in the name, then to 8. Pure logic -- no SDK call -- so it is
    unit-testable without a camera.

    NOTE: this is the WIRE's native depth, not necessarily what the production
    unpack DELIVERS -- 8-bit mode delivers the Mono10 wire reduced to 8-bit (see
    _ids_delivery_significant_bits). Use this for native-depth reasoning (the
    unpack benchmark oracle); use the delivery helper for the depth stamp on a
    live/still frame.
    """
    # Empty/None is a failed-read sentinel from get_pixel_format(), not a format:
    # default to 8 QUIETLY (a warning here would misreport a transient read glitch
    # as a color/packed camera). Also keeps the startswith calls below None-safe.
    if not wire_format_name:
        return 8
    if wire_format_name.startswith('Mono12'):
        return 12
    if wire_format_name.startswith('Mono10'):
        return 10
    if wire_format_name.startswith('Mono8'):
        return 8
    match = re.search(r'Mono(\d+)', wire_format_name)
    if match:
        return int(match.group(1))
    # A non-empty, non-Mono name. This driver's cameras are mono-only, so a
    # color / packed name is unexpected -- and its trailing bit count does not map
    # to significant bits the way Mono<N> does (IDS's own Mono12g24IDS packs 12
    # significant bits into a 24-bit wire), so a naive "last digits" read would
    # mislabel it. Default to 8 but WARN, so a color camera ever reaching here is
    # loud rather than silently stamped 8-bit.
    _cam_log.warning(
        f'[CAM Class ] Unrecognized wire format {wire_format_name!r} for '
        'significant-bits derivation; defaulting to 8 (mono formats expected)'
    )
    return 8


def _ids_ipl_target(wire_format_name: str):
    """IPL ConvertTo target that unpacks a wire format to its NATIVE depth.

    Mono10/Mono12 unpack to a right-aligned uint16; Mono8 stays 8-bit. This is
    the oracle target used by the unpack benchmark (it compares ConvertTo to the
    numpy reference at full native depth); the production live/still unpack uses
    _ids_delivery_target instead. The SDK does the unpack (faster than a
    hand-unpacker and the alignment is the SDK's own); drivers/ids_unpack.py is
    the bench cross-check for the target.
    """
    if wire_format_name.startswith('Mono12'):
        return ids_peak_ipl.PixelFormatName_Mono12
    if wire_format_name.startswith('Mono10'):
        return ids_peak_ipl.PixelFormatName_Mono10
    return ids_peak_ipl.PixelFormatName_Mono8


def _ids_delivery_target(wire_format_name: str):
    """IPL ConvertTo target for the PRODUCTION live/still unpack.

    Differs from _ids_ipl_target (native depth) for the Mono10 wire: this body
    has no native Mono8 wire format, so 8-bit image mode captures the
    lowest-bandwidth native (Mono10) and reduces to 8-bit on the host (see
    modules.image_mode.select_capture_pixel_format). Mono10 is therefore only
    ever the 8-bit-mode wire, so unpack it straight to Mono8 in ONE ConvertTo
    pass -- this halves the unpack output (uint8 vs uint16) and lets the display
    skip its second host downconvert. Mono12 (the 12-bit modes) keeps native
    uint16 so the full depth survives to save/analysis.

    The Mono10->Mono8 reduction is the SDK's bit-shift, within <=1 LSB of the
    host rescale LUT it replaces (tests/test_ids_hardware.py cross-checks this on
    real frames). If a future image mode ever drives Mono10 wire while wanting
    >8-bit output, this coupling must become an explicit delivered-depth the
    camera config sets; no such mode exists today.
    """
    if wire_format_name.startswith('Mono12'):
        return ids_peak_ipl.PixelFormatName_Mono12
    return ids_peak_ipl.PixelFormatName_Mono8


def _ids_delivery_significant_bits(wire_format_name: str) -> int:
    """Payload depth the production unpack delivers, paired with
    _ids_delivery_target: Mono12 -> 12 (native uint16); everything else,
    including the 8-bit-mode Mono10 wire, -> 8 (delivered as uint8)."""
    if wire_format_name.startswith('Mono12'):
        return 12
    return 8


def _unpack_buffer(buffer, wire_format_name: str, crop_spec):
    """Unpack one finished IDS buffer to its delivered array.

    BufferToImage + ConvertTo to the delivery target (native uint16 for the
    12-bit modes; 8-bit uint8 directly for the 8-bit-mode Mono10 wire, see
    _ids_delivery_target), crop the oversize-then-crop surplus (``crop_spec`` =
    (x0, y0, w, h), or None for a full-frame delivery), then copy to a
    contiguous, target-sized array that outlives the SDK image and the re-queued
    buffer. Shared by the live unpack worker and the still-capture path so the
    crop is applied identically on both -- the delivered frame matches
    get_frame_size() regardless of which path produced it. center_crop is
    imported here (drivers must not import modules/ at top level); it is a
    sys.modules hit after first use.
    """
    from modules.image_utils import center_crop

    target = _ids_delivery_target(wire_format_name)
    img = ids_peak_ipl_extension.BufferToImage(buffer)
    if img.PixelFormat() != target:
        img = img.ConvertTo(target)
    view = img.get_numpy()
    if crop_spec is not None:
        view = center_crop(view, *crop_spec)
    return view.copy()


# GenTL SFNC-standard DataStream statistics counters read by the diagnostic
# snapshot. The names are GenTL-standard; the access path (DataStream nodemap)
# is bench-confirmed separately -- the snapshot self-reports an access error so
# a bench run shows whether the path resolved.
_DIAG_STREAM_COUNTERS = (
    'StreamDeliveredFrameCount',
    'StreamLostFrameCount',
    'StreamUnderrunCount',
    'StreamStartedFrameCount',
    'StreamAnnouncedBufferCount',
)


# ids_peak NodeAccessStatus codes (the SDK returns the integer enum value).
# Mapped to symbolic names so a log bundle reads 'ReadOnly' not '3'. 3/4 are
# bench-confirmed on the U3-34L (the throughput component reads 3 = ReadOnly,
# the writable feature nodes read 4); 0/1 (NotImplemented/NotAvailable) order
# differs between GenApi and the IDS enum, so unknown codes fall back to a
# labelled integer rather than a guessed name.
_NODE_ACCESS_STATUS_NAMES = {
    2: 'WriteOnly',
    3: 'ReadOnly',
    4: 'ReadWrite',
}


def _access_status_name(raw) -> str:
    """Symbolic name for an ids_peak NodeAccessStatus value; falls back to a
    labelled integer (unknown code) or its string form (non-integer)."""
    try:
        code = int(raw)
    except (TypeError, ValueError):
        return str(raw)
    return _NODE_ACCESS_STATUS_NAMES.get(code, f'AccessStatus({code})')


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

        # Oversize-then-crop framing state. set_frame_size acquires the next
        # legal AOI at or above the request (the sensor's 48-px width grid
        # cannot hit an arbitrary size) and records the centered sub-rectangle
        # (x0, y0, w, h) the unpack worker crops back to exactly what was asked.
        # It is the single source of truth for the delivered (public) frame size
        # -- get_frame_size() reads (w, h) from it, get_acquired_aoi() reports
        # the larger hardware AOI. None before the first set_frame_size or after
        # a geometry change invalidates it (see _invalidate_framing).
        self._crop_spec: tuple[int, int, int, int] | None = None

        # The offset-independent sensor max (width, height), cached from a read
        # taken with the offsets at zero (Width/Height .Maximum() shrinks as the
        # offset grows, so a live read once a centering offset is applied would
        # under-report). Refreshed each set_frame_size; cleared on a geometry
        # change. None until the first set_frame_size (get_max_frame_size then
        # reads live, where the offsets are still at their zero default).
        self._sensor_max: tuple[int, int] | None = None

        # Recovery contract: DeviceLost is the terminal removal signal for this
        # fixed-cable, reconnect-disabled body (uEye+ U3 default). The
        # callback wrapper AND its registration handle must both stay alive -- GC
        # of the wrapper auto-deregisters the callback -- so they are held as
        # attrs. _device_key (the GenTL key captured at open) filters the
        # manager-wide DeviceLost down to our device; _async_teardown_started
        # guards the one-shot deferred close/destroy off the SDK callback thread.
        self._device_key = None
        self._device_lost_callback = None
        self._device_lost_callback_handle = None
        self._async_teardown_started = False
        # _device_serial is the stable device identity across re-enumeration
        # (the GenTL descriptor/key are not): the unplug presence probe and a
        # denied-open retry both re-match by it.
        self._device_serial = None
        # Capability values resolved live from the nodemap at connect, not
        # hardcoded per model: the analog GainSelector enum entry this body
        # actually exposes, and its maximum square binning factor. Defaults are
        # safe pre-connect fallbacks; _query_dynamic_capabilities overwrites them.
        self._gain_selector = None
        self._max_binning = 2
        # True when the model name matched no curated profile and the generic
        # IDS fallback is in use -- gates deriving binning sizes from the nodemap
        # (a curated body keeps its intentional list).
        self._profile_is_generic = False
        # Sticky latch: once a user disconnect() begins it owns the teardown --
        # a removal signal racing it must not schedule a second, competing
        # teardown, and a connect() during the window runs the full bring-up
        # rather than reading the dying handle as connected.
        self._disconnect_requested = False

        super().__init__()

    def connect(self) -> bool:
        global _ids_library_initialized
        # Idempotent: a second connect on a live camera would collide with our
        # own exclusive Control handle (BAD_ACCESS "open already"), and the
        # failure path would then tear down a perfectly healthy connection.
        # "Live" is a truthiness test -- active is False before the first
        # connect and None after a teardown; only an open device handle is
        # truthy -- AND requires that no removal/disconnect teardown owns the
        # lifecycle: during that window the handle is dead-but-not-yet-nulled,
        # and the right move is the full bring-up below (which starts by
        # clearing the stale lifecycle flags). _active read directly: the
        # `active` property takes _state_lock itself.
        with self._state_lock:
            already_connected = bool(self._active) and not (
                self._device_removed or self._async_teardown_started or self._disconnect_requested
            )
        if already_connected:
            _cam_log.info('[CAM Class ] connect skipped: already connected')
            return True
        try:
            # Initialize device manager
            ids_peak.Library.Initialize()
            _ids_library_initialized = True
            self.device_manager = ids_peak.DeviceManager.Instance()
            self.device_manager.Update()

            # Search for devices. The emptiness check and the index are not
            # atomic -- the producer can report a phantom entry that fails to
            # index (seen as `IndexError` on [0] with no camera attached) -- so
            # a failed index IS the no-camera outcome, reported as such rather
            # than as a raw IndexError.
            devices = self.device_manager.Devices()
            if devices.empty():
                raise ConnectionError('Could not find IDS camera')
            try:
                descriptor = devices[0]
            except IndexError as e:
                raise ConnectionError('Could not find IDS camera') from e
            self.active, descriptor = self._open_device_with_retry(
                descriptor,
                refresh=lambda: self._refresh_descriptor_matching(descriptor),
            )
            # The key comes from the descriptor the device was ACTUALLY opened
            # from (a retry may have swapped in a re-enumerated one): DeviceLost
            # is delivered for EVERY device in the system, so the callback
            # filters on this key to act only on our camera's removal.
            self._device_key = descriptor.Key()
            self.data_stream = self.active.DataStreams()[0].OpenDataStream()
            self.remote_nodemap = self.active.RemoteDevice().NodeMaps()[0]
            self._device_removed = False
            self._async_teardown_started = False
            # Fresh lifecycle: clear the sticky disconnect latch left from a
            # prior teardown so a new bring-up never inherits it.
            self._disconnect_requested = False

            # Register the terminal removal signal now that our device is open --
            # events only transmit from registration time, and we cannot lose a
            # device we have not opened yet.
            self._register_device_callbacks()

            try:
                self.model_name = self.active.ModelName()
                # Read the serial from the DESCRIPTOR (the same surface the
                # post-reset re-discovery matches against) so the re-match is
                # symmetric, not from the opened Device which may format it
                # differently.
                self._device_serial = descriptor.SerialNumber()
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
            # Fresh handler starts with an empty dispatch list; re-push any
            # durable listeners (manual record / plugins) so a reconnect on the
            # same instance does not silently stop delivering frames to them.
            self._reapply_frame_callbacks()

            self.init_camera_config()
            # connect() returns CONFIGURED but NOT grabbing; the single
            # start fires later via open_and_start() (the start gate).

            logger.info('[CAM Class ] Connected to IDS camera')
            return True

        except ConnectionError as er:
            _cam_log.warning(f'[CAM Class ] IDS camera connect failed: {er}')
            self._cleanup_partial_connect()
        except Exception as ex:
            # No-device / no-GenTL-path is the common case here and is an
            # expected probe outcome; log the type + message, not the stack.
            _cam_log.error(f'[CAM Class ] IDS camera connect failed: {type(ex).__name__}: {ex}')
            self._cleanup_partial_connect()

        return False

    def _cleanup_partial_connect(self) -> None:
        """Undo a failed/partial connect(): both failure branches route here so a
        removal signal or a stream binding never survives an aborted bring-up.

        Unregisters the DeviceLost callback FIRST, while device_manager is still
        valid: if registration succeeded before a later step threw, the self-bound
        callback would otherwise outlive this aborted connect and fire a spurious
        teardown on the next removal event of any device. Then drops the opened
        handles -- including cam_image_handler, which pins the just-opened data
        stream (leaving it set keeps the USB3 endpoint bound so a retry rebinds
        the same stream). Idempotent: safe when nothing was opened or registered
        (the no-device ConnectionError path runs it before any handle exists).
        """
        self._unregister_device_callbacks()
        self.active = None
        self.remote_nodemap = None
        self.data_stream = None
        self._pixel_format_cache = None
        self.cam_image_handler = None

    def _open_device_with_retry(self, descriptor, refresh=None):
        """OpenDevice(Control) with a short bounded retry on an access denial.

        Control access is exclusive and can be transiently unavailable (see the
        _OPEN_CONTROL_* constants). Only an access-denial fault retries; any
        other open failure propagates immediately, as does the last denial once
        the retry budget is spent -- the caller's failure handling is unchanged.

        Returns ``(device, descriptor)``: the opened device AND the descriptor
        it was actually opened from, because `refresh` (called before each
        retry) may swap in a re-enumerated descriptor and the caller must read
        the serial/key from the descriptor that is live, not the stale one it
        passed in. `refresh` returning None keeps the current descriptor; a
        raise from `refresh` propagates, so a caller can abort the retry from
        inside its refresh callback.
        """
        attempt = 0
        while True:
            try:
                return descriptor.OpenDevice(ids_peak.DeviceAccessType_Control), descriptor
            except Exception as e:
                # Two transient shapes retry: an access DENIAL (the device is
                # exclusively held and the hold is clearing), and NOT_AVAILABLE
                # (the OS driver file is not openable yet -- seen right after a
                # DeviceReset, when the device re-enumerates a beat before its
                # kernel endpoint is ready). Anything else propagates.
                is_transient = _exc_is(
                    e, 'BadAccessException', 'bad_access', 'access denied', 'open already'
                ) or _exc_is(e, 'NotAvailableException', 'not_available', 'not available')
                if not is_transient or attempt >= _OPEN_CONTROL_RETRIES:
                    raise
                attempt += 1
                delay = _OPEN_CONTROL_BACKOFF_S * attempt
                _cam_log.warning(
                    f'[CAM Class ] OpenDevice(Control) transiently unavailable (held or '
                    f'not ready; attempt {attempt}/{_OPEN_CONTROL_RETRIES}), retrying in '
                    f'{delay:.1f}s: {e}'
                )
                time.sleep(delay)
                if refresh is not None:
                    fresh = refresh()
                    if fresh is not None:
                        descriptor = fresh

    def _refresh_descriptor_matching(self, original):
        """Re-enumerate and return the fresh descriptor for the SAME physical
        device `original` points at, or None to keep retrying the one in hand.

        Retry companion for a denied open: the denial can mean the prior
        holder's release re-enumerated the device, invalidating the descriptor
        in hand. The re-match is by serial -- never a blind first-device pick,
        which could bind a different camera on a multi-device system. When the
        serial cannot be read or the walk faults, the answer is None (retry
        the current descriptor); a refresh is an opportunistic improvement,
        never a new failure mode, so it does not raise.
        """
        try:
            serial = original.SerialNumber()
            self.device_manager.Update()
            descriptor, _read_faulted = self._find_descriptor_by_serial(serial)
            return descriptor
        except Exception as e:
            logger.debug(f'[CAM Class ] descriptor refresh for open retry failed: {e}')
            return None

    def disconnect(self) -> bool:
        try:
            # Unregister the DeviceLost callback first, while device_manager is
            # still set -- a stale registration would outlive the camera.
            self._unregister_device_callbacks()
            # Latch ownership of the teardown: a removal signal racing this
            # disconnect must not schedule a second, competing teardown.
            with self._state_lock:
                self._disconnect_requested = True
            if self.active:
                if self._device_removed:
                    # The handle is dead: the remote-nodemap AcquisitionStop /
                    # StopAcquisition / Flush / RevokeBuffers in stop_grabbing()
                    # can hang or abort on a removed device, so skip them and run
                    # only the poll/unpack teardown (KillWait + thread joins),
                    # which never touches the removed remote nodemap.
                    try:
                        if self.cam_image_handler:
                            self.cam_image_handler.stop()
                    except Exception as e:
                        logger.debug(f'[CAM Class ] handler stop on removed device ignored: {e}')
                else:
                    grab_stop_ran = False
                    try:
                        if self.is_grabbing():
                            self.stop_grabbing()
                            grab_stop_ran = True
                    except Exception as e:
                        logger.debug(f'[CAM Class ] stop_grabbing during disconnect ignored: {e}')
                    # Grab-thread quiescence is disconnect's own obligation,
                    # not a side effect of stop_grabbing(): is_grabbing()
                    # reads the ACQUISITION state, which a failed-start
                    # rollback can leave False while the grab threads are
                    # still alive -- and a live thread pins the data stream,
                    # keeping the device's exclusive Control claim held past
                    # this disconnect so the next OpenDevice(Control) collides
                    # with our own zombie hold. Quiesce directly whenever
                    # stop_grabbing() (which joins the threads itself) did not
                    # run, so the worst case stays ONE join ceiling, never two.
                    # stop() is idempotent and non-raising (its SDK touches
                    # are self-guarded).
                    if not grab_stop_ran and self.cam_image_handler is not None:
                        self.cam_image_handler.stop()
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
        finally:
            # Always drop the handler -- it pins the data stream, and ids_peak
            # exposes no explicit DataStream/Device Close() (release is by dropping
            # the last Python reference). A partial connect (handler built, active
            # never set), an error mid-teardown, or a not-connected call must not
            # leave the stream (USB3 endpoint) bound, or a later reconnect rebinds
            # the same stream instead of a fresh one. On the active paths the grab
            # threads were already quiesced above; a partial-connect handler was
            # never started, so dropping the reference is the only teardown needed.
            self.cam_image_handler = None
            # Clear the start gate + last-frame buffer on EVERY path (happy,
            # not-connected, or error): a removal teardown can null active
            # while the gate is still OPEN, so a same-instance reconnect must
            # not inherit a stranded-open gate (blank live view) or a stale
            # image.
            self._reset_lifecycle_state()
        return False

    def _register_device_callbacks(self) -> None:
        """Register the terminal DeviceLost signal on the DeviceManager.

        DeviceLost is the only removal event this body emits: reconnect is
        disabled by default on uEye+ U3 cameras, so an unplug fires DeviceLost
        (DeviceDisconnected fires only with reconnect enabled, which we never turn
        on). Both the wrapper object AND the returned handle are stored -- if the
        wrapper is garbage-collected the SDK silently deregisters the callback.
        Guarded so an SDK build without the API degrades to the in-loop typed
        DeviceLostException fallback rather than failing the connect.

        Future reconnect support extends here: enable ReconnectEnable and also
        register DeviceDisconnected/DeviceReconnected callbacks. The
        _handle_device_lost single owner and this register/unregister scaffolding
        are reusable as-is.
        """
        dm = self.device_manager
        if dm is None:
            return
        try:
            self._device_lost_callback = dm.DeviceLostCallback(self._on_device_lost)
            self._device_lost_callback_handle = dm.RegisterDeviceLostCallback(
                self._device_lost_callback
            )
            logger.info('[CAM Class ] Registered DeviceLost callback')
        except Exception as e:
            self._device_lost_callback = None
            self._device_lost_callback_handle = None
            logger.debug(f'[CAM Class ] DeviceLost callback unavailable: {e}')

    def _unregister_device_callbacks(self) -> None:
        """Deregister the DeviceLost callback by handle and drop the refs.

        Idempotent -- safe when nothing was registered or device_manager is gone.
        """
        dm = self.device_manager
        handle = self._device_lost_callback_handle
        if dm is not None and handle is not None:
            try:
                dm.UnregisterDeviceLostCallback(handle)
            except Exception as e:
                logger.debug(f'[CAM Class ] DeviceLost unregister ignored: {e}')
        self._device_lost_callback = None
        self._device_lost_callback_handle = None

    def _on_device_lost(self, key) -> None:
        """DeviceManager DeviceLost handler -- fires on an SDK-owned thread.

        Receives the GenTL device key of the lost device; DeviceLost is delivered
        for every device in the system, so act only when it is ours. Wrapped in
        BaseException because the SDK swallows callback exceptions -- a leak here
        would silently drop the removal signal.

        The non-matching branch logs too: without it, "DeviceLost never fired"
        and "fired but the key did not match ours" are indistinguishable in a
        bundle -- and a key mismatch would silently strand a real unplug as a
        display stall instead of a clean disconnect. Logged at INFO (not debug)
        so it survives a debug-off field run; DeviceLost only fires on an actual
        device-removal event, so it is not a per-frame chatter source.
        """
        try:
            if key != self._device_key:
                _cam_log.info(
                    f'[CAM Class ] DeviceLost fired for another device '
                    f'(key={key}, ours={self._device_key}); ignoring'
                )
                return
            _cam_log.warning(f'[CAM Class ] DeviceLost callback for our device (key={key})')
            self._handle_device_lost()
        except BaseException as e:
            try:
                _cam_log.error(f'[CAM Class ] _on_device_lost guard caught {type(e).__name__}: {e}')
            except BaseException:
                pass

    def _handle_device_lost(self) -> None:
        """Single owner of camera removal AND of the uniform wedge drop: flip
        the disconnected flag (cheap, thread-safe) and schedule the
        close/destroy off the callback thread.

        Triggered by the DeviceLost callback (SDK thread); by a typed
        DeviceLostException in the poll loop; by the sustained-stall presence
        probe; and by every dead-stream ceiling -- the invalid-handle wedge
        classifiers (wait + buffer paths), the only-bad-frames ceiling, and
        the hard stall ceiling. A wedge deliberately exits here as a clean
        disconnect the user recovers by reconnecting. All idempotent via the
        _device_removed and _async_teardown_started guards.
        """
        with self._state_lock:
            if self._disconnect_requested:
                # A user disconnect() is already in flight and owns the teardown;
                # a removal signal racing it (the presence probe finishing its
                # enumeration after disconnect() began, or a late DeviceLost) must
                # not schedule a second, competing teardown.
                return
        self._mark_disconnected()
        self._schedule_async_teardown()

    def _probe_device_presence(self):
        """Actively check whether our device is still enumerated. PURE query:
        returns True (present), False (absent), or None (inconclusive) -- it
        never latches removal or tears down. The caller (the poll thread) owns
        the confirm-twice debounce and the routing to the single removal owner.

        Called on a SUSTAINED stall, because DeviceLost does not fire on a USB
        unplug for the U3-34L -- without this a real removal is indistinguishable
        from a wedged-but-present stall and the live view hangs until the next
        explicit op. This is the USB-unplug fallback; it does NOT replace
        DeviceLost (still the owner when it fires). The producer can also keep a
        PHANTOM entry enumerated after an unplug, so a present reading is not
        proof of life either -- the hard stall ceiling owns that case.

        Returns None (inconclusive, NOT proof of removal) when a
        teardown/disconnect has begun, when there is no device manager / serial
        yet, or when the Update/enumeration itself faults. A single truncated
        Update must never be read as removal -- that is why the caller requires
        repeated absences (_PRESENCE_PROBE_CONFIRMATIONS) before it concludes a
        removal.
        """
        with self._state_lock:
            if self._device_removed or self._async_teardown_started or self._disconnect_requested:
                return None
        dm = self.device_manager
        serial = self._device_serial
        if dm is None or not serial:
            return None
        try:
            # A short update timeout bounds how long this blocks the poll thread;
            # restored in finally so a later connect() Update() on this
            # process-wide singleton never inherits the tightened value (which
            # would risk missing a present-but-slow-to-enumerate camera).
            try:
                dm.SetDeviceUpdateTimeout(_PRESENCE_PROBE_UPDATE_TIMEOUT_MS)
            except Exception as e:
                logger.debug(
                    f'[CAM Class ] presence probe: SetDeviceUpdateTimeout unavailable: {e}'
                )
            try:
                dm.Update()
                descriptor, read_faulted = self._find_descriptor_by_serial(serial)
                if descriptor is not None:
                    return True
                # Our serial was not in the list. Only call that ABSENT if the
                # walk was clean: a descriptor read fault means we may have failed
                # to read OUR descriptor's serial, so the device could still be
                # present -- inconclusive (None), never a step toward removal.
                return None if read_faulted else False
            finally:
                try:
                    dm.SetDeviceUpdateTimeout(_DEVICE_UPDATE_TIMEOUT_MS)
                except Exception as e:
                    logger.debug(
                        f'[CAM Class ] presence probe: update-timeout restore unavailable: {e}'
                    )
        except Exception as e:
            # An Update/enumeration fault is not proof of removal; inconclusive.
            logger.debug(f'[CAM Class ] presence probe inconclusive: {e}')
            return None

    def _find_descriptor_by_serial(self, serial):
        """Walk the enumerated devices for one whose serial matches.

        Returns (descriptor_or_None, read_faulted): the matching descriptor (or
        None if no clean match), and whether any per-descriptor SerialNumber()
        read raised. Shared by the denied-open descriptor refresh (which ignores
        read_faulted and just keeps the descriptor it has) and the unplug
        presence probe (which treats a fault-without-match as inconclusive, not
        absent, so a transient read fault can't drive a false removal). The
        caller must call device_manager.Update() first; this only walks the
        current Devices() list.
        """
        read_faulted = False
        for descriptor in list(self.device_manager.Devices()):
            try:
                if descriptor.SerialNumber() == serial:
                    return descriptor, read_faulted
            except Exception as e:
                read_faulted = True
                logger.debug(f'[CAM Class ] descriptor read ignored: {e}')
        return None, read_faulted

    def _schedule_async_teardown(self) -> None:
        """Run disconnect() (the SDK close/destroy) on a daemon thread.

        IDS's own callbacks may reconfigure the stream inline, but never close or
        destroy the device on the callback thread -- so the lightweight
        _mark_disconnected runs inline while the close/destroy defers here. The
        _async_teardown_started latch makes this one-shot under concurrent
        triggers (the callback and the poll-loop fallback).
        """
        with self._state_lock:
            if self._async_teardown_started:
                return
            self._async_teardown_started = True

        def _run_teardown():
            try:
                time.sleep(0.05)  # let the SDK callback return before close/destroy
                self.disconnect()
            except Exception as e:
                logger.debug(f'[CAM Class ] async teardown ignored: {e}')

        threading.Thread(target=_run_teardown, name='IDSAsyncTeardown', daemon=True).start()

    def is_connected(self) -> bool:
        if self.active in (False, None):
            self._device_removed = True
            return False
        return not self._device_removed

    def _diag_node_value(self, name: str, enum: bool = False):
        """Read one remote-nodemap value for the snapshot, sentinel on failure.

        Mirrors the Pylon snapshot's ``_safe_node``: a missing/unreadable node
        records a ``<missing>`` string instead of raising, so one absent node
        never aborts the whole snapshot. ``enum=True`` reads the current
        symbolic entry (PixelFormat, throughput component).
        """
        try:
            node = self.remote_nodemap.FindNode(name)
            if node is None:
                return '<missing>'
            if enum:
                return node.CurrentEntry().SymbolicValue()
            return node.Value()
        except Exception as e:
            return f'<missing: {type(e).__name__}>'

    def _diag_probe_node(self, nodemap, name: str) -> dict:
        """Probe one optional node's presence + access on a given nodemap.

        Reports whether the node exists on THIS body, its access status
        (writable vs read-only vs not-available), and its current value or
        symbolic entries. Some capabilities a setter might want -- an internal
        test pattern, chunk data, USB3 transfer-size tuning -- are exposed only
        on certain bodies, and a node the IDS manual advertises can still be
        absent or read-only on a given camera. This lets a single bench snapshot
        confirm a node is actually usable here before any setter is wired to it.
        Never raises: a missing node records ``present=False``.
        """
        try:
            node = nodemap.FindNode(name)
        except Exception as e:
            return {'present': False, 'detail': f'{type(e).__name__}: {e}'}
        if node is None:
            return {'present': False}
        info: dict = {'present': True}
        try:
            info['access'] = (
                _access_status_name(node.AccessStatus())
                if hasattr(node, 'AccessStatus')
                else '<no AccessStatus>'
            )
        except Exception as e:
            info['access'] = f'<error: {type(e).__name__}>'
        # Enum nodes report symbolic entries; scalar nodes a value. Resolve the
        # enum shape first and fall back to a scalar read on any failure, so a
        # node is never left both entry- and value-shaped (a non-enum node that
        # happens to expose AvailableEntries falls through cleanly to Value()).
        try:
            entries = [entry.SymbolicValue() for entry in node.AvailableEntries()]
        except Exception:
            entries = None
        if entries is not None:
            info['entries'] = entries
            try:
                info['current'] = node.CurrentEntry().SymbolicValue()
            except Exception as e:
                info['current'] = f'<error: {type(e).__name__}>'
        else:
            try:
                info['value'] = node.Value()
            except Exception as e:
                info['value'] = f'<error: {type(e).__name__}>'
        return info

    def _probe_feature_nodes(self) -> dict:
        """Presence + access of the optional-feature candidate nodes on this
        body, grouped by nodemap (remote, then DataStream). A few capabilities a
        setter or diagnostic might want -- an internal test pattern, chunk
        data, USB3 transfer-size tuning, an in-software DeviceReset -- are exposed
        only on some IDS bodies and are not confirmed on the U3-34L0XCP-M. Used by
        both the diagnostic snapshot and the one-shot free-run log so a
        normal-startup bundle already shows which exist + are writable, without a
        separate probe run. DeviceReset is a command node (no readable value); its
        presence + access is the signal -- whether a reset could be issued in
        software at all, rather than by physical replug.
        """
        remote = {
            name: self._diag_probe_node(self.remote_nodemap, name)
            for name in ('TestPattern', 'ChunkModeActive', 'ChunkSelector', 'DeviceReset')
        }
        stream: dict = {}
        try:
            nodemaps = self.data_stream.NodeMaps()
            stream_nm = nodemaps[0] if nodemaps else None
        except Exception as e:
            stream_nm = None
            stream['_access_error'] = f'NodeMaps() raised: {type(e).__name__}: {e}'
        if stream_nm is None:
            # Distinguish 'nodemap never resolved' from 'probed and absent' so
            # the record stays honest about which (mirrors _read_stream_stats).
            stream.setdefault('_access_error', 'no DataStream nodemaps')
        else:
            for name in (
                'U3vStreamChannelBulkTransferSize',
                'U3vStreamChannelTransferRequestCount',
            ):
                stream[name] = self._diag_probe_node(stream_nm, name)
        return {'remote': remote, 'stream': stream}

    def _read_stream_stats(self) -> dict:
        """Read the GenTL DataStream statistics counters, defensively.

        Counter names are GenTL SFNC-standard; the DataStream-nodemap access
        path is not exercised elsewhere in this driver, so an access failure
        is recorded under ``_access_error`` (the snapshot stays honest about
        whether the path resolved on this body/SDK) rather than raising. Each
        counter reads to a ``<missing>`` sentinel so the delta math downstream
        skips it cleanly.
        """
        try:
            nodemaps = self.data_stream.NodeMaps()
        except Exception as e:
            return {'_access_error': f'NodeMaps() raised: {type(e).__name__}: {e}'}
        if not nodemaps:
            return {'_access_error': 'no DataStream nodemaps'}
        stream_nm = nodemaps[0]
        stats: dict = {}
        for name in _DIAG_STREAM_COUNTERS:
            try:
                node = stream_nm.FindNode(name)
                stats[name] = node.Value() if node is not None else '<missing>'
            except Exception as e:
                stats[name] = f'<missing: {type(e).__name__}>'
        return stats

    def read_diagnostic_snapshot(
        self,
        duration_s: float = 3.0,
        drain_camera_side_errors: bool = True,
    ) -> dict:
        """Capture a single diagnostic snapshot of camera + stream state.

        Parity with the Pylon driver: reads camera identity, current
        configuration, temperatures, and buffer-pool state, then samples the
        GenTL DataStream statistics counters across a ``duration_s`` window and
        computes per-counter deltas + derived rates (observed_fps,
        loss_rate_pct, losses_per_second). Does NOT change grab state; when the
        camera is not grabbing the deltas are near-zero (counters do not
        advance without an active grab loop), a sentinel rather than an error.

        Every node read is defensive (``_diag_node_value`` / ``_read_stream_stats``)
        so a missing node records a sentinel rather than raising.
        ``drain_camera_side_errors`` is accepted for signature parity with the
        Pylon driver; IDS exposes no equivalent camera-side error log, so it is
        a no-op here.
        """
        result: dict = {
            'connected': False,
            'supported': True,
            'duration_s_requested': float(duration_s),
            'duration_s_actual': 0.0,
            'camera': {},
            'config': {},
            'temperatures': {},
            'buffers': {},
            'stats_pre': {},
            'stats_post': {},
            'deltas': {},
            'derived': {},
            'feature_nodes': {},
            'errors': [],
        }

        if not self.active or self.remote_nodemap is None:
            result['errors'].append('camera not connected')
            return result
        result['connected'] = True

        for name, key in (
            ('DeviceModelName', 'model_name'),
            ('DeviceSerialNumber', 'serial'),
            ('DeviceFirmwareVersion', 'firmware_version'),
            ('DeviceVersion', 'device_version'),
        ):
            result['camera'][key] = self._diag_node_value(name)

        for name, key, is_enum in (
            ('PixelFormat', 'pixel_format', True),
            ('Width', 'width', False),
            ('Height', 'height', False),
            ('ExposureTime', 'exposure_us', False),
            ('Gain', 'gain', False),
            ('DeviceLinkThroughputLimit', 'dltl_value_bps', False),
            ('DeviceLinkThroughputLimitComponent', 'dltl_component', True),
            ('AcquisitionFrameRate', 'acquisition_frame_rate', False),
            ('PayloadSize', 'payload_size_bytes', False),
            ('BinningVertical', 'binning_vertical', False),
            ('BinningHorizontal', 'binning_horizontal', False),
        ):
            result['config'][key] = self._diag_node_value(name, enum=is_enum)

        result['temperatures'] = self.get_all_temperatures()

        # Buffer-pool state from proven DataStream accessors.
        for accessor, key in (
            (lambda: self.data_stream.IsGrabbing(), 'is_grabbing'),
            (lambda: len(self.data_stream.AnnouncedBuffers()), 'announced_count'),
            (lambda: self.data_stream.NumBuffersAnnouncedMinRequired(), 'min_required'),
        ):
            try:
                result['buffers'][key] = accessor()
            except Exception as e:
                result['buffers'][key] = f'<missing: {type(e).__name__}>'

        # Optional-feature candidate nodes (test pattern, chunk data, USB3
        # transfer-size tuning) -- presence + access on this body, so the
        # snapshot settles whether a setter can rely on each one.
        result['feature_nodes'] = self._probe_feature_nodes()

        # Statistics sampling window.
        result['stats_pre'] = self._read_stream_stats()
        t0 = time.monotonic()
        try:
            if duration_s > 0:
                time.sleep(duration_s)
        except Exception as e:
            result['errors'].append(f'sleep raised: {type(e).__name__}: {e}')
        dt = time.monotonic() - t0
        result['duration_s_actual'] = dt
        result['stats_post'] = self._read_stream_stats()

        # Deltas only where both pre and post returned numeric counters AND the
        # counter did not go backwards: GenTL counters reset to 0 on
        # StartAcquisition, so post < pre means a stop/start happened inside the
        # window and the delta is meaningless (recorded None) rather than a
        # negative rate.
        for name in _DIAG_STREAM_COUNTERS:
            pre = result['stats_pre'].get(name)
            post = result['stats_post'].get(name)
            if isinstance(pre, (int, float)) and isinstance(post, (int, float)) and post >= pre:
                result['deltas'][name] = post - pre
            else:
                result['deltas'][name] = None

        # Derived rates, only when a real sampling window was requested
        # (duration_s > 0). Gating on dt alone (always > 0) would emit bogus
        # rates over a ~microsecond span for a 0-duration snapshot, reading as a
        # measured-and-clean stream that was never actually observed.
        delivered_d = result['deltas'].get('StreamDeliveredFrameCount')
        lost_d = result['deltas'].get('StreamLostFrameCount')
        if duration_s > 0 and dt > 0:
            if isinstance(delivered_d, (int, float)):
                result['derived']['observed_fps'] = delivered_d / dt
                total = delivered_d + lost_d if isinstance(lost_d, (int, float)) else None
                if isinstance(total, (int, float)) and total > 0:
                    result['derived']['loss_rate_pct'] = 100.0 * lost_d / total
            if isinstance(lost_d, (int, float)):
                result['derived']['losses_per_second'] = lost_d / dt

        return result

    def _load_profile(self):
        """Resolve the static profile, substituting a generic IDS profile when
        the model name matched no curated entry.

        The base lookup falls through to a cross-vendor Mono8 'Unknown' profile
        for an unrecognized model; for an IDS body that is wrong (packed
        Mono10/12, IDS AOI granularity). Substitute the generic IDS profile so
        the body is still driven as IDS, then let _query_dynamic_capabilities /
        init_camera_config fill the capability fields from the live nodemap.
        """
        super()._load_profile()
        if self.profile.driver != 'ids':
            from drivers.camera_profiles import ids_default_profile

            self._profile_is_generic = True
            _cam_log.warning(
                f'[CAM Class ] Unrecognized IDS model {self.model_name!r}; using the '
                f'generic IDS profile (capabilities read live from the nodemap). '
                f'Add a profile entry for curated sensor metadata.'
            )
            self.profile = ids_default_profile(self.model_name)

    @staticmethod
    def _select_gain_selector_name(available, preferred='AnalogAll'):
        """Pick the GainSelector enum entry to drive analog gain, from the body's
        live entries -- so a body that names it 'All' or some 'Analog*' variant
        instead of the literal 'AnalogAll' still gets gain control. The old
        hardcoded 'AnalogAll' silently failed every gain write on such a body.

        preferred if present; else the first entry containing 'Analog'; else
        'All' if present; else the first advertised entry. None only when the
        enum advertises nothing. Pure-logic + staticmethod for unit-testability.
        """
        if preferred and preferred in available:
            return preferred
        for entry in available:
            if 'Analog' in entry:
                return entry
        if 'All' in available:
            return 'All'
        return available[0] if available else None

    def _resolve_gain_selector(self):
        """Resolve the analog GainSelector entry from the live nodemap enum, or
        None if the body has no GainSelector / the read fails."""
        try:
            entries = tuple(
                e.SymbolicValue()
                for e in self.remote_nodemap.FindNode('GainSelector').AvailableEntries()
            )
            preferred = 'AnalogAll'
            gain = getattr(self.profile, 'gain', None)
            if gain and getattr(gain, 'gain_selector', None):
                preferred = gain.gain_selector
            return self._select_gain_selector_name(entries, preferred)
        except Exception as e:
            _cam_log.debug(f'[CAM Class ] Could not resolve GainSelector: {e}')
            return None

    def _query_dynamic_capabilities(self):
        """Query IDS SDK for gain/exposure ranges and merge into profile."""
        if not self.active or not self.remote_nodemap:
            return

        try:
            # Resolve the analog GainSelector entry this body exposes (cached
            # for gain()), and select it BEFORE reading the Gain range so the
            # range reflects the selector the setter will use.
            self._gain_selector = self._resolve_gain_selector()
            if self._gain_selector:
                try:
                    self.remote_nodemap.FindNode('GainSelector').SetCurrentEntry(
                        self._gain_selector
                    )
                except Exception as e:
                    logger.debug(f'[CAM Class ] Could not pre-select GainSelector: {e}')

            # Gain range. The IDS Gain node is a linear multiplier (min ~1.0x)
            # on the IMX676 bodies, but LVP's gain model is dB (shared with the
            # Pylon driver), so the profile range is reported in dB:
            # dB = 20 * log10(factor). The linear-multiplier assumption is
            # per-body: SFNC permits Gain to be expressed natively in dB, and a
            # body that does so would be double-converted here. Reading the Gain
            # node's reported unit to branch the conversion is the robust fix but
            # needs a bench body to confirm the unit string -- deferred until one
            # is available; gain stays correct on the IMX676 bodies meanwhile.
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

            # Sensor pixel pitch (micrometers). Read live so the micron scale
            # bar and click-to-center distance are correct for ANY IDS body,
            # not only ones with a curated profile. SensorPixelWidth is a Float
            # node in um (IDS peak ImageFormatControl). Fill only when the
            # profile did not supply one (an unrecognized body's generic profile
            # carries 0.0) -- never override a curated, bench-validated value --
            # but always log the hardware value so a mismatch is visible.
            try:
                sensor_px_um = self.remote_nodemap.FindNode('SensorPixelWidth').Value()
                logger.info(f'[CAM Class ] SensorPixelWidth: {sensor_px_um:.4f} um')
                if sensor_px_um and sensor_px_um > 0 and not self.profile.pixel_size_um:
                    self.profile.pixel_size_um = float(sensor_px_um)
            except Exception as e:
                logger.debug(f'[CAM Class ] Could not query SensorPixelWidth: {e}')

            # Binning ceiling. Keep a recognized body's CURATED sizes (the
            # setter cap follows them); for the generic fallback only, derive the
            # ceiling from the live node so an unrecognized body that supports 4x
            # is not refused. Advertise the common power-of-two grid up to the
            # ceiling rather than a contiguous range -- never offer an
            # intermediate factor (e.g. 3x) the node would reject on SetValue.
            self._max_binning = max(self.profile.binning_sizes) if self.profile.binning_sizes else 1
            if self._profile_is_generic:
                try:
                    bv_max = int(self.remote_nodemap.FindNode('BinningVertical').Maximum())
                    bh_max = int(self.remote_nodemap.FindNode('BinningHorizontal').Maximum())
                    max_bin = min(bv_max, bh_max)
                    if max_bin >= 1:
                        self._max_binning = max_bin
                        self.profile.binning_sizes = [s for s in (1, 2, 4, 8, 16) if s <= max_bin]
                        logger.info(
                            f'[CAM Class ] Binning ceiling {max_bin}x '
                            f'(sizes {self.profile.binning_sizes})'
                        )
                except Exception as e:
                    logger.debug(f'[CAM Class ] Could not query binning range: {e}')

            # Two Pylon-parity capabilities are deliberately NOT resolved here:
            # they are Basler-proprietary (Bsl*) nodes with no SFNC-standard (or
            # IDS) equivalent the live nodemap advertises, so they are genuinely
            # absent from the IDS surface -- a capability-derived absence, not a
            # stubbed model assumption:
            #   - conversion-gain mode (pylon set_conversion_gain_mode /
            #     BslConversionGainMode): a dual high/low charge-to-voltage gain
            #     select. The IMX676 exposes only the standard Gain node (resolved
            #     above); there is no advertised conversion-gain selector to bind.
            #   - line-noise reduction (pylon set_line_noise_reduction /
            #     BslLineNoiseReductionEnable): a Basler readout-FPN filter. No
            #     equivalent IDS node, and the production unpack does no host-side
            #     line-noise correction.
            # Consistent with every capability above, a future IDS body that
            # advertised an equivalent node would be wired in here alongside
            # gain/exposure/pixel-pitch/binning -- never assumed from the model
            # string.

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
                # Mono8). Operators need this in the log to diagnose any future
                # logical-to-camera mismatch.
                supported = self.get_supported_pixel_formats()
                logger.info(f'[CAM Class ] Supported PixelFormat entries: {list(supported)}')
                # Couple to the live nodemap, not a model-keyed static profile:
                # any IDS body is driven by what its PixelFormat node actually
                # advertises, the same way gain / exposure / AOI are read live.
                # Record the real formats on the profile so a body whose model
                # string matched no static profile entry still reports hardware
                # truth downstream and selects a valid format here.
                if supported:
                    self.profile.pixel_formats = list(supported)
                    preferred = self._select_default_pixel_format(supported)
                else:
                    # The live AvailableEntries read returned nothing (transient
                    # SDK error). Fall back to the curated profile's formats
                    # rather than leaving the format unset -- an unset format
                    # streams at the UserSet default depth and would feed the
                    # unpack pipeline the wrong bit depth.
                    _cam_log.warning(
                        '[CAM Class ] Live PixelFormat read empty; falling back '
                        'to the profile formats'
                    )
                    preferred = self._select_default_pixel_format(self.profile.pixel_formats)
                if preferred:
                    self.set_pixel_format(preferred)
                else:
                    _cam_log.error(
                        '[CAM Class ] No usable mono PixelFormat available; '
                        'leaving the format unset'
                    )
                # Apply the horizontal flip only when this body advertises
                # ReverseX as writable -- a capability check via the node's
                # access status, not an assumption every body has it. Orientation
                # nodes vary by body/mount; gating here keeps a body without a
                # writable ReverseX from skipping the rest of init (TriggerMode,
                # exposure, set_frame_size, free-run config all follow).
                _reverse_x = self._diag_probe_node(self.remote_nodemap, 'ReverseX')
                if _reverse_x.get('access') in ('ReadWrite', 'WriteOnly'):
                    self.remote_nodemap.FindNode('ReverseX').SetValue(True)
                else:
                    logger.info(
                        f'[CAM Class ] ReverseX not writable on this body '
                        f'({_reverse_x.get("access", "absent")}); leaving default orientation'
                    )
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

    def _assume_dead_stall_s(self) -> float:
        """The stall age past which teardown treats the device as dead.

        Scaled to the current wait window rather than fixed: the poll wait is
        exposure-scaled, so at a long exposure a single healthy inter-frame gap
        could exceed a fixed threshold and a live camera would have its stop
        misclassified. Requires at least a few consecutive empty wait windows,
        never less than the configured floor.
        """
        handler = self.cam_image_handler
        timeout_s = (handler.timeout_ms / 1000.0) if handler is not None else 0.0
        return max(_TEARDOWN_ASSUME_DEAD_STALL_S, 3.0 * timeout_s)

    def stop_grabbing(self):
        # Same owner-side connected-device precondition as start_grabbing:
        # multiple independent callers may stop a camera whose connect failed,
        # and the None handles below would otherwise surface as a spurious
        # "stop_grabbing ignored error" implying an SDK fault that never was.
        # Logged at info, not warning like the start-side twin: a skipped stop
        # leaves nothing running, while a skipped start costs the live view.
        if self.active is None or self.remote_nodemap is None or self.data_stream is None:
            _cam_log.info('[CAM Class ] stop_grabbing skipped: camera not connected')
            return
        # A stream stalled past the assume-dead threshold means the device has
        # stopped answering (an unplug the producer still enumerates as a
        # phantom entry, or an unrecoverable wedge). The REMOTE-device writes
        # below (AcquisitionStop node execute, TLParamsLocked) each block the
        # calling thread ~10 s against dead hardware before failing -- on the
        # app-close path that reads as a frozen UI and ends in a force-quit --
        # so they are skipped when suspect. Everything HOST-side still runs:
        # the grab-thread quiesce, the local StopAcquisition (which keeps
        # is_grabbing() honest so a later start_grabbing can restart a
        # merely-stalled camera), and the pool flush/revoke (local handle
        # only). The skipped TLParamsLocked release self-heals: the next
        # start_grabbing re-asserts the lock.
        handler = self.cam_image_handler
        stall_age = handler.stall_age_s() if handler is not None else 0.0
        device_suspect = stall_age >= self._assume_dead_stall_s()
        if device_suspect:
            _cam_log.warning(
                f'[CAM Class ] stop_grabbing: stream stalled {stall_age:.0f}s -- device '
                'presumed dead; skipping remote-device writes, host-side stop only'
            )
        elif _cam_log is not None:
            _cam_log.info('ids AcquisitionStop + StopAcquisition + Flush + RevokeBuffers')
        try:
            # When no handler exists yet, treat as quiesced (nothing to race).
            threads_quiesced = True
            if self.cam_image_handler:
                threads_quiesced = self.cam_image_handler.stop()

            if not device_suspect:
                self.remote_nodemap.FindNode('AcquisitionStop').Execute()
                self.remote_nodemap.FindNode('AcquisitionStop').WaitUntilDone()
            self.data_stream.StopAcquisition()

            # Release the transport-layer parameter lock taken in start_grabbing
            # (IDS brackets acquisition with TLParamsLocked 1/0).
            if not device_suspect:
                try:
                    self.remote_nodemap.FindNode('TLParamsLocked').SetValue(0)
                except Exception as e:
                    logger.debug(f'[CAM Class ] TLParamsLocked=0 not available: {e}')

            # Revoke ONLY once the poll + worker threads are provably dead.
            # Revoking under a live consumer leaves it holding an invalid buffer
            # handle (the IDSPoll InvalidInstanceException). If the threads could
            # not be quiesced, leave the buffers announced rather than pull them
            # out from under a running thread -- the next start_grabbing() reclaims
            # the stale pool once the threads have quiesced (it must, or a fresh
            # pool announced on top grows the announced count every cycle until the
            # stream wedges).
            if threads_quiesced:
                self._flush_and_revoke_pool()
            else:
                self._breadcrumb_unrevoked_pool()
        except Exception as e:
            _cam_log.warning(f'[CAM Class ] stop_grabbing ignored error: {e}')

    def _breadcrumb_unrevoked_pool(self) -> None:
        """Log the count of buffers left announced when a stop skipped revoke
        (threads not quiesced), so a bench run can watch that the next clean start
        reclaims them rather than the count climbing cycle over cycle. The count
        query is self-guarded so the breadcrumb still emits even if AnnouncedBuffers()
        raises -- the whole point is that this warning is visible in the log.
        """
        try:
            count = len(self.data_stream.AnnouncedBuffers())
        except Exception as e:
            count = -1
            logger.debug(f'[CAM Class ] AnnouncedBuffers query for breadcrumb failed: {e}')
        _cam_log.warning(
            f'[CAM Class ] stop_grabbing left {count if count >= 0 else "?"} buffers '
            'announced (grab threads not quiesced, revoke skipped); the next clean '
            'start_grabbing() reclaims them'
        )

    def _flush_and_revoke_pool(self) -> int:
        """Flush the data stream and revoke every announced buffer; return the
        count revoked. The single owner of the flush-then-revoke idiom, so its
        two safety properties hold everywhere it is used (stop_grabbing, rollback,
        stale-pool reclaim):

          - Snapshots AnnouncedBuffers() with list() BEFORE revoking, so revoking
            cannot mutate the collection mid-iteration and skip buffers.
          - Guards each RevokeBuffer independently, so one failure does not strand
            the rest of the pool.

        Touches only the LOCAL data-stream handle (never the remote nodemap), so
        it is safe on any valid stream even after a removed-device path. The
        caller MUST hold the contract that no grab thread is live (a revoke under
        a live consumer strands it with an invalid handle). Propagates a failure
        of the initial AnnouncedBuffers() query -- the caller decides whether that
        is fatal (start_grabbing aborts rather than stack a fresh pool on top).
        """
        announced = list(self.data_stream.AnnouncedBuffers())
        try:
            self.data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
        except Exception as e:
            logger.debug(f'[CAM Class ] Flush during pool revoke ignored: {e}')
        for buffer in announced:
            try:
                self.data_stream.RevokeBuffer(buffer)
            except Exception as e:
                logger.debug(f'[CAM Class ] RevokeBuffer ignored: {e}')
        return len(announced)

    def _reclaim_stale_pool(self) -> bool:
        """Flush + revoke any buffers a prior stop left announced, before a fresh
        pool is allocated. Called from start_grabbing only after the grab threads
        are confirmed quiesced, so no live consumer holds a buffer. Touches only
        the local data-stream handle (never the remote nodemap), so it is safe on
        any valid stream even after a removed-device path. No-op when the pool is
        already empty (the clean-stop case, where stop_grabbing already revoked).

        Returns True when the pool is confirmed clear (empty or revoked), False
        when it could not be queried/reclaimed. The caller MUST NOT announce a
        fresh pool on a False -- doing so stacks it on the un-reclaimed stale pool,
        which is the leak that grows the announced count until the stream wedges.
        """
        if self.data_stream is None:
            return True
        try:
            count = self._flush_and_revoke_pool()
        except Exception as e:
            logger.debug(f'[CAM Class ] stale-pool reclaim query failed: {e}')
            return False
        if count:
            _cam_log.warning(
                f'[CAM Class ] Reclaimed {count} buffers left announced by a prior '
                'stop that skipped revoke, before announcing a fresh pool'
            )
        return True

    def start_grabbing(self):
        # Connected-device precondition, owned HERE because this is a public
        # restartable primitive with several independent call sites (the
        # bring-up start gate, the API-layer streaming restart, diagnostics)
        # -- gating each caller would re-scatter the same check. The
        # pylon driver's start_grabbing carries the identical owner-side guard;
        # both drivers share the base contract that start_grabbing is safe to
        # fire on a camera whose connect failed, declining loudly rather than
        # dressing the None handles up as a failed start + rollback.
        if self.active is None or self.remote_nodemap is None or self.data_stream is None:
            _cam_log.warning('[CAM Class ] start_grabbing skipped: camera not connected')
            return
        if self.is_grabbing():
            if _cam_log is not None:
                _cam_log.info('ids start_grabbing SKIPPED: already grabbing')
            return
        # A prior stop() whose grab threads never quiesced leaves the stream
        # wedged: its buffer pool is un-revoked AND its threads still hold the data
        # stream. Announcing a fresh pool on top is the leak that grows the
        # announced count every cycle, and a fresh consumer cannot start under the
        # live one. Abort cleanly instead of stacking; the wedge ceilings own the
        # escalation to the removal drop. (Not hit in normal reconfigure:
        # stop_grabbing blocks on the join ceiling, so a returning stop means
        # threads already died.)
        if self.cam_image_handler is not None and not self.cam_image_handler.threads_quiesced():
            _cam_log.error(
                '[CAM Class ] start_grabbing aborted: prior grab threads have not '
                'quiesced (stream wedged); not stacking a fresh buffer pool'
            )
            return
        if _cam_log is not None:
            _cam_log.info('ids start_grabbing: alloc buffers + StartAcquisition + AcquisitionStart')
        try:
            # Reclaim any pool a prior unclean-but-now-quiesced stop left announced
            # BEFORE allocating fresh, so the announced count cannot grow cycle over
            # cycle (the mid-suite wedge). Safe: threads are confirmed dead above.
            # If the stale pool could NOT be reclaimed (its state is unknowable),
            # do not announce a fresh pool on top of it -- abort like the
            # not-quiesced case rather than stack and leak.
            if not self._reclaim_stale_pool():
                _cam_log.error(
                    '[CAM Class ] start_grabbing aborted: could not reclaim the prior '
                    'buffer pool (data stream unhappy); not stacking a fresh pool'
                )
                return

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
            _cam_log.warning(f'[CAM Class ] start_grabbing failed, rolling back: {e}')
            self._rollback_failed_start()

    def _rollback_failed_start(self):
        """Return to a clean stopped state after start_grabbing raised partway.

        start_grabbing announces a buffer pool and takes the transport-layer lock
        (TLParamsLocked=1) before StartAcquisition. If a later step raises, the
        camera is left half-started: the lock stays set and the announced buffers
        stay in the pool, yet is_grabbing() is False -- so disconnect() skips
        stop_grabbing() and never releases either, and the next start_grabbing()
        announces a fresh pool on top of the orphaned one. Best-effort undo each
        step, independently guarded so one failure does not strand the rest, and
        in teardown order: quiesce the handler FIRST (so no live thread holds a
        buffer), then stop acquisition, release the lock, and revoke the pool --
        but revoke ONLY if the handler confirmed its grab threads died, mirroring
        stop_grabbing: revoking under a thread that could not be joined strands it
        with an invalid handle. If they did not quiesce, leave the pool announced
        for the next clean start_grabbing() to reclaim.
        """
        handler = self.cam_image_handler
        quiesced = True
        if handler is not None:
            try:
                quiesced = handler.stop()
            except Exception as e:
                logger.debug(f'[CAM Class ] rollback handler.stop ignored: {e}')
                quiesced = False  # could not confirm the threads died -- do not revoke
        # Each undo step runs only when its handle exists: a start that failed
        # before a handle was bound (or on a torn-down camera) has nothing to
        # undo there, and attempting it would log a spurious SDK-fault line.
        if self.data_stream is not None:
            try:
                self.data_stream.StopAcquisition()
            except Exception as e:
                logger.debug(f'[CAM Class ] rollback StopAcquisition ignored: {e}')
        if self.remote_nodemap is not None:
            try:
                self.remote_nodemap.FindNode('TLParamsLocked').SetValue(0)
            except Exception as e:
                logger.debug(f'[CAM Class ] rollback TLParamsLocked=0 ignored: {e}')
        if self.data_stream is not None:
            if quiesced:
                try:
                    self._flush_and_revoke_pool()
                except Exception as e:
                    logger.debug(f'[CAM Class ] rollback buffer revoke ignored: {e}')
            else:
                self._breadcrumb_unrevoked_pool()

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
        new_max = self._maximize_acquisition_frame_rate()
        if new_max is not None:
            logger.info(f'[CAM Class ] AcquisitionFrameRate set to max: {new_max} fps')

    def _maximize_acquisition_frame_rate(self):
        """Set AcquisitionFrameRate to its current maximum (free-run = no software
        cap) and return that maximum, or None when the node is unavailable.

        The maximum is EXPOSURE-COUPLED: a long exposure lowers it and the camera
        clamps the current rate down to it, while a later shorter exposure raises
        the maximum but leaves the rate at the clamped-low value. So this must be
        re-asserted after every exposure change (see exposure_t), not only at grab
        start -- otherwise the live rate stays stuck at the slow rate set during
        the long exposure, and the startup rate stays low when the saved exposure
        loads after the one-time free-run config.
        """
        if not self.active or self.remote_nodemap is None:
            return None
        try:
            fr = self.remote_nodemap.FindNode('AcquisitionFrameRate')
            new_max = fr.Maximum()
            fr.SetValue(new_max)
            return new_max
        except Exception as e:
            logger.debug(f'[CAM Class ] AcquisitionFrameRate not available: {e}')
            return None

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
            # The node's access status tells WHY a non-Link current persisted: a
            # read-only node is a firmware/body limitation we cannot lift, while a
            # writable one means the 'Link' set failed for another reason -- so the
            # bundle distinguishes the two without a probe. The U3-34L reports this
            # ReadOnly (bench 2026-06-28), locking the rate to Sensor mode.
            access = (
                _access_status_name(comp.AccessStatus())
                if hasattr(comp, 'AccessStatus')
                else '<no AccessStatus>'
            )
            if current == 'Link':
                logger.info(
                    f'[CAM Class ] Free-run: ThroughputLimitComponent=Link applied '
                    f'(available={entries}, access={access})'
                )
            else:
                _cam_log.warning(
                    f'[CAM Class ] Free-run: ThroughputLimitComponent={current}, NOT Link '
                    f'(available={entries}, access={access}) -- rate stays Sensor-throttled '
                    f'below the wire ceiling'
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
        frame = self.get_acquired_aoi() or {}
        # State the DELIVERED depth, not just the wire format: the Mono10 wire is
        # delivered as 8-bit (8-bit mode), Mono12 as native uint16. Logging only
        # the wire format left a bundle unable to say whether a frame arrived as
        # uint8 or uint16 -- the gap that made a host-side-depth question need a
        # bench round-trip to answer.
        wire = self.get_pixel_format()
        delivered_bits = _ids_delivery_significant_bits(wire)
        logger.info(
            f'[CAM Class ] Free-run state: frame={frame.get("width")}x{frame.get("height")} '
            f'pixel_format={wire} delivers={delivered_bits}-bit '
            f'({"uint8" if delivered_bits <= 8 else "uint16"}) '
            f'DeviceLinkThroughputLimit={dltl}/{dltl_max} B/s '
            f'AcquisitionFrameRate={rate}/{rate_max} fps'
        )

        # Optional-feature candidate nodes, logged on every bring-up so a normal
        # log bundle already records which exist + are writable on this body.
        # _probe_feature_nodes is internally defensive (each read is sentineled),
        # so it never raises here.
        logger.info(f'[CAM Class ] Optional-feature nodes: {self._probe_feature_nodes()}')

    def _drain_finished_buffers(self, n_frames: int, per_frame, *, label: str) -> None:
        """Pause the unpack worker, drive the data stream directly for n_frames,
        and hand each completed (non-incomplete) buffer's IPL image to
        per_frame(img); always re-queue the buffer and restore the grab pipeline.

        Shared spine for the bench scaffolds (benchmark_unpack /
        crosscheck_8bit_unpack): they differ only in the per-frame comparison,
        so the worker-pause + WaitForFinishedBuffer drain + teardown lives here
        once. The live unpack path is untouched. ``label`` tags the warning/debug
        lines so a log shows which scaffold drove the stream.
        """
        # Pause the worker so this loop owns the finished-buffer flow, and clear
        # any KillWait the stop posted so the first wait is not aborted on arrival.
        handler = self.cam_image_handler
        if handler is not None:
            handler.stop()
        try:
            self.data_stream.FlushPendingKillWaits()
        except Exception as e:
            logger.debug(f'[CAM Class ] {label}: FlushPendingKillWaits unavailable: {e}')

        try:
            for _ in range(n_frames):
                try:
                    buffer = self.data_stream.WaitForFinishedBuffer(2000)
                except Exception as e:
                    _cam_log.warning(f'[CAM Class ] {label}: WaitForFinishedBuffer: {e}')
                    continue
                try:
                    if buffer.IsIncomplete():
                        continue
                    img = ids_peak_ipl_extension.BufferToImage(buffer)
                    per_frame(img)
                finally:
                    self.data_stream.QueueBuffer(buffer)
        finally:
            if handler is not None:
                handler.start()

    def probe_afl_packed_acceptance(self) -> dict:
        """One-shot bench probe: does ids_peak_afl accept a PACKED image?

        The IMX676 body has no camera-side auto-exposure/gain, so a future host
        auto-brightness routine must run through ids_peak_afl. Its Process()
        takes an IPL image, but IDS does not document whether the packed
        Mono10g40IDS / Mono12g24IDS wire formats are accepted or must be
        unpacked first. This drives one packed frame -- and its unpacked copy --
        through afl Process() and records which the library accepts: the single
        fact that decides whether the auto routine can feed the raw buffer or
        must unpack every frame first (an INVALID_IMAGE_FORMAT on the packed
        image is the "must unpack" answer).

        Introspective and defensive: the ids_peak_afl surface (module, Manager,
        Controller) is captured, and the controller-type enum is resolved by
        trying the known binding spellings, so a wrong assumption still yields
        the real API rather than a bare failure. ExposureTime is snapshot and
        restored because Process() writes that node itself. The live unpack
        worker is paused and restored by the shared drain spine; the display
        path is untouched.
        """
        result = {
            'afl_importable': False,
            'afl_module': None,
            'afl_version': None,
            'afl_import_errors': [],
            'wire_format': None,
            'packed': None,
            'unpacked': None,
        }
        if not self.active or not self.data_stream:
            result['error'] = 'camera not connected / no data stream'
            return result

        # AFL is optional and not on the live path -- import lazily so a host
        # without the auto-features library still loads the driver. The binding
        # has shipped under more than one import path; try the known ones.
        afl = None
        for path in ('ids_peak_afl.ids_peak_afl', 'ids_peak_afl', 'ids_peak.ids_peak_afl'):
            try:
                afl = __import__(path, fromlist=['_'])
                result['afl_module'] = path
                break
            except Exception as e:
                result['afl_import_errors'].append(f'{path}: {type(e).__name__}: {e}')
        if afl is None:
            return result
        result['afl_importable'] = True
        result['afl_version'] = getattr(afl, '__version__', '?')
        result['afl_module_names'] = [n for n in dir(afl) if not n.startswith('__')]

        def _try_process(mgr, image, tag):
            # Record acceptance, and on rejection the exact exception/status so
            # a format rejection is distinguishable from an unrelated failure.
            try:
                mgr.Process(image)
                return {'tag': tag, 'accepted': True}
            except Exception as e:
                info = {
                    'tag': tag,
                    'accepted': False,
                    'exception': type(e).__name__,
                    'message': str(e),
                }
                for attr in ('code', 'Code', 'status', 'Status'):
                    member = getattr(e, attr, None)
                    if member is None:
                        continue
                    try:
                        info['status'] = member() if callable(member) else member
                    except Exception:
                        continue
                    break
                return info

        # Snapshot exposure AND gain first: the brightness controller drives BOTH,
        # so if either cannot be captured we must NOT run Process -- we could not
        # put the camera back, and this probe must leave live state unchanged.
        def _snap(node_name):
            try:
                node = self.remote_nodemap.FindNode(node_name)
                return node, node.Value()
            except Exception as e:
                result.setdefault('snapshot_errors', {})[node_name] = f'{type(e).__name__}: {e}'
                return None, None

        exposure_node, exposure_before = _snap('ExposureTime')
        gain_node, gain_before = _snap('Gain')
        if exposure_before is None or gain_before is None:
            result['error'] = (
                'could not snapshot exposure/gain; skipping Process to avoid '
                'leaving the camera at an auto-nudged setting'
            )
            return result

        lib_inited = False
        try:
            try:
                afl.Library.Init()
                lib_inited = True
            except Exception as e:
                # Do NOT proceed to Manager/Process on an uninitialized native
                # library -- an unpinned binding may hard-crash rather than raise.
                result['lib_init_error'] = f'{type(e).__name__}: {e}'
                return result

            mgr = None
            try:
                mgr = afl.Manager(self.remote_nodemap)
                result['manager_names'] = [n for n in dir(mgr) if not n.startswith('_')]
                ctype = None
                for name in (
                    'PEAK_AFL_CONTROLLER_TYPE_BRIGHTNESS',
                    'CONTROLLER_TYPE_BRIGHTNESS',
                    'ControllerType_Brightness',
                ):
                    if hasattr(afl, name):
                        ctype = getattr(afl, name)
                        result['controller_type_attr'] = name
                        break
                if ctype is not None:
                    ctrl = mgr.CreateController(ctype)
                    mgr.AddController(ctrl)
                    result['controller_names'] = [n for n in dir(ctrl) if not n.startswith('_')]
                else:
                    # No controller means Process() would run on an empty Manager
                    # and likely 'accept' without touching the pixel path -- a false
                    # positive. Refuse to conclude rather than report a wrong answer.
                    result['controller_type_attr'] = 'NONE_MATCHED'
                    result['error'] = (
                        'no brightness controller-type enum matched; cannot probe '
                        '(Process without a controller would be a false positive)'
                    )
                    return result
            except Exception as e:
                result['manager_setup_error'] = f'{type(e).__name__}: {e}'

            if mgr is None:
                return result

            wire = self.get_pixel_format()
            result['wire_format'] = wire
            if wire is None:
                # get_pixel_format() returns None on a nodemap read failure;
                # _ids_ipl_target(None) would raise. Degrade to a recorded error.
                result['error'] = 'could not read pixel format'
                return result
            target = _ids_ipl_target(wire)

            # Probe only the FIRST complete frame (processing more frames would
            # repeatedly nudge exposure/gain); within it, try the packed image then
            # an unpacked copy. Both nudges are undone by the finally restore.
            probed = {'done': False}

            def _probe(img):
                if probed['done']:
                    return
                probed['done'] = True
                result['packed'] = _try_process(mgr, img, f'packed:{wire}')
                try:
                    unpacked = img.ConvertTo(target)
                    result['unpacked'] = _try_process(mgr, unpacked, 'unpacked')
                except Exception as e:
                    result['unpacked'] = {
                        'tag': 'unpacked',
                        'error': f'convert failed: {type(e).__name__}: {e}',
                    }

            # Request several frames so an incomplete first buffer (which the
            # drain spine skips) does not leave the probe with no frame.
            self._drain_finished_buffers(5, _probe, label='afl-probe')
            if not probed['done']:
                result['error'] = 'no complete frame arrived to probe'
        finally:
            # Restore BOTH exposure and gain -- the brightness controller drives
            # both, and this probe must leave live state unchanged.
            for node, before, key in (
                (exposure_node, exposure_before, 'exposure'),
                (gain_node, gain_before, 'gain'),
            ):
                if node is not None and before is not None:
                    try:
                        node.SetValue(before)
                    except Exception as e:
                        result[f'{key}_restore_error'] = f'{type(e).__name__}: {e}'
            # Drop the Manager/Controller BEFORE Library.Exit() so their C++-backed
            # destructors do not run against a torn-down library (use-after-free).
            ctrl = None
            mgr = None
            if lib_inited:
                try:
                    afl.Library.Exit()
                except Exception as e:
                    logger.debug(f'[CAM Class ] afl-probe: Library.Exit unavailable: {e}')
        # Emit the verdict through the driver log so it is captured by --driver-log
        # (the returned dict is print()ed by the test, which the log bundle drops).
        packed = result.get('packed') or {}
        unpacked = result.get('unpacked') or {}
        _cam_log.info(
            '[CAM Class ] afl-probe RESULT: '
            f'importable={result.get("afl_importable")} wire={result.get("wire_format")!r} '
            f'PACKED_accepted={packed.get("accepted")} (exc={packed.get("exception")} '
            f'status={packed.get("status")}) UNPACKED_accepted={unpacked.get("accepted")} '
            f'reason={result.get("error") or result.get("manager_setup_error") or result.get("lib_init_error")}'
        )
        return result

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
        frame = self.get_acquired_aoi() or {}
        width, height = frame.get('width'), frame.get('height')
        results.update(wire_format=wire, width=width, height=height)
        if not width or not height:
            results['error'] = 'could not read frame size'
            return results
        target = _ids_ipl_target(wire)

        convert_ms, numpy_ms = [], []

        def _compare(img):
            # ConvertTo path (the current production unpack + the oracle).
            # Bind the converted image to a local before get_numpy(): get_numpy()
            # returns a view that does NOT keep the IPL image alive, so a chained
            # ConvertTo(...).get_numpy().copy() frees the converted buffer the
            # instant get_numpy() returns and copy() then reads freed memory
            # (access violation). Use a SEPARATE name from img -- img stays the
            # packed source the numpy path reads just below.
            t0 = time.perf_counter()
            conv_img = img.ConvertTo(target)
            conv = conv_img.get_numpy().copy()
            convert_ms.append((time.perf_counter() - t0) * 1000.0)

            # numpy path: the raw packed wire bytes (uint8 since SDK 2.21)
            # decoded by our own unpacker. ConvertTo does not mutate img, so the
            # original is still the packed source here.
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

        self._drain_finished_buffers(n_frames, _compare, label='benchmark')

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

    def crosscheck_8bit_unpack(self, n_frames: int = 100) -> dict:
        """Bench gate for the direct 10->8 delivery: prove the SDK's
        packed->Mono8 ConvertTo matches the prior native-then-rescale path.

        The production unpack now delivers the 8-bit-mode Mono10 wire straight to
        8-bit (one ConvertTo pass) instead of unpacking to native uint16 and
        running the host rescale LUT. That swaps the host's exact linear rescale
        (value/max*255) for the SDK's bit-shift, which should differ by <=1 LSB.
        This loop verifies that on REAL frames before the direct path is trusted:

          A (new): img.ConvertTo(Mono8)
          B (old): convert_to_8bit(img.ConvertTo(native), native_significant_bits)

        Returns the per-pixel max abs diff and the count of pixels exceeding
        1 LSB. The hardware test asserts max_abs_diff <= 1 and over_1lsb == 0.
        Only meaningful on a packed Mono10/Mono12 wire (an already-8-bit wire is
        a no-op match).
        """
        import numpy as np

        from modules.image_utils import convert_to_8bit

        results = {
            'n_requested': n_frames,
            'n_compared': 0,
            'max_abs_diff': 0,
            'pixels_over_1lsb': 0,
            'wire_format': None,
            'skipped': None,
            'error': None,
        }
        if not self.active or not self.data_stream:
            results['error'] = 'camera not connected / no data stream'
            return results

        wire = self.get_pixel_format()
        results['wire_format'] = wire
        direct_target = _ids_delivery_target(wire)
        native_target = _ids_ipl_target(wire)
        native_bits = ids_significant_bits(wire)

        # Only the Mono10 wire reduces (direct Mono8 vs native Mono10): there the
        # SDK bit-shift must match the host rescale. On Mono12/Mono8 the delivery
        # target already equals the native target, so there is no 10->8 reduction
        # to validate -- comparing a uint16 'direct' against an 8-bit oracle would
        # be a guaranteed false mismatch. Skip rather than false-alarm.
        if direct_target == native_target:
            results['skipped'] = (
                f'no 8-bit reduction on wire {wire!r} (delivery target == native); '
                f'run in 8-bit mode (Mono10 wire) to exercise the cross-check'
            )
            return results

        def _compare(img):
            # Bind each ConvertTo result before get_numpy() (the view does not
            # keep the IPL image alive) and copy out, same lifetime rule
            # benchmark_unpack documents.
            direct_img = img.ConvertTo(direct_target)
            direct = direct_img.get_numpy().copy()
            native_img = img.ConvertTo(native_target)
            native = native_img.get_numpy().copy()
            oracle = convert_to_8bit(native, native_bits)

            diff = np.abs(direct.astype(np.int16) - oracle.astype(np.int16))
            results['n_compared'] += 1
            results['max_abs_diff'] = max(results['max_abs_diff'], int(diff.max()))
            results['pixels_over_1lsb'] += int((diff > 1).sum())

        self._drain_finished_buffers(n_frames, _compare, label='crosscheck')
        return results

    def set_frame_size(self, w, h) -> dict | bool:
        """Deliver exactly the requested frame size via oversize-then-crop.

        The IMX676 AOI snaps to a coarse grid (48 px wide, 4 px tall), so a
        request like 1900 cannot be set exactly. Rather than silently floor it
        (the old behavior delivered 1872 for a 1900 request), acquire the next
        legal AOI UP, center it on the sensor, and record the sub-rectangle the
        unpack worker crops back to the exact request. The hardware AOI (the
        oversized acquisition) is diagnostic only; the delivered, public size is
        the cropped target.

        Returns the delivered size as ``{'width': w, 'height': h}`` on success
        so the caller knows what was actually applied without a read-back (a
        live get_frame_size() can spuriously drop the camera on a transient
        error); returns False when the camera is inactive or the apply fails.
        """
        if not self.active:
            # Expected during disconnect/teardown; log so a dropped resize is
            # visible in a bundle rather than a silent no-op.
            _cam_log.debug('[CAM Class ] set_frame_size skipped: camera inactive')
            return False

        try:
            from modules.aoi_geometry import plan_aoi

            nodemap = self.remote_nodemap
            width_node = nodemap.FindNode('Width')
            height_node = nodemap.FindNode('Height')
            offset_x_node = nodemap.FindNode('OffsetX')
            offset_y_node = nodemap.FindNode('OffsetY')

            # Width/Height Minimum is offset-independent, so read it once here
            # and reuse for the grid phase (size_min below). The offset minimums
            # DO depend on the offsets, so they are read later with the offsets
            # zeroed.
            w_min, h_min = width_node.Minimum(), height_node.Minimum()
            # The request is the crop TARGET, not the acquisition floor -- pass
            # it through verbatim. plan_aoi rounds the ACQUISITION up to the next
            # legal AOI (always >= the node minimum) and crops it back to exactly
            # this target, so a request below the node minimum is still delivered
            # at the requested size: a 950-wide frame at 2x binning, where
            # Width.Min is 1056, acquires 1056 and crops to 950 (square). Flooring
            # the target up to the minimum here instead delivered the floored size
            # (1056x950, non-square) because the crop then had nothing to trim.
            target = (int(w), int(h))
            # Alignment step from the SDK nodemap, not the profile: the hardware
            # increment is authoritative (48 wide on the IMX676 bodies), and an
            # unrecognized model falls back to a default profile whose alignment
            # (4) the SDK rejects. Couple to the hardware, not a static spec.
            step = (width_node.Increment(), height_node.Increment())
            bias = self._optical_center_bias()

            with self.update_camera_config():
                # Zero the offsets first so Width/Height range over the full
                # sensor (an AOI's max width shrinks as its X offset grows) and
                # the max we read is the true sensor max, not max-minus-offset.
                offset_x_node.SetValue(0)
                offset_y_node.SetValue(0)

                # This offset-zero read is the only place the true (offset-
                # independent) sensor max is visible; cache it for
                # get_max_frame_size, which is called with offsets applied.
                max_size = (width_node.Maximum(), height_node.Maximum())
                self._sensor_max = max_size

                # The live Width/Height node bounds: min is the grid phase, inc
                # the alignment step, max the offset-zeroed sensor max. The 1x
                # minimum is otherwise only inferred from the delivered size --
                # logging it makes the real hardware floor observable in a bundle.
                _cam_log.info(
                    f'[CAM Class ] set_frame_size nodes '
                    f'W[min={w_min} inc={step[0]} max={max_size[0]}] '
                    f'H[min={h_min} inc={step[1]} max={max_size[1]}]'
                )

                # Each node's Minimum is the grid PHASE, not just a request
                # floor: the legal set is Min + k*Inc, and a binned Height
                # reports Min=418 with Inc=4 -- off the plain-multiple grid, so a
                # multiple-of-Inc snap is rejected. Width/Height Minimum (w_min,
                # h_min) was read above; the offset minimums are read here with
                # the offsets zeroed so they are the true, offset-independent
                # phase. Couple to the hardware, not a static spec.
                size_min = (w_min, h_min)
                offset_min = (offset_x_node.Minimum(), offset_y_node.Minimum())

                plan = plan_aoi(
                    target=target,
                    step=step,
                    max_size=max_size,
                    offset_step=(offset_x_node.Increment(), offset_y_node.Increment()),
                    size_min=size_min,
                    offset_min=offset_min,
                    bias=bias,
                )

                width_node.SetValue(plan.acq_width)
                height_node.SetValue(plan.acq_height)
                offset_x_node.SetValue(plan.offset_x)
                offset_y_node.SetValue(plan.offset_y)

                # Record the crop INSIDE the stopped window: update_camera_config
                # restarts the grab (and reallocs buffers to the new AOI) on exit,
                # so the window must be in place before the unpack worker resumes,
                # or it would crop the new-sized buffer against the old one. None
                # when the AOI already matches the request (needs_crop False) so
                # the unpack worker skips the per-frame slice entirely;
                # get_frame_size then falls back to the acquired AOI, which equals
                # the delivered size on that path.
                self._crop_spec = (
                    (plan.crop_x0, plan.crop_y0, plan.crop_width, plan.crop_height)
                    if plan.needs_crop
                    else None
                )

            if (plan.crop_width, plan.crop_height) != target:
                # plan_aoi clamps to the sensor: a request within one alignment
                # step of the max can't be supplied in full. The delivered size
                # is honest (get_frame_size reports it), but flag the shortfall.
                _cam_log.warning(
                    f'[CAM Class ] set_frame_size delivered '
                    f'{plan.crop_width}x{plan.crop_height}, smaller than requested '
                    f'{target[0]}x{target[1]} (near sensor max); get_frame_size() '
                    f'reports the delivered size'
                )

            _cam_log.info(
                f'[CAM Class ] set_frame_size target={target[0]}x{target[1]} '
                f'acq={plan.acq_width}x{plan.acq_height} '
                f'off=({plan.offset_x},{plan.offset_y}) bias={bias} '
                f'crop=({plan.crop_x0},{plan.crop_y0},{plan.crop_width},{plan.crop_height})'
            )
            return {'width': plan.crop_width, 'height': plan.crop_height}
        except Exception as e:
            # A partially-applied AOI (offsets zeroed, or Width/Height set but
            # the crop not yet recorded) must not leave a stale crop window for
            # the unpack worker. Invalidate so frames pass through at the full
            # AOI until the next successful set_frame_size re-applies it.
            self._invalidate_framing()
            _cam_log.error(f'[CAM Class ] set_frame_size failed: {e}')
            return False

    def _invalidate_framing(self) -> None:
        """Drop the recorded crop window so the unpack worker passes frames
        through at the full AOI.

        Called when the AOI geometry changes out from under the crop -- a
        binning change (which resizes the buffer) or a failed set_frame_size
        (which may have committed a new AOI before recording the matching
        window). get_frame_size() then falls back to the live hardware AOI until
        the next successful set_frame_size records a window that fits the new
        buffer. Without this the worker crops every new-sized frame against the
        old window and drops them all (a frozen/black preview).

        Also drops the cached sensor max: a binning change halves it, so the
        cached value would otherwise be stale until the next set_frame_size.
        """
        self._crop_spec = None
        self._sensor_max = None

    def _optical_center_bias(self) -> tuple[int, int]:
        """The optical-center AOI offset bias, in displayed pixels.

        Neutral (0, 0) today: the AOI centers geometrically, which is correct for
        every unit. set_frame_size already threads the return value through
        plan_aoi's ``bias``, so the optical-center work (planned ~2 weeks out)
        implements only this method's body -- read the per-unit optical center
        (motorconfig ImageCenter, sensor pixels), reorient it into the delivered
        array frame (aoi_geometry.reorient_image_center, with the sensor's
        mounted orientation pinned by a one-time bench collimator calibration),
        and divide by the active binning. It must return without raising:
        set_frame_size catches exceptions, so a raise would swallow into a silent
        failure to resize.
        """
        return (0, 0)

    def get_min_frame_size(self) -> dict:
        if not self.active:
            return {}

        # Oversize-then-crop delivers any aligned size BELOW the hardware AOI
        # minimum (set_frame_size acquires the AOI floor and crops down), so the
        # DELIVERABLE minimum is the alignment granularity -- the same deliverable
        # space get_pixel_alignment reports -- NOT the Width/Height node Minimum
        # (the smallest ACQUIRABLE AOI, e.g. 1056x418 at 2x binning). Reporting
        # the AOI minimum made the UI clamp a half-resolution request up to it,
        # delivering a non-square frame (1056x950 for a 950x950 request at 2x).
        return dict(self.profile.alignment)

    def get_max_frame_size(self) -> dict:
        if not self.active:
            return {}

        # Prefer the cached offset-zero read: Width/Height .Maximum() shrinks as
        # the offset grows (Width.Max = SensorWidth - OffsetX, floored to the
        # increment), so a live read with the centering offset applied would
        # under-report the sensor max. set_frame_size caches the true max from
        # its offset-zero read.
        if self._sensor_max is not None:
            return {'width': self._sensor_max[0], 'height': self._sensor_max[1]}

        try:
            # No cache yet (before the first set_frame_size): the offsets are
            # still at their zero default, so a live Maximum() read is the true
            # sensor max.
            return {
                'width': self.remote_nodemap.FindNode('Width').Maximum(),
                'height': self.remote_nodemap.FindNode('Height').Maximum(),
            }
        except Exception as e:
            _cam_log.error(f'[CAM Class ] get_max_frame_size failed: {e}')
            return {}

    def get_frame_size(self):
        """The delivered (public) frame size -- the cropped target, not the AOI.

        Oversize-then-crop acquires a larger AOI than requested and trims it, so
        the consumer-facing size is the recorded crop window's (w, h). Falls back
        to the hardware AOI before the first set_frame_size, or after a geometry
        change has invalidated the crop (see _invalidate_framing).
        """
        if not self.active:
            return
        if self._crop_spec is not None:
            return {'width': self._crop_spec[2], 'height': self._crop_spec[3]}
        return self.get_acquired_aoi()

    def get_acquired_aoi(self):
        """The hardware AOI actually set on the sensor (the oversized
        acquisition before the software crop). Diagnostic only -- the public
        frame size is the cropped target from get_frame_size()."""
        if not self.active:
            return

        try:
            return {
                'width': self.remote_nodemap.FindNode('Width').Value(),
                'height': self.remote_nodemap.FindNode('Height').Value(),
            }
        except Exception as e:
            _cam_log.error(f'[CAM Class ] get_acquired_aoi failed: {e}')
            return None

    @staticmethod
    def _select_default_pixel_format(supported) -> str | None:
        """Pick the connect-time default PixelFormat from the camera's actual
        advertised entries -- coupling to the live nodemap, not a model-keyed
        static profile, so ANY IDS body is driven by what it really exposes.

        MONO ONLY: this driver's unpack/blit path delivers a single-channel
        frame, so a colour (Bayer/RGB) entry is never a valid default -- return
        None rather than feed the mono pipeline a mosaic. Among mono entries,
        prefer lowest delivered bandwidth: by bit depth (8 < 10 < 12), then a
        PACKED entry (IDS 'g..IDS' / 'p' grouped names, fewer bytes on the wire)
        over an unpacked one of the same depth. Returns None when the body
        advertises no mono entry (a colour-only body, or an empty/failed read).

        Pure-logic + staticmethod for unit-testability without an SDK
        connection (same pattern as _resolve_logical_format_name).
        """
        mono = [f for f in supported if f.startswith('Mono')]
        if not mono:
            return None

        def _rank(fmt):
            if fmt.startswith('Mono8'):
                depth = 8
            elif fmt.startswith('Mono10'):
                depth = 10
            elif fmt.startswith('Mono12'):
                depth = 12
            else:
                depth = 99  # Mono14/Mono16/unknown -- valid mono, but heavier
            packed = ('g' in fmt) or ('IDS' in fmt) or fmt.endswith('p')
            return (depth, 0 if packed else 1, fmt)

        return sorted(mono, key=_rank)[0]

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
                # Update the cache INSIDE the config guard, while the grab loop
                # is still paused. update_camera_config()'s __exit__ restarts
                # grabbing; caching after it would leave a window where frames
                # flow under the new format but get_pixel_format()/
                # get_camera_info() still report the old cached value.
                self._pixel_format_cache = resolved
            return True
        except Exception as e:
            # A transient PixelFormat write failure is not a removal: raise
            # HardwareError so it propagates, but do NOT _mark_disconnected --
            # that drops the camera mid-resize over a recoverable fault.
            # set_binning_size (same update_camera_config machinery) handles it
            # this way; matching keeps the geometry setters consistent.
            _cam_log.error(f'[CAM Class ] set_pixel_format({resolved}) failed: {e}')
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
        # Guard on _device_removed, not just active: _mark_disconnected() sets the
        # removed flag but deliberately leaves active set (the handle release is
        # deferred to disconnect()), so a live AvailableEntries() read on a removed
        # remote nodemap can hang. This early-return keeps a removal-window query
        # off the dead nodemap.
        if not self.active or self._device_removed:
            return ()
        try:
            return tuple(
                pf.SymbolicValue()
                for pf in self.remote_nodemap.FindNode('PixelFormat').AvailableEntries()
            )
        except Exception as e:
            # An empty tuple is indistinguishable from "no formats" at the call
            # site -- a caller doing formats[0] then raises IndexError far from the
            # real cause. Name a removal or a wedged nodemap distinctly so the log
            # carries the reason, but still return () (removal stays owned by the
            # DeviceLost callback, so this query never marks disconnected itself).
            if _exc_is(e, 'DeviceLostException', 'removed'):
                _cam_log.error(f'[CAM Class ] get_supported_pixel_formats: device removed ({e})')
            elif _exc_is(e, 'InvalidInstanceException', 'invalid', 'nodemap', 'bufferhandle'):
                _cam_log.error(
                    f'[CAM Class ] get_supported_pixel_formats: nodemap/stream wedged ({e})'
                )
            else:
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
            # Clamp UP to the live node minimum, mirroring the Pylon driver.
            # The live-view slider initializes to its 0.01ms default and is
            # applied at startup before the saved exposure loads; that 10us
            # request is below the camera's live ExposureTime.Minimum() (which
            # also drifts up as AOI / frame rate change), and an unclamped
            # SetValue throws OUT_OF_RANGE. Reconcile the request to the
            # hardware's reported floor so it lands at the nearest valid value
            # instead of erroring. The max bound stays a reject above.
            exp_node = self.remote_nodemap.FindNode('ExposureTime')
            us_value = max(float(exposure_ms) * 1000, exp_node.Minimum())
            exp_node.SetValue(us_value)
            self._last_exposure_ms = us_value / 1000.0
            rate_note = ''
            if self.cam_image_handler:
                # Update grab timeout so long exposures don't cause perpetual timeouts.
                self.cam_image_handler.timeout_ms = max(2000, int(exposure_ms * 2 + 500))
                # Re-assert the free-run frame rate: AcquisitionFrameRate's maximum
                # is exposure-coupled, so a long exposure clamps the rate down and a
                # later shorter exposure does NOT restore it on its own. Without this
                # the live fps stays stuck at the slow rate set during the long
                # exposure (the "fps does not recover" symptom).
                new_max = self._maximize_acquisition_frame_rate()
                if new_max is not None:
                    rate_note = f' AcquisitionFrameRate->{new_max:.1f}fps'
            if _cam_log is not None:
                _cam_log.info(
                    f'ids ExposureTime.SetValue({us_value:.0f}us) (={exposure_ms}ms){rate_note}'
                )
            logger.debug(f'[CAM Class ] Exposure set to {exposure_ms}ms')
            return True
        except Exception as e:
            if _cam_log is not None:
                _cam_log.error(f'ids ExposureTime.SetValue({exposure_ms}ms) FAILED: {e}')
            _cam_log.error(f'[CAM Class ] Exposure set failed (likely out of bounds): {e}')
            return False

    def get_min_exposure(self) -> float | None:
        """Return the camera's LIVE minimum exposure in milliseconds.

        The ExposureTime node minimum drifts above the connect-time value once
        other settings change (pixel clock, frame rate, AOI), so read it live
        each call rather than trusting the value cached in the profile at
        connect. Falls back to the cached profile floor if the node read fails.
        """
        if not self.active:
            return super().get_min_exposure()
        try:
            return self.remote_nodemap.FindNode('ExposureTime').Minimum() / 1000.0
        except Exception as e:
            _cam_log.warning(f'[CAM Class ] get_min_exposure live read failed: {e}')
            return super().get_min_exposure()

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

    def get_sdk_info(self) -> dict:
        """IDS peak SDK provenance (name + version) for diagnostics.

        Best-effort, mirroring the Pylon driver's get_sdk_info: the version is
        a provenance label, so an unreadable value is reported as None
        (unknown) -- ids_peak.Library.Version() is the runtime accessor, the
        module __version__ the fallback.
        """
        version = None
        for reader in (
            lambda: str(ids_peak.Library.Version()),
            lambda: getattr(ids_peak, '__version__', None),
        ):
            try:
                version = reader()
            except Exception:
                continue
            if version is not None:
                break
        return {'name': 'IDS peak', 'version': version}

    def get_all_temperatures(self) -> dict:
        """Return {selector: degC, ...} per DeviceTemperatureSelector entry; {} if unreadable.

        Mirrors the Pylon driver's shape using the GenICam-standard
        DeviceTemperature / DeviceTemperatureSelector nodes through the IDS
        Peak FindNode API. A body that exposes a single sensor (no selector)
        reports it under 'Device'. Returns {} when the camera is inactive or
        the body exposes no temperature telemetry (FindNode raises on an
        absent node). Never raises.
        """
        if not self.active or self.remote_nodemap is None:
            return {}
        try:
            temp = self.remote_nodemap.FindNode('DeviceTemperature')
        except Exception as e:
            _cam_log.debug(f'[CAM Class ] DeviceTemperature node absent: {e}')
            return {}
        if temp is None:
            return {}
        # The selector is optional -- a single-sensor body may expose only
        # DeviceTemperature. With a selector, read every available entry.
        try:
            selector = self.remote_nodemap.FindNode('DeviceTemperatureSelector')
        except Exception:
            selector = None
        temps: dict[str, float] = {}
        try:
            if selector is not None:
                # Restore the selector afterwards so a later DeviceTemperature
                # read (or a concurrent reader) is not left pointed at the last
                # iterated sensor.
                try:
                    original = selector.CurrentEntry().SymbolicValue()
                except Exception:
                    original = None
                try:
                    for entry in selector.AvailableEntries():
                        name = entry.SymbolicValue()
                        try:
                            selector.SetCurrentEntry(name)
                            temps[name] = float(temp.Value())
                        except Exception as e:
                            _cam_log.debug(f'[CAM Class ] temperature read for {name} failed: {e}')
                finally:
                    if original is not None:
                        try:
                            selector.SetCurrentEntry(original)
                        except Exception as e:
                            _cam_log.debug(
                                f'[CAM Class ] restoring DeviceTemperatureSelector failed: {e}'
                            )
            # Fall back to the bare DeviceTemperature when there is no selector
            # OR the selector yielded nothing readable (a vestigial/empty
            # selector still leaves a single sensor readable).
            if not temps:
                temps['Device'] = float(temp.Value())
        except Exception as e:
            _cam_log.warning(f'[CAM Class ] get_all_temperatures failed: {e}')
            return {}
        return temps

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

    def _set_remote_node(self, node_name: str, value, method_label: str) -> bool:
        """Write a value to an OPTIONAL remote-device node iff this body exposes
        it, returning whether it was applied.

        Orientation (ReverseX) and GigE-transport (GevSCPS*/GevSCPD) nodes are
        present on some IDS bodies and absent on others. The IDS binding raises
        from FindNode when a node is absent, so presence detection requires the
        catch; this returns the applied/not-applied status so callers treat the
        node as a capability to query, not a node to assume. Same status-return
        contract as _set_data_stream_int_node (the DataStream-nodemap analogue).
        Never raises.
        """
        if not self.active or self.remote_nodemap is None:
            return False
        try:
            node = self.remote_nodemap.FindNode(node_name)
            if node is None:
                return False
            node.SetValue(value)
            return True
        except Exception as e:
            _cam_log.debug(
                f'[CAM Class ] {method_label}: optional node {node_name} not '
                f'applied ({type(e).__name__}: {e})'
            )
            return False

    def set_gev_packet_size(self, size_bytes: int) -> bool:
        """Set GevSCPSPacketSize (GigE-Vision stream packet size / MTU) iff this
        body exposes it. Returns False on a USB3 body (node absent) and writes
        it on a GigE (uEye+ GV) body -- capability, queried from the nodemap,
        not a hardcoded transport verdict (mirrors the Pylon driver)."""
        return self._set_remote_node('GevSCPSPacketSize', int(size_bytes), 'set_gev_packet_size')

    def set_gev_inter_packet_delay(self, delay_ticks: int) -> bool:
        """Set GevSCPD (GigE-Vision inter-packet delay) iff this body exposes it.
        Returns False on a USB3 body (node absent); writes it on a GigE body."""
        return self._set_remote_node('GevSCPD', int(delay_ticks), 'set_gev_inter_packet_delay')

    def _set_data_stream_int_node(self, node_name: str, value: int, method_label: str) -> bool:
        """Write an integer to a DataStream-nodemap node (the USB3 transfer-
        tuning channel parameters). Returns True on success; False if the camera
        is inactive, the DataStream nodemap is unavailable, the node is absent on
        this body, or the SDK rejects the write -- the channel parameters are
        typically locked while the stream is grabbing, so a runtime write can
        fail and the caller must stop the stream first if it needs the change to
        take. Never raises.
        """
        if not self.active or self.data_stream is None:
            return False
        try:
            nodemaps = self.data_stream.NodeMaps()
            stream_nm = nodemaps[0] if nodemaps else None
            if stream_nm is None:
                return False
            node = stream_nm.FindNode(node_name)
            if node is None:
                return False
            node.SetValue(int(value))
            return True
        except Exception as e:
            _cam_log.warning(
                f'[CAM Class ] IDS {method_label}({value}) failed: {type(e).__name__}: {e}'
            )
            return False

    def set_max_transfer_size(self, value_bytes: int) -> bool:
        """Set the DataStream U3vStreamChannelBulkTransferSize -- the IDS analogue
        of Pylon's StreamGrabber MaxTransferSize (bytes per USB transfer the SDK
        requests). Bench-confirmed ReadWrite on the U3-34L0XCP-M. Returns False
        if inactive / node absent / the SDK rejects (e.g. locked while grabbing).
        """
        return self._set_data_stream_int_node(
            'U3vStreamChannelBulkTransferSize', int(value_bytes), 'set_max_transfer_size'
        )

    def set_num_max_queued_urbs(self, value: int) -> bool:
        """Set the DataStream U3vStreamChannelTransferRequestCount -- the IDS
        analogue of Pylon's StreamGrabber NumMaxQueuedUrbs (count of in-flight
        USB transfer requests). Bench-confirmed ReadWrite on the U3-34L0XCP-M
        (observed range up to 6). Returns False if inactive / node absent / the
        SDK rejects the write.
        """
        return self._set_data_stream_int_node(
            'U3vStreamChannelTransferRequestCount', int(value), 'set_num_max_queued_urbs'
        )

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
            size: Binning factor. The ceiling is read live from the binning
                node maximum at connect (self._max_binning), not hardcoded, so a
                body that supports 4x is not refused.

        Returns:
            bool: True on success. False only when the camera is inactive
                or size is out of range (caller-correctable guards).
                Hardware-level failure raises HardwareError.

        Raises:
            HardwareError: SDK call failed.
        """
        if not self.active:
            return False

        if size < 1 or size > self._max_binning:
            _cam_log.error(f'[CAM Class ] Unsupported bin size: {size} (max {self._max_binning})')
            return False

        try:
            logger.debug(
                f'[CAM Class ] Binning {self.get_binning_size()} -> {size}, frame {self.get_frame_size()}'
            )
            with self.update_camera_config():
                self.remote_nodemap.FindNode('BinningVertical').SetValue(size)
                self.remote_nodemap.FindNode('BinningHorizontal').SetValue(size)
                # Zero the offsets: the crop is invalidated just below (frames
                # pass at the full binned AOI until set_frame_size re-centers),
                # and a leftover centering offset would make get_max_frame_size's
                # live fallback read Width.Maximum() = sensor - offset, i.e.
                # under-report the binned sensor max in the window before the UI
                # re-applies set_frame_size.
                self.remote_nodemap.FindNode('OffsetX').SetValue(0)
                self.remote_nodemap.FindNode('OffsetY').SetValue(0)
                # Binning resizes the AOI buffer, so the recorded crop window no
                # longer fits. Invalidate INSIDE the stopped window (before the
                # grab restarts on exit) so the unpack worker passes frames
                # through at the full binned AOI instead of cropping against a
                # stale window and dropping every frame. The UI re-applies
                # set_frame_size with the new displayed size right after.
                self._invalidate_framing()

            logger.debug(
                f'[CAM Class ] Binning set to {self.get_binning_size()}, frame now {self.get_frame_size()}'
            )
            return True
        except Exception as e:
            _cam_log.error(f'[CAM Class ] set_binning_size failed: {e}')
            raise HardwareError(f'set_binning_size({size}) failed: {type(e).__name__}: {e}') from e

    def get_binning_size(self) -> int:
        # READ-FAILURE sentinel is -1, not 1: 1 is a legal binning factor (1x =
        # no binning), so returning 1 on a failed read would be indistinguishable
        # from a real 1x camera -- any caller that validates getters by value
        # would accept the in-band 1 and silently de-bin a 2x camera, whereas
        # -1 is rejected (binning must be >= 1). The INACTIVE case stays 1: no
        # camera means no binning, and callers already treat inactive as the
        # 1x default.
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
            return -1

    # grab() inherited from Camera base class

    def grab_new_capture(self, timeout_s):
        """Return a frame captured AFTER this call, via the shared latest-frame
        path -- never by driving the data stream directly.

        The handler's poll thread is the single WaitForFinishedBuffer consumer and
        _requeue the single QueueBuffer; a still capture that ran its own
        WaitForFinishedBuffer + QueueBuffer (the prior implementation) raced the
        live poll/worker on the same stream -- stealing finished buffers and
        re-queueing outside _requeue_lock, which the SDK does not promise is
        concurrency-safe. Instead, snapshot the worker's frame-generation, wait for
        it to advance (a genuinely new frame, unpacked + cropped + depth-stamped by
        the worker), then read that frame through grab() -- which stores it under
        _array_lock with its paired significant_bits. This mirrors the Pylon
        driver, which also consumes the handler's frame rather than the stream.
        """
        handler = self.cam_image_handler
        if not handler:
            return False, None
        since = handler.frame_generation()
        if not handler.wait_for_new_frame(since, timeout_s):
            return False, None
        return self.grab()

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
        published through the profile (see _query_dynamic_capabilities). The
        converted factor is reconciled to the Gain node's reported [Minimum,
        Maximum] before writing: the dB->factor round-trip overshoots the node
        maximum by a float epsilon exactly at the cap, so without this the camera
        rejects its own reported maximum gain (the same self-clamp the FX2
        register conversion does). A genuine SDK failure still returns False.

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
            # Select the analog gain entry this body actually exposes (resolved
            # at connect; re-resolve if missing) BEFORE reading the range and
            # writing -- the Gain Min/Max depend on the active selector. If none
            # can be resolved, fail loud rather than write the factor against
            # whatever selector is currently active (that would land analog gain
            # on the wrong amplifier and still report success).
            selector = self._gain_selector or self._resolve_gain_selector()
            if not selector:
                _cam_log.error(
                    '[CAM Class ] Cannot set gain: no GainSelector entry resolved on this body'
                )
                return False
            self.remote_nodemap.FindNode('GainSelector').SetCurrentEntry(selector)
            gain_node = self.remote_nodemap.FindNode('Gain')
            # Reconcile to the node's reported range: the dB->factor conversion
            # overshoots Gain.Maximum() by a float epsilon at the cap, so the
            # camera rejects its own reported maximum gain without this clamp.
            factor = min(max(factor, gain_node.Minimum()), gain_node.Maximum())
            if _cam_log is not None:
                _cam_log.info(
                    f'ids GainSelector={selector} Gain.SetValue({factor:.3f}) (={value} dB)'
                )
            gain_node.SetValue(factor)
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

    def set_test_pattern(self, enabled: bool = False, pattern: str = 'Black') -> bool:
        """Apply the IDS TestPattern node: 'Off' when disabled, else the named
        pattern. Bench-confirmed ReadWrite on the U3-34L0XCP-M with entries
        Off / Black / White / ColorBar / ColorBarVertical / Gray / LightGrey;
        a pattern outside that set is rejected by the SDK. Returns True on
        success, False if inactive / node absent / the SDK rejects the entry
        (the bool signals the caller, matching the driver's other setters).
        Takes effect on subsequent free-run frames."""
        if not self.active or self.remote_nodemap is None:
            return False
        entry = pattern if enabled else 'Off'
        try:
            self.remote_nodemap.FindNode('TestPattern').SetCurrentEntry(entry)
            _cam_log.info(f'[CAM Class ] ids TestPattern.SetCurrentEntry({entry})')
            return True
        except Exception as e:
            _cam_log.warning(
                f'[CAM Class ] IDS set_test_pattern(enabled={enabled}, pattern={pattern}) '
                f'failed: {type(e).__name__}: {e}'
            )
            return False


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

    # Removal is owned solely by the DeviceLost callback (with a typed
    # DeviceLostException fallback in the poll loop), so the poll loop no longer
    # accrues consecutive failures toward an auto-disconnect -- the old
    # N-consecutive-timeouts heuristic mislabeled a host stall as a removal.

    # The worker wakes at least this often to re-check the stop request even
    # when no frames are arriving (a stalled stream must still shut down).
    _WORKER_POLL_S = 0.5

    # Upper bound on how long stop() will work to terminate the grab threads
    # before giving up. A poll thread parked in WaitForFinishedBuffer with a
    # long (exposure-scaled) timeout only unparks via KillWait, so stop() may
    # need several KillWait+join rounds; this bounds the total so teardown can
    # never hang indefinitely on a wedged thread.
    _STOP_JOIN_CEILING_S = 10.0

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
        # Frame-generation gate: the worker bumps this each time it stores a
        # frame; a still capture (grab_new_capture) snapshots it and waits for it
        # to advance, so it returns a genuinely-new frame WITHOUT driving the data
        # stream itself -- the poll thread stays the single WaitForFinishedBuffer
        # consumer and _requeue the single QueueBuffer.
        self._frame_generation = 0
        self._frame_gen_cond = threading.Condition()
        # Sustained-stall tracking for the active device-presence probe. Touched
        # only by the poll thread: _stall_started is the monotonic time of the
        # first timeout in the current no-frame run (None once a frame arrives);
        # _last_presence_probe rate-limits the probe across timeouts;
        # _absence_confirmations counts consecutive absent probes toward removal
        # (reset by a frame, a present/inconclusive probe, or a restart).
        self._stall_started = None
        self._last_presence_probe = 0.0
        self._absence_confirmations = 0
        # Bad-frame run toward the wedge-escalation ceiling: frames that ARRIVED
        # but yielded no usable image (incomplete, or an unreadable handle).
        # Distinct from the presence-probe stall above: a bad frame still arrives
        # (the device is delivering), so it never times out and never accrues
        # toward the presence probe -- a stream of only-bad frames needs its own
        # ceiling or it freezes forever. Reset by a complete frame.
        self._bad_frames_since_complete = 0

    def start(self):
        # Never launch a second consumer while EITHER grab thread is still alive
        # -- a live thread racing a fresh one on the same data stream / slot
        # corrupts the pipeline. A non-None ref whose thread has already EXITED is
        # a prior stop() that could not join within its ceiling (the ref is left
        # set); once the thread is actually dead, clear the stale refs and start
        # fresh rather than stay permanently wedged on a dead reference.
        poll_alive = self._poll_thread is not None and self._poll_thread.is_alive()
        worker_alive = self._worker_thread is not None and self._worker_thread.is_alive()
        if poll_alive or worker_alive:
            return
        self._poll_thread = None
        self._worker_thread = None
        self._stop_event.clear()
        # This handler is reused across stop/start (binning/exposure changes), so
        # clear the stall clock: a stale _stall_started from an earlier no-frame
        # run would make the FIRST timeout after restart satisfy the sustained
        # window immediately and probe (a poll-thread hitch) instead of waiting
        # out the intended debounce.
        self._stall_started = None
        self._last_presence_probe = 0.0
        self._absence_confirmations = 0
        self._bad_frames_since_complete = 0
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

    def stop(self) -> bool:
        """Terminate the poll + worker threads and report whether BOTH are
        confirmed dead. Returns False when a thread could not be joined inside
        the ceiling; the caller must then NOT revoke buffers -- a buffer revoked
        under a still-running poll/worker thread is touched as an invalid handle.

        KillWait is re-posted every round, not once: a single pre-join KillWait
        can miss a poll thread that has not yet entered WaitForFinishedBuffer
        (it was between iterations), leaving it to wait out the full
        exposure-scaled timeout. Re-posting guarantees the abort lands once the
        thread is parked, so join returns promptly instead of timing out.
        """
        self._stop_event.set()
        self._slot.stop()
        deadline = time.monotonic() + self._STOP_JOIN_CEILING_S
        for thread in (self._poll_thread, self._worker_thread):
            while thread is not None and thread.is_alive() and time.monotonic() < deadline:
                # Unblock a poll thread parked in WaitForFinishedBuffer; harmless
                # for the worker, which exits via the slot's stop sentinel.
                try:
                    self.data_stream.KillWait()
                except Exception as e:
                    logger.debug(f'[CAM Class ] KillWait unavailable: {e}')
                thread.join(timeout=0.5)
        poll_alive = self._poll_thread is not None and self._poll_thread.is_alive()
        worker_alive = self._worker_thread is not None and self._worker_thread.is_alive()
        # Drop a reference only once its thread is provably dead, so a leaked
        # live thread stays visible to start()'s guard (which would otherwise
        # launch a second poll thread onto the same data stream).
        if not poll_alive:
            self._poll_thread = None
        if not worker_alive:
            self._worker_thread = None
        quiesced = not poll_alive and not worker_alive
        if not quiesced:
            _cam_log.error(
                '[CAM Class ] IDS grab threads did not terminate within '
                f'{self._STOP_JOIN_CEILING_S}s (poll_alive={poll_alive} '
                f'worker_alive={worker_alive}); skipping buffer revoke this stop'
            )
        return quiesced

    def threads_quiesced(self) -> bool:
        """True when neither grab thread is alive -- the buffer pool is safe to
        revoke and a fresh start will not race a live consumer.

        Re-evaluates aliveness live rather than trusting stop()'s return: a stop()
        that could not join within its ceiling leaves the ref set (returning
        False), but the thread may have exited since, and the next start_grabbing()
        must be able to see that it is now safe to reclaim the pool and restart.
        """
        poll = self._poll_thread
        worker = self._worker_thread
        poll_dead = poll is None or not poll.is_alive()
        worker_dead = worker is None or not worker.is_alive()
        return poll_dead and worker_dead

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

            # A finished buffer arrived -> the device is delivering frames, so
            # any sustained-stall accrual toward the presence probe is cleared
            # (including a partial absent-confirmation streak from an earlier
            # stall in this same session).
            self._stall_started = None
            self._absence_confirmations = 0

            if self._stop_event.is_set():
                self._requeue(buffer)
                break

            # Accessing a finished buffer can raise if its handle has gone invalid
            # -- the data stream wedged (revoked / reset) out from under the live
            # poll thread. This access is OUTSIDE the WaitForFinishedBuffer guard
            # above, so an unhandled raise here would kill the poll thread with a
            # traceback; classify it instead.
            try:
                incomplete = buffer.IsIncomplete()
            except Exception as e:
                if self._handle_buffer_error(e):
                    break
                # Non-removal, non-wedge fault: return the buffer to the pool and
                # keep polling (removal is owned solely by DeviceLost), mirroring
                # the keep-polling path of _handle_wait_error. This buffer ARRIVED
                # but yielded no usable image, so it counts toward the wedge
                # ceiling too -- a stream whose handles keep raising would
                # otherwise freeze forever the same way an all-incomplete one does.
                self._requeue(buffer)
                if self._note_bad_frame():
                    break
                continue

            if incomplete:
                # Incomplete = USB packet loss / bandwidth saturation: a degraded
                # frame, not a removal. ABSORB an isolated blip -- log, re-queue,
                # keep streaming; never mark disconnected (removal is owned by the
                # DeviceLost callback). But COUNT it toward the wedge ceiling.
                self._log_incomplete(buffer)
                self._requeue(buffer)
                if self._note_bad_frame():
                    break
                continue

            # A COMPLETE frame: the stream produced a usable image -- reset the
            # bad-frame run so earlier blips do not accrue toward escalation across
            # a good frame.
            self._bad_frames_since_complete = 0

            # Hand the buffer to the worker. A buffer this displaces is re-queued
            # by the slot; the worker re-queues this one once it has unpacked it.
            self._slot.put(buffer)

    def _note_bad_frame(self) -> bool:
        """Count a frame that arrived but yielded no usable image (incomplete, or
        an unreadable handle). Returns True when the run since the last complete
        frame has crossed the wedge ceiling and the caller must stop the loop.

        A stream delivering only bad frames never times out, so the sustained-
        stall presence probe never fires -- this ceiling is the only net that
        keeps it from freezing forever. On crossing it, escalate to the Category-2
        drop path (the same uniform teardown an unplug takes). Re-checks the stop
        event first: a stop()/reconfigure racing the ceiling frame must just break,
        never trigger a spurious teardown of a camera that is only being stopped.
        """
        self._bad_frames_since_complete += 1
        if self._bad_frames_since_complete < _MAX_BAD_FRAMES_BEFORE_WEDGE:
            return False
        if self._stop_event.is_set():
            return True
        _cam_log.error(
            f'[CAM Class ] {self._bad_frames_since_complete} frames since the last '
            'complete frame -- stream wedged (delivering only unusable frames); '
            'escalating to teardown (Category-2 drop)'
        )
        self._parent._handle_device_lost()
        return True

    def _handle_wait_error(self, e: Exception) -> bool:
        """Classify a WaitForFinishedBuffer error; return True to stop the loop.

        An AbortedException reaching here is a SPURIOUS abort, never our own
        teardown: _poll_loop checks the stop event BEFORE calling this, so a
        requested stop has already broken the loop. The only abort left is a
        KillWait posted by a previous stop() that outlived its
        FlushPendingKillWaits and landed on a freshly (re)started poll thread's
        first wait -- seen under rapid reconfigure (e.g. fast binning toggles).
        Flush the stale abort and keep polling (return False); returning True
        here would strand the live view with a dead poll thread and no frames
        until the next reconfigure. A timeout is a STALL -- keep polling, never
        disconnect. A typed DeviceLostException is an authoritative removal
        (fallback to the DeviceLost callback) and routes to the single removal
        owner. Any other fault is logged and polling continues; removal is owned
        solely by DeviceLost, not by a consecutive-failure count.
        """
        if _exc_is(e, 'AbortedException', 'abort'):
            try:
                self.data_stream.FlushPendingKillWaits()
            except Exception as fe:
                logger.debug(
                    f'[CAM Class ] FlushPendingKillWaits (spurious abort) unavailable: {fe}'
                )
            return False
        if _exc_is(e, 'TimeoutException', 'timeout'):
            # A timeout is a STALL, not a removal: no frame arrived within the
            # (exposure-scaled) window -- a host stall, a throughput hiccup, or a
            # wedged-but-present camera. Keep polling unless the sustained-stall
            # probe has confirmed a USB removal (DeviceLost never fires for it),
            # in which case stop the loop promptly like the typed-removal path
            # below rather than spin against a device already routed to teardown.
            logger.debug(f'[CAM Class ] poll timeout (stall, not removal): {e!r}')
            return self._check_sustained_stall()
        if _exc_is(e, 'DeviceLostException', 'removed', 'device'):
            # Authoritative typed removal -- the fallback if the DeviceLost
            # callback ever misses. Routes to the same single removal owner.
            _cam_log.warning(f'[CAM Class ] Device removal detected in grab loop: {e}')
            self._parent._handle_device_lost()
            return True
        if _exc_is(e, 'InvalidInstanceException', 'invalid', 'bufferhandle'):
            # The data stream was revoked/reset out from under the live wait, so
            # WaitForFinishedBuffer itself raises an invalid-handle fault. Without
            # this case the fault fell through to keep-polling below and the poll
            # thread HOT-SPUN on it thousands of times a second against a dead
            # stream (observed on the bench: a single wedge produced tens of
            # thousands of identical warnings and never stopped). Treat it exactly
            # like the same fault on buffer access: stop the loop and route to
            # the uniform removal drop -- the DeviceReset self-heal this used to
            # attempt never once succeeded on hardware (four distinct failure
            # modes) and its reset reboots the camera under whatever runs next,
            # so a wedge exits as a clean disconnect the user recovers by
            # reconnecting. Mirrors _handle_buffer_error.
            _cam_log.error(
                '[CAM Class ] IDS data-stream handle invalid in wait (stream wedged): '
                f'{type(e).__name__}: {e!r} -- routing to the removal drop'
            )
            self._parent._handle_device_lost()
            return True
        # Any other wait error: log and keep polling. Removal is owned solely by
        # DeviceLost (callback + typed fallback above), so a transient SDK fault
        # no longer accrues toward an auto-disconnect -- the old N-consecutive
        # heuristic mislabeled a host stall as a removal. !r escapes non-ASCII.
        _cam_log.warning(f'[CAM Class ] ImageHandler poll exception: {type(e).__name__}: {e!r}')
        return False

    def _handle_buffer_error(self, e: Exception) -> bool:
        """Classify an error from accessing a finished buffer; return True to stop
        the loop, False to keep polling.

        WaitForFinishedBuffer already succeeded, so this is not a wait error -- it
        is the finished buffer's handle raising on access. Three cases, mirroring
        _handle_wait_error so removal stays owned by the single DeviceLost owner:

        - A typed DeviceLostException is an authoritative removal (the fallback if
          the DeviceLost callback is ever missed): route to the single removal
          owner and stop the loop.
        - An invalid buffer handle means the data stream wedged (revoked / reset)
          out from under the live poll thread; the buffer is unusable. Stop the
          loop rather than hot-spin on a stream yielding only invalid buffers,
          and route to the uniform removal drop: the wedge exits as a clean
          disconnect the user recovers by reconnecting.
        - Any other fault: keep polling. Removal is owned solely by DeviceLost, so
          an unknown transient fault must not tear the poll thread down for good.
        """
        if _exc_is(e, 'DeviceLostException', 'removed', 'device'):
            _cam_log.warning(f'[CAM Class ] Device removal detected accessing buffer: {e}')
            self._parent._handle_device_lost()
            return True
        if _exc_is(e, 'InvalidInstanceException', 'invalid', 'bufferhandle'):
            _cam_log.error(
                '[CAM Class ] IDS finished-buffer handle invalid (stream wedged): '
                f'{type(e).__name__}: {e!r} -- routing to the removal drop'
            )
            self._parent._handle_device_lost()
            return True
        _cam_log.warning(
            f'[CAM Class ] ImageHandler buffer access exception: {type(e).__name__}: {e!r}'
        )
        return False

    def stall_age_s(self) -> float:
        """Seconds since the STARTED stream last produced a finished buffer,
        0.0 when healthy, not yet started, or already stopped. Read by teardown
        to decide whether the device is likely dead -- remote-device writes
        against unplugged hardware block the calling thread for seconds before
        failing, so a long-stalled stream skips them.

        Zero once the grab threads are quiesced: the stall clock is only
        meaningful while a poll thread is (supposed to be) delivering. Without
        this, a clock armed by a brief blip before a clean stop would keep
        aging in wall-clock time and a much later stop on a healthy stopped
        camera would misclassify it as dead.
        """
        started = self._stall_started
        if started is None or self.threads_quiesced():
            return 0.0
        return time.monotonic() - started

    def _check_sustained_stall(self) -> bool:
        """On a poll timeout, escalate a SUSTAINED stall to an active presence
        probe -- the USB-unplug fallback for the DeviceLost callback that never
        fires on the U3-34L. Returns True iff it concluded a removal (so the
        caller stops the poll loop, like the typed-removal path); False to keep
        polling.

        A single timeout is just a stall (host hiccup, wedged-but-present
        camera); only when the stream has produced no frame for
        _PRESENCE_PROBE_STALL_S do we ask the parent to actively check the device
        is still enumerated, and then no more than once per
        _PRESENCE_PROBE_INTERVAL_S so a long wedge isn't hammered with
        DeviceManager.Update calls.

        The parent's probe is a pure present/absent/inconclusive query; this
        method owns the debounce. Only after _PRESENCE_PROBE_CONFIRMATIONS
        CONSECUTIVE absent probes does it route to the single removal owner --
        a present or inconclusive reading resets the streak -- so a transient
        enumeration hiccup can't tear down a still-connected camera.
        _stall_started and the streak are reset by the poll loop the moment a
        frame arrives.
        """
        now = time.monotonic()
        if self._stall_started is None:
            self._stall_started = now
            return False
        if now - self._stall_started < _PRESENCE_PROBE_STALL_S:
            return False
        # The ceiling scales with the (exposure-scaled) wait window so a long
        # exposure's healthy inter-frame gap can never reach it: it requires
        # several consecutive empty windows, never less than the floor.
        ceiling_s = max(_STALL_ESCALATION_CEILING_S, 5.0 * self.timeout_ms / 1000.0)
        if now - self._stall_started >= ceiling_s:
            # The probe alone cannot conclude removal when the producer keeps a
            # phantom entry enumerated after an unplug -- without this ceiling
            # such a stall lasts forever, removal never latches, and every
            # teardown path keeps treating dead hardware as live. Routed to the
            # same uniform removal drop every dead-stream ceiling takes.
            _cam_log.error(
                f'[CAM Class ] stream stalled {now - self._stall_started:.0f}s with the '
                'device still enumerated -- treating as dead (ghost enumeration or an '
                'unrecoverable stall); routing to the removal drop'
            )
            self._parent._handle_device_lost()
            return True
        if now - self._last_presence_probe < _PRESENCE_PROBE_INTERVAL_S:
            return False
        self._last_presence_probe = now
        present = self._parent._probe_device_presence()
        if present is False:
            self._absence_confirmations += 1
            if self._absence_confirmations >= _PRESENCE_PROBE_CONFIRMATIONS:
                _cam_log.warning(
                    f'[CAM Class ] Device absent from enumeration on '
                    f'{self._absence_confirmations} consecutive probes after a '
                    'sustained poll stall; treating as USB removal (DeviceLost '
                    'did not fire)'
                )
                self._parent._handle_device_lost()
                return True
        else:
            # Present (True) or inconclusive (None): the device is not confirmed
            # gone, so break any absence streak rather than carry it forward.
            self._absence_confirmations = 0
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

    def frame_generation(self) -> int:
        """Current frame-store counter. Snapshot it before a still capture, then
        wait_for_new_frame() past it to get a frame stored after that point."""
        with self._frame_gen_cond:
            return self._frame_generation

    def wait_for_new_frame(self, since: int, timeout_s: float) -> bool:
        """Block until the worker stores a frame newer than ``since`` (or timeout).

        Returns True if a newer frame is available, False on timeout. This is the
        still-capture path's hook into the live worker: it never touches the data
        stream, so a still capture cannot race the poll/worker on WaitForFinished-
        Buffer or QueueBuffer.
        """
        deadline = time.monotonic() + timeout_s
        with self._frame_gen_cond:
            while self._frame_generation <= since:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._frame_gen_cond.wait(remaining)
            return True

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
                with self._frame_gen_cond:
                    self._frame_generation += 1
                    self._frame_gen_cond.notify_all()
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
        """Unpack one finished buffer to its delivered array + that array's depth.

        Delegates to _unpack_buffer (shared with the still-capture path) so the
        oversize-then-crop crop is applied identically on both: BufferToImage +
        ConvertTo to the delivery target, crop to the recorded window, copy out.
        The depth is _ids_delivery_significant_bits (paired with the delivery
        target): 12 for the 12-bit modes' native uint16, 8 for the 8-bit-mode
        Mono10 wire delivered directly as uint8. crop_spec and the buffer size
        change together inside update_camera_config (grab stopped), so they stay
        consistent. Worker-only.
        """
        wire = self._parent.get_pixel_format()
        array = _unpack_buffer(buffer, wire, self._parent._crop_spec)
        return array, _ids_delivery_significant_bits(wire)
