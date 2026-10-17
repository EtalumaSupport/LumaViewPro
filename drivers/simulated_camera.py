# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Simulated Camera -- drop-in replacement for PylonCamera / IDSCamera.

No camera hardware required. Generates synthetic images, tracks all
camera state (exposure, gain, binning, frame size, pixel format), and
supports the full Camera ABC interface.
"""

import collections
import dataclasses
import datetime
import pathlib
from dataclasses import dataclass
import threading
import random
import time
from collections.abc import Callable
from typing import ClassVar

import cv2
import numpy as np

from lvp_logger import logger
from drivers.camera import Camera, Frame, FrameGrid, ImageHandlerBase, link_info
from drivers.camera_profiles import simulated_profile
from drivers.exceptions import CameraNoAnswerError, PartAbsentError
from drivers.registry import camera_registry
from drivers.simulated_specimen import drift_frames, specimen_field

# camera.log hookup: simulator records the same per-driver SDK-call
# trace that real drivers (pyloncamera/idscamera/fx2driver) write, so
# sim-mode runs produce a populated logs/camera.log for verification.
try:
    from lvp_logger import camera_logger as _cam_log
except ImportError:
    _cam_log = None


@dataclass(frozen=True)
class SimulatedStall:
    """When a simulated camera's stream stalls, and for how long.

    A stall is a stretch in which the camera stays connected and grabbing
    and no frame arrives -- what a real camera's link or grab loop can fall
    into without the device being removed. Built where it is asked for (a
    launch argument, a test), so a stall that cannot happen is refused there.

    Attributes:
        after_s: Seconds from the moment it is applied until the stall begins.
        for_s: How long the stall lasts, in seconds.

    Raises:
        ValueError: ``after_s`` is negative or ``for_s`` is not positive.
    """

    after_s: float
    for_s: float

    def __post_init__(self):
        if self.after_s < 0 or self.for_s <= 0:
            raise ValueError(
                f'a stall starts at 0 s or later and lasts more than 0 s; got '
                f'after_s={self.after_s}, for_s={self.for_s}'
            )


@dataclass(frozen=True)
class SensorExposure:
    """What one simulated frame was exposed under: the settings the sensor
    applied when its exposure started, which is what its pixels show.

    A write reaches the camera's register at once -- the frame's chunk
    reports the register -- but a frame already started comes out under the
    settings it started with. This record is that difference made
    readable: a frame's chunk can name the new gain while this names the
    gain its pixels were made at.

    Attributes:
        frame_id: The camera's frame counter, as the frame's ``FrameID``
            chunk carries it.
        started_ticks: The camera's clock when the exposure started, as the
            frame's ``Timestamp`` chunk carries it.
        exposure_us: The exposure applied.
        gain_db: The gain applied.
        black_level: The black level applied.
        illumination_ma: The LED current lighting the sample, summed over
            the lit addresses; 0 for an unlit field.
        x_um: The stage's X when the exposure started; None on a scope
            with no X axis.
        y_um: The stage's Y, likewise.
        z_um: Z when the exposure started; None on a scope with no focus axis.
        z_end_um: Z when the exposure ended (the next frame's start); None
            until then, and on a scope with no focus axis.
        depth_of_field_um: The objective's depth of field the frame was
            focused through; None when no objective is known.
    """

    frame_id: int
    started_ticks: int
    exposure_us: float
    gain_db: float
    black_level: float
    illumination_ma: float
    x_um: float | None
    y_um: float | None
    z_um: float | None
    z_end_um: float | None
    depth_of_field_um: float | None


class _SimImageHandler(ImageHandlerBase):
    """The real drivers' frame buffer, with a wait for the next stored frame.

    The simulator's acquisition thread stores into it exactly as a real
    driver's SDK thread does, so the frame count, the buffered frame and the
    per-frame callbacks all come from the one implementation every driver
    shares. The wait is what ``grab_new_capture`` needs: a frame stored after
    the call, without polling.
    """

    def __init__(self):
        super().__init__()
        self._stored = threading.Condition()

    def _store_frame(
        self, image, timestamp, chunks=None, *, arrived_ns, significant_bits, wire_bytes
    ) -> Frame | None:
        frame = super()._store_frame(
            image,
            timestamp,
            chunks,
            arrived_ns=arrived_ns,
            significant_bits=significant_bits,
            wire_bytes=wire_bytes,
        )
        with self._stored:
            self._stored.notify_all()
        return frame

    def wait_for_frame_after(self, since: int, timeout_s: float) -> bool:
        """True once a frame later than ordinal ``since`` is stored; False on timeout."""
        deadline = time.monotonic() + timeout_s
        with self._stored:
            while self.frames_delivered <= since:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._stored.wait(remaining)
            return True


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """``image`` blurred by a Gaussian of ``sigma`` pixels.

    A wide blur is made on a reduced copy and enlarged back: past a few
    pixels of sigma the fine detail it would keep is gone anyway, and a
    full-size kernel that wide costs tens of milliseconds a frame.
    """
    if sigma <= 8.0:
        return cv2.GaussianBlur(image, (0, 0), sigma)
    factor = sigma / 8.0
    h, w = image.shape
    small = cv2.resize(
        image,
        (max(1, round(w / factor)), max(1, round(h / factor))),
        interpolation=cv2.INTER_AREA,
    )
    small = cv2.GaussianBlur(small, (0, 0), 8.0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


@camera_registry.register('sim', priority=100, is_simulator=True)
class SimulatedCamera(Camera):
    _SENSOR = simulated_profile().native_resolution
    MODEL_NAME = simulated_profile().model_name
    SERIAL_NUMBER = 'SIM-CAM-001'

    TIMING_FAST: ClassVar[dict] = {'grab_delay': 0.0}
    TIMING_REALISTIC: ClassVar[dict] = {'grab_delay': 0.005}  # ~5ms USB transfer overhead

    # How many frames the sensor has started, and not yet delivered, when a
    # setting is written: they come out under the settings they started
    # with. The simulated camera's own constant, never the gate's skip count
    # or the timing record (b) runs on: coupled to either, a test of the gate
    # passes by construction. Two is the depth of a frame exposing and a
    # frame reading out; frames held on the way to the host are the
    # simulated backlog's.
    SENSOR_PIPELINE_FRAMES = 2

    # The LED current at which a lit field renders at the brightness the
    # exposure and gain give it; the light scales in proportion to the
    # current. A camera built with no scope around it is lit at this.
    REFERENCE_ILLUMINATION_MA = 10.0

    # How far the specimen moves on the sensor per micrometre of stage
    # travel, in pixels at binning 1. The specimen is a pattern, not a
    # calibrated slide: this makes the field follow X and Y, so a frame made
    # before a move shows the old field.
    _SPECIMEN_PX_PER_UM = 0.5

    # How many delivered frames' exposure records the camera keeps for
    # ``exposed_under``: a bound on memory, far more than a capture's wait
    # or a test's look back covers.
    EXPOSURE_RECORDS_KEPT = 256

    # The camera's own clock: nanosecond ticks, as a Basler USB3 body's.
    TIMESTAMP_TICK_HZ = 1_000_000_000

    # Free-run delivery ceiling in frames per second. A real camera's
    # sensor readout and link bandwidth bound its frame rate however
    # short the exposure; without a ceiling the simulator free-runs at
    # 1/exposure (1000 fps at 1 ms), which no camera delivers.
    _MAX_DELIVERY_FPS = 40.0

    # Where the sample sits, nominally: the height bring-up places the stage
    # at, as an operator focusing by eye would.
    _NOMINAL_FOCAL_Z_UM = 5000.0

    # Blur, in pixels of sigma at binning 1, per depth of field off focus.
    # Within a fraction of a depth of field the sample stays sharp, and
    # beyond it the blur grows in proportion, steeper for a high-NA lens.
    _BLUR_PX_PER_DOF = 1.5

    # The depth of field a camera built with no scope around it renders
    # with: there is no objective to ask, and its focus still has to fall
    # off on either side of the plane for a standalone test of it.
    _UNWIRED_DOF_UM = 10.0

    # The nodes a write can be refused on (refuse_write), each by the setter
    # the API writes it through.
    WRITABLE_NODES: ClassVar[dict[str, str]] = {
        'gain': 'gain',
        'exposure': 'exposure_t',
        'auto_exposure': 'auto_exposure_t',
        'frame_size': 'set_frame_size',
        'binning': 'set_binning_size',
        'pixel_format': 'set_pixel_format',
        'black_level': 'set_black_level',
        'auto_gain': 'auto_gain',
        'auto_gain_once': 'auto_gain_once',
        'auto_gain_target_brightness': 'update_auto_gain_target_brightness',
    }
    # What a refused write answers, by name: a real SDK's node write
    # answers False, nothing, a bare True that confirms no value, or raises;
    # or answers False because the camera left the bus during the write; or
    # is not answered in time while the camera stays on the bus.
    REFUSED_WRITE_ANSWERS: ClassVar[tuple[str, ...]] = (
        'false',
        'none',
        'true',
        'raise',
        'removed',
        'no_answer',
    )
    # The answers no real driver gives for a node: a frame-size write answers
    # the window it delivers or False, never a bare True.
    _ANSWERS_NO_DRIVER_GIVES: ClassVar[dict[str, frozenset[str]]] = {
        'frame_size': frozenset({'true'}),
    }

    def __init__(
        self,
        width: int = _SENSOR['width'],
        height: int = _SENSOR['height'],
        grab_delay: float = 0.0,
        z_position_func: Callable[[], float | None] | None = None,
        xy_position_func: Callable[[], tuple[float | None, float | None]] | None = None,
        illumination_func: Callable[[], float] | None = None,
        depth_of_field_func: Callable[[], float | None] | None = None,
        focal_spread_um: float = 0.0,
        timing: str = 'fast',
    ):
        # Native (unbinned) sensor size -- the fixed ceiling. _width/_height
        # below are the CURRENT frame size in post-binning (displayed) pixels,
        # matching the Pylon driver's convention: set_frame_size takes the
        # post-binning ROI, and the grabbed image is exactly that size (the
        # sensor delivers native/binning pixels). At binning 1 the frame
        # equals the native size.
        self._native_width = width
        self._native_height = height
        self._width = width
        self._height = height
        self._grab_delay = grab_delay

        self._exposure_us = 10_000.0  # 10 ms in microseconds
        self._gain = 1.0
        self._black_level = 0.0
        self._pixel_format = 'Mono8'
        self._binning = 1
        self._grabbing = False
        self._auto_gain_enabled = False
        self._auto_gain_target_brightness = 0.5
        self._auto_gain_min = 0.0
        self._auto_gain_max = 20.0
        self._auto_exposure_enabled = False
        self._frame_rate_limit_enabled = False
        self._frame_rate_target = 30.0

        self._lock = threading.RLock()

        # The camera's clock: its own epoch, unrelated to the host's, so a
        # mapping between the two has to be measured, as on a real body.
        self._clock_offset_ns = random.randrange(10**12)
        # What the last latch_clock latched; 0, as a body's value node
        # reads before its first latch.
        self._latched_ticks = 0
        # The camera's frame counter: one per exposure started.
        self._frames_started = 0
        # How many made frames wait on the way to the host before the
        # oldest reaches it: a host backlog (``delay_delivery``). 0 delivers
        # each frame as it is made.
        self._backlog_frames = 0
        # What each recently delivered frame was exposed under, by its
        # FrameID, oldest first (``exposed_under``).
        self._exposure_records: collections.OrderedDict[int, SensorExposure] = (
            collections.OrderedDict()
        )

        # A real camera free-runs: while it is grabbing, frames arrive on the
        # SDK's own thread whether or not anyone reads them, and a reader gets
        # the latest one. The simulator does the same on this acquisition
        # thread, storing each frame in the shared image handler. A simulator
        # that made a frame only when one was read showed a frame count that
        # stood still while nothing read it, so a stream that had died and a
        # stream nobody was watching looked the same.
        self._acquisition_thread: threading.Thread | None = None
        self._acquisition_stop = threading.Event()
        # A stretch of monotonic time in which the stream delivers nothing
        # while the camera stays connected and grabbing: the silent stall a
        # real camera's USB link or grab loop can fall into. None when no
        # stall is set. See hold_frames.
        self._held_window: tuple[float, float] | None = None

        # Synthetic image state -- can be set externally for test scenarios
        # 'specimen', 'black', 'white', 'noise', 'focus_target'.
        # 'specimen' is the no-pattern-requested state; anything unrecognized
        # renders it too, with a warning.
        self._test_pattern = 'specimen'

        # The specimen at the full sensor, built when first rendered or
        # loaded; the frames cut from it at the current binning and size, one
        # per drift step; and each source's last defocused frame, reused
        # while Z, the objective and the frame are unchanged. Each cache
        # entry is a (key, value) pair, rebuilt when its key changes.
        self._specimen_native: np.ndarray | None = None
        self._specimen_cut: tuple | None = None
        self._drift_index = 0
        self._defocused: dict[str, tuple] = {}
        self._focus_target_cache: tuple | None = None

        # Z-dependent focus. The sample's true plane is drawn once, within
        # focal_spread_um of the nominal height: a simulated scope passes a
        # spread, so every session needs a real autofocus; a camera built
        # on its own is in focus at the nominal height. set_focal_z fixes it.
        self._z_position = self._NOMINAL_FOCAL_Z_UM
        self._focal_z = self._NOMINAL_FOCAL_Z_UM + random.uniform(-focal_spread_um, focal_spread_um)
        logger.info(f'[SimCamera ] The simulated sample is in focus at Z {self._focal_z:.1f} um')
        self._z_position_func = z_position_func  # Optional: auto-query Z from motor
        # Optional: auto-query the stage's X and Y, so the field follows them.
        self._xy_position_func = xy_position_func
        # Optional: auto-query how much light is on the sample. Unwired, the
        # camera renders as though the field were lit, which is what a camera
        # constructed on its own with no scope around it has to assume.
        self._illumination_func = illumination_func
        # Optional: auto-query the objective's depth of field (um); None when
        # no objective is known.
        self._depth_of_field_func = depth_of_field_func

        # Apply timing preset (overrides grab_delay if preset given)
        self.set_timing_mode(timing)

        # Let the base class call connect()
        super().__init__()

    def set_timing_mode(self, mode: str) -> None:
        """Switch timing mode: 'fast' or 'realistic'.

        Args:
            mode: One of ``'fast'``, ``'realistic'``.

        Raises:
            ValueError: ``mode`` is not a known preset.
        """
        if mode == 'realistic':
            preset = self.TIMING_REALISTIC
        elif mode == 'fast':
            preset = self.TIMING_FAST
        else:
            raise ValueError(f"Unknown timing mode: {mode!r}. Use 'fast' or 'realistic'.")
        self._grab_delay = preset['grab_delay']
        self._timing_mode = mode

    def load_specimen(self, image_dir: str | None = None) -> None:
        """Load the specimen this camera images, and show it.

        The first image in ``image_dir`` (png, jpg, tiff), else in
        data/sim_images/, as grayscale at the full sensor size; with none,
        the generated specimen field: one sample, as a real one is, seen
        drifting a pixel around a circle so a running stream looks alive.

        Args:
            image_dir: Path to a directory holding the image. None looks in
                data/sim_images/ relative to the app root.
        """
        if image_dir is None:
            for candidate in [
                pathlib.Path(__file__).resolve().parent.parent / 'data' / 'sim_images',
                pathlib.Path('.') / 'data' / 'sim_images',
            ]:
                if candidate.is_dir():
                    image_dir = candidate
                    break

        loaded = None
        if image_dir is not None and pathlib.Path(image_dir).is_dir():
            files = [
                fp
                for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')
                for fp in sorted(pathlib.Path(image_dir).glob(ext))
            ]
            for fp in files:
                image = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    logger.warning(f'[SimCamera ] Could not load {fp}')
                    continue
                loaded = cv2.resize(
                    image, (self._native_width, self._native_height), interpolation=cv2.INTER_AREA
                )
                logger.info(f'[SimCamera ] Loaded the specimen image: {fp.name}')
                break

        self._specimen_native = (
            loaded
            if loaded is not None
            else specimen_field(self._native_height, self._native_width)
        )
        self._specimen_cut = None
        self._defocused = {}
        self._test_pattern = 'specimen'

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        """Mark the simulated camera as active and load its profile.

        Returns:
            bool: Always True.
        """
        with self._lock:
            self.active = True
            self.model_name = self.MODEL_NAME
            self._device_serial = self.SERIAL_NUMBER
            self._device_removed = False

            self._load_profile()
            self.init_camera_config()
            # The camera stamps each frame on its own clock, read at connect
            # as the Pylon driver reads its body's tick rate.
            self.timestamp_tick_frequency_hz = self.TIMESTAMP_TICK_HZ
            # A fresh handler per connection, as the real drivers build one:
            # its frame count starts over, and the durable callback registry
            # is pushed onto it so a listener registered before a reconnect
            # keeps receiving frames.
            self.cam_image_handler = _SimImageHandler()
            self._reapply_frame_callbacks()
            # connect() returns CONFIGURED but NOT grabbing; the single
            # start fires later via open_and_start() (the start gate).

            if _cam_log is not None:
                _cam_log.info(f'sim Connected: {self.model_name} ({self._device_serial})')
            logger.info(f'[CAM Sim   ] Connected: {self.model_name} ({self._device_serial})')
            return True

    def _disconnect(self) -> bool:
        """Mark the simulated camera as disconnected.

        Returns:
            bool: True when the camera was active before this call,
                False when it was already disconnected.
        """
        with self._lock:
            if not self.active:
                return False
            self._grabbing = False
            self.active = None
            if _cam_log is not None:
                _cam_log.info('sim Disconnected')
            logger.info('[CAM Sim   ] Disconnected')
        # Stop acquisition OUTSIDE self._lock (mirroring stop_grabbing): the
        # acquisition loop generates frames under self._lock, so joining it
        # while holding that lock stalls the full join timeout whenever it is
        # mid-frame.
        self._stop_acquisition()
        # Reset base lifecycle state OUTSIDE self._lock too:
        # _reset_lifecycle_state takes _lifecycle_lock, and holding sim's _lock
        # across it would create a _lock -> _lifecycle_lock acquisition order
        # (the base config path takes them the other way). Acquisition is
        # stopped above, so nothing writes the frame buffer from here on.
        self._reset_lifecycle_state()
        return True

    def is_connected(self) -> bool:
        """Whether the simulated camera is currently connected.

        Returns:
            bool: True when active and the device-removed flag is clear.
        """
        if self.active in (False, None):
            return False
        return not self._device_removed

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def init_camera_config(self) -> None:
        """Reset simulated camera to default config (Mono8, 10 ms, gain=1, bin=1)."""
        if not self.active:
            return
        self._pixel_format = 'Mono8'
        self._exposure_us = 10_000.0  # 10 ms
        self._gain = 1.0
        # Reset at every connect, as the real bodies reload their Default
        # user set.
        self._black_level = 0.0
        self._binning = 1

    # ------------------------------------------------------------------
    # Grabbing
    # ------------------------------------------------------------------
    def is_grabbing(self) -> bool:
        """Return whether the simulated camera is currently acquiring.

        Returns:
            bool: True when ``start_grabbing()`` has been called and
                ``stop_grabbing()`` has not.
        """
        return self._grabbing

    def _start_grabbing(self) -> None:
        """Begin acquiring frames in the simulator."""
        with self._lock:
            self._grabbing = True
            if _cam_log is not None:
                _cam_log.info('sim start_grabbing')
            logger.info('[CAM Sim   ] start_grabbing')
        self._start_acquisition()

    def _stop_grabbing(self) -> None:
        """Stop acquiring frames in the simulator."""
        with self._lock:
            self._grabbing = False
            if _cam_log is not None:
                _cam_log.info('sim stop_grabbing')
            logger.info('[CAM Sim   ] stop_grabbing')
        self._stop_acquisition()

    def _start_acquisition(self) -> None:
        """Spawn the acquisition thread if not already running."""
        if self._acquisition_thread is not None and self._acquisition_thread.is_alive():
            return
        self._acquisition_stop.clear()
        self._acquisition_thread = threading.Thread(
            target=self._acquisition_loop,
            name='SimCameraAcquisition',
            daemon=True,
        )
        self._acquisition_thread.start()

    def _stop_acquisition(self) -> None:
        """Signal the acquisition thread to exit and join with a short timeout.

        A listener that stops grabbing from inside its own frame callback runs
        on the acquisition thread, which cannot join itself; the stop flag
        alone ends the loop once the callback returns.
        """
        self._acquisition_stop.set()
        t = self._acquisition_thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._acquisition_thread = None

    def leave_bus(self) -> None:
        """The camera leaves the bus, as a real one is marked when its
        device is removed; its own reconnect brings it back."""
        self._mark_disconnected()

    @classmethod
    def answers_for(cls, node: str) -> tuple[str, ...]:
        """The refused-write answers ``node`` can be given: those a real driver gives for it."""
        never = cls._ANSWERS_NO_DRIVER_GIVES.get(node, frozenset())
        return tuple(a for a in cls.REFUSED_WRITE_ANSWERS if a not in never)

    def refuse_write(self, node: str, answer: str, count: int = 1) -> None:
        """The next ``count`` writes of ``node`` answer ``answer`` and move nothing.

        Args:
            node: One of ``WRITABLE_NODES``.
            answer: One of ``REFUSED_WRITE_ANSWERS``: ``'false'``,
                ``'none'`` or ``'true'`` is returned; ``'raise'`` raises
                ``RuntimeError`` as a failed node write; ``'removed'``
                takes the camera off the bus and returns False, as a
                camera pulled during the write does; ``'no_answer'``
                raises ``CameraNoAnswerError`` as a write the camera did
                not answer in time, leaving it on the bus.
            count: How many writes in a row meet it, 1 or more: a write
                sent again at once meets the second.

        Raises:
            ValueError: ``node`` or ``answer`` is not one of those, or the
                answer is one no real driver gives for that node.
        """
        if node not in self.WRITABLE_NODES:
            raise ValueError(f'{node!r} is not one of {tuple(self.WRITABLE_NODES)}')
        taken = self.answers_for(node)
        if answer not in taken:
            raise ValueError(f'{answer!r} is not one of {taken} for {node!r}')
        setter = self.WRITABLE_NODES[node]

        # The refusal stands in for the setter on this camera for ``count``
        # writes: the last of them takes it away again, so the next reaches
        # the node.
        left = [count]

        def refused(*args, **kwargs):
            left[0] -= 1
            if left[0] <= 0:
                vars(self).pop(setter, None)
            logger.warning(f'[CAM Sim   ] {node} write refused ({answer}, simulated)')
            if answer == 'raise':
                raise RuntimeError(f'simulated {node} node write failed')
            if answer == 'removed':
                self.leave_bus()
                return False
            if answer == 'no_answer':
                raise CameraNoAnswerError(f'simulated {node} write: the camera did not answer')
            return {'false': False, 'none': None, 'true': True}[answer]

        vars(self)[setter] = refused

    def allow_write(self, node: str) -> None:
        """Take back a ``refuse_write`` on ``node`` that no write has met yet."""
        vars(self).pop(self.WRITABLE_NODES[node], None)

    def hold_frames(self, stall: SimulatedStall) -> None:
        """Stall the stream as ``stall`` says, starting its clock now.

        The camera stays connected and grabbing throughout; only the frames
        stop. This is how a simulated scope shows what happens when a stream
        stops delivering, without hardware.
        """
        start = time.monotonic() + stall.after_s
        self._held_window = (start, start + stall.for_s)
        logger.info(
            f'[CAM Sim   ] frames will stop in {stall.after_s:g} s for {stall.for_s:g} s '
            '(simulated stall)'
        )

    def release_frames(self) -> None:
        """End a stall set by ``hold_frames``, or take back one not yet begun."""
        self._held_window = None

    def delay_delivery(self, frames: int) -> None:
        """Hold ``frames`` made frames on the way to the host from now on.

        A frame reaches the host only once ``frames`` more have been made
        behind it, as a frame waits in a link or a driver queue that has
        fallen behind: it arrives later than it was made, numbered and
        timed at its arrival. 0 delivers each frame as it is made, and the
        frames still held arrive at once.
        """
        self._backlog_frames = frames

    def exposed_under(self, frame_id: int) -> SensorExposure | None:
        """What the delivered frame ``frame_id`` was exposed under; None for
        one this camera did not deliver or no longer keeps a record of."""
        with self._lock:
            return self._exposure_records.get(frame_id)

    def _frame_interval_s(self, exposure_us: float) -> float:
        """The free-run frame period at ``exposure_us``: the exposure, never
        shorter than the delivery ceiling's period. The pacing and the
        reported frame rate both read it, so the report cannot drift from
        the pacing."""
        return max(exposure_us / 1_000_000.0, 1.0 / self._MAX_DELIVERY_FPS)

    def get_resulting_frame_rate(self) -> float | None:
        if not self.active:
            return None
        return 1.0 / self._frame_interval_s(self._exposure_us)

    def get_link_info(self) -> dict | None:
        if not self.active:
            return None
        return link_info(transport='USB3')

    def _acquisition_loop(self) -> None:
        """Store one new frame per frame interval while grabbing.

        The interval is the exposure, never shorter than the delivery
        ceiling's period (``_MAX_DELIVERY_FPS``). Frames are due on a
        fixed schedule, so the host time spent generating and delivering
        a frame comes out of the interval rather than adding to it -- a
        real camera's frame period does not include host work. A loop
        that falls behind (generation slower than the interval) delivers
        the next frame at once and re-anchors, never bursting to catch up.

        Each frame goes through the image handler's ``_store_frame``, which
        counts it, buffers it with its depth and fires the per-frame
        callbacks.

        A frame is exposed under the settings in force when it starts
        (``SensorExposure``), and the sensor has ``SENSOR_PIPELINE_FRAMES``
        frames started when a setting is written; those come out under the
        old settings, timed and rendered by them, and the new ones reach the
        output after them. Each frame carries the chunks a Basler body
        stamps: its counter and its exposure start on the camera's clock
        from the snapshot, and the exposure and gain from the register as
        the frame is tagged, so the chunk leads the content as it does on
        the bench.
        """
        next_due = time.monotonic()
        stalled = False
        # Made frames on their way to the host, oldest first (delay_delivery).
        on_the_way: collections.deque = collections.deque()
        # The frames started and not yet delivered, oldest first.
        with self._lock:
            in_flight = collections.deque(
                self._start_exposure() for _ in range(self.SENSOR_PIPELINE_FRAMES)
            )
        while not self._acquisition_stop.is_set():
            handler = self.cam_image_handler
            if handler is None or not self._grabbing:
                return
            held = self._held_window
            now = time.monotonic()
            if held is not None and held[0] <= now < held[1]:
                # Stalled: nothing reaches the host, and the stream picks up
                # on its own schedule when the stall ends.
                if not stalled:
                    stalled = True
                    logger.info('[CAM Sim   ] simulated stall: frames stopped')
                if self._acquisition_stop.wait(
                    min(self._frame_interval_s(self._exposure_us), held[1] - now)
                ):
                    return
                next_due = time.monotonic()
                continue
            if stalled:
                stalled = False
                logger.info('[CAM Sim   ] simulated stall: frames resumed')
            with self._lock:
                started = self._start_exposure()
                # The newest frame's exposure ends as this one starts.
                if in_flight:
                    in_flight[-1] = dataclasses.replace(in_flight[-1], z_end_um=started.z_um)
                in_flight.append(started)
                exposure = in_flight.popleft()
                image = self._generate_image(exposure)
                bits = self.significant_bits
                chunks = {
                    'FrameID': exposure.frame_id,
                    'Timestamp': exposure.started_ticks,
                    'ExposureTime': self._exposure_us,
                    'Gain': self._gain,
                }
                next_exposure_us = in_flight[0].exposure_us
            # The transfer from a real camera to the host takes time after the
            # frame is exposed; the realistic preset charges it here, between
            # the frame being made and the host having it.
            if self._grab_delay > 0:
                time.sleep(self._grab_delay)
            # A stall that began while this frame was being made holds it
            # too: it had not reached the host, and nothing reaches the host
            # during a stall. Without this, the one frame in the making when
            # a stall is set arrives after it, so a stall set just before a
            # single grab sometimes delivers that grab a frame.
            held = self._held_window
            if held is not None and held[0] <= time.monotonic() < held[1]:
                continue
            # A frame reaches the host once the backlog behind it is made;
            # with none it goes at once, and a backlog taken back releases
            # every frame it held.
            on_the_way.append((image, bits, chunks, exposure))
            while len(on_the_way) > self._backlog_frames:
                self._deliver(handler, *on_the_way.popleft())
            # The next frame's own exposure is the inter-frame interval,
            # bounded below by the delivery ceiling.
            next_due = max(next_due + self._frame_interval_s(next_exposure_us), time.monotonic())
            if self._acquisition_stop.wait(next_due - time.monotonic()):
                return

    # ------------------------------------------------------------------
    # Frame size
    # ------------------------------------------------------------------
    def _deliver(
        self,
        handler: _SimImageHandler,
        image: np.ndarray,
        bits: int,
        chunks: dict,
        exposure: SensorExposure,
    ) -> None:
        """Hand one made frame to the host, and keep what it was exposed under."""
        with self._lock:
            self._exposure_records[exposure.frame_id] = exposure
            while len(self._exposure_records) > self.EXPOSURE_RECORDS_KEPT:
                self._exposure_records.popitem(last=False)
        # The simulated link is USB3 carrying the format unpacked, as a
        # Basler USB3 camera carries Mono12: the generated array is the
        # acquired window in its wire container.
        handler._store_frame(
            image,
            datetime.datetime.now(),
            chunks,
            arrived_ns=time.monotonic_ns(),
            significant_bits=bits,
            wire_bytes=image.nbytes,
        )

    def _frame_grid(self) -> FrameGrid:
        """Windows of 4 x 4 steps, up to the sensor at the current binning.

        Sizes are post-binning (displayed) pixels, so the ceiling is the native
        sensor size divided by the binning factor -- the same constraint Pylon
        enforces via ``Width.Max`` at the active binning.
        """
        return FrameGrid(
            step=(4, 4),
            max_size=(self._native_width // self._binning, self._native_height // self._binning),
        )

    def _set_hardware_window(self, plan) -> bool:
        # A geometry change stops the stream, applies, and restarts it, as it
        # does on a real body (pixel format and binning below do the same), so
        # no frame made under the old setting is stored after the change.
        with self.update_camera_config(), self._lock:
            self._width = plan.acq_width
            self._height = plan.acq_height
            if _cam_log is not None:
                _cam_log.info(f'sim set_frame_size({self._width}x{self._height})')
        return True

    def get_min_frame_size(self) -> dict:
        """Return the simulator's minimum supported frame size.

        Returns:
            dict: ``{'width': 4, 'height': 4}``.
        """
        return {'width': 4, 'height': 4}

    def get_max_frame_size(self) -> dict:
        """Return the maximum frame size at the current binning.

        Post-binning ceiling = native sensor size / binning, matching Pylon's
        binning-dependent ``Width.Max`` / ``Height.Max``.

        Returns:
            dict: ``{'width': int, 'height': int}``.
        """
        return {
            'width': self._native_width // self._binning,
            'height': self._native_height // self._binning,
        }

    def _hardware_frame_size(self) -> dict:
        return {'width': self._width, 'height': self._height}

    # ------------------------------------------------------------------
    # Pixel format
    # ------------------------------------------------------------------
    def set_pixel_format(self, pixel_format: str) -> bool:
        """Set the simulated camera pixel format.

        Args:
            pixel_format: Format identifier (must be one its profile lists).

        Returns:
            bool: True on success, False when the format is not supported.
        """
        if pixel_format not in self.profile.pixel_formats:
            if _cam_log is not None:
                _cam_log.error(f'sim set_pixel_format({pixel_format}) UNSUPPORTED')
            logger.error(f'[CAM Sim   ] Unsupported pixel format: {pixel_format}')
            return False
        with self.update_camera_config(), self._lock:
            self._pixel_format = pixel_format
            if _cam_log is not None:
                _cam_log.info(f'sim set_pixel_format({pixel_format})')
        return True

    def get_pixel_format(self) -> str:
        """Return the simulated camera's current pixel format.

        Returns:
            str: One of the profile's ``pixel_formats``.
        """
        return self._pixel_format

    def get_supported_pixel_formats(self) -> tuple:
        """Return the formats its profile lists.

        Returns:
            tuple: The profile's ``pixel_formats``.
        """
        return tuple(self.profile.pixel_formats)

    # ------------------------------------------------------------------
    # Exposure
    # ------------------------------------------------------------------
    def exposure_t(self, exposure_ms: float) -> float | bool:
        """Set exposure time in milliseconds, returning the microseconds
        actually in effect.

        Refuses (``False``) rather than silently no-opping when the
        simulator is inactive or the request exceeds ``max_exposure``: the
        hardware value does not move on either path, so a caller that treated
        those as applied would record a chunk-match target the simulator never
        stamps.

        Args:
            exposure_ms: Exposure time in milliseconds.
        """
        if not self.active:
            return False
        if exposure_ms > self.max_exposure:
            if _cam_log is not None:
                _cam_log.warning(
                    f'sim ExposureTime.SetValue({exposure_ms}ms) CLAMPED max={self.max_exposure}ms'
                )
            logger.warning(
                f'[CAM Sim   ] Exposure {exposure_ms}ms exceeds max ({self.max_exposure}ms)'
            )
            return False
        with self._lock:
            self._exposure_us = float(exposure_ms) * 1000.0
            if _cam_log is not None:
                _cam_log.info(
                    f'sim ExposureTime.SetValue({float(exposure_ms) * 1000.0:.0f}us) (={exposure_ms}ms)'
                )
            logger.debug(f'[CAM Sim   ] Exposure set to {exposure_ms}ms')
            return self._exposure_us

    def get_exposure_t(self) -> float:
        """Return exposure time in milliseconds.

        Returns:
            float: Exposure in ms, or -1 when the simulator is not active.
        """
        if not self.active:
            return -1
        return self._exposure_us / 1000.0

    def auto_exposure_t(self, state: bool = True) -> bool:
        """Enable or disable simulated auto-exposure (state stored only).

        Args:
            state: True to enable, False to disable.

        Returns:
            bool: Always True.
        """
        self._auto_exposure_enabled = state
        return True

    # ------------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------------
    def get_all_temperatures(self) -> dict:
        """Return synthetic temperature telemetry.

        Returns:
            dict: ``{'sensor': 35.0, 'board': 40.0}``.

        Raises:
            HardwareError: No camera is active.
        """
        if not self.active:
            raise PartAbsentError('Camera temperature read: no camera is active')
        return {'sensor': 35.0, 'board': 40.0}

    def supports_temperature(self) -> bool:
        return bool(self.active)

    def supports_clock_latch(self) -> bool:
        return bool(self.active)

    def latch_clock(self) -> None:
        with self._lock:
            self._latched_ticks = self.camera_clock_ticks()

    def latched_clock_ticks(self) -> int:
        with self._lock:
            return self._latched_ticks

    # ------------------------------------------------------------------
    # Frame rate
    # ------------------------------------------------------------------
    def set_max_acquisition_frame_rate(self, enabled: bool, fps: float = 1.0) -> None:
        """Enable or disable the simulated frame-rate cap.

        Args:
            enabled: True to enforce ``fps`` as the upper bound.
            fps: Cap value in frames per second.
        """
        with self._lock:
            self._frame_rate_limit_enabled = enabled
            if enabled:
                self._frame_rate_target = fps
            if _cam_log is not None:
                _cam_log.info(f'sim set_max_acquisition_frame_rate(enabled={enabled}, fps={fps})')

    # ------------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------------
    def _set_hardware_binning(self, size: int) -> bool:
        """Set hardware binning factor for the simulator.

        Args:
            size: Binning factor (1-4 inclusive).

        Returns:
            bool: True on success, False when ``size`` is unsupported.
        """
        if size < 1 or size > 4:
            if _cam_log is not None:
                _cam_log.error(f'sim set_binning_size({size}) UNSUPPORTED')
            logger.error(f'[CAM Sim   ] Unsupported bin size: {size}')
            return False
        with self.update_camera_config(), self._lock:
            self._binning = size
            # Frame is in post-binning pixels, so a larger binning shrinks the
            # post-binning ceiling (native / binning); clamp the current frame
            # down to it, the same way Pylon clamps Width to the reduced
            # Width.Max. The UI re-pushes set_frame_size after a binning change,
            # but isolated binning changes (no frame re-push) still stay valid.
            self._width = min(self._width, self._native_width // size)
            self._height = min(self._height, self._native_height // size)
            if _cam_log is not None:
                _cam_log.info(f'sim set_binning_size({size})')
        return True

    def get_binning_size(self) -> int:
        """Return the simulator's current binning factor.

        Returns:
            int: Binning factor (1 = no binning).
        """
        return self._binning

    # ------------------------------------------------------------------
    # Z-dependent focus simulation
    # ------------------------------------------------------------------
    def set_z_position(self, z: float) -> None:
        """Set current Z position (um) for focus simulation.

        Args:
            z: Current Z stage position in micrometers.
        """
        self._z_position = float(z)

    def get_z_position(self) -> float:
        """Return the current Z position used for focus simulation.

        Returns:
            float: Z position in micrometers.
        """
        return self._z_position

    def set_focal_z(self, z: float) -> None:
        """Set the Z position (um) where the sample is in focus.

        Args:
            z: Focal Z position in micrometers.
        """
        self._focal_z = float(z)
        self._defocused = {}

    def get_focal_z(self) -> float:
        """Return the Z position (um) where the sample is in focus.

        Returns:
            float: Focal Z position in micrometers.
        """
        return self._focal_z

    def get_nominal_focal_z(self) -> float:
        """Return the Z position (um) the sample nominally sits at.

        Where bring-up places the stage, as an operator focusing by eye
        would: within the camera's focal spread of the true focus, which an
        autofocus then finds.

        Returns:
            float: Nominal sample Z position in micrometers.
        """
        return self._NOMINAL_FOCAL_Z_UM

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------
    def _make_focus_target(self, h: int, w: int) -> np.ndarray:
        """A sharp focus target with multi-scale features, in 8-bit grey levels.

        Edges at several spatial frequencies, so Vollath F4 (and other focus
        metrics) produce a smooth, peaked response as it is defocused.
        """
        key = (h, w)
        if self._focus_target_cache is not None and self._focus_target_cache[0] == key:
            return self._focus_target_cache[1]

        img = np.zeros((h, w), dtype=np.float32)

        # Grid of fine lines (high frequency -- most sensitive to defocus)
        grid_spacing = 8
        img[::grid_spacing, :] = 255 * 0.4
        img[:, ::grid_spacing] = 255 * 0.4

        # Scattered bright spots (simulates point-like features)
        rng = np.random.RandomState(42)  # deterministic
        n_spots = max(20, (h * w) // 5000)
        ys = rng.randint(0, h, n_spots)
        xs = rng.randint(0, w, n_spots)
        for y, x in zip(ys, xs, strict=False):
            y0 = max(0, y - 2)
            y1 = min(h, y + 3)
            x0 = max(0, x - 2)
            x1 = min(w, x + 3)
            img[y0:y1, x0:x1] = 255 * 0.8

        # Medium-frequency checkerboard (16px blocks)
        block = 16
        yy = np.arange(h) // block
        xx = np.arange(w) // block
        checker = (yy[:, None] + xx[None, :]) % 2
        img += checker * 255 * 0.2

        target = np.clip(img, 0, 255).astype(np.uint8)
        self._focus_target_cache = (key, target)
        return target

    def _specimen_frame(self, h: int, w: int) -> tuple[np.ndarray, str]:
        """The in-focus specimen as this frame sees it, and which drift step it is.

        The frame is a region of the sensor, centred, at the current binning:
        the specimen is reduced once when the binning changes and each drift
        step's frame cut from it, never re-scaled per frame, so it keeps its
        proportions.
        """
        key = (h, w, self._binning)
        if self._specimen_cut is None or self._specimen_cut[0] != key:
            if self._specimen_native is None:
                self._specimen_native = specimen_field(self._native_height, self._native_width)
            sensor = self._specimen_native
            if self._binning > 1:
                sensor = cv2.resize(
                    sensor,
                    (self._native_width // self._binning, self._native_height // self._binning),
                    interpolation=cv2.INTER_AREA,
                )
            self._specimen_cut = (key, drift_frames(sensor, h, w))
            self._defocused = {}
        step = self._drift_index % len(self._specimen_cut[1])
        self._drift_index += 1
        return self._specimen_cut[1][step], f'specimen {step}'

    def camera_clock_ticks(self) -> int:
        """The camera's clock now, in ``TIMESTAMP_TICK_HZ`` ticks: what a latch of it answers."""
        return time.monotonic_ns() + self._clock_offset_ns

    def _start_exposure(self) -> SensorExposure:
        """Start a frame's exposure under the settings in force now. Call with ``_lock`` held.

        Z and the objective are asked here, as the frame starts; a Z the
        scope cannot read (no focus axis) and an unknown objective are
        None, which render in focus. Unwired, the camera stands at its own
        Z with a nominal depth of field, as a camera built with no scope
        around it must.
        """
        if self._z_position_func is not None:
            z = self._z_position_func()
            if z is not None:
                self._z_position = z
        else:
            z = self._z_position
        if self._depth_of_field_func is None:
            dof = self._UNWIRED_DOF_UM
        else:
            dof = self._depth_of_field_func()
        if self._illumination_func is None:
            lit_ma = self.REFERENCE_ILLUMINATION_MA
        else:
            lit_ma = float(self._illumination_func())
        x, y = (None, None) if self._xy_position_func is None else self._xy_position_func()
        self._frames_started += 1
        return SensorExposure(
            frame_id=self._frames_started,
            started_ticks=self.camera_clock_ticks(),
            exposure_us=self._exposure_us,
            gain_db=self._gain,
            black_level=self._black_level,
            illumination_ma=lit_ma,
            x_um=x,
            y_um=y,
            z_um=z,
            z_end_um=None,
            depth_of_field_um=dof,
        )

    def _blur_sigma_px(self, exposure: SensorExposure) -> float:
        """The defocus blur ``exposure`` was made with, as a sigma in pixels."""
        if exposure.z_um is None or exposure.depth_of_field_um is None:
            # No focus axis to read (a manual scope is focused by hand), or
            # no objective known: there is no focus to model.
            return 0.0
        dof = exposure.depth_of_field_um
        defocus = abs(exposure.z_um - self._focal_z)
        # A binned pixel is larger, so the same blur spans fewer of them.
        return self._BLUR_PX_PER_DOF * defocus / dof / self._binning

    def _defocus(self, sharp: np.ndarray, source: str, exposure: SensorExposure) -> np.ndarray:
        """``sharp`` as the objective imaged it for ``exposure``."""
        sigma = self._blur_sigma_px(exposure)
        if sigma < 0.3:
            return sharp
        key = (sharp.shape, self._binning, round(sigma, 2))
        cached = self._defocused.get(source)
        if cached is not None and cached[0] == key:
            return cached[1]
        blurred = _gaussian_blur(sharp, sigma)
        self._defocused[source] = (key, blurred)
        return blurred

    def _render_sample(
        self,
        sharp: np.ndarray,
        source: str,
        dtype,
        max_val: int,
        brightness: float,
        exposure: SensorExposure,
        *,
        specimen: bool,
    ) -> np.ndarray:
        """A sample frame: defocused, scaled to ``dtype`` and lit.

        The ``specimen`` is what a live view shows: floored so it stays
        visible at the short default exposures. A test pattern is not, so
        its signal follows exposure, gain and focus exactly. Zero is exempt
        and stays zero: it means no light reached the sample, and a floor
        that lifts darkness back to a visible field is what made an unlit
        capture indistinguishable from a lit one.
        """
        if brightness <= 0.0:
            return np.zeros(sharp.shape, dtype=dtype)
        if specimen:
            brightness = max(0.5, brightness)
        img = self._defocus(sharp, source, exposure)
        if dtype == np.uint16:
            return (img.astype(np.float32) * (max_val / 255.0 * brightness)).astype(dtype)
        if brightness != 1.0:
            return cv2.convertScaleAbs(img, alpha=brightness)
        # A frame of its own: the cached one would be shared with every
        # frame after it, and a reader may write into what it was handed.
        return np.array(img)

    @staticmethod
    def _illumination_scale(exposure: SensorExposure) -> float:
        """How much light reached the sample for ``exposure``: the LED current
        against ``REFERENCE_ILLUMINATION_MA``, and 0.0 when nothing was lit.

        A camera sees what the illumination gives it, so an unlit field is
        black however long the exposure or however high the gain. Without
        this the simulator could not produce a dark frame at all, and the
        whole class of illumination failures -- an LED that did not come
        on, a channel left dark through a run -- was reproducible only on
        a bench.

        Unwired, the sample is lit (``_start_exposure``): a camera built
        with no scope around it has no illumination to ask about, and
        rendering it black would make every standalone camera test describe
        a fault.

        The light scales with the current, so a change of current reaches
        the image as a change of exposure or gain does, and a long exposure
        can be kept off full scale by lighting it less.
        """
        return exposure.illumination_ma / SimulatedCamera.REFERENCE_ILLUMINATION_MA

    def _at_stage_xy(self, img: np.ndarray, exposure: SensorExposure) -> np.ndarray:
        """The specimen field ``img`` as seen with the stage where ``exposure`` started.

        The field moves across the sensor against the stage's travel, and
        what leaves one edge comes back at the other: the specimen is a
        repeating pattern. A scope without X or Y holds that axis still.
        """
        px_per_um = self._SPECIMEN_PX_PER_UM / self._binning
        dx = round((exposure.x_um or 0.0) * px_per_um)
        dy = round((exposure.y_um or 0.0) * px_per_um)
        if not dx and not dy:
            return img
        return np.roll(img, (-dy, -dx), axis=(0, 1))

    def _generate_image(self, exposure: SensorExposure) -> np.ndarray:
        """Generate a synthetic image as ``exposure`` made it, at the current geometry and pattern."""
        h = self._height
        w = self._width

        if self._pixel_format in ('Mono10', 'Mono12'):
            dtype = np.uint16
            max_val = 4095 if self._pixel_format == 'Mono12' else 1023
        else:
            dtype = np.uint8
            max_val = 255

        # Scale brightness by exposure and gain; the gain is held in dB, so it
        # multiplies the signal by 10^(dB/20)
        raw = (exposure.exposure_us / 1_000_000.0) * 10.0 ** (exposure.gain_db / 20.0) * 10.0
        brightness = min(1.0, raw * self._illumination_scale(exposure))
        if self._test_pattern == 'black':
            img = np.zeros((h, w), dtype=dtype)
        elif self._test_pattern == 'white':
            img = np.full((h, w), max_val, dtype=dtype)
        elif self._test_pattern == 'noise':
            img = np.random.randint(0, int(max_val * brightness) + 1, (h, w), dtype=dtype)
        elif self._test_pattern == 'focus_target':
            img = self._render_sample(
                self._make_focus_target(h, w),
                'focus_target',
                dtype,
                max_val,
                brightness,
                exposure,
                specimen=False,
            )
        else:
            # No specific pattern was asked for: the specimen. Built lazily,
            # because this is reachable before load_specimen() has run -- a
            # camera-init failure skips that call entirely.
            if self._test_pattern != 'specimen':
                logger.warning(
                    f'[SimCamera ] Unknown test pattern '
                    f'{self._test_pattern!r} -- rendering the specimen field'
                )
            frame, source = self._specimen_frame(h, w)
            img = self._at_stage_xy(
                self._render_sample(
                    frame, source, dtype, max_val, brightness, exposure, specimen=True
                ),
                exposure,
            )

        if exposure.black_level:
            # The offset lifts every pixel, and a pixel lifted past full scale
            # saturates there, as a sensor's output does; the integer cast
            # would otherwise wrap a bright pixel to near zero.
            img = np.minimum(np.rint(img.astype(np.float32) + exposure.black_level), max_val)
            img = img.astype(dtype)
        return img

    def grab_new_capture(self, timeout_s: float) -> Frame | None:
        """Wait for a frame stored after this call, then return it.

        The acquisition thread is the only producer, as a real camera's SDK
        thread is, so a still capture waits for the stream rather than making
        a frame of its own: the frame it gets is one the stream delivered,
        counted once, under the settings in effect when it was made.

        Args:
            timeout_s: Wall-clock seconds to wait for the next frame.

        Returns:
            The stored ``Frame``, or None when the camera is not grabbing or
            no frame arrived within ``timeout_s``.
        """
        handler = self.cam_image_handler
        if not self._grabbing or handler is None:
            return None
        if not handler.wait_for_frame_after(handler.frames_delivered, timeout_s):
            return None
        return self.grab()

    # ------------------------------------------------------------------
    # Gain
    # ------------------------------------------------------------------
    def get_gain(self) -> float:
        """Return the simulated camera gain.

        Returns:
            float: Gain in dB, or -1 when the camera is not active.
        """
        if not self.active:
            return -1
        return self._gain

    # ------------------------------------------------------------------
    # Black level: an offset in DN of the delivered pixel format, added to
    # every rendered frame, so the value reported is what the image shows.
    # ------------------------------------------------------------------
    _BLACK_LEVEL_RANGE = (0.0, 64.0)

    def supports_black_level(self) -> bool:
        return bool(self.active)

    def get_black_level(self) -> float | None:
        if not self.active:
            return None
        return self._black_level

    def get_black_level_range(self) -> tuple[float, float] | None:
        if not self.active:
            return None
        return self._BLACK_LEVEL_RANGE

    def set_black_level(self, value: float) -> float | bool | None:
        """Set the simulated black level; refused outside its range, as a
        real node refuses. Returns as ``gain`` does."""
        if not self.active:
            return None
        low, high = self._BLACK_LEVEL_RANGE
        if not low <= float(value) <= high:
            logger.warning(f'[CAM Sim   ] Black level {value} outside [{low}, {high}]')
            return False
        with self._lock:
            self._black_level = float(value)
        return float(value)

    def gain(self, value: float) -> float | bool | None:
        """Set the simulated camera gain.

        Refuses (``False``) a value outside the range the profile declares,
        as a real body's gain node does, so a refused gain is reachable in
        the simulator and not only on a bench. The simulated gain does not
        move on a refusal.

        While continuous auto-gain runs, the write is accepted and answered
        but the gain stays where the loop holds it: a Basler body takes a
        Gain write under GainAuto=Continuous, reads it back, and its loop
        takes the node back at its next update, so a value meant for the
        sensor must be written once the loop is off.

        Args:
            value: Gain in dB.

        Returns:
            float | bool | None: See ``Camera.gain``.
        """
        if not self.active:
            return None
        low, high = self.min_gain, self.max_gain
        if (low is not None and float(value) < low) or float(value) > high:
            logger.warning(f'[CAM Sim   ] Gain {value} dB outside [{low}, {high}] dB')
            return False
        with self._lock:
            if not self._auto_gain_enabled:
                self._gain = float(value)
            if _cam_log is not None:
                _cam_log.info(f'sim Gain.SetValue({float(value):.3f})')
            logger.debug(f'[CAM Sim   ] Gain set to {value}')
        return float(value)

    def init_auto_gain_focus(
        self,
        auto_target_brightness: float = 0.5,
        min_gain: float | None = None,
        max_gain: float | None = None,
    ) -> bool:
        """Initialize auto-gain ROI and parameters (no-op in simulation).

        Args:
            auto_target_brightness: Normalized brightness target (0.0-1.0).
            min_gain: Optional lower bound in dB.
            max_gain: Optional upper bound in dB.

        Returns:
            bool: Always True.
        """
        with self._lock:
            self._auto_gain_target_brightness = auto_target_brightness
            if min_gain is not None:
                self._auto_gain_min = min_gain
            if max_gain is not None:
                self._auto_gain_max = max_gain
        return True

    def _converged_gain(self) -> float:
        """Return the gain the simulated auto-gain loop settles on.

        The midpoint of the auto-gain bounds, held inside the range the
        gain node declares: a real body's auto loop drives the same node a
        manual write does and cannot leave its range, so a caller whose
        bounds reach past the profile (a ``current.json`` carried over from
        a body with a higher ceiling) still gets a gain this camera accepts
        when it writes it back.
        """
        midpoint = (self._auto_gain_min + self._auto_gain_max) / 2.0
        low = self.min_gain if self.min_gain is not None else midpoint
        return min(max(midpoint, low), self.max_gain)

    def auto_gain(
        self,
        state: bool = True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
        ae_max_exposure_ms: float | None = None,
    ) -> bool:
        """Enable or disable simulated continuous auto-gain.

        On enable, the simulator converges immediately by setting gain
        to the midpoint of [min_gain_db, max_gain_db].

        Args:
            state: True to enable, False to disable.
            target_brightness: Normalized brightness target (0.0-1.0).
            min_gain_db: Optional lower bound in dB.
            max_gain_db: Optional upper bound in dB.
            ae_max_exposure_ms: Accepted for interface parity; the
                simulator has no auto-exposure bound to apply.

        Returns:
            bool: Always True.
        """
        with self._lock:
            self._auto_gain_enabled = state
            if state:
                self._auto_gain_target_brightness = target_brightness
                if min_gain_db is not None:
                    self._auto_gain_min = min_gain_db
                if max_gain_db is not None:
                    self._auto_gain_max = max_gain_db
                self._gain = self._converged_gain()
            if _cam_log is not None:
                _cam_log.info(
                    f'sim auto_gain(state={state}, target={target_brightness}, min_db={min_gain_db}, max_db={max_gain_db})'
                )
        return True

    def auto_gain_once(
        self,
        state: bool = True,
        target_brightness: float = 0.5,
        min_gain_db: float | None = None,
        max_gain_db: float | None = None,
        ae_max_exposure_ms: float | None = None,
    ) -> bool:
        """Run a single simulated auto-gain iteration.

        Converges on the midpoint of [min_gain_db, max_gain_db], held inside
        the gain node's range (see ``_converged_gain``).

        Args:
            state: True to run, False to no-op.
            target_brightness: Normalized brightness target (0.0-1.0).
            min_gain_db: Optional lower bound in dB.
            max_gain_db: Optional upper bound in dB.

        Returns:
            bool: Always True.
        """
        if state:
            with self._lock:
                self._auto_gain_target_brightness = target_brightness
                if min_gain_db is not None:
                    self._auto_gain_min = min_gain_db
                if max_gain_db is not None:
                    self._auto_gain_max = max_gain_db
                self._gain = self._converged_gain()
        return True

    def update_auto_gain_target_brightness(self, auto_target_brightness: float) -> bool:
        """Update the auto-gain target brightness.

        Args:
            auto_target_brightness: Normalized brightness target (0.0-1.0).

        Returns:
            bool: Always True.
        """
        with self._lock:
            self._auto_gain_target_brightness = auto_target_brightness
        return True

    def update_auto_gain_min_max(
        self, min_gain_db: float | None, max_gain_db: float | None
    ) -> bool:
        """Update auto-gain bounds.

        Args:
            min_gain_db: Minimum gain in dB, or None to leave unchanged.
            max_gain_db: Maximum gain in dB, or None to leave unchanged.

        Returns:
            bool: Always True.
        """
        with self._lock:
            if min_gain_db is not None:
                self._auto_gain_min = min_gain_db
            if max_gain_db is not None:
                self._auto_gain_max = max_gain_db
        return True

    # ------------------------------------------------------------------
    # Test pattern
    # ------------------------------------------------------------------
    def set_test_pattern(self, enabled: bool = False, pattern: str = 'Black') -> None:
        """Enable or disable the simulator's test pattern generator.

        Args:
            enabled: True to enable the pattern, False to revert to the
                specimen field the live view normally shows.
            pattern: Pattern name (case-insensitive). Common values:
                ``'black'``, ``'white'``, ``'noise'``, ``'focus_target'``.
        """
        if enabled:
            self._test_pattern = pattern.lower()
        else:
            # Back to the specimen field, not a static ramp: turning a test
            # pattern OFF must restore what the view showed before it went on.
            self._test_pattern = 'specimen'
