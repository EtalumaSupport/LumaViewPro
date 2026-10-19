# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""CameraAPI -- sub-API for the camera's settings and their stores.

CameraAPI owns the camera's state: the camera cache, the held-layer record
and the auto-gain arm (under _state_lock), the camera lock, the camera
change listeners and the dispatch of camera commands onto the camera lane.
It writes the camera and saves the stored camera settings through the
writer a session binds (``Lumascope.bind_settings``). The capture
(``ImagingAPI``) holds a reference to it and calls its private hold, lock,
resume and dispatch members.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import functools
import logging as _logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import modules.binning as binning
import modules.common_utils as common_utils
import modules.image_mode as image_mode
import modules.settings_paths as settings_paths
from modules.finite_number import refuse_unless_finite_number
from modules.request_fields import key_path, refuse_unless_kind, required
from drivers.camera import Frame
from drivers.exceptions import CameraNoAnswerError, CameraReadFailedError
from lib import profile_trace
from lib.lumascope_error import Fault
from lvp_logger import logger
from modules.exceptions import (
    ArgumentRefusedError,
    AutoGainNotSettledError,
    CameraSettingOutOfRangeError,
    CameraSettingRejected,
    CameraSettingUnsupportedError,
    ExposureAtMaximumNotice,
    ExposureAtMinimumNotice,
    HardwareCommandRefusedError,
    MissingPart,
)
from modules.activity_claim import current_taking
from modules.frame_validity import FrameValidity
from modules.notification_center import notifications
from modules.sequential_io_executor import IOTask
from modules.api_surface import api, api_fields

if TYPE_CHECKING:
    from drivers.camera import Camera
    from modules.lumascope_api._lumascope import Lumascope

_api_log = _logging.getLogger('LVP.api')


# What a dispatched camera command does when no camera is connected: it is
# refused, naming the camera, unless its dispatch names the answer it is
# satisfied with -- an off whose end state already holds -- or leaves the
# question to its body, which alone can read what it was asked to do.
_REFUSED_WHEN_ABSENT = object()
_DECIDED_IN_BODY = object()

# The camera cache with no camera behind it: at construction, and again when
# the scope disconnects, so no read describes a camera that has left. Every
# value is one the matching is_valid_* predicate rejects, except 'binning',
# whose absent default is also a legal factor (see ImagingAPI.__init__).
_NO_CAMERA_CACHE = {
    'active': False,
    'gain_db': -1.0,
    'exposure_ms': 0.0,
    'frame_size': {'width': 0, 'height': 0},
    'pixel_format': None,
    'binning': 1,
}


def _camera_not_connected(member: str) -> HardwareCommandRefusedError:
    """The refusal of ``member`` for want of a camera, raised by whoever asked.

    Built from the asker's own answer, never a second poll: a camera that
    answered connected on a re-ask would leave the refusal unraised.
    """
    part = MissingPart.CAMERA
    return HardwareCommandRefusedError(part.reason, member, missing=part)


def _no_camera_cache() -> dict:
    """A fresh copy of ``_NO_CAMERA_CACHE``: its frame dicts are not shared."""
    return {k: dict(v) if isinstance(v, dict) else v for k, v in _NO_CAMERA_CACHE.items()}


class AutoGainConvergence(enum.Enum):
    """Where a continuous auto-gain arm landed when the capture locked it.

    CONVERGED: exposure inside the layer class's usable range.
    MAXED: exposure pinned at the class ceiling -- the scene is too dark
        for the range, so target brightness was not reached.
    AT_MINIMUM: exposure at or below the class usable floor -- the scene
        is too bright for the range.
    FAILED: the camera reported no usable achieved value, so the capture
        ran without an exposure/gain evidence gate.
    The limit states are outcomes, not failures: the frame is saved with
    the achieved values and a run continues.
    """

    CONVERGED = 'CONVERGED'
    MAXED = 'MAXED'
    AT_MINIMUM = 'AT_MINIMUM'
    FAILED = 'FAILED'


@dataclasses.dataclass(frozen=True)
class _AutoGainArm:
    """A commanded continuous auto-gain arm, held until a capture locks it.

    resume_after_capture: a live-view arm is re-armed after the capture
    so the view keeps adjusting; a protocol step's arm stays locked Off
    until the next step arms again.
    """

    settings: dict
    resume_after_capture: bool


@dataclasses.dataclass(frozen=True)
class LayerCameraRequest:
    """A layer's stored camera settings, read from its store, for the camera to take."""

    gain_db: float
    exposure_ms: float
    auto_gain: bool
    auto_gain_settings: dict


@dataclasses.dataclass(frozen=True)
class _HeldLayer:
    """The layer whose stored settings the camera holds.

    ``request`` is what the layer's store asked for: the comparison that
    decides whether a layer must be put on the camera again. It is None
    when the camera refused part of it, so the next ask writes again.
    ``taken`` is what the camera took, which differs from the request when
    a value above the camera's range was capped.
    """

    layer: str
    request: LayerCameraRequest | None
    taken: dict | None


@api_fields(
    'ceiling_ms',
    'exposure_ms',
    'floor_ms',
    'gain_db',
    'resume_after_capture',
    'settings',
    'state',
    'stored_exposure_ms',
)
@dataclasses.dataclass(frozen=True)
class AutoGainLock:
    """The result of locking an auto-gain arm; ``state`` is None when no
    arm was recorded (nothing was locked, no driver traffic happened)."""

    state: AutoGainConvergence | None
    exposure_ms: float | None = None
    gain_db: float | None = None
    floor_ms: float | None = None
    ceiling_ms: float | None = None
    resume_after_capture: bool = False
    settings: dict | None = None
    # The exposure a caller STORES as the layer's manual setting after the
    # lock: the achieved value floored to the class's usable floor. The
    # auto loop can drive the camera below that floor on a bright scene;
    # a setting stored that low makes near-black captures on ordinary
    # scenes, so the store keeps the floor while exposure_ms keeps the
    # truth. Decided here so a GUI and a REST caller store the same value.
    stored_exposure_ms: float | None = None

    def __post_init__(self) -> None:
        # Both achieved values or neither: a consumer that keeps the lock's
        # gain and a snapshot's exposure would leave a mixed camera state.
        if (self.exposure_ms is None) != (self.gain_db is None):
            raise ValueError(
                'AutoGainLock carries exposure_ms and gain_db together or not at all; '
                f'got exposure_ms={self.exposure_ms!r} gain_db={self.gain_db!r}'
            )


def stored_exposure_after_lock(exposure_ms: float, floor_ms: float | None) -> float:
    """The exposure to store as a manual setting after an auto-gain lock."""
    return max(exposure_ms, floor_ms) if floor_ms is not None else exposure_ms


@api_fields('applied', 'capped', 'stored')
@dataclasses.dataclass(frozen=True)
class AppliedCameraSetting:
    """What a stored camera setting actually becomes on the attached body.

    A stored gain or exposure is the user's committed intent and outlives
    whichever camera happens to be attached. A smaller body cannot reach it,
    so the value written to hardware is the cap while the stored value is
    left alone -- put a capable camera back and the intent applies again.

    The three facts travel together because both consumers need all three:
    the apply path writes ``applied``, and a display has to show ``stored``
    while saying the camera is holding it down. Handing out a bare float
    would make each consumer recompute ``capped`` for itself, which is the
    second answerer this type exists to prevent.
    """

    stored: float | bool
    applied: float | bool
    capped: bool

    def __post_init__(self) -> None:
        # capped is not a caller's opinion: it is whether the write differs
        # from the intent. A consumer that renders one while testing the
        # other would report a limit that is not being applied.
        if self.capped != (self.applied != self.stored):
            raise ValueError(
                f'AppliedCameraSetting: capped={self.capped!r} contradicts '
                f'stored={self.stored!r} applied={self.applied!r}'
            )


def cap_stored_value(stored: float, cap: float | None) -> AppliedCameraSetting:
    """Resolve a stored setting against a published maximum.

    An unknown cap (no camera, or a driver that publishes none) narrows
    nothing: a missing bound is not a bound of zero, and inventing one here
    would apply a limit no hardware asked for. ``stored`` is a finite
    number: its callers refuse any other, since ``nan <= cap`` is False and
    NaN would be applied as the camera's maximum.
    """
    value = float(stored)
    if cap is None or value <= cap:
        return AppliedCameraSetting(stored=value, applied=value, capped=False)
    return AppliedCameraSetting(stored=value, applied=float(cap), capped=True)


def _rejected_gain_words(gain_db: float) -> str:
    """The sentence for a gain the camera refused.

    One home for every raise of a gain refusal: the public setter, the
    layer apply and the reports of refusals whose flight ends in the API.
    It says what the camera holds, not what happens next: a live view goes
    on at that gain, while a protocol step does not capture at all.
    """
    return (
        f'The camera rejected the gain change to {float(gain_db):.1f} dB. '
        'It is still at its previous gain. Check that the value is '
        'within the camera limits.'
    )


def _rejected_mode_words(what: str) -> str:
    """The sentence for a mode change the camera refused."""
    return f'The camera did not take the {what} change. It is still as it was.'


def _rejected_black_level_words(black_level: float) -> str:
    """The sentence for a black level the camera refused; see
    ``_rejected_gain_words``. A camera holding its black level automatically
    refuses a manual one, which the sentence names."""
    return (
        f'The camera did not accept the black level {float(black_level):g}. '
        'It is still at its previous black level. A camera that sets its '
        'black level automatically takes no manual value.'
    )


def _value_rejection(setting: str, requested: float) -> CameraSettingRejected:
    """The typed refusal of a camera value setting, in its one set of words."""
    return CameraSettingRejected(
        setting, requested, message=_REJECTED_VALUE_WORDS[setting](requested)
    )


def _refuse_unless_auto_gain_settings(settings: object, argument: str) -> None:
    """Refuse an auto-gain settings dictionary the write cannot read.

    Read on the caller's thread, before the dispatch: the write reads the
    keys on the camera lane, where a missing one failed as the camera's
    rejection of a change it was never sent.

    Raises:
        ArgumentRefusedError: ``'wrong_kind'``, ``'missing_key'`` or
            ``'not_a_number'``, naming the key's path under ``argument``;
            ``'gain_range_inverted'``, a minimum gain above the maximum.
    """
    refuse_unless_kind(settings, 'dict', argument)
    for key in ('target_brightness', 'min_gain_db', 'max_gain_db'):
        refuse_unless_kind(required(settings, key, argument), 'number', key_path(argument, key))
    for key in ('max_exposure_ms', 'min_exposure_ms'):
        refuse_unless_kind(settings.get(key), 'number', key_path(argument, key), nullable=True)
    # The pair is judged here, where every arm passes, and not at the
    # settings write: each bound is written alone, so a write judging the
    # pair would make a client order its two writes.
    if settings['min_gain_db'] > settings['max_gain_db']:
        raise ArgumentRefusedError(
            'gain_range_inverted',
            argument=argument,
            value={key: settings[key] for key in ('min_gain_db', 'max_gain_db')},
        )


def _no_answer_rejection(rejected: CameraSettingRejected) -> CameraSettingRejected:
    """The same setting's rejection when the camera answered neither try.

    Its own words, not the refusal's: a refusal leaves the camera at its
    previous value, but a write nobody answered may have been taken.
    """
    noun = rejected.setting.removesuffix('_db').removesuffix('_ms').replace('_', ' ')
    return CameraSettingRejected(
        rejected.setting,
        rejected.requested,
        message=(
            f'The camera did not answer a {noun} change, sent twice, so whether '
            'it took it is not known. Check the USB connection if this repeats.'
        ),
        reason='camera_no_answer',
    )


def _mode_rejection(setting: str, requested: object, what: str) -> CameraSettingRejected:
    """The typed refusal of a mode change, in its one set of words."""
    return CameraSettingRejected(setting, requested, message=_rejected_mode_words(what))


def _absent_mode_refusal(
    setting: str, requested: object, what: str, *, offered: tuple, consequence: str
) -> CameraSettingUnsupportedError:
    """The refusal of a camera setting the attached camera does not have.

    Nothing reached the camera. ``offered`` is what the camera can still be
    asked for (an auto mode's off, a toggle's standard setting, or nothing),
    and ``consequence`` the sentence saying what stays as it is.
    """
    return CameraSettingUnsupportedError(
        setting,
        requested,
        offered=offered,
        message=f'This camera has no {what}. {consequence}',
    )


# What stays as it is when an auto mode the camera lacks is refused.
_AUTO_MODE_STAYS = 'The gain and exposure stay as they are set.'


def camera_range_words(low: float | None, high: float | None, unit: str) -> str:
    """A camera's range for a person, naming only the ends it declares.

    The one wording of a range, shared by the setters' refusal and the run's.
    """
    if low is None:
        return f'at most {high:g} {unit}'
    if high is None:
        return f'at least {low:g} {unit}'
    return f'{low:g} to {high:g} {unit}'


def _value_in_effect(result: object, requested: float, scale: float = 1.0) -> float:
    """The value a camera write left in effect, in the API's unit.

    A driver that can say what it applied answers with a number, and that
    number is the truth: a body that clamps, snaps or quantizes applies
    something other than the request. Any other applied answer -- a driver
    that cannot report a value -- leaves the request as the best knowledge.
    ``bool`` is an ``int`` subclass, so a bare ``True`` is excluded
    explicitly rather than read as 1.0.

    Args:
        result: The driver's answer to the write (not ``False``).
        requested: The value asked for, in the API's unit.
        scale: Multiplier from the driver's unit to the API's.
    """
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        return float(result) * scale
    return requested


def _rejected_exposure_words(exposure_ms: float) -> str:
    """The sentence for an exposure the camera refused; see
    ``_rejected_gain_words``."""
    return (
        f'The camera rejected the exposure change to {float(exposure_ms):g} ms. '
        'It is still at its previous exposure. Check that the value is '
        'within the camera limits.'
    )


_REJECTED_VALUE_WORDS = {
    'gain_db': _rejected_gain_words,
    'exposure_ms': _rejected_exposure_words,
    'black_level': _rejected_black_level_words,
}

# The camera settings a restore puts back beyond gain and exposure, in the
# order it writes them: a binning write resets the frame, and the black
# level's range is the pixel format's.
_GEOMETRY_ORDER = ('binning', 'frame_size', 'pixel_format', 'black_level')

# The saved geometry settings: the binning, the frame and the unbinned region
# it was framed from, saved together whenever the camera's geometry changes.
_STORED_GEOMETRY = (
    'binning.size',
    'frame.width',
    'frame.height',
    'frame.native_width',
    'frame.native_height',
)

# Each live-read setting's noun, driver read, validity check and coercion.
_LIVE_SETTING_READS = {
    'binning': (
        'binning',
        lambda driver: driver.get_binning_size(),
        common_utils.is_valid_binning_size,
        int,
    ),
    'frame_size': (
        'frame size',
        lambda driver: driver.get_frame_size(),
        common_utils.is_valid_frame_size,
        lambda v: {'width': int(v['width']), 'height': int(v['height'])},
    ),
    'pixel_format': (
        'pixel format',
        lambda driver: driver.get_pixel_format(),
        common_utils.is_valid_pixel_format,
        str,
    ),
}


def _describe_geometry(snapshot: dict) -> str:
    """The snapshot's settings beyond gain and exposure, for the log."""
    return ' '.join(f'{key}={snapshot[key]}' for key in _GEOMETRY_ORDER if key in snapshot)


@api_fields('setting', 'value')
@dataclasses.dataclass(frozen=True)
class CameraChanged:
    """A camera event: the setting changed, ``'gain'`` or ``'exposure'``, and its new value."""

    setting: str
    value: float


class CameraAPI:
    """Camera sub-API. Owns the camera's setters, reads, holds and applies,
    the camera's state (cache, held layer, auto-gain arm, listeners) and the
    saving of the stored camera settings.
    """

    def __init__(self, scope: Lumascope, frame_validity: FrameValidity) -> None:
        self._scope = scope
        # The one frame validity, built by the scope and shared with the
        # capture: every camera write invalidates it, the capture drains it.
        self.frame_validity = frame_validity

        # _state_lock guards _auto_gain_arm and _held_layer; _cam_lock
        # serializes access to the camera driver itself (any path that
        # touches the SDK reads/writes goes through this lock).
        self._state_lock = threading.Lock()
        self._cam_lock = profile_trace.TimedLock(threading.RLock(), name='camera._cam_lock')

        # Camera change listeners -- push-based UI update mechanism.
        # Each listener is called with (param: str, value: float) whenever
        # camera gain or exposure changes. param is 'gain' or 'exposure'.
        # Fires from the thread that caused the change, so listeners MUST
        # schedule UI work via Clock.schedule_once.
        self._camera_listeners_lock = threading.Lock()
        self._camera_listeners: list = []

        # The commanded continuous auto-gain arm, or None. Only the API
        # commands the auto mode and no driver reads it back, so this is
        # the single record that an arm is standing. Consumed atomically
        # by the capture that locks it (under _state_lock).
        self._auto_gain_arm: _AutoGainArm | None = None

        # The layer whose stored settings the camera holds, or None when it
        # holds no layer's: only ``_hold_layer_impl`` sets a layer, and every
        # raw camera door clears it (``_no_layers_write``). Under _state_lock.
        self._held_layer: _HeldLayer | None = None

        # When True, programmatic value-range warnings (sub-0.1ms exposure,
        # future similar setters) are silenced. Internal callers that sweep
        # full ranges (camera characterization, dynamic-range tests) enter
        # this via suppress_value_warnings(); the warnings exist for L1
        # researchers who type microsecond values thinking ms.
        self._suppress_value_warnings = False

        # Camera state cache -- the single store every public camera getter
        # answers from. Updated when the camera connects, after every
        # set_gain_db/set_exposure/etc. write-through, and by each getter's
        # validated live read. Every entry is either a validated hardware
        # reading or its seed below, and the seeds are the documented
        # camera-absent defaults -- so a driver failure sentinel can never
        # be cached or returned, and for every key except 'binning' the
        # matching is_valid_* predicate rejects the seed, making "never
        # successfully read" detectable from the value itself. gain_db
        # seeds to -1.0 (not 0.0) because 0 dB is a legal gain and would
        # read as hardware truth. 'binning' seeds to 1, which is also its
        # absent default AND a legal factor: never-read and genuinely-1x1
        # are indistinguishable by design -- both honestly answer 1.
        self._camera_cache_lock = threading.Lock()
        self._camera_cache = _no_camera_cache()
        # Per-key write generation, bumped by every authoritative cache
        # write (_commit_camera_writes). A validated live READ snapshots
        # the generation before touching the driver and commits only if
        # it is unchanged, so a read that was in flight while a setter
        # landed can never overwrite the setter's newer truth with the
        # pre-write hardware value.
        self._camera_cache_write_gen: dict[str, int] = {}
        # Per-key monotonic timestamp of the last WARNING about a failed
        # camera read; failures inside the window log at debug instead,
        # so a dead camera polled at frame rate warns once per window,
        # not per frame.
        self._camera_read_warn_ts: dict[str, float] = {}
        self._populate_camera_cache()

    @property
    def _driver(self) -> Camera | None:
        """Resolve the camera driver via the composition root each access.

        Lumascope's `_camera_driver` slot is reassigned on disconnect /
        reconnect and during tests that hot-swap drivers. Re-resolving
        here keeps CameraAPI in sync without rebinding.
        """
        return self._scope._camera_driver

    @property
    def _binning_size(self) -> int:
        """Last-known binning factor (cache-backed; no SDK read).

        Per-frame consumers (scale-bar sizing, FOV math) read this instead
        of get_binning_size() so the hot path never touches the SDK.
        """
        with self._camera_cache_lock:
            return self._camera_cache['binning']

    def _populate_camera_cache(self) -> None:
        """Populate camera cache from hardware. Called at init and on reconnect.

        One canonical read path: every key refreshes through the same
        validated-read commit the public getters answer by, so a failed or
        sentinel read never overwrites a cached last-known-good value,
        one raising key cannot abort the rest of the round, and populate
        cannot drift from the getters' validation rules. The camera's
        limits are not here: they are ``capabilities.camera_limits``, read
        once at connect.
        """
        if not self._driver or not self._driver.active:
            with self._camera_cache_lock:
                self._camera_cache['active'] = False
            return

        try:
            self.get_binning_size()
            self.get_gain_db()
            self.get_exposure_ms()
            self._get_frame_size()
            self._get_pixel_format()
            with self._camera_cache_lock:
                self._camera_cache['active'] = True
            logger.info('[SCOPE API ] Camera cache populated')
        except Exception as e:
            logger.warning(f'[SCOPE API ] Failed to populate camera cache: {e}')
            with self._camera_cache_lock:
                self._camera_cache['active'] = bool(self._driver and self._driver.active)

    def _refresh_cache_from_hardware_after_auto(self) -> None:
        """Resync gain + exposure cache to hardware after an auto cycle.

        The auto-gain / auto-exposure SDK paths drive hardware values
        directly without going through this layer's cache, so when the
        auto cycle toggles off the cache may hold a stale pre-auto
        value. Without this refresh, the cache-equality skip at
        ``set_gain_db`` / ``set_exposure_ms`` short-circuits subsequent
        setter calls and hardware silently stays at the converged auto
        value -- the user-visible failure shape was a protocol's first
        run capturing at an unintended exposure inherited from a
        pre-scan live-mode AG cycle.

        Soft-fails on hardware-read exceptions by clearing the cached
        values to a sentinel so the next setter falls through (better
        than a stale cached value silently passing the equality check).
        """
        if not self._driver or not self._driver.active:
            return
        try:
            gain = self._driver.get_gain()
            exp = self._driver.get_exposure_t()
        except Exception as e:
            self._commit_camera_writes({'gain_db': -1.0, 'exposure_ms': -1.0})
            logger.warning(
                f'[SCOPE API ] cache refresh after auto-off failed: {e}; '
                f'cache invalidated to force next setter through.'
            )
            return
        # A non-physical reading (the drivers' negative failed-read
        # sentinel) routes to the SAME -1.0 invalidation the exception
        # path above writes -- NOT keep-prior: the pre-auto cached value
        # is known-stale here (hardware moved during the auto cycle), and
        # keeping it would let the setter equality check short-circuit.
        # Committed as an authoritative write (generation bump): a stale
        # getter read racing this resync must not resurrect the pre-auto
        # value.
        resync = {}
        if gain is not None:
            resync['gain_db'] = float(gain) if common_utils.is_valid_gain_db(gain) else -1.0
        if exp is not None:
            resync['exposure_ms'] = float(exp) if common_utils.is_valid_exposure_ms(exp) else -1.0
        if resync:
            self._commit_camera_writes(resync)
        # Diagnostic: record where hardware auto-gain/exposure actually
        # converged when the cycle ended. A converged gain that stayed near
        # the floor on a dim scene means the settle window ended before AG
        # ramped (under-converged -> dark capture); a gain at the ceiling
        # means the scene needs more light, not more settle frames.
        logger.debug(
            f'[AG CONVERGE] auto cycle ended; camera converged to gain={gain} dB exposure={exp} ms'
        )

    def _invalidate_camera_cache(self) -> None:
        """Return the camera cache to the no-camera state, on disconnect.

        Every key, not only ``'active'``: a cached read that went on
        describing the departed camera once decided a write -- a frame edit
        at the old size compared equal to the stale cache, skipped the
        camera and was stored with no camera behind it. Committed as an
        authoritative write, so a read in flight cannot re-commit a value of
        the camera that left.
        """
        self._commit_camera_writes(_no_camera_cache())
        self._forget_held_layer()

    def _forget_held_layer(self) -> None:
        """Record that the camera holds no layer's stored settings."""
        with self._state_lock:
            self._held_layer = None

    def _no_layers_write(self, impl):
        """``impl``, run once the camera is recorded as holding no layer.

        A raw door writes numbers that are no layer's, so after it the camera
        holds none, whatever the write did; cleared before the write, since a
        write that raises may have changed the camera all the same.
        """

        def write(*args, **kwargs):
            self._forget_held_layer()
            return impl(*args, **kwargs)

        return write

    def _fire_camera_listeners(self, param: str, value: float) -> None:
        """Notify all camera listeners of a setting change."""
        with self._camera_listeners_lock:
            listeners = list(self._camera_listeners)
        notifications.tell_listeners(listeners, param, value, category='Camera')

    def _frame_under_the_arm(self) -> Frame | None:
        """The camera's newest frame taken under the standing auto-gain arm.

        The lock wants what the camera's auto loop has reached NOW, so the
        newest frame is the answer, but only one that arrived after the arm
        did (its ``auto_gain`` change): a frame from before the arm holds
        the values the loop started from, not what it reached. When the
        newest is older, a newer one is waited for, within a grab's bound.
        Nothing that judges or records a particular frame reads this: those
        read the chunks of the frame they hold (``Frame.chunks``).

        Returns:
            The frame, or None when no camera is attached or none arrived
            under the arm within the bound.
        """
        driver = self._driver
        if driver is None:
            return None
        armed = self.frame_validity.changes().get('auto_gain')
        frame = driver.grab()
        deadline = time.monotonic() + max(self._grab_sizing_exposure_s() * 3, 1.0)
        while armed is not None and (frame is None or frame.seq <= armed.at_seq):
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            frame = driver.grab_new_capture(timeout_s=left)
        return frame

    def _camera_write(
        self,
        write_fn: Callable[[], object],
        *,
        member: str,
        refusal: Callable[[], CameraSettingRejected] | None,
        invalidates: tuple[str, ...] = (),
        force_invalidate: tuple[str, ...] = (),
        targets: tuple[tuple[str, float | None], ...] = (),
        force_clear: tuple[str, ...] = (),
        cache_update: dict[str, object] | None = None,
        target_from_result: tuple[str, ...] = (),
    ) -> object:
        """Single sanctioned path for a camera-state write and its validity
        consequence. Every camera setter routes its hardware write through here
        so the write and the frame-validity invalidation it requires are
        declared together -- a new setter cannot write a camera node and forget
        to invalidate.

        Order is load-bearing: the write happens first, then ``force_invalidate``
        fires unconditionally (the always-mark-RED contract for the manual value
        setters, so a hardware-rejected write still expires validity rather than
        leaving a stale frame acceptable), then the applied-only block runs.

        A write either applied or raises, so no setter answers a refusal with
        a value a caller could forget to check. An explicit ``False`` is the
        driver's refusal and raises ``refusal()``; a ``None`` return (drivers
        without a confirmation signal) and any truthy result count as applied.
        A write the camera did not answer (``CameraNoAnswerError``) is sent
        once more, at once: every setting write is value-idempotent, and a
        refusal is not resent, since the camera would refuse it again. Not
        answered the second time either, its value is not known, so the
        chunk target of each source it would have recorded is cleared (the
        next captures settle by frame count), its applied-only invalidations
        fire, and the refusal is raised as ``'camera_no_answer'``. A write
        that raises anything else is refused as ``refusal()``; each raise
        carries the driver's exception as its cause, so no setter lets a
        driver's own exception out of the API.

        A camera declared removed by the time a sent write returns is refused
        as an absent one, whatever the driver answered: a camera that is gone
        has no value in effect, and its absence is one answer however the
        command met it. Asked here, after the unconditional block and before
        anything is recorded, because every camera write passes through. The
        removal latch is asked, not ``active``: ``_mark_disconnected()``
        leaves the SDK handle attached until its teardown runs off this
        thread, so a write to a vanished camera still reaches the driver and
        comes back looking like a refusal.

        Args:
            write_fn: Zero-arg callable performing the driver write (including
                any lock it needs) and returning the driver's result.
            member: The public command the write belongs to, named by the
                refusal when the camera is found gone.
            refusal: Builds the ``CameraSettingRejected`` this write raises
                when the driver raises, in the words of the setting's
                refusal. Required, so a new setter states it; ``None`` only
                for a write that sends nothing to the camera.
            invalidates: Sources to invalidate only when the write was applied.
            force_invalidate: Sources to invalidate unconditionally, regardless
                of the result.
            targets: ``(source, value)`` pairs passed to ``set_target`` when the
                write was applied (``value`` None clears the target).
            force_clear: Sources whose chunk target is cleared (set to None)
                unconditionally -- the mode/one-shot setters drop their manual
                target whether or not the driver reported the write applied, so
                chunk-match falls back to skip-frames settling (target
                maintenance stays outside the applied gate). Clear-only by
                construction: an unconditional write can only ever drop a
                target, never record one for a possibly-rejected value.
            cache_update: Keys to write into the ``_camera_cache`` snapshot when
                the write was applied.
            target_from_result: Sources whose chunk target is taken from the
                driver's own return value instead of from ``targets``. A driver
                may clamp, snap or quantize the request before the hardware
                sees it; the frame then carries chunk data describing what was
                APPLIED, so a target recorded from the request can never match
                and every subsequent frame is rejected. Declaring the target
                here -- rather than computing it at the call site -- is what
                keeps a transforming setter from silently reintroducing that
                mismatch. A driver returning a non-numeric result (applied, but
                unable to report a value) falls back to the ``targets`` entry
                for that source.

        Returns:
            The driver write's result, never ``False``, so the caller can
            read the value the driver applied and fire its listeners.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera: it was declared removed by the time the write
                returned. Chained to the driver's exception when it raised.
            CameraSettingRejected: The camera still present, the driver
                refused the write (``'camera_rejected'``), raised from it
                (``'camera_rejected'``, chained), or answered neither try
                (``'camera_no_answer'``, chained).
        """
        try:
            try:
                result = write_fn()
            except CameraNoAnswerError as unanswered:
                if refusal is None:
                    raise
                _api_log.info(f'{member}: the camera did not answer ({unanswered}); sent again')
                result = write_fn()
        except CameraNoAnswerError as ex:
            if refusal is None:
                raise
            for source, _value in targets:
                self.frame_validity.set_target(source, None)
            for source in invalidates:
                self.frame_validity.invalidate(source)
            if self.camera_removed:
                raise _camera_not_connected(member) from ex
            raise _no_answer_rejection(refusal()) from ex
        except Exception as ex:
            if refusal is None:
                raise
            if self.camera_removed:
                raise _camera_not_connected(member) from ex
            raise refusal() from ex
        finally:
            for source in force_invalidate:
                self.frame_validity.invalidate(source)
            for source in force_clear:
                self.frame_validity.set_target(source, None)
        if refusal is not None and self.camera_removed:
            raise _camera_not_connected(member)
        if result is False:
            raise refusal()
        for source in invalidates:
            self.frame_validity.invalidate(source)
        for source, value in targets:
            if (
                source in target_from_result
                and isinstance(result, (int, float))
                and not isinstance(result, bool)
            ):
                # bool is an int subclass, so a driver reporting a bare
                # True would otherwise stamp a 1.0 target and reject
                # every frame -- the failure this parameter exists to
                # prevent, reintroduced by the check meant to prevent it.
                value = float(result)
            self.frame_validity.set_target(source, value)
        if cache_update:
            self._commit_camera_writes(cache_update)
        return result

    # Failed camera reads inside this window after a WARNING log at debug;
    # mirrors the notification framework's short non-fatal dedup window so
    # a dead camera polled at frame rate warns once per window, not per
    # frame, while the first failure of a streak is always visible in the
    # main log.
    _READ_FAILURE_WARN_INTERVAL_S = 5.0

    def _commit_camera_writes(self, updates: dict) -> None:
        """Cache write-through for authoritative values.

        Authoritative means the value did not come from a read: a driver
        write that was applied, or a deliberate invalidation. Bumps each
        key's write generation so a validated live read that was already
        in flight when this landed cannot commit its now-stale value over
        the newer truth.
        """
        with self._camera_cache_lock:
            self._camera_cache.update(updates)
            for key in updates:
                self._camera_cache_write_gen[key] = self._camera_cache_write_gen.get(key, 0) + 1

    def _log_camera_read_failure(self, key: str, cause: object) -> None:
        """Surface a failed camera read without per-frame log spam.

        The read failures this layer absorbs (getters answer
        last-known-good) must still be visible in the main log, or a
        camera whose every read fails looks healthy in a tech-support
        bundle. First failure per key per window logs WARNING; repeats
        within the window log debug.
        """
        now = time.monotonic()
        with self._camera_cache_lock:
            last_warn = self._camera_read_warn_ts.get(key, 0.0)
            warn = (now - last_warn) >= self._READ_FAILURE_WARN_INTERVAL_S
            if warn:
                self._camera_read_warn_ts[key] = now
        if warn:
            logger.warning(
                f'[SCOPE API ] camera {key} read failed ({cause}); '
                f'answering last-known-good until a read succeeds'
            )
        else:
            _api_log.debug(f'camera {key} read failed: {cause}')

    @api
    @property
    def camera_removed(self) -> bool:
        """Whether this scope's camera was declared removed by its driver.

        The driver's removal latch, set once by whatever saw the camera go
        (an SDK removal callback, a bus probe, the grab loop) and cleared
        only by the driver's next connect. Narrower than ``Lumascope.camera_connected``, which also
        reads False when a connection query merely failed. False when the
        scope has no camera at all -- nothing was removed.

        Returns:
            bool: True once the camera has been declared removed.
        """
        return self._driver is not None and self._driver.is_device_removed()

    # --- Setters ---
    def _set_gain_db_impl(self, gain_db: float) -> float | None:
        """Set the camera gain, and answer with the gain now in effect.

        The composition primitive the auto-gain lock, the exposure ceiling,
        the camera restore and the layer apply build on; a write that did
        not apply raises (``_camera_write``), and each of them catches it,
        carries on at the gain the camera holds, and answers for it in its
        own outcome. Nothing is logged or shown here: a refusal is reported
        once, by whoever ends its flight.

        Args:
            gain_db: Gain value in dB.

        Returns:
            float | None: The gain in dB now in effect -- the driver's own
                answer when it gives one, since a body that clamps or
                quantizes applies something other than the request, else the
                request. ``None`` when no camera is active.

        Raises:
            CameraSettingRejected: The write did not apply
                (``_camera_write``); the gain is not recorded, since frames
                keep streaming at the old one and IDS has no chunk backstop
                to catch the mismatch downstream.
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, when it was removed during the write
                (``_camera_write``).
        """
        if not self._driver or not self._driver.active:
            return
        # The validity invalidate must never be gated by the software cache:
        # a cache desynced from hardware once short-circuited it, so a frame at
        # a stale gain was captured as valid. force_invalidate marks 'gain' RED
        # on every write (even a rejected one); the value the driver says it
        # applied is recorded as the chunk target and cached only when the
        # write was not rejected. The driver compares against live hardware and
        # skips a truly redundant SDK write; the cache-equality check here gates
        # only the UI listener + info log, where a missed redundant update is
        # harmless.
        prior = self.gain_db_cached

        def _write_gain():
            with self._cam_lock:
                return self._driver.gain(gain_db)

        ok = self._camera_write(
            _write_gain,
            member='set_gain_db',
            refusal=lambda: _value_rejection('gain_db', gain_db),
            force_invalidate=('gain',),
            targets=(('gain', float(gain_db)),),
            target_from_result=('gain',),
        )
        applied = _value_in_effect(ok, float(gain_db))
        self._commit_camera_writes({'gain_db': applied})
        if abs(applied - prior) >= 0.001:
            _api_log.info(f'set_gain_db {applied}dB')
            self._fire_camera_listeners('gain', applied)
        return applied

    def _set_exposure_ms_impl(self, exposure_ms: float) -> float | None:
        """Set the camera exposure time, and answer with the exposure now in
        effect.

        Silent, and raising when the write did not apply, for the reason
        given on ``_set_gain_db_impl``.

        Args:
            exposure_ms: Exposure time in milliseconds.

        Returns:
            float | None: The exposure in milliseconds now in effect --
                converted from the driver's microseconds when it reports
                them, since a body that clamps to its floor or quantizes onto
                a row grid applies something other than the request, else the
                request. ``None`` when no camera is active.

        Raises:
            CameraSettingRejected: The write did not apply
                (``_camera_write``); the exposure is not recorded.
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, when it was removed during the write
                (``_camera_write``).
        """
        if not self._driver or not self._driver.active:
            return
        # The validity invalidate must never be gated by the software cache:
        # a cache desynced from hardware once short-circuited it, capturing a
        # frame at a stale exposure as valid. Always invalidate + drive the
        # setter; the cache-equality check gates only the UI listener + log.
        prior = self.exposure_ms_cached
        # Sanity-check threshold: 5 microseconds. Pylon physical
        # ExposureTime minimum across Basler USB3 sensors is 10-35 us;
        # below 5 us is impossible on any sensor we ship with and
        # indicates a unit-confusion bug (e.g. seconds-treated-as-ms).
        # Bright-field captures legitimately use 0.03 ms (30 us) on
        # bright samples, so the threshold sits below that range.
        if exposure_ms < 0.005 and not self._suppress_value_warnings:
            import traceback

            _caller = ''.join(traceback.format_stack(limit=6)[-4:-1]).strip()
            logger.warning(
                f'[SCOPE API ] set_exposure_ms({exposure_ms}ms) is below '
                f'any Basler sensor physical minimum -- camera '
                f'will clamp the request. Confirm the value is '
                f'in milliseconds, not seconds or microseconds.\n'
                f'Call stack:\n{_caller}'
            )

        # Record the applied exposure for chunk-match. ChunkExposureTime is
        # microseconds; the API takes milliseconds. Convert at the seam so the
        # chunk value and frame_validity's tolerance share units. force_invalidate
        # marks 'exposure' RED on every write; target + cache only when the write
        # was not rejected.
        def _write_exposure():
            with self._cam_lock:
                return self._driver.exposure_t(exposure_ms)

        ok = self._camera_write(
            _write_exposure,
            member='set_exposure_ms',
            refusal=lambda: _value_rejection('exposure_ms', exposure_ms),
            force_invalidate=('exposure',),
            targets=(('exposure', float(exposure_ms) * 1000.0),),
            target_from_result=('exposure',),
        )
        applied = _value_in_effect(ok, float(exposure_ms), scale=0.001)
        self._commit_camera_writes({'exposure_ms': applied})
        if abs(applied - prior) >= 0.001:
            _api_log.info(f'set_exposure {applied}ms')
            self._fire_camera_listeners('exposure', applied)
        return applied

    # --- Public dispatch ---
    # These three are what every caller reaches: an SDK script, a REST
    # handler, the GUI -- and the run, the autofocus sweep and the diagnostics,
    # which call them under their taking so the lane admits their work while
    # they hold the scope. From a task already on the lane's worker the lane
    # runs the body inline.

    # How long a dispatched camera write waits on the camera worker before
    # giving up. A gain or exposure write is a short SDK call behind at most
    # a few queued camera commands, so this is a liveness bound rather than
    # a budget -- if it ever expires, the worker is wedged and the caller
    # should hear about it instead of blocking forever. Deliberately not a
    # public parameter: an external caller has no way to know what value
    # would be right.
    _CAMERA_WRITE_TIMEOUT_S = 5.0

    # A command whose body may stop and restart the grab is slower than a
    # value write: a grab restart on a Pylon body has been measured at 11 s
    # (frame size, pixel format, and on Pylon the conversion-gain mode and
    # line-noise reduction too), and the dispatcher's wait bounds QUEUE TIME
    # plus execution -- a restart queued behind another one must not time
    # out while both are healthy. Such a command takes this bound once per
    # restart it may make. The bound stays a liveness verdict, not a
    # budget: `fut.result` ABANDONS on timeout without cancelling, so a
    # timed-out write still lands later and the caller's view of the camera
    # diverges -- which is why this must be sized so a healthy write can
    # never hit it.
    _CAMERA_GEOMETRY_TIMEOUT_S = 30.0

    # A restore may restart the grab three times (binning, frame size, pixel
    # format) and then writes the black level, gain, exposure and the
    # auto-gain arm.
    _CAMERA_RESTORE_TIMEOUT_S = 3 * _CAMERA_GEOMETRY_TIMEOUT_S + 4 * _CAMERA_WRITE_TIMEOUT_S

    def _dispatch_camera(
        self,
        impl,
        name,
        args=(),
        kwargs=None,
        *,
        timeout_s,
        override=False,
        falsifies_recording=False,
        satisfied_when_absent=_REFUSED_WHEN_ABSENT,
    ):
        """Run one camera command for an external caller, on the right thread.

        The body runs on the scope's camera lane, serialized against every
        other camera-bus operation, and this blocks until it has. A lane that
        will not accept work tells the caller so, because the alternative is
        `put` returning None and the command disappearing with nothing raised
        and nothing logged.

        The lane's ``call`` decides a refusal and raises it to the caller:
        the lane is closed, or a run, a diagnostic or a home holds the scope and this
        call is not made under its taking, or ``falsifies_recording`` is set
        and a recording holds the scope. Once admitted, the task asks for the
        camera before the body runs, and again when the body raises a fault
        (``_asking_for_the_camera``), so a camera the command finds removed
        is refused as one already gone; after ``disconnect()`` the lane's
        own ``scope_disconnected`` is what the caller hears.

        ``override`` admits the task whatever holds the scope, on the key the
        session registered: only for a read that must touch the camera from
        its lane's thread but changes nothing a holder depends on.

        ``satisfied_when_absent`` is what the command answers with no camera
        connected: an off whose end state already holds, or a read with no
        camera to describe. Left out, the command is refused naming the
        camera.
        """
        kwargs = kwargs or {}
        key = self._scope._camera_override_key if override else None
        task = IOTask(
            action=self._asking_for_the_camera(impl, name, satisfied_when_absent),
            args=args,
            kwargs=kwargs,
            falsifies_recording=falsifies_recording,
        )
        return self._scope._camera_executor.call(task, name, timeout_s, override=key)

    def _submit_camera(self, impl, name, args=(), *, waiter):
        """Put one camera command on the camera lane without waiting for it.

        The command is admitted or refused on the calling thread, as
        ``_dispatch_camera`` admits it, so a refusal raises here; its outcome
        settles ``waiter``, a Future the caller made and marked running. With
        no camera connected the outcome is the refusal naming the camera.

        Raises:
            HardwareCommandRefusedError: the lane is closed, or a run, a
                diagnostic or a home holds the scope and this call is not made under
                its taking.
        """
        task = IOTask(
            action=self._asking_for_the_camera(impl, name, _REFUSED_WHEN_ABSENT), args=args
        )
        self._scope._camera_executor.submit(task, name, waiter=waiter)

    def _asking_for_the_camera(self, impl, member: str, satisfied_when_absent):
        """``impl``, run only once the camera is asked for, on the lane.

        The camera's one presence question for an outside command, asked
        after the lane admits it and before the body writes: with no camera
        connected the command is refused naming it, or answers
        ``satisfied_when_absent`` when its dispatch names one. A body that
        is what finds the camera gone is answered the same way: its refusal
        naming the camera (``_camera_write``'s), or any fault it raises with
        the camera declared removed by then (``camera_removed``), since a
        camera that is not there has no fault of its own to report; the
        fault is the refusal's cause. So an off is satisfied whenever the
        camera went. A refusal or defect of any other kind keeps its type.
        A body handed ``_DECIDED_IN_BODY`` asks for the camera itself, and
        a camera it finds removed is refused, never satisfied. Bring-up's
        direct ``_impl`` calls do not pass through here and ask for
        themselves.
        """
        decided_in_body = satisfied_when_absent is _DECIDED_IN_BODY
        refused_when_absent = decided_in_body or satisfied_when_absent is _REFUSED_WHEN_ABSENT

        @functools.wraps(impl)
        def ask_then_run(*args, **kwargs):
            if not decided_in_body and not self._scope.camera_connected:
                if refused_when_absent:
                    raise _camera_not_connected(member)
                return satisfied_when_absent
            try:
                return impl(*args, **kwargs)
            except HardwareCommandRefusedError as refusal:
                if refused_when_absent or refusal.missing is not MissingPart.CAMERA:
                    raise
                return satisfied_when_absent
            except Fault as fault:
                if not self.camera_removed:
                    raise
                if refused_when_absent:
                    raise _camera_not_connected(member) from fault
                return satisfied_when_absent

        return ask_then_run

    def refuse_camera_not_connected(self, member: str) -> None:
        """Refuse when no camera is connected.

        Asked by every outside camera command on the camera lane before it
        writes. Offered alone to a caller that must refuse before work of its
        own starts: Go To Step asks it before moving, and a frame listener
        before it is registered. Off the lane after ``disconnect()`` it
        answers as a command on the lane would, ``scope_disconnected``.

        A consult seam, not part of the L2 API surface: an L2 caller's camera
        command asks it itself.

        Raises:
            HardwareCommandRefusedError: ``'scope_disconnected'`` after
                ``disconnect()``; ``'not_connected'``, naming the camera.
                Nothing was sent.
        """
        if self._scope._camera_executor.pending_shutdown:
            raise HardwareCommandRefusedError('scope_disconnected', member)
        if not self._scope.camera_connected:
            raise _camera_not_connected(member)

    def _refuse_out_of_range(
        self,
        setting: str,
        requested: float,
        minimum: float | None,
        maximum: float | None,
        *,
        noun: str,
        unit: str,
    ) -> None:
        """Refuse a request outside the camera's declared range.

        Checked on the public setters, from the cached limits, so every
        camera answers alike: a body that would refuse the value and one that
        would silently clamp it both never see it. The impls check no range;
        their callers (the layer apply, the auto-gain lock, the restore)
        write values that are already the camera's own or capped to it. An
        undeclared end is not checked.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, the request is not a
                finite number, which no range admits.
            CameraSettingOutOfRangeError: The request is outside the range.
        """
        refuse_unless_finite_number(requested, setting)
        value = float(requested)
        if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
            raise CameraSettingOutOfRangeError(
                setting,
                value,
                minimum,
                maximum,
                message=(
                    f"{noun.capitalize()} {value:g} {unit} is outside this camera's range, "
                    f'{camera_range_words(minimum, maximum, unit)}. The {noun} was not changed.'
                ),
            )

    @api
    def set_gain_db(self, gain_db: float) -> float | None:
        """Set the camera gain, wait for it, and answer with the gain in effect.

        See ``_set_gain_db_impl`` for the value contract; this adds the
        dispatch described on ``_dispatch_camera`` and the raise below.

        The raise is here rather than in the impl because this is the L2
        surface -- an SDK, headless or REST caller -- where the refusal ends
        its flight at the caller. Success is observed by the returned value; a
        refusal cannot be mistaken for one by a caller that forgets to check a
        return code.

        Returns:
            float | None: The gain in dB now in effect, which differs from the
                request when the camera snapped it. ``None`` when the camera's
                teardown landed after the lane asked for it.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, the gain is not a
                finite number; nothing was sent to the camera.
            CameraSettingOutOfRangeError: The gain is outside the range the
                camera declares; nothing was sent to it.
            CameraSettingRejected: A live driver refused the gain or raised
                from the write (chained). It carries the words its reporter
                shows; nothing is shown here.
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected or one removed during the write.
        """
        # The range is the connected camera's: with none, or the scope
        # disconnected, that is the answer, not a range it no longer has.
        self.refuse_camera_not_connected('set_gain_db')
        limits = self._scope.capabilities.camera_limits
        self._refuse_out_of_range(
            'gain_db',
            gain_db,
            limits.gain_db_min if limits else None,
            limits.gain_db_max if limits else None,
            noun='gain',
            unit='dB',
        )
        return self._dispatch_camera(
            self._no_layers_write(self._set_gain_db_impl),
            'set_gain_db',
            args=(gain_db,),
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
        )

    @api
    def set_exposure_ms(self, exposure_ms: float) -> float | None:
        """Set the camera exposure time, wait for it, and answer with the
        exposure in effect.

        See ``_set_exposure_ms_impl`` for the value contract and the
        unit-confusion warning it carries. The raise is placed here, on the
        L2 surface, for the reason given on ``set_gain_db``.

        Returns:
            float | None: The exposure in milliseconds now in effect, which
                differs from the request when the camera clamped it to its
                floor or quantized it. ``None`` when the camera's teardown
                landed after the lane asked for it.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, the exposure is not a
                finite number; nothing was sent to the camera.
            CameraSettingOutOfRangeError: The exposure is outside the range
                the camera declares; nothing was sent to it.
            CameraSettingRejected: A live driver refused the exposure or
                raised from the write (chained). It carries the words its
                reporter shows; nothing is shown here.
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected or one removed during the write.
        """
        # The range is the connected camera's: with none, or the scope
        # disconnected, that is the answer, not a range it no longer has.
        self.refuse_camera_not_connected('set_exposure_ms')
        limits = self._scope.capabilities.camera_limits
        self._refuse_out_of_range(
            'exposure_ms',
            exposure_ms,
            limits.exposure_ms_min if limits else None,
            limits.exposure_ms_max if limits else None,
            noun='exposure',
            unit='ms',
        )
        return self._dispatch_camera(
            self._no_layers_write(self._set_exposure_ms_impl),
            'set_exposure_ms',
            args=(exposure_ms,),
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
        )

    @api
    def set_auto_gain(
        self, state: bool, settings: dict, *, resume_after_capture: bool = True
    ) -> bool | None:
        """Enable or disable automatic gain adjustment, and wait for it.

        See ``_set_auto_gain_impl`` for the value contract; this adds the
        dispatch described on ``_dispatch_camera`` and the raise below.

        With no camera connected, turning auto-gain off is satisfied: True,
        as a camera already in manual answers.

        Raises:
            CameraSettingRejected: The camera refused the change. Nothing
                was recorded, and nothing is shown here.
            CameraSettingUnsupportedError: Turning auto-gain on for a camera
                without it.
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, for turning auto-gain on with no camera connected or
                one removed during the write; turning it off is satisfied
                either way.
            ArgumentRefusedError: ``'missing_key'``, ``settings`` lacks a
                key the write reads; ``'wrong_kind'`` or ``'not_a_number'``,
                it or one of its values is not of the kind read;
                ``'gain_range_inverted'``, its minimum gain is above its
                maximum.
        """
        _refuse_unless_auto_gain_settings(settings, 'settings')
        return self._dispatch_camera(
            self._no_layers_write(self._set_auto_gain_impl),
            'set_auto_gain',
            args=(state, settings),
            kwargs={'resume_after_capture': resume_after_capture},
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
            satisfied_when_absent=_REFUSED_WHEN_ABSENT if state else True,
        )

    def _set_auto_gain_impl(
        self, state: bool, settings: dict, *, resume_after_capture: bool = True
    ) -> bool | None:
        """Enable or disable automatic gain adjustment.

        A write that did not apply raises, as the value impls do, unreported
        and recording no arm: the lock would otherwise write back values
        from a loop that never ran.

        Args:
            state: True to enable auto gain, False to disable.
            settings: Dict with 'target_brightness', 'min_gain_db', 'max_gain_db',
                and optionally 'max_exposure_ms' / 'min_exposure_ms' (the
                per-channel-class bounds of the exposure AG/AE may settle
                in; the caller supplies them since it knows the layer).
            resume_after_capture: with ``state=True``, whether a capture
                that locks this arm re-arms it afterwards (live view) or
                leaves the camera at the locked values (a protocol step).

        Returns:
            bool | None: The driver's answer (``None`` or ``True``; see
                ``Camera.gain``); ``None`` when no camera is active. ``True``,
                with nothing written, for turning off a mode a camera
                without hardware auto-gain never had.

        Raises:
            CameraSettingUnsupportedError: Turning auto-gain on for a camera
                without it. Nothing reaches the camera.
            CameraSettingRejected: The write did not apply (``_camera_write``).
        """

        if not self._driver or not self._driver.active:
            return None
        if not self._camera_has_auto_gain():
            # Such a camera is already manual, so off is its state rather
            # than a write: no validity invalidation, no target clear, no
            # cache resync. Its driver answers no auto-mode call truthfully,
            # so it is never asked.
            if state:
                raise _absent_mode_refusal(
                    'auto_gain',
                    state,
                    'automatic gain',
                    offered=(False,),
                    consequence=_AUTO_MODE_STAYS,
                )
            return True

        def _write_auto_gain():
            return self._driver.auto_gain(
                state,
                target_brightness=settings['target_brightness'],
                min_gain_db=settings['min_gain_db'],
                max_gain_db=settings['max_gain_db'],
                ae_max_exposure_ms=settings.get('max_exposure_ms'),
            )

        # Auto-gain dynamically adjusts the value, so clear the manual gain
        # target (chunk-match falls back to skip-frames calibration). Arming
        # hardware continuous AG needs the camera several frames to settle
        # against the lit scene, so invalidate 'auto_gain' to hold capture for
        # the settle count. The mode flip leaves the gain value node
        # unchanged, so these are forced, not gated on a value delta.
        arm_settle = state
        if arm_settle:
            self._clamp_exposure_to_ceiling_before_arm(settings.get('max_exposure_ms'))
        result = self._camera_write(
            _write_auto_gain,
            member='set_auto_gain',
            refusal=lambda: _mode_rejection('auto_gain', state, 'auto-gain'),
            force_invalidate=('gain', 'auto_gain') if arm_settle else ('gain',),
            force_clear=('gain',),
        )
        with self._state_lock:
            self._auto_gain_arm = (
                _AutoGainArm(dict(settings), resume_after_capture) if arm_settle else None
            )
        # Hardware-truth wins over cache after the auto cycle ends.
        if not state:
            self._refresh_cache_from_hardware_after_auto()
        return result

    @api
    def lock_auto_gain(self) -> AutoGainLock:
        """Lock a standing continuous auto-gain arm and return the result.

        See ``_lock_auto_gain_impl`` for the contract; this adds only the
        dispatch described on ``_dispatch_camera``. A caller leaving
        auto-gain stores ``stored_exposure_ms`` / ``gain_db`` as the manual
        setting; when no arm stands the result's ``state`` is None and
        nothing was written. With no camera connected no arm can stand, and
        that is the answer.
        """
        return self._dispatch_camera(
            self._no_layers_write(self._lock_auto_gain_impl),
            'lock_auto_gain',
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
            satisfied_when_absent=AutoGainLock(state=None),
        )

    def _lock_auto_gain_impl(self) -> AutoGainLock:
        """Turn a standing continuous auto-gain arm into locked manual values.

        The camera's auto loop chose an exposure and gain the caller never
        requested, so the capture gate's recorded targets are stale. The
        lock writes the achieved values back as manual values through the
        ordinary setters, which re-targets the gate; the next frame must
        then prove them like any manual setting. The achieved values come
        from the chunks of the newest frame taken under the arm when the
        camera has them (the same node and unit the gate compares), else
        from the cache the disarm just resynced from hardware; with no frame
        under the arm within a grab's bound, the lock found nothing usable.

        Returns an AutoGainLock; ``state`` is None when no arm was
        recorded, and nothing was written.
        """
        with self._state_lock:
            arm, self._auto_gain_arm = self._auto_gain_arm, None
        if arm is None:
            return AutoGainLock(state=None)
        settings = arm.settings
        floor = settings.get('min_exposure_ms') or None
        ceiling = settings.get('max_exposure_ms') or None
        # Read under the arm, before the disarm: a frame from before the arm
        # carries the values the loop started from.
        frame = self._frame_under_the_arm()
        try:
            self._set_auto_gain_impl(False, settings)
        except CameraSettingRejected as rejected:
            self._report_refused_write(rejected)
        chunks = (frame.chunks if frame is not None else None) or {}
        exp_us = chunks.get('ExposureTime')
        gain = chunks.get('Gain')
        if frame is None:
            # No frame came under the arm: what the loop reached is not known.
            exp_ms: object = None
            gain = None
        elif isinstance(exp_us, (int, float)) and isinstance(gain, (int, float)):
            exp_ms = exp_us / 1000.0
        else:
            exp_ms = self.exposure_ms_cached
            gain = self.gain_db_cached
        if not (common_utils.is_valid_exposure_ms(exp_ms) and common_utils.is_valid_gain_db(gain)):
            # No usable achieved value: drop the stale exposure target too
            # (the disarm dropped the gain target) so the capture proceeds
            # without an exposure/gain gate rather than rejecting forever.
            self._camera_write(
                lambda: None, member='lock_auto_gain', refusal=None, force_clear=('exposure',)
            )
            logger.warning(
                '[AG CONVERGE] locked: state=FAILED -- the camera reported no '
                f'usable achieved exposure/gain (exposure={exp_ms} gain={gain}); '
                'the capture proceeds ungated on exposure and gain'
            )
            lock = AutoGainLock(
                AutoGainConvergence.FAILED,
                None,
                None,
                floor,
                ceiling,
                arm.resume_after_capture,
                settings,
            )
            self._notify_auto_gain_outcome(lock)
            return lock
        exp_ms = float(exp_ms)
        gain = float(gain)
        # The write-back pins the values auto-gain just reached, which the
        # camera already holds, so a refused write-back leaves the lock's
        # state true and the capture proceeds; the refusal itself is reported.
        for write_back, value in (
            (self._set_exposure_ms_impl, exp_ms),
            (self._set_gain_db_impl, gain),
        ):
            try:
                write_back(value)
            except CameraSettingRejected as rejected:
                self._report_refused_write(rejected)
        if ceiling is not None and exp_ms >= ceiling * 0.99:
            state = AutoGainConvergence.MAXED
        elif floor is not None and exp_ms <= floor:
            state = AutoGainConvergence.AT_MINIMUM
        else:
            state = AutoGainConvergence.CONVERGED
        line = (
            f'[AG CONVERGE] locked: state={state.value} exposure={exp_ms:.3f} ms '
            f'gain={gain:.2f} dB (class floor={floor} ms ceiling={ceiling} ms)'
        )
        # All four states at info: a converged lock logged only at debug left
        # no trace in a customer bundle of what the camera settled at.
        logger.info(line)
        lock = AutoGainLock(
            state,
            exp_ms,
            gain,
            floor,
            ceiling,
            arm.resume_after_capture,
            settings,
            stored_exposure_ms=stored_exposure_after_lock(exp_ms, floor),
        )
        self._notify_auto_gain_outcome(lock)
        return lock

    def _notify_auto_gain_outcome(self, lock: AutoGainLock) -> None:
        """Tell an attended user what the lock found; a protocol run is
        unattended and gets the log line only.

        A live-view arm (``resume_after_capture``) is the attended case:
        the user toggled auto-gain off, took a manual capture, or started a
        manual autofocus. A protocol step's arm is not, and during a run
        only a fatal, run-aborting error may reach the screen. The limit
        states are information -- the setting keeps the class floor or
        ceiling and cannot show the raw value -- and a lock that found
        nothing usable is an error, because the frame was taken without an
        exposure check.
        """
        if not lock.resume_after_capture:
            return
        if lock.state is AutoGainConvergence.MAXED:
            outcome = ExposureAtMaximumNotice(lock.ceiling_ms)
        elif lock.state is AutoGainConvergence.AT_MINIMUM:
            outcome = ExposureAtMinimumNotice(lock.exposure_ms, lock.floor_ms)
        elif lock.state is AutoGainConvergence.FAILED:
            outcome = AutoGainNotSettledError()
        else:
            return
        notifications.report_outcome(outcome, solicited=False, category='Auto-gain')

    def _clamp_exposure_to_ceiling_before_arm(self, ceiling_ms: object) -> None:
        """Bring the exposure inside the auto loop's range before enabling it.

        The loop's upper bound is the class ceiling, but writing the bound
        does not move an exposure already above it: the camera then leaves
        the exposure where it is and adjusts gain only, the lock reads the
        pinned value, and the stored setting clips it to the slider -- the
        live view halves in brightness at toggle-off. Clamping through the
        ordinary setter re-targets the gate to the ceiling as well, so the
        next frame proves it. The cache is a setter's write or a hardware
        resync at every arm (both arming callers write the exposure just
        before; every disarm resyncs), so the sentinel guard is the only
        skip.
        """
        if not isinstance(ceiling_ms, (int, float)) or ceiling_ms <= 0:
            return
        current_ms = self.exposure_ms_cached
        if not common_utils.is_valid_exposure_ms(current_ms) or current_ms <= ceiling_ms:
            return
        logger.info(
            f'[AG ARM] exposure {current_ms:.3f} ms above the class ceiling '
            f'{ceiling_ms:g} ms; clamped to the ceiling before arming'
        )
        try:
            self._set_exposure_ms_impl(float(ceiling_ms))
        except CameraSettingRejected as rejected:
            # The arm goes ahead at the exposure the camera holds.
            self._report_refused_write(rejected)

    def _report_refused_write(
        self, rejection: CameraSettingRejected | HardwareCommandRefusedError
    ) -> None:
        """Report a camera refusal whose flight ends in this class.

        The auto-gain lock's write-back and disarm, the exposure-ceiling clamp,
        the camera restore and the live-view re-arm each carry on with the
        camera as it is, so no caller hears the refusal; this is where it is
        reported, once, through the one reporter. No person asked for these
        writes.
        """
        notifications.report_outcome(rejection, solicited=False, category='Camera')

    def _resume_auto_gain_impl(self, lock: AutoGainLock) -> None:
        """Re-arm continuous auto-gain after a capture locked a live-view arm."""
        if lock.state is None or not lock.resume_after_capture or lock.settings is None:
            return
        # Run after the capture's frame is in hand, often from its finally: a
        # camera found gone here is reported, never raised over that frame.
        try:
            self._set_auto_gain_impl(True, lock.settings, resume_after_capture=True)
        except (CameraSettingRejected, HardwareCommandRefusedError) as not_rearmed:
            self._report_refused_write(not_rearmed)

    @api
    def set_auto_exposure_time(self, state: bool = True) -> bool | None:
        """Enable or disable automatic exposure adjustment, and wait for it.

        See ``_set_auto_exposure_time_impl`` for the value contract; this
        adds the dispatch described on ``_dispatch_camera`` and the raise
        below.

        With no camera connected, turning auto-exposure off is satisfied:
        True, as a camera already in manual answers.

        Raises:
            CameraSettingRejected: The camera refused the change; nothing is
                shown here.
            CameraSettingUnsupportedError: Turning auto-exposure on for a
                camera without it.
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, for turning auto-exposure on with no camera connected
                or one removed during the write; turning it off is satisfied
                either way.
        """
        return self._dispatch_camera(
            self._no_layers_write(self._set_auto_exposure_time_impl),
            'set_auto_exposure_time',
            args=(state,),
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
            satisfied_when_absent=_REFUSED_WHEN_ABSENT if state else True,
        )

    def _set_auto_exposure_time_impl(self, state: bool = True) -> bool | None:
        """Enable or disable automatic exposure adjustment.

        Args:
            state: True to enable auto exposure, False to disable.

        Returns:
            bool | None: The driver's answer. ``None`` when no camera is
                active. ``True``, with nothing written, for turning off a mode
                a camera without hardware auto-exposure never had.

        Raises:
            CameraSettingUnsupportedError: Turning auto-exposure on for a
                camera without it. Nothing reaches the camera.
            CameraSettingRejected: The write did not apply
                (``_camera_write``); the cache is not resynced.
        """

        if not self._driver or not self._driver.active:
            return None
        if not self._camera_has_auto_exposure():
            # See _set_auto_gain_impl: off is such a camera's state, not a write.
            if state:
                raise _absent_mode_refusal(
                    'auto_exposure',
                    state,
                    'automatic exposure',
                    offered=(False,),
                    consequence=_AUTO_MODE_STAYS,
                )
            return True
        # Auto-exposure dynamically adjusts the value, so clear the manual
        # exposure target (chunk-match falls back to skip-frames calibration).
        # The mode flip leaves the exposure value node unchanged, so the
        # invalidate + target-clear are forced, not gated on a value delta.
        result = self._camera_write(
            lambda: self._driver.auto_exposure_t(state),
            member='set_auto_exposure_time',
            refusal=lambda: _mode_rejection('auto_exposure', state, 'auto-exposure'),
            force_invalidate=('exposure',),
            force_clear=('exposure',),
        )
        # Hardware-truth wins over cache after the auto cycle ends.
        if not state:
            self._refresh_cache_from_hardware_after_auto()
        return result

    def _camera_setting_rejection(
        self, setting: str, requested, body: str
    ) -> CameraSettingRejected:
        """Build the typed rejection for a camera-setting apply.

        Callers ``raise self._camera_setting_rejection(...)`` so the raise
        is explicit at every rejection site. Nothing is logged or shown
        here: the rejection is reported once, by whoever ends its flight,
        in the words it carries.
        """
        return CameraSettingRejected(setting, requested, message=body)

    # ------------------------------------------------------------------
    # The camera's stored settings: applied, then saved
    # ------------------------------------------------------------------

    def _saved_after(self, impl, member: str, updates: dict[str, object]):
        """A camera-lane body: ``impl``, then the save of ``updates``.

        Each stored setting's write and its save are one lane task, so no
        other camera command lands between them: saved on the caller's
        thread after the task, two callers' binnings left one saved beside a
        camera at the other. A body answering falsy found no camera, and
        nothing is saved.
        """

        def write_then_save(*args):
            if not impl(*args):
                raise _camera_not_connected(member)
            self._scope.store_settings(updates)

        return write_then_save

    @api
    def set_high_conversion_gain(self, enabled: bool) -> None:
        """Turn the camera's high conversion gain on or off, then save it.

        High conversion gain lowers the sensor's read-noise floor at the cost
        of dynamic range. The one writer of ``camera.high_conversion_gain``:
        the setting is saved only once the camera took it, so the saved
        settings never name a mode the camera is not in. Off on a camera
        without the mode is its state, and is saved.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected or one removed during the write.
                Nothing is saved.
            CameraSettingUnsupportedError: ``'conversion_gain_mode_unsupported'``,
                on, for a camera without the mode. Nothing reaches the camera or
                is saved.
            CameraSettingRejected: The camera refused it or raised from it
                (chained). Nothing is saved.
        """
        self._dispatch_camera(
            self._saved_after(
                self._set_conversion_gain_mode_impl,
                'set_high_conversion_gain',
                {'camera.high_conversion_gain': enabled},
            ),
            'set_high_conversion_gain',
            args=('High' if enabled else 'Low',),
            timeout_s=self._CAMERA_GEOMETRY_TIMEOUT_S,
        )

    @api
    def set_line_noise_reduction(self, enabled: bool) -> None:
        """Turn the camera's line-noise filter on or off, then save it.

        The one writer of ``camera.line_noise_reduction``, saved only once
        the camera took it, as ``set_high_conversion_gain`` is.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected or one removed during the write.
                Nothing is saved.
            CameraSettingUnsupportedError: ``'line_noise_reduction_unsupported'``,
                on, for a camera without the filter. Nothing reaches the camera
                or is saved.
            CameraSettingRejected: The camera refused it or raised from it
                (chained). Nothing is saved.
        """
        self._dispatch_camera(
            self._saved_after(
                self._set_line_noise_reduction_impl,
                'set_line_noise_reduction',
                {'camera.line_noise_reduction': enabled},
            ),
            'set_line_noise_reduction',
            args=(enabled,),
            timeout_s=self._CAMERA_GEOMETRY_TIMEOUT_S,
        )

    @api
    def set_image_mode(self, mode: str) -> None:
        """Capture in ``mode``: apply the camera format it needs, then save it.

        The one writer of ``image_mode`` for every host. The mode names a
        capture depth, and the format is chosen from the ones this camera
        reports, so a sensor is never asked for a format it lacks; a pixel
        format is set only this way, so a capture's depth and its save
        policy agree. A camera that reports none (none is connected) has
        nothing to apply; the mode is saved and bring-up applies it.

        Raises:
            SettingRefusedError: ``'out_of_range'``, ``mode`` is not an image
                mode. Nothing is applied or saved.
            HardwareCommandRefusedError: A recording holds the scope: its
                frames are written at the depth it started with.
                ``'not_connected'``, naming the camera, when it went away
                between the format query and the apply. Nothing is saved.
            CameraSettingRejected: The camera refused the format. Nothing is
                saved, so captures are never tagged with a depth the camera
                is not delivering.
        """
        settings_paths.refuse_outside_range('image_mode', mode)
        capture_depth = image_mode.resolve_image_mode(mode)['capture_depth']
        target = image_mode.select_capture_pixel_format(
            capture_depth, self._scope.capabilities.camera_pixel_formats
        )
        if target is None:
            self._scope.store_settings({'image_mode': mode})
            return
        self._dispatch_camera(
            self._saved_after(self._set_pixel_format_impl, 'set_image_mode', {'image_mode': mode}),
            'set_image_mode',
            args=(target,),
            timeout_s=self._CAMERA_GEOMETRY_TIMEOUT_S,
            falsifies_recording=True,
        )

    @api
    def set_binning_size(self, size: int) -> dict | None:
        """Bin the camera by ``size``, keep the framed region, and save both.

        The one writer of the binning for every host. The framed region is
        saved unbinned (``frame.native_width`` / ``frame.native_height``), so
        a new binning divides that region by the new factor rather than
        rescaling the last displayed size, and cycling the binning
        round-trips. The binning goes first and the frame after it, because
        the camera's frame limits depend on the binning in force; one caller
        running both in order is what keeps a frame edit from being worked
        out against a binning the camera has not reached.

        Returns:
            The frame the camera delivers, ``{'width', 'height'}``.

        Raises:
            HardwareCommandRefusedError: A recording holds the scope: its
                frames and their pixel size follow the binning it started
                with. ``'not_connected'``, naming the camera, with none
                connected or one removed during the write. Nothing is saved.
            CameraSettingUnsupportedError: ``'binning_unsupported'``, this
                camera does not offer ``size``. Nothing reaches the camera.
            CameraSettingRejected: The camera refused the binning (nothing is
                saved) or the frame after it (the binning is saved, with the
                frame the camera reports holding at it, and no frame when it
                has never reported one).
        """
        self._refuse_unusable_binning(size, 'set_binning_size')
        # One lane task restarts the grab twice: the binning, then the frame.
        return self._dispatch_camera(
            self._bin_and_save_impl,
            'set_binning_size',
            args=(size,),
            timeout_s=2 * self._CAMERA_GEOMETRY_TIMEOUT_S,
            falsifies_recording=True,
        )

    def _bin_and_save_impl(self, size: int) -> dict | None:
        """On the camera lane: bin by ``size`` and save it, then frame the saved region at it."""
        native = self._native_frame()
        if not self._set_binning_size_impl(size):
            raise _camera_not_connected('set_binning_size')
        # The camera's reading, not the cache, whose seed is 0x0 until a
        # frame is read: a camera that never reported one saves no frame.
        held = self._get_frame_size()
        updates = {'binning.size': binning.binning_size_int_to_str(size)}
        if held is not None:
            updates['frame.width'] = int(held['width'])
            updates['frame.height'] = int(held['height'])
        self._scope.store_settings(updates)
        return self._apply_frame_impl(native, self._target_frame(native, size))

    @api
    def set_frame_size(self, width: int, height: int) -> dict | None:
        """Frame the camera at ``width`` x ``height`` at the binning it holds, and save it.

        The one writer of the frame for every host. The size is what the
        person sees and captures (post-binning); the unbinned region it
        implies is saved beside it. A size the camera already delivers is
        not written again.

        Returns:
            The frame the camera delivers: the request floored to even
            sides.

        Raises:
            HardwareCommandRefusedError: A recording holds the scope: its
                frames are fitted to the geometry it started with.
                ``'not_connected'``, naming the camera, with none connected
                or one removed during the write; ``'scope_disconnected'``
                after ``disconnect()``. Nothing is saved.
            CameraSettingOutOfRangeError: ``'frame_width_out_of_range'`` or
                ``'frame_height_out_of_range'``, the size, floored to even
                sides, is below the camera's minimum frame or above the
                scope's maximum at the binning held: the sensor, or the
                model's smaller maximum (``capabilities.camera_limits``).
                Nothing reaches the camera or is saved; an undeclared end is
                not checked.
            CameraSettingRejected: The camera refused the frame. Nothing is
                saved.
        """
        return self._dispatch_camera(
            self._frame_and_save_impl,
            'set_frame_size',
            args=(width, height),
            timeout_s=self._CAMERA_GEOMETRY_TIMEOUT_S,
            falsifies_recording=True,
        )

    def _frame_and_save_impl(self, width: int, height: int) -> dict | None:
        """On the camera lane: frame at ``width`` x ``height`` at the binning held, and save it."""
        factor = self._binning_size
        native = {'width': width * factor, 'height': height * factor}
        target = binning.native_to_displayed(native, factor, self.get_pixel_alignment())
        return self._apply_frame_impl(native, target)

    @api(reads=True)
    def frame_at_binning(self, size: int) -> dict:
        """The frame ``set_binning_size(size)`` will ask the camera for; nothing is applied.

        The same arithmetic the apply uses, answered from the saved region
        and the camera's limits, so a display can show the new frame beside
        the new binning while the apply is still running. The camera's own
        answer (its grid, its minimum at the new binning) can differ; the
        saved frame after the apply is the truth.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected.
            CameraSettingUnsupportedError: ``'binning_unsupported'``, this
                camera does not offer ``size``, as ``set_binning_size`` refuses
                it.
        """
        self._refuse_unusable_binning(size, 'frame_at_binning')
        return self._target_frame(self._native_frame(), size)

    def _refuse_unusable_binning(self, size: int, member: str) -> None:
        """Refuse a binning with no camera to take it, or one this camera does not offer.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera and ``member``, with none connected.
            CameraSettingUnsupportedError: ``size`` is not one of
                ``capabilities.camera_binning_sizes``.
        """
        # Asked before the offered sizes: a scope with no camera offers
        # none, and "does not support" would name the wrong cause.
        self.refuse_camera_not_connected(member)
        offered = self._scope.capabilities.camera_binning_sizes
        if size not in offered:
            raise CameraSettingUnsupportedError(
                'binning',
                size,
                offered,
                message=(
                    f'This camera does not support {binning.binning_size_int_to_str(size)} binning.'
                ),
            )

    def _native_frame(self) -> dict:
        """The saved unbinned region.

        Returned as it is saved, never re-capped: a small reading during a
        reconnect would otherwise shrink the region for good. The saved frame
        always carries it: the shipped settings hold it, settings preparation
        adds it to an older file, and bring-up saves one that gives the frame
        the camera delivered.
        """
        frame = self._scope.read_setting('frame')
        return {'width': int(frame['native_width']), 'height': int(frame['native_height'])}

    def _target_frame(self, native: dict, factor: int) -> dict:
        """The displayed frame a saved ``native`` region gives at a new ``factor``.

        The region divided by the binning and floored to even sides, so it
        follows from the region alone. It is raised to the camera's
        minimum: a Pylon camera floors only to its maximum and refuses a
        smaller request outright, and no one asked for this frame -- a
        binning change derives it, so there is no request to refuse. A frame
        a person asks for is refused out of range instead (``set_frame_size``).
        """
        target = binning.native_to_displayed(native, factor, self.get_pixel_alignment())
        limits = self._scope.capabilities.camera_limits
        minimum = limits.frame_min if limits else None
        if minimum is not None:
            target = {
                'width': max(target['width'], minimum.width),
                'height': max(target['height'], minimum.height),
            }
        return target

    def _refuse_frame_out_of_range(self, target: dict) -> None:
        """Refuse a displayed frame outside the camera's range at the binning it holds.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected; ``'scope_disconnected'`` after
                ``disconnect()``.
            CameraSettingOutOfRangeError: A width or height below the
                camera's minimum frame, or above the scope's maximum at the
                current binning.
        """
        # The range is the connected camera's: with none, or the scope
        # disconnected, that is the answer, not a range it no longer has.
        self.refuse_camera_not_connected('set_frame_size')
        limits = self._scope.capabilities.camera_limits
        minimum = limits.frame_min if limits else None
        maximum = limits.frame_max if limits else None
        factor = self._binning_size
        for axis in ('width', 'height'):
            self._refuse_out_of_range(
                f'frame_{axis}',
                target[axis],
                getattr(minimum, axis) if minimum else None,
                getattr(maximum, axis) // factor if maximum else None,
                noun=f'frame {axis}',
                unit='px',
            )

    def _apply_frame_impl(self, native: dict, target: dict) -> dict | None:
        """On the camera lane: apply the displayed ``target``, then save it and the
        ``native`` region it came from.

        A frame the camera already delivers is not written again, but is
        saved with its region; the lane task asked for the camera first, so
        nothing is saved for a camera that is not there.
        """
        if target == self.frame_size_cached:
            delivered = target
        else:
            self._refuse_frame_out_of_range(target)
            delivered = self._set_frame_size_impl(target['width'], target['height'])
            if delivered is None:
                return None
        self._scope.store_settings(
            {
                'frame.native_width': int(native['width']),
                'frame.native_height': int(native['height']),
                'frame.width': int(delivered['width']),
                'frame.height': int(delivered['height']),
            }
        )
        return dict(delivered)

    def _delivered_geometry(self, meant: dict) -> dict:
        """The saved geometry keys for the binning and frame the camera holds.

        Asked after the camera's geometry was written directly -- by
        bring-up, which may run at a binning or frame other than the saved
        one, and by the restore. ``meant`` holds the ``_STORED_GEOMETRY``
        keys the writer meant (the saved ones, or the snapshot's). The
        binning and frame are the camera's, saved together, since a frame
        beside a binning the camera is not at describes a region the sensor
        does not have. The region meant stays where it gives the delivered
        frame at that binning (``_target_frame``, the arithmetic a binning
        change uses, raised to the camera's minimum), else it is the
        delivered frame times the binning, so the next binning change is
        worked out from what the sensor holds: asked whether or not the
        binning and frame changed, since a settings file saved before the
        region existed takes the shipped one, which need not give its frame.
        The frame is the camera's reading, not the cache's 0x0 seed: a camera
        that never reported one gives the binning alone, and no frame or
        region is saved.
        """
        delivered = self._get_frame_size()
        factor = self.get_binning_size()
        held = {'binning.size': binning.binning_size_int_to_str(factor)}
        if delivered is None:
            return held
        held['frame.width'] = int(delivered['width'])
        held['frame.height'] = int(delivered['height'])
        native = {'width': meant['frame.native_width'], 'height': meant['frame.native_height']}
        if self._target_frame(native, factor) != delivered:
            native = {'width': delivered['width'] * factor, 'height': delivered['height'] * factor}
        held['frame.native_width'] = int(native['width'])
        held['frame.native_height'] = int(native['height'])
        return held

    def _store_delivered_geometry(self) -> None:
        """Save the binning and frame bring-up's camera took (``_delivered_geometry``).

        Bring-up applies the saved binning and frame to the camera directly.
        A binning the camera does not offer is replaced by the one it
        reports, and a saved frame above the scope's maximum is refitted to
        it, so what the camera delivers can differ from what was saved;
        saving the delivered pair keeps the settings and every reader of them
        on the geometry the camera actually holds. With no camera connected
        nothing was delivered, so nothing is saved.
        """
        if not self._scope.camera_connected:
            return
        saved = {path: self._scope.read_setting(path) for path in _STORED_GEOMETRY}
        self._store_changed(self._delivered_geometry(saved))

    def _store_changed(self, updates: dict[str, object]) -> None:
        """Save each of ``updates`` whose saved value differs, in one batch; nothing when none does."""
        changed = {
            path: value
            for path, value in updates.items()
            if self._scope.read_setting(path) != value
        }
        if changed:
            logger.info(f'[SCOPE API ] the camera holds {changed}; saved')
            self._scope.store_settings(changed)

    def _max_frame_unbinned(self) -> dict | None:
        """The largest frame the scope delivers, unbinned
        (``capabilities.camera_limits.frame_max``), or None when it is unknown.
        """
        limits = self._scope.capabilities.camera_limits
        size = limits.frame_max if limits else None
        if size is None:
            return None
        return {'width': size.width, 'height': size.height}

    def _set_frame_size_impl(self, w: int, h: int) -> dict | None:
        """Set the camera frame size in pixels.

        Success is observed by receiving the DELIVERED geometry; rejection
        is a raise. A caller therefore cannot record a rejected or clamped
        apply as current geometry -- the only value available to record is
        the one the camera actually took (a rejected resize once left the
        UI, FOV math, and saved settings claiming a size the camera never
        held, and the retry was absorbed by a dedupe record built from the
        request).

        Args:
            w: Frame width in pixels.
            h: Frame height in pixels.

        Returns:
            dict | None: The DELIVERED ``{'width', 'height'}`` -- the
                clamped/snapped geometry actually in effect, which may
                differ from the request. None when no camera is active (an
                outside caller is refused before this runs).

        Raises:
            CameraSettingRejected: A live driver rejected the apply. It
                carries the words its reporter shows; nothing is shown here.
        """

        if not self._driver or not self._driver.active:
            return None

        def _refused() -> CameraSettingRejected:
            return self._camera_setting_rejection(
                'frame_size',
                {'width': w, 'height': h},
                f'The camera did not accept the frame size {w}x{h}. '
                f'It remains at the previous size -- try again, or check '
                f'the USB connection if this repeats.',
            )

        # A frame-size change reallocates buffers; the pipeline must flush, so
        # invalidate unconditionally. Cache the DELIVERED size, not the request:
        # a driver may clamp or snap the request to its legal grid
        # (oversize-then-crop, alignment multiples), and FOV / pixel-size math
        # reads this cache, so it must reflect what the camera actually
        # delivers. The delivered geometry comes from the write's own return
        # value -- a separate get_frame_size() read-back is deliberately
        # avoided because a transient read error there can spuriously drop the
        # camera. A falsy result means the WRITE failed, so the prior cache
        # entry (still describing the hardware) is left in place.
        delivered = self._camera_write(
            lambda: self._driver.set_frame_size(w, h),
            member='set_frame_size',
            refusal=_refused,
            force_invalidate=('frame_size',),
        )
        if not delivered:
            raise _refused()
        delivered_size = {'width': int(delivered['width']), 'height': int(delivered['height'])}
        self._commit_camera_writes({'frame_size': dict(delivered_size)})
        return delivered_size

    def _set_binning_size_impl(self, size: int) -> bool:
        """Set camera pixel binning size.

        Args:
            size: Binning factor (1 = no binning, 2 = 2x2, etc.).

        Returns:
            bool: True when the driver applied the binning. False only
                when the camera is absent (an outside caller is refused
                before this runs).

        Raises:
            CameraSettingRejected: A live driver rejected the apply or
                raised from it (chained). A raise rather than a status,
                so a rejected binning cannot be
                recorded as current by a caller that drops the return --
                a rejected binning silently poisons every native-ROI /
                FOV / stitch derivation built on the recorded factor.
        """
        if not self._driver or not self._driver.active:
            return False

        def _refused() -> CameraSettingRejected:
            return self._camera_setting_rejection(
                'binning',
                size,
                f'The camera did not accept {size}x{size} binning. It remains '
                f'at the previous binning -- try again, or check the USB '
                f'connection if this repeats.',
            )

        # Binning realloc only takes effect when the driver applied it,
        # so the invalidate is gated on the driver result.
        # The factor is committed to the cache only when the driver
        # applied it. A rejected write must not leave the requested
        # value where scale-bar / FOV math reads it -- the hardware
        # is still at the previous binning.
        self._camera_write(
            lambda: self._driver.set_binning_size(size=size),
            member='set_binning_size',
            refusal=_refused,
            invalidates=('binning',),
            cache_update={'binning': int(size)},
        )
        # Both cached geometries are binning-dependent: the sensor
        # minimum halves at 2x, and the delivered frame size halves
        # with it. Left stale, the UI's frame-size clamp reads the 1x
        # minimum and FOV math reads the 1x frame size until the next
        # full hardware refresh. set_binning_size returns no geometry,
        # so this refresh reads the getters -- a deliberate exception
        # to the no-read-back preference, accepted because binning
        # changes are rare user actions, not per-frame traffic. The
        # validated-read path keeps a failed refresh from caching a
        # sentinel (the prior geometry stays in place).
        self._live_validated_read(
            'min_frame_size',
            lambda driver: driver.get_min_frame_size(),
            common_utils.is_valid_frame_size,
            lambda v: {'width': int(v['width']), 'height': int(v['height'])},
        )
        self._get_frame_size()
        _api_log.info(f'set_binning {size}x{size} -> True')
        return True

    def _set_pixel_format_impl(self, pixel_format: str) -> bool:
        """Set the camera pixel format.

        Args:
            pixel_format: Format string (e.g. 'Mono8', 'Mono12').

        Returns:
            bool: True when the driver applied the format. False only
                when the camera is absent / inactive (an outside caller is
                refused before this runs).

        Raises:
            CameraSettingRejected: A live driver rejected the format
                (unsupported) or raised from the apply (chained). A raise
                rather than a status, so a caller that drops the
                return cannot record a rejected format as current --
                capture depth, saved-file tagging, and data-rate math all
                key off the recorded format.
        """
        if not self._driver or not self._driver.active:
            return False

        def _refused() -> CameraSettingRejected:
            return self._camera_setting_rejection(
                'pixel_format',
                pixel_format,
                f'The camera did not accept the pixel format {pixel_format}. '
                f'It remains at the previous format -- try again, or check '
                f'the USB connection if this repeats.',
            )

        # Format change reallocates geometry only when the driver applied
        # it, so invalidate + snapshot are gated on the driver result.
        self._camera_write(
            lambda: self._driver.set_pixel_format(pixel_format),
            member='set_pixel_format',
            refusal=_refused,
            invalidates=('pixel_format',),
            cache_update={'pixel_format': pixel_format},
        )
        return True

    def _set_conversion_gain_mode_impl(self, mode: str) -> bool:
        """Set the camera sensor conversion gain mode.

        High conversion gain lowers the sensor read-noise floor (better
        low-light signal-to-noise) at the cost of dynamic range; Low is
        the standard wide-range mode. A camera without the mode
        (``capabilities.camera_supports_conversion_gain_mode``) is already
        at Low, so Low is its state rather than a write.

        Args:
            mode: 'High' (low noise) or 'Low' (wide dynamic range).

        Returns:
            bool: True when the camera is in ``mode``. False only when the
                camera is absent (an outside caller is refused before this
                runs).

        Raises:
            CameraSettingUnsupportedError: ``'High'`` on a camera without
                the mode, or a mode that is neither ``'High'`` nor ``'Low'``.
                Nothing reaches the camera.
            CameraSettingRejected: The driver refused the change or raised
                from it (chained).
        """
        if not self._driver or not self._driver.active:
            return False
        if not self._scope.capabilities.camera_supports_conversion_gain_mode:
            if mode == 'Low':
                return True
            raise _absent_mode_refusal(
                'conversion_gain_mode',
                mode,
                'high conversion gain',
                offered=('Low',),
                consequence='It stays at its standard conversion gain.',
            )
        if mode not in ('High', 'Low'):
            raise CameraSettingUnsupportedError(
                'conversion_gain_mode',
                mode,
                offered=('High', 'Low'),
                message=f'There is no conversion gain mode {mode!r}: the modes are High and Low.',
            )
        self._camera_write(
            lambda: self._driver.set_conversion_gain_mode(mode),
            member='set_conversion_gain_mode',
            refusal=lambda: _mode_rejection('conversion_gain_mode', mode, 'conversion gain mode'),
            invalidates=('conversion_gain_mode',),
        )
        return True

    def _set_line_noise_reduction_impl(self, enabled: bool) -> bool:
        """Enable or disable the camera line-noise reduction filter.

        A camera-side filter that smooths horizontal stripe artifacts in
        the sensor readout. A camera without it
        (``capabilities.camera_supports_line_noise_reduction``) is already
        unfiltered, so off is its state rather than a write.

        Args:
            enabled: True turns the filter on; False off.

        Returns:
            bool: True when the filter is as asked. False only when the
                camera is absent (an outside caller is refused before this
                runs).

        Raises:
            CameraSettingUnsupportedError: Turning the filter on for a camera
                without it. Nothing reaches the camera.
            CameraSettingRejected: The driver refused the change or raised
                from it (chained).
        """
        if not self._driver or not self._driver.active:
            return False
        if not self._scope.capabilities.camera_supports_line_noise_reduction:
            if not enabled:
                return True
            raise _absent_mode_refusal(
                'line_noise_reduction',
                enabled,
                'line noise reduction',
                offered=(False,),
                consequence='Its readout stays unfiltered.',
            )
        self._camera_write(
            lambda: self._driver.set_line_noise_reduction(enabled=enabled),
            member='set_line_noise_reduction',
            refusal=lambda: _mode_rejection(
                'line_noise_reduction', enabled, 'line noise reduction'
            ),
            invalidates=('line_noise_reduction',),
        )
        return True

    @api(reads=True)
    def get_black_level(self) -> float | None:
        """Read the camera's black level, live.

        The value is the camera's own black level parameter, in the camera's
        own units, as the vendors' EMVA 1288 sheets state it: on a Basler
        body one unit moves the gray value by a model-specific step (0.0625 DN
        at 12-bit depth on the daA3840), on an IDS body it is DN of the current
        pixel format, on the FX2 it is the sensor's Row Black Target. The
        offset it makes in DN is measured, not derived from it.

        Not cached: a failed read has no last-known-good to answer with.

        Returns:
            float | None: The black level; None when no camera is active or
                the camera reports none.

        Raises:
            HardwareError: The camera reports a black level and the read
                failed.
        """
        driver = self._driver
        if not driver or not driver.active:
            return None
        return driver.get_black_level()

    @api(reads=True)
    def get_black_level_range(self) -> tuple[float, float] | None:
        """Read the range ``set_black_level`` accepts, live, in the units of
        ``get_black_level``.

        The camera's own minimum and maximum for its current pixel format;
        the range changes with the format. ``set_black_level`` refuses a value
        outside it before anything reaches the camera.

        Not cached: a failed read has no last-known-good to answer with.

        Returns:
            tuple[float, float] | None: ``(minimum, maximum)``; None when no
                camera is active or the camera has no black level setting.
                A camera holding its black level automatically still answers
                its range; ``set_black_level`` is then refused by the camera.

        Raises:
            HardwareError: The camera has a black level setting and the read
                failed.
        """
        driver = self._driver
        if not driver or not driver.active:
            return None
        return driver.get_black_level_range()

    @api(reads=True)
    def get_resulting_frame_rate(self) -> float | None:
        """Read the frame rate the camera reports its current settings allow,
        live, in frames per second.

        The camera's own figure: Basler's resulting acquisition frame rate,
        IDS's AcquisitionFrameRate maximum, the simulator's pacing rate. The
        FX2 reports none. It is what the camera says it can do, not what
        reaches the host; what reaches the host is ``get_delivered_rate``.

        Not cached: a failed read has no last-known-good to answer with.

        Returns:
            float | None: Frames per second; None when no camera is active or
                the camera reports none.

        Raises:
            HardwareError: The camera reports a frame rate and the read
                failed.
        """
        driver = self._driver
        if not driver or not driver.active:
            return None
        return driver.get_resulting_frame_rate()

    @api
    def set_black_level(self, value: float) -> float | None:
        """Set the camera's black level, wait for it, and answer with the
        value in effect.

        See ``get_black_level`` for what the value means. The value holds
        until the next connect, when the camera reloads its default user set.

        Returns:
            float | None: The black level now in effect, which differs from
                the request when the camera snapped it.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected or one removed during the write.
            CameraSettingUnsupportedError: This camera has no black level
                setting (``capabilities.camera_supports_black_level`` is
                False). Nothing reached the camera.
            ArgumentRefusedError: ``'not_a_number'``, the value is not a
                finite number; nothing was sent to the camera.
            CameraSettingOutOfRangeError: The value is outside the range the
                camera reports for its current pixel format; nothing was sent.
            CameraSettingRejected: The camera refused it, as a camera holding
                its black level automatically does, or raised from the write
                (chained).
            HardwareError: The camera's probe, range read or read-back
                failed; the black level in effect is unknown.
        """
        return self._dispatch_camera(
            self._set_black_level_impl,
            'set_black_level',
            args=(value,),
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
        )

    def _set_black_level_impl(self, value: float) -> float | None:
        """Set the black level; the refusals of ``set_black_level`` are raised
        here, on the camera lane, because the support probe and the range are
        live camera reads. A write that did not apply raises
        ``CameraSettingRejected`` (``_camera_write``).
        """
        driver = self._driver
        if not driver or not driver.active:
            return None
        if not driver.supports_black_level():
            raise _absent_mode_refusal(
                'black_level',
                value,
                'black level setting',
                offered=(),
                consequence='The black level stays as it is.',
            )
        low, high = driver.get_black_level_range()
        self._refuse_out_of_range('black_level', value, low, high, noun='black level', unit='units')

        def _write_black_level():
            with self._cam_lock:
                return self._driver.set_black_level(float(value))

        return self._camera_write(
            _write_black_level,
            member='set_black_level',
            refusal=lambda: _value_rejection('black_level', value),
            invalidates=('black_level',),
        )

    def _live_validated_read(
        self,
        key: str,
        reader: Callable[[Camera], object],
        is_valid: Callable[[object], bool],
        coerce: Callable[[Any], Any],
    ) -> object | None:
        """Attempt one live driver read: validate, commit to cache, return.

        Returns None when no camera is active or the read raised or came
        back as the driver's failure sentinel (-1 / None / {}). The commit
        is generation-guarded: if an authoritative write
        (_commit_camera_writes) landed while the driver read was in
        flight, the stale read is discarded and the newer cache value is
        returned instead, so a slow read can never overwrite a setter's
        write-through.

        Callers decide what None means: the value getters fall back to
        the cached last-known-good; the live-confirmed consumers
        (metadata writers, state snapshots) omit the field, because for
        them a stale value recorded as truth is worse than no value.
        """
        driver = self._driver
        if not driver or not driver.active:
            return None
        with self._camera_cache_lock:
            gen_before = self._camera_cache_write_gen.get(key, 0)
        try:
            live = reader(driver)
        except Exception as ex:
            self._log_camera_read_failure(key, ex)
            return None
        if not is_valid(live):
            self._log_camera_read_failure(key, f'sentinel value {live!r}')
            return None
        value = coerce(live)
        with self._camera_cache_lock:
            if self._camera_cache_write_gen.get(key, 0) == gen_before:
                self._camera_cache[key] = value
            else:
                value = self._camera_cache[key]
        return value

    def _validated_camera_read(
        self,
        key: str,
        reader: Callable[[Camera], object],
        is_valid: Callable[[object], bool],
        coerce: Callable[[Any], Any],
        absent: object,
    ) -> object:
        """Refresh-then-read: the single path a public camera getter answers by.

        Attempts the live driver read on every call (so a getter tracks
        hardware while auto-gain moves it), commits the result to the camera
        cache only when it validates, and answers from the cache either way.
        A driver failure sentinel (-1 / None / {}) or a raising read can
        therefore never cross this boundary: consumers receive the
        last-known-good value, or ``absent`` (the documented camera-absent
        default) when no camera is active or no valid value has ever been
        read.

        Args:
            key: Camera-cache key backing this getter.
            reader: Live read, called with the resolved driver.
            is_valid: Shared validity predicate for this value.
            coerce: Normalizes a validated live value for caching.
            absent: Documented return when no camera is active or no valid
                value is known.

        Returns:
            The validated live value, the cached last-known-good, or
            ``absent``.
        """
        driver = self._driver
        if not driver or not driver.active:
            return absent
        live = self._live_validated_read(key, reader, is_valid, coerce)
        if live is not None:
            return live
        with self._camera_cache_lock:
            cached = self._camera_cache[key]
        return cached if is_valid(cached) else absent

    def _grab_sizing_exposure_s(self) -> float:
        """The exposure in seconds, for sizing a grab's wait and a frame's cost.

        Read live, so a mid-window exposure change sizes the next grab. 0
        when the camera has never reported one: every wait sized from it
        has a floor of its own, and with no exposure to scale it is the
        floor that sizes it.
        """
        exposure_ms = self.get_exposure_ms()
        return exposure_ms / 1000 if exposure_ms is not None else 0.0

    @api(reads=True)
    def get_live_camera_settings(self) -> dict:
        """Live-confirmed camera settings, omitting any field whose read
        did not just succeed.

        For consumers that record what the hardware was at a specific
        moment (saved-image metadata, state snapshots): the value getters
        (get_gain_db / get_exposure_ms) deliberately hide read failures
        behind the last-known-good cache, which is right for control flow
        but would record a value the frame was not captured at. Here,
        unknown stays unknown.

        Returns:
            dict: Any of 'gain_db', 'exposure_ms', 'frame_size'
                (``{'width': int, 'height': int}``), and 'pixel_format' --
                only fields whose live driver read succeeded and
                validated. Empty when no camera is active.
        """
        settings = {}
        gain = self._live_validated_read(
            'gain_db',
            lambda driver: driver.get_gain(),
            common_utils.is_valid_gain_db,
            float,
        )
        if gain is not None:
            settings['gain_db'] = gain
        exposure = self._live_validated_read(
            'exposure_ms',
            lambda driver: driver.get_exposure_t(),
            common_utils.is_valid_exposure_ms,
            float,
        )
        if exposure is not None:
            settings['exposure_ms'] = exposure
        frame_size = self._live_validated_read(
            'frame_size',
            lambda driver: driver.get_frame_size(),
            common_utils.is_valid_frame_size,
            lambda v: {'width': int(v['width']), 'height': int(v['height'])},
        )
        if frame_size is not None:
            settings['frame_size'] = dict(frame_size)
        pixel_format = self._live_validated_read(
            'pixel_format',
            lambda driver: driver.get_pixel_format(),
            common_utils.is_valid_pixel_format,
            str,
        )
        if pixel_format is not None:
            settings['pixel_format'] = pixel_format
        return settings

    @api(reads=True)
    def get_gain_db(self) -> float | None:
        """Get the current camera gain.

        Returns:
            float | None: Gain in dB -- the live reading when it succeeds,
                else the last-known-good value. None when no camera is
                active or gain has never been read.
        """
        return self._validated_camera_read(
            'gain_db',
            lambda driver: driver.get_gain(),
            common_utils.is_valid_gain_db,
            float,
            None,
        )

    @api(reads=True)
    def get_exposure_ms(self) -> float | None:
        """Get the current camera exposure time.

        Returns:
            float | None: Exposure time in milliseconds -- the live reading
                when it succeeds, else the last-known-good value. None when
                no camera is active or exposure has never been read.
        """
        return self._validated_camera_read(
            'exposure_ms',
            lambda driver: driver.get_exposure_t(),
            common_utils.is_valid_exposure_ms,
            float,
            None,
        )

    def _get_frame_size(self) -> dict | None:
        """Get the current camera frame size.

        Returns:
            dict | None: Contains 'width' and 'height' in pixels -- the live
                reading when it succeeds, else the last-known-good value.
                None when no camera is active or the size has never been
                read.
        """
        frame_size = self._validated_camera_read(
            'frame_size',
            lambda driver: driver.get_frame_size(),
            common_utils.is_valid_frame_size,
            lambda v: {'width': int(v['width']), 'height': int(v['height'])},
            None,
        )
        return dict(frame_size) if frame_size is not None else None

    def _get_pixel_format(self) -> str | None:
        """Get the current camera pixel format.

        Returns:
            str | None: Pixel format string (e.g. 'Mono8') -- the live
                reading when it succeeds, else the last-known-good value
                (seeded by connect-time configuration and the
                set_pixel_format write-through). None when no camera is
                active or no format is known.
        """
        return self._validated_camera_read(
            'pixel_format',
            lambda driver: driver.get_pixel_format(),
            common_utils.is_valid_pixel_format,
            str,
            None,
        )

    def get_width(self) -> int | None:
        """Get the current frame width setting.

        Returns:
            int | None: Current width in pixels -- last-known-good when the
                live read fails. None when no camera is active or the size
                has never been read.
        """
        frame_size = self._get_frame_size()
        return int(frame_size['width']) if frame_size else None

    def get_height(self) -> int | None:
        """Get the current frame height setting.

        Returns:
            int | None: Current height in pixels -- last-known-good when the
                live read fails. None when no camera is active or the size
                has never been read.
        """
        frame_size = self._get_frame_size()
        return int(frame_size['height']) if frame_size else None

    @api(reads=True)
    def get_binning_size(self) -> int:
        """Get the current camera binning size.

        Returns:
            int: Current binning factor, always >= 1 -- last-known-good when
                the live read fails, 1 when no camera is active. The driver's
                -1 failed-read sentinel never surfaces here: a -1 would flow
                into frame-geometry arithmetic (native size = displayed *
                binning) as a sign flip.
        """
        return self._validated_camera_read(
            'binning',
            lambda driver: driver.get_binning_size(),
            common_utils.is_valid_binning_size,
            int,
            1,
        )

    @api(reads=True)
    def get_pixel_alignment(self) -> dict:
        """Return the camera's deliverable frame-size granularity.

        Every camera delivers the size asked for exactly, by acquiring the next
        window up on its grid and cropping back, so the only constraint is even
        dimensions (H.264): ``{2, 2}``, the camera profile's step.

        Returns:
            dict: ``{'width': int, 'height': int}``.
        """
        default = {'width': 2, 'height': 2}
        if not self._driver or not self._driver.active:
            return default
        try:
            return dict(self._driver.profile.alignment)
        except (AttributeError, TypeError):
            return default

    # --- State / lifecycle properties ---
    @property
    def active_cached(self) -> bool:
        """Whether the camera is connected and active (reads cache).

        Returns:
            bool: True if the camera is currently active.
        """
        with self._camera_cache_lock:
            return self._camera_cache['active']

    @api
    @property
    def gain_db_cached(self) -> float:
        """Current camera gain in dB (reads cache).

        Returns:
            float: Cached gain value in dB.
        """
        with self._camera_cache_lock:
            return self._camera_cache['gain_db']

    @api
    @property
    def longest_exposure_ms(self) -> float | None:
        """The longest exposure the camera may be using now, in ms.

        While continuous auto-gain is armed the camera picks its own
        exposure up to the ceiling the arm was given (the sensor maximum
        when it was given none), and the cached exposure is the value from
        before the arm; otherwise the cached exposure is the one in force.
        A consumer bounding how long a frame may take -- a feed-death
        watch -- needs the former, not the cache.

        Returns:
            float | None: The bound in ms; None when the arm has no ceiling
                and no camera reports a maximum.
        """
        with self._state_lock:
            arm = self._auto_gain_arm
        if arm is None:
            return self.exposure_ms_cached
        ceiling = arm.settings.get('max_exposure_ms')
        if ceiling:
            return float(ceiling)
        limits = self._scope.capabilities.camera_limits
        return limits.exposure_ms_max if limits else None

    @api
    @property
    def exposure_ms_cached(self) -> float:
        """Current camera exposure time in ms (reads cache).

        Returns:
            float: Cached exposure time in milliseconds.
        """
        with self._camera_cache_lock:
            return self._camera_cache['exposure_ms']

    @api
    @property
    def frame_size_cached(self) -> dict:
        """Current camera frame size as {'width': int, 'height': int} (reads cache).

        Returns:
            dict: Copy of the cached frame size dict.
        """
        with self._camera_cache_lock:
            return dict(self._camera_cache['frame_size'])

    @api
    @property
    def min_frame_size_cached(self) -> dict | None:
        """``capabilities.camera_limits.frame_min`` as ``{'width', 'height'}``,
        or None when it is unknown or no camera is connected.
        """
        limits = self._scope.capabilities.camera_limits
        if limits is None or limits.frame_min is None:
            return None
        return {'width': limits.frame_min.width, 'height': limits.frame_min.height}

    @api
    @property
    def max_exposure_ms_cached(self) -> float | None:
        """``capabilities.camera_limits.exposure_ms_max``, or None when it is
        unknown or no camera is connected.
        """
        limits = self._scope.capabilities.camera_limits
        return limits.exposure_ms_max if limits else None

    @api
    @property
    def max_gain_db_cached(self) -> float | None:
        """``capabilities.camera_limits.gain_db_max``, or None when it is
        unknown or no camera is connected.
        """
        limits = self._scope.capabilities.camera_limits
        return limits.gain_db_max if limits else None

    @api
    @property
    def min_gain_db_cached(self) -> float | None:
        """``capabilities.camera_limits.gain_db_min``, or None when the camera
        declares none or no camera is connected: a missing floor is not a
        floor of zero.
        """
        limits = self._scope.capabilities.camera_limits
        return limits.gain_db_min if limits else None

    @api
    @property
    def min_exposure_ms_cached(self) -> float | None:
        """``capabilities.camera_limits.exposure_ms_min``, or None when the
        camera declares none or no camera is connected."""
        limits = self._scope.capabilities.camera_limits
        return limits.exposure_ms_min if limits else None

    @api(reads=True)
    def applied_gain_db_for(self, stored_gain_db: float) -> AppliedCameraSetting:
        """What a stored gain becomes on the attached camera.

        The one place the gain cap is applied. A caller that narrows a
        stored value itself -- against this cap or against a widget's
        range -- is a second answerer, and the store it writes back is
        how a user's setting gets destroyed by connecting a smaller body.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, the gain is not a
                finite number.
        """
        refuse_unless_finite_number(stored_gain_db, 'stored_gain_db')
        limits = self._scope.capabilities.camera_limits
        return cap_stored_value(stored_gain_db, limits.gain_db_max if limits else None)

    @api(reads=True)
    def applied_exposure_ms_for(self, stored_exposure_ms: float) -> AppliedCameraSetting:
        """What a stored exposure becomes on the attached camera.

        See ``applied_gain_db_for``; the same contract for the other
        quantity, so both travel the same path to hardware and to display.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, the exposure is not a
                finite number.
        """
        refuse_unless_finite_number(stored_exposure_ms, 'stored_exposure_ms')
        limits = self._scope.capabilities.camera_limits
        return cap_stored_value(stored_exposure_ms, limits.exposure_ms_max if limits else None)

    @api(reads=True)
    def applied_auto_gain_for(self, stored_auto_gain: bool) -> AppliedCameraSetting:
        """What a stored auto-gain preference becomes on the attached camera.

        See ``applied_gain_db_for``; the same contract for the mode. A camera
        without hardware auto-gain runs manual whatever the layer stored, and
        the stored preference is left alone so a camera that has the mode
        takes it again.
        """
        stored = bool(stored_auto_gain)
        applied = stored and self._camera_has_auto_gain()
        return AppliedCameraSetting(stored=stored, applied=applied, capped=applied != stored)

    def _camera_has_auto_gain(self) -> bool:
        """Whether the attached camera has hardware auto-gain
        (``capabilities.camera_supports_auto_gain``). The IDS and FX2 drivers
        have no auto-gain to drive, so nothing may ask them to change it.
        """
        return self._scope.capabilities.camera_supports_auto_gain

    def _camera_has_auto_exposure(self) -> bool:
        """Whether the attached camera has hardware auto-exposure
        (``capabilities.camera_supports_auto_exposure``)."""
        return self._scope.capabilities.camera_supports_auto_exposure

    @api
    @property
    def pixel_format_cached(self) -> str | None:
        """Current camera pixel format (e.g. 'Mono8', 'Mono12') (reads cache).

        Returns:
            str | None: Cached pixel format string, or None when no format
                has ever been successfully read or configured. A failed
                driver read never overwrites this entry
                (validate-before-store at cache population), so a known
                format survives transient read failures.
        """
        with self._camera_cache_lock:
            return self._camera_cache['pixel_format']

    # --- Save / restore ---
    def save_camera_state(self, tag: str) -> dict:
        """Snapshot every camera setting with a getter, for restoration.

        Omit-if-unknown: a field enters the snapshot only when a usable
        value exists (the getters answer last-known-good, so a missing
        field means the value was NEVER successfully read). Restore can
        therefore trust every field it finds, and name the ones it
        cannot restore. ``auto_gain_arm`` is always present: the standing
        continuous arm, or None when none stood -- a run that ends with
        its step-end disarm otherwise leaves the loop off while the layer
        toggle shows on, until a slider write happens to re-arm it.

        With a camera active the snapshot also holds its frame size and
        pixel format, its binning where it offers more than one size, and its
        black level where that can be set (``capabilities``). These are read
        live, and a failed read raises: the camera cannot report its own
        setting, and a restore that left it out would leave the scope changed
        without saying so. With them, ``stored`` holds the saved image mode,
        binning, frame and unbinned region, which the restore saves back: three
        image modes share one format, so the mode cannot be read back from the
        camera. A scope no session composed has no settings to read them from.

        Args:
            tag: Descriptive name for the snapshot (for logging).

        Returns:
            dict: Snapshot suitable for passing to ``restore_camera_state``.

        Raises:
            HardwareError: A setting the camera offers could not be read.
            ConfigError: ``'scope_not_composed'``, no session bound this scope.
        """
        snapshot = {'tag': tag}
        snapshot.update(self._camera_geometry_now())
        gain_db = self.get_gain_db()
        if common_utils.is_valid_gain_db(gain_db):
            snapshot['gain_db'] = gain_db
        else:
            # The warning belongs HERE, not at restore: an absent field at
            # restore time can be a deliberate caller trim (autofocus keeps
            # the fields its run explicitly targeted), but at save time a
            # missing value has exactly one meaning -- the camera never
            # reported it -- and a later partial restore would otherwise be
            # untraceable.
            logger.warning(
                f'[SCOPE API ] save_camera_state tag={tag}: gain has never '
                f'been successfully read; snapshot omits it and the restore '
                f'will leave gain unchanged'
            )
        exposure_ms = self.get_exposure_ms()
        if common_utils.is_valid_exposure_ms(exposure_ms):
            snapshot['exposure_ms'] = exposure_ms
        else:
            logger.warning(
                f'[SCOPE API ] save_camera_state tag={tag}: exposure has '
                f'never been successfully read; snapshot omits it and the '
                f'restore will leave exposure unchanged'
            )
        if snapshot.keys() & set(_GEOMETRY_ORDER):
            # The saved settings the restore puts back beside the camera's:
            # three image modes share one format, so the mode cannot be read
            # back from the camera, and the region is the saved intent.
            snapshot['stored'] = {
                path: self._scope.read_setting(path) for path in ('image_mode', *_STORED_GEOMETRY)
            }
        with self._state_lock:
            snapshot['auto_gain_arm'] = self._auto_gain_arm
            snapshot['held_layer'] = self._held_layer
        _api_log.info(
            f'save_camera_state tag={tag}: '
            f'gain={snapshot.get("gain_db", "never-read")} '
            f'exp={snapshot.get("exposure_ms", "never-read")} '
            f'arm={snapshot["auto_gain_arm"] is not None} '
            f'{_describe_geometry(snapshot)}'
        )
        return snapshot

    def _camera_geometry_now(self) -> dict:
        """The camera's restorable settings beyond gain and exposure, read live.

        Empty when no camera is active. Binning appears only where the camera
        offers more than one size, the black level only where it can be set.

        Raises:
            HardwareError: A setting the camera offers could not be read.
        """
        driver = self._driver
        if not driver or not driver.active:
            return {}
        caps = self._scope.capabilities
        now = {key: self._read_setting_live(key) for key in ('frame_size', 'pixel_format')}
        if len(caps.camera_binning_sizes) > 1:
            now['binning'] = self._read_setting_live('binning')
        if caps.camera_supports_black_level:
            black_level = self.get_black_level()
            if black_level is None:
                raise CameraReadFailedError("The camera's black level could not be read")
            now['black_level'] = black_level
        return now

    def _read_setting_live(self, key: str) -> object:
        """One live read of a restorable camera setting.

        Raises:
            HardwareError: The read failed or answered the driver's sentinel.
        """
        noun, reader, is_valid, coerce = _LIVE_SETTING_READS[key]
        value = self._live_validated_read(key, reader, is_valid, coerce)
        if value is None:
            raise CameraReadFailedError(f"The camera's {noun} could not be read")
        return value

    def restore_camera_state(self, snapshot: dict) -> None:
        """Restore every camera setting a saved state holds, and wait.

        See ``_restore_camera_state_impl`` for the contract; this adds the
        dispatch described on ``_dispatch_camera``, so the restore is one
        task on the camera lane and its writes cannot interleave with
        another caller's. A camera already declared removed when the restore
        starts has nothing to put back: it returns with nothing written, its
        removal already told by whatever noticed it.

        Args:
            snapshot: Return value from ``save_camera_state``.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera: none is connected and the snapshot holds a setting
                to restore, or the restore found it removed.
        """
        return self._dispatch_camera(
            self._restore_camera_state_impl,
            'restore_camera_state',
            args=(snapshot,),
            timeout_s=self._CAMERA_RESTORE_TIMEOUT_S,
            satisfied_when_absent=_DECIDED_IN_BODY,
        )

    def _restore_camera_state_impl(self, snapshot: dict) -> None:
        """Restore every camera setting a saved state holds.

        The binning, frame size, pixel format and black level go back first,
        in ``_GEOMETRY_ORDER``, each only where it differs from the camera's
        value now: an unchanged frame size, format or binning write would
        restart grabbing for nothing. Then gain, exposure and the arm.

        Fields absent from the snapshot are skipped and named in the log:
        either the caller deliberately trimmed them (autofocus keeps the
        values its run explicitly targeted) or they were never readable at
        save time -- save_camera_state already WARNed about the latter.
        The auto-gain loop goes back to the snapshot's state: re-armed after
        the setters when an arm was recorded; disarmed before them when none
        was and one stands now (a run aborted between a step's arm and its
        disarm), since a value written under the camera's loop is taken back
        by it; untouched when the field is absent. A re-arm clamps the
        exposure to the class ceiling like any arm.

        Args:
            snapshot: Return value from ``save_camera_state``.
        """
        if not snapshot:
            return
        tag = snapshot.get('tag', '?')
        # A removed camera has nothing to restore, and each setter would
        # report its absence again after the removal was already reported
        # by whatever noticed it -- a run's ending, the stall check.
        if self.camera_removed:
            _api_log.info(f'restore_camera_state tag={tag}: camera removed; nothing restored')
            return
        # Decided here rather than at the lane's door, because only the
        # snapshot says whether there is anything to restore: one taken with
        # no camera holds no setting and no arm, and restoring it is
        # satisfied with none connected.
        if not self._scope.camera_connected:
            if any(key in snapshot for key in ('gain_db', 'exposure_ms', *_GEOMETRY_ORDER)) or (
                snapshot.get('auto_gain_arm') is not None
            ):
                raise _camera_not_connected('restore_camera_state')
            return
        # An ABSENT field is a legitimate trim (skip quietly); a PRESENT
        # field that fails validation is a caller bug -- the sanctioned
        # producer only ever emits valid fields -- so that case warns.
        gain_db = snapshot.get('gain_db')
        gain_known = common_utils.is_valid_gain_db(gain_db)
        if not gain_known and 'gain_db' in snapshot:
            logger.warning(
                f'[SCOPE API ] restore_camera_state tag={tag}: snapshot '
                f'carries a non-physical gain ({gain_db!r}); gain left as-is'
            )
        exposure_ms = snapshot.get('exposure_ms')
        exposure_known = common_utils.is_valid_exposure_ms(exposure_ms)
        if not exposure_known and 'exposure_ms' in snapshot:
            logger.warning(
                f'[SCOPE API ] restore_camera_state tag={tag}: snapshot '
                f'carries a non-physical exposure ({exposure_ms!r}); '
                f'exposure left as-is'
            )
        # The log of record goes BEFORE the hardware writes: a setter may
        # raise a typed exception mid-restore, and the post-mortem then
        # needs the line stating what this restore was about to do (a
        # partial restore with no record once misattributed wrong-
        # brightness images to protocol settings).
        arm_recorded = 'auto_gain_arm' in snapshot
        arm = snapshot.get('auto_gain_arm')
        with self._state_lock:
            standing = self._auto_gain_arm
        if arm_recorded and arm is not None:
            arm_action = 're-armed'
        elif arm_recorded and standing is not None:
            arm_action = 'disarmed'
        else:
            arm_action = 'unchanged'
        _api_log.info(
            f'restore_camera_state tag={tag}: '
            f'gain={gain_db if gain_known else "skipped"} '
            f'exp={exposure_ms if exposure_known else "skipped"} '
            f'arm={arm_action} '
            f'{_describe_geometry(snapshot)}'
        )
        # The camera holds the snapshot's layer only once every one of its
        # settings is back; until then, and after a trimmed or refused restore,
        # it holds none.
        self._forget_held_layer()
        self._restore_camera_geometry_and_save(snapshot)
        writes = []
        if arm_action == 'disarmed':
            writes.append(lambda: self._set_auto_gain_impl(False, dict(standing.settings)))
        if gain_known:
            writes.append(lambda: self._set_gain_db_impl(gain_db))
        if exposure_known:
            writes.append(lambda: self._set_exposure_ms_impl(exposure_ms))
        if arm_action == 're-armed':
            writes.append(
                lambda: self._set_auto_gain_impl(
                    True, dict(arm.settings), resume_after_capture=arm.resume_after_capture
                )
            )
        # Each refusal is reported and the restore goes on to its next write,
        # so one setting the camera did not take leaves no other unrestored.
        refused = False
        for write in writes:
            try:
                write()
            except CameraSettingRejected as rejected:
                refused = True
                self._report_refused_write(rejected)
        if gain_known and exposure_known and arm_recorded and not refused:
            with self._state_lock:
                self._held_layer = snapshot.get('held_layer')

    def _restore_camera_geometry_and_save(self, snapshot: dict) -> None:
        """Write back the snapshot's settings beyond gain and exposure, then save them.

        The camera is written through the private bodies and then saved as
        bring-up saves (``_delivered_geometry``): where the camera took the
        snapshot's binning and frame, the snapshot's saved keys go back
        exactly, its image mode with them; where a write was refused part
        way, what the camera holds is saved, and the refusal is raised. Only
        a key whose value changes is written, so a run or an autofocus that
        changed nothing saves nothing.

        Raises:
            HardwareError: A setting could not be read to compare; what the
                camera holds is saved first.
            CameraSettingRejected: The camera refused binning, frame size or
                format; the restore stops there, as a setter's raise does,
                with what the camera holds saved.
        """
        stored = snapshot.get('stored')
        try:
            self._restore_camera_geometry(snapshot)
        except Exception:
            # Whatever stopped it -- a refused write, a failed compare read
            # after an earlier write landed -- the camera may have changed,
            # so what it holds is saved; a camera that left holds nothing.
            if stored is not None and self._scope.camera_connected:
                self._store_changed(self._delivered_geometry(stored))
            raise
        if stored is not None and self._scope.camera_connected:
            self._store_changed(
                {'image_mode': stored['image_mode'], **self._delivered_geometry(stored)}
            )

    def _restore_camera_geometry(self, snapshot: dict) -> None:
        """Write back each of the snapshot's settings beyond gain and exposure
        that differs from the camera's value now, in ``_GEOMETRY_ORDER``.

        Raises:
            HardwareError: A setting could not be read to compare.
            CameraSettingRejected: The camera refused binning, frame size or
                format; the restore stops there, as a setter's raise does.
        """
        for key in _GEOMETRY_ORDER:
            if key not in snapshot:
                continue
            wanted = snapshot[key]
            now = self.get_black_level() if key == 'black_level' else self._read_setting_live(key)
            if now == wanted:
                continue
            if key == 'binning':
                self._set_binning_size_impl(wanted)
            elif key == 'frame_size':
                self._set_frame_size_impl(wanted['width'], wanted['height'])
            elif key == 'pixel_format':
                self._set_pixel_format_impl(wanted)
            else:
                try:
                    self._set_black_level_impl(wanted)
                except CameraSettingRejected as rejected:
                    self._report_refused_write(rejected)

    # --- Camera config orchestration ---
    def hold_layer(self, layer: str, request: LayerCameraRequest) -> dict | None:
        """Put ``layer``'s stored settings on the camera unless it holds them, and wait.

        See ``_hold_layer_impl``; this adds the dispatch described on
        ``_dispatch_camera``. In-process only: the Session reads the layer's
        store and hands it in.
        """
        _refuse_unless_auto_gain_settings(request.auto_gain_settings, 'auto_gain_settings')
        return self._dispatch_camera(
            self._hold_layer_impl,
            'hold_layer',
            args=(layer, request),
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
        )

    def get_held_layer(self) -> str | None:
        """The layer whose stored settings the camera holds, read from the record.

        Returns:
            str | None: The layer the Session last put on the camera whole;
                None when the camera holds no layer's settings: a raw camera
                write came after it (``set_gain_db``, ``set_exposure_ms``,
                ``apply_layer_camera_settings`` and the other raw doors), a
                restore was trimmed or refused, the camera refused part of the
                layer, or no camera is connected.
        """
        with self._state_lock:
            held = self._held_layer
        if held is None or held.request is None:
            return None
        return held.layer

    def _holding(self, layer: str, request: LayerCameraRequest) -> _HeldLayer | None:
        """The record, when the camera holds ``layer`` at ``request``; else None."""
        with self._state_lock:
            held = self._held_layer
        if held is not None and held.layer == layer and held.request == request:
            return held
        return None

    def _hold_layer_for_capture(self, layer: str, request: LayerCameraRequest, member: str) -> None:
        """``_hold_layer_impl`` from inside a capture's own camera-lane task.

        The capture is already the task its taking admitted, so this is not
        admitted again against whatever holds the scope now: a run that took
        the scope after the capture was put on the lane would otherwise
        refuse the capture it waits for. A write falsifies a recording, so
        one needed while a recording runs is refused; a layer the camera
        already holds needs none and is never refused. With the camera
        holding no layer's settings (a raw camera write came last) nothing
        is written: the raw values stand, and the frame's record says so.

        Raises:
            HardwareCommandRefusedError: ``'exclusive_activity_running'``,
                naming the recording, when a write is needed while one runs;
                nothing was written.
            CameraSettingRejected: as ``_hold_layer_impl`` says.
        """
        with self._state_lock:
            held = self._held_layer
        if held is None or self._holding(layer, request) is not None:
            return
        claim = self._scope._activity_claim
        holder = claim.enter_falsifying_change(current_taking()) if claim is not None else None
        if holder is not None and holder.kind == 'recording':
            raise HardwareCommandRefusedError('exclusive_activity_running', member, holder.kind)
        # Any other holder took the scope after this capture was admitted;
        # while it holds, no recording can start, so the write is not counted.
        try:
            self._hold_layer_impl(layer, request)
        finally:
            if claim is not None and holder is None:
                claim.leave_falsifying_change()

    def _hold_layer_impl(self, layer: str, request: LayerCameraRequest) -> dict | None:
        """Put ``layer``'s stored settings on the camera unless it holds them.

        The one writer of the camera's held layer. Nothing is written when
        the camera already holds ``layer`` at this request: every camera
        write invalidates the frames, so a needless one costs the next
        capture a drain.

        Returns:
            The gain and exposure in effect, as
            ``_apply_layer_camera_settings_impl`` returns them; None when no
            camera is active.

        Raises:
            CameraSettingRejected: as ``_apply_layer_camera_settings_impl``
                says. The camera is recorded as holding ``layer`` at what it
                took, with no request, so the next ask writes again.
        """
        held = self._holding(layer, request)
        if held is not None:
            return held.taken
        self._forget_held_layer()
        try:
            taken = self._apply_layer_camera_settings_impl(
                request.gain_db,
                request.exposure_ms,
                layer=layer,
                auto_gain=request.auto_gain,
                auto_gain_settings=request.auto_gain_settings,
            )
        except CameraSettingRejected:
            now = {'gain_db': self.get_gain_db(), 'exposure_ms': self.get_exposure_ms()}
            with self._state_lock:
                self._held_layer = _HeldLayer(layer, None, now)
            raise
        if taken is not None:
            with self._state_lock:
                self._held_layer = _HeldLayer(layer, request, taken)
        return taken

    @api
    def apply_layer_camera_settings(
        self,
        gain_db: float,
        exposure_ms: float,
        auto_gain: bool = False,
        auto_gain_settings: dict | None = None,
        resume_after_capture: bool = True,
        layer: str = '(unspecified)',
    ) -> dict | None:
        """Apply per-layer camera settings in one batched call, and wait.

        See ``_apply_layer_camera_settings_impl`` for the contract; this
        adds only the dispatch described on ``_dispatch_camera``. The
        batch is one dispatched task, so the three writes stay atomic on
        the camera lane.

        Args:
            layer: The layer these settings belong to, for the log line.
                Optional here and required on the impl: this member is the
                external-caller surface, where an omitted layer should not
                be a TypeError, and the default keeps such a call
                self-describing in api.log rather than anonymous.

        Raises:
            ArgumentRefusedError: ``'missing_key'``, ``auto_gain_settings``
                lacks a key the auto-gain write reads; ``'not_a_number'``,
                one is not a number; ``'gain_range_inverted'``, its minimum
                gain is above its maximum.
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected; nothing was written.
            CameraSettingRejected: The camera refused the gain, the
                exposure or the auto-gain change, as
                ``_apply_layer_camera_settings_impl`` says.
        """
        if auto_gain_settings is not None:
            _refuse_unless_auto_gain_settings(auto_gain_settings, 'auto_gain_settings')
        return self._dispatch_camera(
            self._no_layers_write(self._apply_layer_camera_settings_impl),
            'apply_layer_camera_settings',
            args=(gain_db, exposure_ms),
            kwargs={
                'layer': layer,
                'auto_gain': auto_gain,
                'auto_gain_settings': auto_gain_settings,
                'resume_after_capture': resume_after_capture,
            },
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
        )

    def _apply_layer_camera_settings_impl(
        self,
        gain_db: float,
        exposure_ms: float,
        *,
        layer: str,
        auto_gain: bool = False,
        auto_gain_settings: dict | None = None,
        resume_after_capture: bool = True,
    ) -> dict | None:
        """Apply per-layer camera settings in a single batched call.

        Sets gain, exposure, and auto-gain state. Replaces 3 separate
        IOTask queues with a single call for atomicity. Turning auto-gain
        off goes first and arming it goes last, so gain and exposure are
        written with the camera's loop off. Arming may clamp
        the exposure just handed in down to the layer class's auto-exposure
        ceiling (the loop cannot bring an exposure above its bound back
        inside it); the log line below reports the request, the arm's own
        line records the clamp.

        Every write is attempted even when an earlier one is refused, so one
        refused setting does not leave the others at a previous layer's
        values; the refusal is raised once all of them have run.

        Args:
            gain_db: Camera gain in dB.
            exposure_ms: Exposure time in milliseconds.
            layer: The layer these settings belong to. Required, and
                keyword-only: two layers sharing a gain and an exposure
                otherwise write identical log lines, and one apply becomes
                indistinguishable from two. Every caller already holds it.
            auto_gain: The layer's stored auto-gain preference. A camera
                without hardware auto-gain applies it as manual
                (``applied_auto_gain_for``); the store keeps it.
            auto_gain_settings: Dict with target_brightness, min_gain_db, max_gain_db
                               (required if auto_gain is True).

        Returns:
            dict | None: ``{'gain_db': ..., 'exposure_ms': ...}``, each the
                value now in effect (see ``_set_gain_db_impl``). ``None``
                when no camera is active.

        Raises:
            CameraSettingRejected: The camera did not take the gain, the
                exposure or the auto-gain change. When more than one, ``setting``
                names each, comma-separated, ``requested`` holds their values in
                that order, the message carries each sentence, and the reason is
                ``'camera_no_answer'`` when any of them got no answer.
        """
        if not self._driver or not self._driver.active:
            return None
        # These arrive as the layer's STORED values, which a smaller camera
        # need not be able to reach. Capping here is what lets the store keep
        # the user's intent: the write below carries a value this body takes,
        # so the chunk target and the cache record what the sensor is actually
        # at, and no driver is asked for a value it would refuse (pylon) or
        # silently self-clamp while reporting success (IDS, FX2) -- that
        # divergence is why the cap cannot be left to the driver.
        gain = self.applied_gain_db_for(gain_db)
        exposure = self.applied_exposure_ms_for(exposure_ms)
        auto = self.applied_auto_gain_for(auto_gain)

        def _write_mode():
            return self._set_auto_gain_impl(
                auto.applied, settings=auto_gain_settings, resume_after_capture=resume_after_capture
            )

        # The camera's loop owns gain and exposure while it runs: a value
        # written under it is answered as applied and then taken back by the
        # loop's next update, and turning the loop off afterwards freezes the
        # loop's value. So a manual layer's values are written once the loop
        # is off, and an auto layer's seed values before it is armed (the
        # arm's ceiling clamp reads the exposure just written).
        refused: list[CameraSettingRejected] = []

        def _attempt(write):
            try:
                return write()
            except CameraSettingRejected as rejected:
                refused.append(rejected)
                return None

        if auto_gain_settings is not None and not auto.applied:
            _attempt(_write_mode)
        gain_result = _attempt(lambda: self._set_gain_db_impl(gain.applied))
        exposure_result = _attempt(lambda: self._set_exposure_ms_impl(exposure.applied))
        if auto_gain_settings is not None and auto.applied:
            _attempt(_write_mode)
        # Both numbers when the camera held one down, so a bundle shows the
        # intent that was stored next to the value the sensor took; one
        # number would read as the user having chosen the lower one.
        capped_note = ''
        if gain.capped or exposure.capped:
            capped_note = f' capped(stored gain={gain.stored}dB exp={exposure.stored}ms)'
        if auto.capped:
            capped_note += ' capped(stored auto_gain=True: no hardware auto-gain)'
        _api_log.info(
            f'apply_layer_camera_settings layer={layer} gain={gain.applied}dB '
            f'exp={exposure.applied}ms auto_gain={auto.applied}{capped_note}'
        )
        if len(refused) == 1:
            raise refused[0]
        if refused:
            # Not answered when any part got no answer: what the camera holds
            # is then not known, whatever the other parts said.
            raise CameraSettingRejected(
                ','.join(r.setting for r in refused),
                tuple(r.requested for r in refused),
                message=' '.join(str(r) for r in refused),
                reason=(
                    'camera_no_answer'
                    if any(r.reason == 'camera_no_answer' for r in refused)
                    else 'camera_rejected'
                ),
            ) from refused[0]
        return {'gain_db': gain_result, 'exposure_ms': exposure_result}

    @api
    def update_auto_gain_target_brightness(self, target_brightness: float) -> bool | None:
        """Set the auto-gain target brightness, and wait for it.

        See ``_update_auto_gain_target_brightness_impl`` for the settle
        contract; this adds the dispatch described on ``_dispatch_camera``
        and the raise below.

        Raises:
            CameraSettingRejected: The camera refused the change; nothing is
                shown here.
            CameraSettingUnsupportedError: The camera has no hardware
                auto-gain, so there is no target to set.
            ArgumentRefusedError: ``'not_a_number'``, the target is not a
                finite number.
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected or one removed during the write.
        """
        refuse_unless_finite_number(target_brightness, 'target_brightness')
        return self._dispatch_camera(
            self._no_layers_write(self._update_auto_gain_target_brightness_impl),
            'update_auto_gain_target_brightness',
            args=(target_brightness,),
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
        )

    def _update_auto_gain_target_brightness_impl(self, target_brightness: float) -> bool | None:
        """Set the auto-gain target brightness on the camera.

        Args:
            target_brightness: Target brightness value (0.0 to 1.0).

        Returns:
            bool | None: The driver's answer. ``None`` when no camera is
                active.

        Raises:
            CameraSettingRejected: The write did not apply (``_camera_write``).
        """
        if not self._driver or not self._driver.active:
            return None
        if not self._camera_has_auto_gain():
            raise _absent_mode_refusal(
                'auto_gain_target_brightness',
                target_brightness,
                'automatic gain',
                offered=(False,),
                consequence=_AUTO_MODE_STAYS,
            )
        # Changing the target re-drives the auto-gain loop: gain (and, under
        # auto-exposure, exposure) converge to a new operating point, so a frame
        # grabbed before they resettle is captured at the old brightness. Route
        # through the sanctioned write path and mark the settle sources RED so
        # capture waits for the convergence -- the same sources set_auto_gain
        # arms, since this is the same convergence.
        return self._camera_write(
            lambda: self._driver.update_auto_gain_target_brightness(target_brightness),
            member='update_auto_gain_target_brightness',
            refusal=lambda: _mode_rejection(
                'auto_gain_target_brightness', target_brightness, 'auto-gain target brightness'
            ),
            force_invalidate=('gain', 'auto_gain'),
        )

    def auto_gain_once(
        self,
        state: bool,
        target_brightness: float,
        min_gain_db: float,
        max_gain_db: float,
        ae_max_exposure_ms: float | None = None,
    ) -> None:
        """Run one-shot auto-gain, and wait for it.

        See ``_auto_gain_once_impl`` for the settle contract; this adds
        the dispatch described on ``_dispatch_camera`` and the raise below.

        Raises:
            CameraSettingRejected: The camera refused the one-shot; nothing
                is shown here.
            CameraSettingUnsupportedError: The camera has no hardware
                auto-gain.
            HardwareCommandRefusedError: ``'not_connected'``, naming the
                camera, with none connected or one removed during the write.
        """
        self._dispatch_camera(
            self._no_layers_write(self._auto_gain_once_impl),
            'auto_gain_once',
            args=(state, target_brightness, min_gain_db, max_gain_db),
            kwargs={'ae_max_exposure_ms': ae_max_exposure_ms},
            timeout_s=self._CAMERA_WRITE_TIMEOUT_S,
        )

    def _auto_gain_once_impl(
        self,
        state: bool,
        target_brightness: float,
        min_gain_db: float,
        max_gain_db: float,
        ae_max_exposure_ms: float | None = None,
    ) -> None:
        """Run auto-gain for a single frame on the camera.

        Args:
            state: True to enable one-shot auto-gain.
            target_brightness: Target brightness (0.0 to 1.0).
            min_gain_db: Minimum gain in dB.
            max_gain_db: Maximum gain in dB.
            ae_max_exposure_ms: Optional per-channel-class upper bound (ms)
                on the exposure auto-exposure may drive to.

        Raises:
            CameraSettingRejected: The write did not apply
                (``_camera_write``); the arm and the cache are left as they
                were.
        """
        if not self._driver or not self._driver.active:
            return
        if not self._camera_has_auto_gain():
            if state:
                raise _absent_mode_refusal(
                    'auto_gain',
                    state,
                    'automatic gain',
                    offered=(False,),
                    consequence=_AUTO_MODE_STAYS,
                )
            return
        # One-shot AG changes both gain and exposure on the camera from a single
        # driver call; the pipeline still needs frames to flush the converged
        # values, so both validity markers must go RED until they settle. The
        # converged values are chosen by the SDK, so clear any manual chunk-match
        # target and fall back to skip-frames settling. Both invalidations and
        # target-clears are forced -- one call, two sources, no value write the
        # authority could observe.
        self._camera_write(
            lambda: self._driver.auto_gain_once(
                state=state,
                target_brightness=target_brightness,
                min_gain_db=min_gain_db,
                max_gain_db=max_gain_db,
                ae_max_exposure_ms=ae_max_exposure_ms,
            ),
            member='auto_gain_once',
            refusal=lambda: _mode_rejection('auto_gain', state, 'one-shot auto-gain'),
            force_invalidate=('gain', 'exposure'),
            force_clear=('gain', 'exposure'),
        )
        with self._state_lock:
            self._auto_gain_arm = None
        # One-shot AG always ends with the auto cycle complete and the
        # SDK toggled back to Off internally; hardware holds the
        # converged value while LVP's cache is still pre-auto.
        self._refresh_cache_from_hardware_after_auto()

    @contextlib.contextmanager
    def suppress_value_warnings(self) -> Iterator[None]:
        """Suppress programmatic value-range warnings (sub-0.1ms exposure
        and similar) for the duration of the `with` block.

        Used by sweep-style internal callers (camera characterization
        dynamic_range / linearity stages) that walk the full setting
        range deliberately. The warnings exist for L1 researchers who
        type microsecond values thinking ms; they're noise when the
        char tool is exercising the API as designed.

        Restores the prior flag value (not unconditionally False) so
        nested `with` blocks behave correctly. Restoration runs on
        exception too -- exiting an exception-aborted char run leaves
        the API in a clean state for the next user action.
        """
        prior = self._suppress_value_warnings
        self._suppress_value_warnings = True
        try:
            yield
        finally:
            self._suppress_value_warnings = prior

    # --- Frame-flow listeners ---
    @api(in_process=True, event=CameraChanged)
    def add_camera_listener(self, listener: Callable[[str, float], None]) -> None:
        """Register a callback for camera setting changes.

        The listener is called with ``(param, value)`` whenever camera
        gain or exposure changes.  *param* is ``'gain'`` or ``'exposure'``.
        It fires from the thread that caused the change, so listeners
        **must** schedule UI work via ``Clock.schedule_once``.

        Note: this fires on set_gain_db/set_exposure_ms (user actions),
        NOT on every camera frame grab -- zero overhead on display framerate.

        Args:
            listener: ``callable(param: str, value: float)``
        """
        with self._camera_listeners_lock:
            self._camera_listeners.append(listener)

    @api(in_process=True)
    def remove_camera_listener(self, listener: Callable[[str, float], None]) -> None:
        """Unregister a camera listener.

        Args:
            listener: A callable previously passed to
                ``add_camera_listener``. Silently ignores listeners that
                are not currently registered.
        """
        with self._camera_listeners_lock:
            try:
                self._camera_listeners.remove(listener)
            except ValueError:
                pass
