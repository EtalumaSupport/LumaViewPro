# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Frame validity tracking for hardware state changes.

When hardware state changes (LED on/off, gain/exposure change, motor movement),
frames from the camera may not yet reflect the new state due to:
  1. Camera pipeline latency (2-3 frames to flush)
  2. Physical hardware settle time (motor moves take seconds)

Frame validity is the SINGLE source of truth for capture readiness. No capture
should proceed until frame_validity confirms all pending state changes have
settled. This includes both camera pipeline flush AND physical completion.

For camera-only sources (LED, gain, exposure): settled = frame count met.
For motion sources (xy_move, z_move, turret): settled = axis has physically
stopped moving (via settle callback) AND the frame count met by frames that
arrived after the stop.

A frame only counts toward a change it could actually show: each pending
source records which frame the camera had already delivered when its
register was written, and a frame credits it only if it arrived later.
Without that, a frame the live preview grabbed before the write but
counted after it would retire part of the wait it was never subject to.

Whether one frame counts for one change is ``FrameValidity.judge``, by the
rule the change's source runs (``FrameRule``): the clock rule, for a camera
with a clock latch and a measured delay for the source, or any move on such
a camera, counts a frame that started exposing, on the camera's clock,
after the change plus that delay (a move's change is its stop); the credit
rule, for every other source, counts a frame once the source's frame count
was met before it. Which rule each source runs is decided at
bring-up and published as ``capabilities.frame_rules``.

Usage:
    fv = FrameValidity(lambda: camera.frames_delivered)
    fv.set_settle_check(my_axis_check_fn)  # Register motion completion callback
    fv.invalidate(FrameValiditySource.Z_MOVE, began_ns=None)  # Z axis started moving
    fv.frames_until_valid()                # Returns >0 (motion not complete)
    # ... frames are grabbed by live view, counter increments ...
    fv.frames_until_valid()                # Still >0 if Z still moving
    # ... Z arrives at target, settle check returns True ...
    fv.frames_until_valid()                # Returns 0 -- next frame is valid

Excluding a source asks whether the frame is valid ignoring that change,
so the frame can predate it: fit for showing a frame during motion, never
for one that is measured or recorded (autofocus waits out each Z move, or a
step's score would be the previous step's):
    fv.is_valid_for(exclude_sources=(FrameValiditySource.Z_MOVE,))
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
import math
import threading
import time
import types
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, ClassVar

from drivers.camera import Frame
from lib import profile_trace
from modules.api_surface import api_fields

if TYPE_CHECKING:
    from modules.lumascope_api.frame_record import ClockMapping

# The camera timing record's version this release reads, and what its
# validity group's values must carry (``FrameValidity.check_timing_record``).
TIMING_RECORD_SCHEMA = 2
TIMING_MEASURES = ('apply_delay_us', 'delivered_frames')
TIMING_PROVENANCE = ('run_id', 'host', 'os', 'lvp_version', 'plugin_version', 'measured_date')


def _is_amount(value) -> bool:
    """Whether a timing record's measured number is a number, 0 or more."""
    return isinstance(value, int | float) and not isinstance(value, bool) and value >= 0


class FrameValiditySource(enum.StrEnum):
    """What changed, for a capture to wait out: each a key of ``FrameValidity.SKIP_FRAMES``."""

    LED = 'led'
    GAIN = 'gain'
    EXPOSURE = 'exposure'
    XY_MOVE = 'xy_move'
    Z_MOVE = 'z_move'
    TURRET = 'turret'
    AUTO_GAIN = 'auto_gain'
    PIXEL_FORMAT = 'pixel_format'
    FRAME_SIZE = 'frame_size'
    BINNING = 'binning'
    BLACK_LEVEL = 'black_level'
    CONVERSION_GAIN_MODE = 'conversion_gain_mode'
    LINE_NOISE_REDUCTION = 'line_noise_reduction'


class Counter(enum.Enum):
    """Who counted a frame toward a change: the capture's drain, a grab
    (a capture's own, or an ungated ``get_image``), or the live preview.

    A capture's wait can be met by frames another consumer counted, so the
    record of a wait names who counted each frame.
    """

    DRAIN = 'drain'
    GRAB = 'grab'
    PREVIEW = 'preview'


@api_fields('seq', 'by')
@dataclasses.dataclass(frozen=True)
class Credit:
    """One frame counted toward a change, and who counted it."""

    seq: int
    by: Counter


class Rule(enum.Enum):
    """How a frame is judged against a change of one source."""

    CLOCK = 'clock'
    CREDIT = 'credit'


@api_fields('rule', 'delay_us', 'provenance')
@dataclasses.dataclass(frozen=True)
class FrameRule:
    """The rule a source's changes are judged by, and where it came from.

    Attributes:
        rule: ``Rule.CLOCK`` when the camera has a clock latch and its timing
            record holds a measured delay for the source, or the source is a
            move; ``Rule.CREDIT`` otherwise.
        delay_us: Under the clock rule, the source's measured delay: the
            larger of its directions' ``apply_delay_us``, each the least
            start over its trials of the first frame showing the change. A
            frame starting later than the change plus this shows it. 0 for
            a move with none measured. None under the credit rule.
        provenance: Under the clock rule, the run that measured the delay
            (``TIMING_PROVENANCE``), or that none did (``AT_THE_STOP_RULE``);
            None under the credit rule.

    Raises:
        ValueError: a clock rule without its delay or provenance, or a
            credit rule with either.
    """

    rule: Rule
    delay_us: float | None
    provenance: Mapping[str, str] | None

    def __post_init__(self):
        clock = self.rule is Rule.CLOCK
        if (self.delay_us is not None) != clock or (self.provenance is not None) != clock:
            raise ValueError(
                f'a {self.rule.value} rule with delay {self.delay_us!r} and '
                f'provenance {self.provenance!r}: the clock rule carries both, the credit rule neither'
            )


CREDIT_RULE = FrameRule(rule=Rule.CREDIT, delay_us=None, provenance=None)

# A move with no measured delay, on a camera with a clock latch: a frame
# counts once it started exposing after the axis was read stopped. A frame
# that started before the stop was exposed while the axis could still move,
# whenever it arrived; the stop is read after the motor stops, so no frame
# that started after it was.
AT_THE_STOP_RULE = FrameRule(
    rule=Rule.CLOCK,
    delay_us=0.0,
    provenance=types.MappingProxyType({'measured': 'none: a frame started after the stop'}),
)


@api_fields(
    'source',
    'at_seq',
    'written_ns',
    'earliest_ns',
    'written_at',
    'skip',
    'rule',
    'credits',
    'needs_in_force',
    'in_force_ns',
)
@dataclasses.dataclass(frozen=True)
class Change:
    """The last recorded change of one source, and the facts a frame is judged against.

    Frames do not consume it: it stays the source's change until the next
    ``invalidate`` replaces it, so a capture long after can still say how
    long after the change its frame came.

    Attributes:
        source: What changed.
        at_seq: The frame the camera had already delivered when the change
            was recorded; for a move, once in force, when its stop was
            stamped (``FrameValidity.motion_settled``). A frame already in
            the camera's pipeline when the
            register was written cannot show the new state, so crediting it
            would retire the skip count against frames that predate the
            change -- and the capture then returns a frame from before the
            write. An ordinal rather than a clock reading, because the
            comparison has to survive the clock moving: wall time runs
            backwards across a DST fall-back, an NTP correction or a host
            resume, and a settle count measured against a timestamp that
            jumped forward never completes. Frame numbers only go up.
        written_ns: Host ``time.monotonic_ns()`` when the change was
            recorded. Every recorder calls ``invalidate`` after a camera
            register or LED write returns, and before a move starts.
        earliest_ns: Host ``time.monotonic_ns()`` of the earliest moment
            the change could have reached the hardware: a write's start, as
            its writer stamped it, since the register may change anywhere
            inside the write; for a move, once in force, the start of the
            last read that saw its axis still moving, else ``written_ns``.
            A delay measured from here is never short; the rule judges
            from ``written_ns`` or ``in_force_ns``, the latest, so the
            pair never accepts a frame early.
        written_at: Host wall-clock time of the same moment, for a reader.
        skip: The frames the change needed, as the count stood when it was
            recorded.
        rule: The rule the change is judged by, as its source's stood when
            it was recorded.
        credits: The frames counted toward it, in the order counted, until
            ``skip`` was met.
        needs_in_force: A motion source the axis must also settle for: the
            frame count alone does not end its wait.
        in_force_ns: Host ``time.monotonic_ns()`` when the axis state the
            settle check reads became settled, stamped by the motion API
            (``FrameValidity.motion_settled``); None until then, and for a
            change that does not wait on an axis.
    """

    source: FrameValiditySource
    at_seq: int
    written_ns: int
    earliest_ns: int
    written_at: datetime.datetime
    skip: int
    rule: FrameRule
    credits: tuple[Credit, ...]
    needs_in_force: bool
    in_force_ns: int | None

    @property
    def remaining(self) -> int:
        """Frames the change still needs counted; 0 once the count is met."""
        return max(0, self.skip - len(self.credits))


@api_fields(
    'change',
    'frame_seq',
    'frames_after',
    'after_write_ms',
    'after_in_force_ms',
    'frame_start_ticks',
    'qualifies_after_ticks',
    'counts',
)
@dataclasses.dataclass(frozen=True)
class Judgment:
    """Whether one frame counts for one change, and the facts that decided it.

    Attributes:
        change: The change judged against, as it stood when judged; its
            ``rule`` is the rule it was judged by.
        frame_seq: The frame's arrival ordinal.
        frames_after: Frames delivered after the change was recorded, up to
            and including this one.
        after_write_ms: Milliseconds from the change's recording to the
            frame's arrival.
        after_in_force_ms: Milliseconds from the change's in-force time to
            the frame's arrival; None when it has none.
        frame_start_ticks: Under the clock rule, when the frame started
            exposing on the camera's clock (its ``Timestamp`` chunk); None
            under the credit rule, and for a frame without the chunk.
        qualifies_after_ticks: Under the clock rule, the camera-clock tick a
            frame must start after to count: the change, its source's delay
            and the clock mapping's uncertainty. None under the credit rule,
            and when it cannot be known (no mapping, or a move not yet in
            force).
        counts: Under the credit rule, the frame arrived after the change,
            the change's count was met by frames before it, and an axis the
            change waits on had settled before it arrived. Under the clock
            rule, the frame started after ``qualifies_after_ticks``; never
            when either is unknown.
    """

    change: Change
    frame_seq: int
    frames_after: int
    after_write_ms: float
    after_in_force_ms: float | None
    frame_start_ticks: int | None
    qualifies_after_ticks: int | None
    counts: bool

    def fields(self) -> dict:
        """The judgment as plain fields, the one shape a saved file and the
        capture's log line both carry. A time the judgment has none of is
        absent, never a stand-in."""
        out = {
            'source': self.change.source,
            'rule': self.change.rule.rule.value,
            'counts': self.counts,
            'frames_after': self.frames_after,
            'after_write_ms': round(self.after_write_ms, 3),
            'skip': self.change.skip,
            'counted_by': [f'{credit.seq}:{credit.by.value}' for credit in self.change.credits],
        }
        if self.after_in_force_ms is not None:
            out['after_in_force_ms'] = round(self.after_in_force_ms, 3)
        if self.frame_start_ticks is not None:
            out['frame_start_ticks'] = self.frame_start_ticks
        if self.qualifies_after_ticks is not None:
            out['qualifies_after_ticks'] = self.qualifies_after_ticks
        return out


@dataclasses.dataclass(frozen=True)
class Owed:
    """What a capture still waits on before a frame can count for every change.

    Attributes:
        frames: Frames still to be delivered for the changes judged by the
            credit rule, and 1 while a move's axis has not settled; 0 when
            none is owed.
        until_ns: Host ``time.monotonic_ns()`` after which a frame can start
            late enough for every change judged by the clock rule (its
            anchor, its delay and the mapping's uncertainty); None when no
            such change has one.
    """

    frames: int
    until_ns: int | None


class FrameValidity:
    """Tracks frame validity after hardware state changes.

    Each hardware state change source has a configurable number of frames
    that must be skipped before the camera output reflects the new state.
    Motion sources additionally require physical completion (axis stopped).
    """

    # Per-source skip frame counts (camera pipeline flush): the counts the
    # credit rule runs. A source whose camera has a clock latch and a
    # measured delay runs the clock rule instead (``frame_rules``). Each
    # instance copies them, so one scope's counts never outlive it into the
    # next scope in the process.
    SKIP_FRAMES: ClassVar[dict[FrameValiditySource, int]] = {
        FrameValiditySource.LED: 2,  # LED on/off or current change (measured: 2 on a2A3536)
        FrameValiditySource.GAIN: 2,  # Camera gain change (measured: 2 on a2A3536)
        # Camera exposure time change. Not a measurement: the a2A3536 timing
        # record's value that once stood beside it has no surviving source run.
        FrameValiditySource.EXPOSURE: 3,
        # A move's frames are counted from its stop (``motion_settled``).
        # Not measurements: no camera without a clock has a timing record.
        FrameValiditySource.XY_MOVE: 2,  # X or Y axis movement
        FrameValiditySource.Z_MOVE: 2,  # Z axis movement
        FrameValiditySource.TURRET: 2,  # Turret rotation
        # Frames for hardware continuous auto-gain to settle against the lit
        # scene after arming. Like led/gain/exposure this is an instrumentation
        # settling count, not a content measurement: the LED must be lit before
        # arming so AG settles on the real scene, not a dark frame. At 10 a dim
        # brightfield scene (low LED current, gain starting at 0) often had not
        # converged when the frame was grabbed -- the capture came out dark.
        # Doubled to give AG more frames to ramp; the [AG CONVERGE] diagnostic
        # measures the real convergence need per camera for the eventual
        # camera_timing/<model>.json value (or a brightness-converge gate).
        FrameValiditySource.AUTO_GAIN: 20,
        # Geometry / format changes restart the grab engine or realloc the
        # camera buffer; the pipeline needs frames to flush the old
        # geometry before a capture reflects the new one. Conservative
        # defaults pending per-camera bench measurement into
        # camera_timing/<model>.json, like the others.
        FrameValiditySource.PIXEL_FORMAT: 3,
        FrameValiditySource.FRAME_SIZE: 3,
        FrameValiditySource.BINNING: 3,
        # Sensor settings that change the next frames like gain does. Unmeasured;
        # gain's count until a bench measurement goes into camera_timing.
        FrameValiditySource.BLACK_LEVEL: 2,
        FrameValiditySource.CONVERSION_GAIN_MODE: 2,
        FrameValiditySource.LINE_NOISE_REDUCTION: 2,
    }

    # One shape for every trace row this class writes. A row records the
    # comparison the gate actually made: the frame's arrival ordinal against
    # the ordinal the write was stamped against, and what the count stands at
    # afterwards. Both sides are in the row so a reader can see WHY a source
    # did or did not clear, rather than inferring it from what is missing.
    _TRACE_FILE = 'frame_validity_trace.csv'
    _TRACE_HEADER = 'ts_ms,event,source,frame_seq,at_seq,remaining,pending_count'

    # Sources that require physical hardware completion in addition to frame count.
    MOTION_SOURCES = frozenset(
        {FrameValiditySource.XY_MOVE, FrameValiditySource.Z_MOVE, FrameValiditySource.TURRET}
    )

    # Sources carrying per-frame chunk metadata, used to REJECT a frame at
    # capture time -- never to accept one. A chunk reports the register value
    # in force when the camera TAGGED the frame, not the light integrated
    # into it, so a chunk that AGREES with the target proves nothing about
    # the pixels; a chunk that DISAGREES proves the frame predates the write.
    # Only the disagreement direction is sound. LED has no chunk equivalent;
    # motion is firmware-gated via _settle_check_fn.
    CHUNK_VALIDATABLE_SOURCES = frozenset({FrameValiditySource.GAIN, FrameValiditySource.EXPOSURE})

    # Maps our source names to the chunk_data dict keys used by camera
    # drivers (chunk_data uses the genicam attribute symbolic names).
    CHUNK_KEY_FOR_SOURCE: ClassVar[dict[FrameValiditySource, str]] = {
        FrameValiditySource.GAIN: 'Gain',
        FrameValiditySource.EXPOSURE: 'ExposureTime',
    }

    # Float-tolerance for chunk-match equality. ChunkExposureTime is in
    # microseconds (the API converts ms -> us when calling set_target);
    # ChunkGain is in dB. Values measured by sweeping set values across
    # the supported range and reading back ChunkGain / ChunkExposureTime
    # on multiple Basler USB3 cameras (a2A3536-31umBAS ace 2; daA3840-45um
    # dart with firmware 1.1.0 and 2.6.0): observed deltas were bit-
    # identical across hardware and firmware -- quantization happens at
    # the Pylon SDK / genicam nodemap layer, not in camera firmware.
    # Gain round-trip error peaked at ~5e-5 dB (float epsilon), exposure was
    # bit-exact in microseconds. Tolerances set ~20x above observed max
    # for safety across future firmware revisions.
    DEFAULT_CHUNK_TOLERANCE: ClassVar[dict[FrameValiditySource, float]] = {
        FrameValiditySource.GAIN: 0.001,  # dB
        FrameValiditySource.EXPOSURE: 2.0,  # microseconds
    }

    def __init__(self, frames_delivered):
        """
        Args:
            frames_delivered: zero-argument callable returning how many frames
                the camera has delivered so far. REQUIRED -- without it a write
                cannot be placed in the frame stream, every frame would credit
                every pending source, and that is precisely the defect this
                class exists to prevent. Passing a callable rather than the
                driver keeps the lookup late: the composition root assigns the
                camera driver after this object is built.
        """
        self._lock = threading.Lock()
        self._frames_delivered = frames_delivered
        self._frame_counter = 0
        self.SKIP_FRAMES = dict(FrameValidity.SKIP_FRAMES)
        # Each source's last change, kept after it settles so a capture can be
        # judged against it; and the sources whose last change has not.
        self._changes: dict[str, Change] = {}
        self._pending: set[str] = set()
        self._settle_check_fn = None  # Optional: (source) -> bool
        self._target_values = {}  # source -> requested value (for chunk-match)
        self._last_counted_seq = None  # ordinal of the last counted frame
        # Each source's rule; the credit rule until bring-up hands over the
        # scope's (use_rules), as for a camera with no measured record.
        self._rules = self.credit_rules()
        # Monotone per-source invalidation history. Unlike _pending, entries
        # are never consumed by frames -- count_frame and reset() leave this
        # map untouched -- so a capture can snapshot it before its grab and
        # compare after to detect an invalidation that frames have already
        # settled. Pending-state snapshots cannot do that: a poller frame
        # plus the capture's own final grab can consume a 2-skip source's
        # entry exactly, leaving pending/frames_until_valid bit-identical
        # around a real mid-window invalidation.
        self._invalidation_counts = {}  # source -> total invalidate() calls

    def set_settle_check(self, fn: Callable[[FrameValiditySource], bool]) -> None:
        """Register a callback that checks if a source has physically settled.

        Args:
            fn: Answers whether the hardware for a source has physically
                completed its state change.
                For motion sources, this typically checks axis state == IDLE.
                For non-motion sources, should return True.

        Called during validity checks for MOTION_SOURCES. Without this
        callback, motion sources settle based on frame count only (legacy
        behavior, incorrect for long moves).
        """
        self._settle_check_fn = fn

    @classmethod
    def credit_rules(cls) -> Mapping[FrameValiditySource, FrameRule]:
        """Every source on the credit rule: the rules of a camera with no measured delay."""
        return types.MappingProxyType(dict.fromkeys(cls.SKIP_FRAMES, CREDIT_RULE))

    @classmethod
    def rules_from_record(
        cls, record: dict | None, has_clock_latch: bool
    ) -> Mapping[FrameValiditySource, FrameRule]:
        """Each source's rule, from a camera's checked timing record.

        On a camera with a clock latch, a source the record holds an
        ``apply_delay_us`` for runs the clock rule at the larger of its
        directions' delays with that direction's provenance: a capture
        waits one delay per source, whichever way the change went. A move
        the record holds no delay for, or every move when there is no
        record, runs it at the stop (``AT_THE_STOP_RULE``). Every other
        source runs the credit rule, as every source does on a camera
        without a latch.

        Args:
            record: The camera's timing record, already checked
                (``check_timing_record``); None for a camera that names none.
            has_clock_latch: Whether the camera's clock can be mapped onto
                the host's (``capabilities.camera_has_clock_latch``).
        """
        rules = dict(cls.credit_rules())
        if not has_clock_latch:
            return types.MappingProxyType(rules)
        rules.update(dict.fromkeys(cls.MOTION_SOURCES, AT_THE_STOP_RULE))
        if record is None:
            return types.MappingProxyType(rules)
        for source, directions in record['validity'].items():
            delays = [v for v in directions.values() if 'apply_delay_us' in v]
            if delays:
                slowest = max(delays, key=lambda v: v['apply_delay_us'])
                rules[FrameValiditySource(source)] = FrameRule(
                    rule=Rule.CLOCK,
                    delay_us=float(slowest['apply_delay_us']),
                    provenance=types.MappingProxyType(dict(slowest['provenance'])),
                )
        return types.MappingProxyType(rules)

    def use_rules(self, rules: Mapping[FrameValiditySource, FrameRule]) -> None:
        """Judge each change recorded from now on by its source's rule in ``rules``.

        Bring-up hands over the scope's ``capabilities.frame_rules``.

        Raises:
            ValueError: ``rules`` leaves a source with a skip count without a rule.
        """
        missing = sorted(set(self.SKIP_FRAMES) - set(rules))
        if missing:
            raise ValueError(f'frame rules name no rule for {missing}')
        with self._lock:
            self._rules = rules

    def invalidate(self, source: FrameValiditySource, *, began_ns: int | None) -> Change:
        """Record that hardware state changed and frames need to settle.

        Args:
            source: What changed.
            began_ns: Host ``time.monotonic_ns()`` when the write that made
                the change began, the change's ``earliest_ns``; None for a
                change recorded before it is made (a move), whose earliest
                is its record.

        Returns:
            Change: The change as recorded, now the source's last change.

        Raises:
            ValueError: ``source`` has no skip count. A source that settled
                on a silent default had a count nobody chose.
        """
        skip = self.SKIP_FRAMES.get(source)
        if skip is None:
            raise ValueError(f'frame validity has no skip count for source {source!r}')
        with self._lock:
            # Read INSIDE the lock: two threads invalidating the same source
            # can otherwise store the earlier ordinal for the later write,
            # which is this whole class of bug reintroduced by a race.
            written_ns = time.monotonic_ns()
            entry = Change(
                source=source,
                at_seq=self._frames_delivered(),
                written_ns=written_ns,
                earliest_ns=written_ns if began_ns is None else began_ns,
                written_at=datetime.datetime.now(),
                skip=skip,
                rule=self._rules[source],
                credits=(),
                needs_in_force=source in self.MOTION_SOURCES and self._settle_check_fn is not None,
                in_force_ns=None,
            )
            self._changes[source] = entry
            self._pending.add(source)
            self._invalidation_counts[source] = self._invalidation_counts.get(source, 0) + 1
            # Snapshot for the trace while the lock is still held. Reading any
            # of this back afterwards lets another thread's invalidate supply
            # it, and the row then describes a state no write produced.
            at_seq = entry.at_seq
            pending_count = len(self._pending)
        if profile_trace.ENABLE_PROFILE_TRACE:
            self._trace_rows([('invalidate', source, '', at_seq, skip, pending_count)])
        return entry

    def count_frame(self, frame_seq: int, *, by: Counter) -> None:
        """Record that a frame was grabbed from the camera.

        Call this after every successful camera grab (grab() or grab_new_capture()).
        A frame credits only the sources whose write it POSTDATES, and only
        when it is newer than every frame already counted toward that
        source; a source clears once its skip count has been met that way. A
        motion source's count starts at its stop, so it clears only when the
        settle check passes and frames after the stop met the count.

        Args:
            frame_seq: The frame's arrival ordinal, from the grab that
                produced it. Required: it is both the frame's identity and
                the evidence that the frame is newer than a given write.
                Multiple consumers poll the same buffered frame concurrently
                (live preview, histogram, capture drains); without identity
                dedupe those polls retire the skip counts with zero new
                frames, and a capture can then accept a frame exposed under
                the previous gain/exposure/LED state.
            by: Who counted it, recorded on each change it credits.
        """
        tracing = profile_trace.ENABLE_PROFILE_TRACE
        rows = []
        with self._lock:
            if frame_seq == self._last_counted_seq:
                # Only while something is pending: the dedupe check runs before
                # the pending loop, so it has no idle bound of its own and a
                # poller would otherwise fill the file forever. While a source
                # IS pending this row is the load-bearing one -- a camera whose
                # ordinal never advances produces nothing but these, and without
                # them the file is empty and reads as "no frames arrived".
                if tracing and self._pending:
                    rows.append(('dedupe', '', frame_seq, '', '', len(self._pending)))
            else:
                self._last_counted_seq = frame_seq
                self._frame_counter += 1
                settled = []
                events = []
                for source in self._pending:
                    pending = self._changes[source]
                    # What a source publishes is frames STILL NEEDED -- a count
                    # that stops at zero rather than running down one per frame
                    # into nonsense. A frame no newer than the last one credited
                    # is one the wait already holds, or older: the preview can
                    # count a frame it grabbed before the drain's newer one,
                    # after it, and the dedupe above sees only the last count.
                    # A move's frame count starts at its stop (motion_settled):
                    # a frame before it was exposed while the axis could move.
                    credited = (
                        frame_seq > pending.at_seq
                        and not (pending.needs_in_force and pending.in_force_ns is None)
                        and pending.remaining > 0
                        and (not pending.credits or frame_seq > pending.credits[-1].seq)
                    )
                    if credited:
                        pending = dataclasses.replace(
                            pending, credits=(*pending.credits, Credit(frame_seq, by))
                        )
                        self._changes[source] = pending
                    if self._is_source_settled_unlocked(source, pending):
                        settled.append(source)
                        events.append(('settled', source, pending.at_seq, pending.remaining))
                    else:
                        events.append(
                            (
                                'credit' if credited else 'nocredit',
                                source,
                                pending.at_seq,
                                pending.remaining,
                            )
                        )
                for s in settled:
                    self._pending.discard(s)
                pending_count = len(self._pending)
                # One row per source, never a joined one: the bench pass
                # criterion pairs each settled row with its own invalidate row
                # by source, which a row naming two sources cannot support.
                if tracing:
                    rows = [
                        (event, source, frame_seq, at_seq, remaining, pending_count)
                        for event, source, at_seq, remaining in events
                    ]
        if rows:
            self._trace_rows(rows)

    def _trace_rows(self, rows) -> None:
        """Write already-built trace rows. Call with the lock RELEASED.

        Every row this class emits is built through here, so the field count
        cannot drift between sites: an arity mismatch raises out of the writer
        rather than being logged, which would surface on whichever thread
        happened to be counting a frame.

        Args:
            rows: sequence of
                ``(event, source, frame_seq, at_seq, remaining, pending_count)``
                tuples, each already snapshotted under the lock.
        """
        ts_ms = int(time.time() * 1000)
        for row in rows:
            profile_trace.trace(
                self._TRACE_FILE,
                self._TRACE_HEADER,
                [ts_ms, *row],
                recording_id=profile_trace.NO_RECORDING,
            )

    def _is_source_settled_unlocked(self, source: FrameValiditySource, pending: Change) -> bool:
        """Check if a source has settled. Must be called with _lock held."""
        if pending.remaining > 0:
            return False
        # Motion sources also require physical completion
        if source in self.MOTION_SOURCES and self._settle_check_fn is not None:
            return self._settle_check_fn(source)
        return True

    def set_target(self, source: FrameValiditySource, value: float | None) -> None:
        """Record the requested value for a chunk-validatable source.

        The API layer (Lumascope.set_gain_db / set_exposure_ms) calls this
        after invalidate() so the capture path can compare a returned
        frame's chunk against what was asked for and REJECT a frame whose
        chunk disagrees. The target never clears a source: settling is by
        frame count alone.

        Args:
            source: What the target is for (``GAIN``, ``EXPOSURE``). Sources outside
                CHUNK_VALIDATABLE_SOURCES are accepted but never consulted.
            value: Target value to compare chunks against. None clears any
                prior target.
        """
        with self._lock:
            if value is None:
                self._target_values.pop(source, None)
            else:
                self._target_values[source] = float(value)

    def target(self, source: FrameValiditySource) -> float | None:
        """Return the recorded target value for a source, or None if unset.

        Capture paths use this to decide whether a returned frame can be
        chunk-verified: a set target plus a present chunk key means the
        frame must match; no target (e.g. hardware auto-gain owns the
        value) means chunk verification does not apply.

        Args:
            source: What the target is for (``GAIN``, ``EXPOSURE``).
        """
        with self._lock:
            return self._target_values.get(source)

    def chunk_match(
        self, source: FrameValiditySource, chunk_value: float | None, tolerance: float | None = None
    ) -> bool:
        """Public float-tolerant equality between a chunk value and the recorded target.

        Used by the capture path's stale-frame rejection, and by tests.

        Args:
            source: What the target is for.
            chunk_value: Observed value from chunk metadata. May be None.
            tolerance: Optional tolerance override; defaults to DEFAULT_CHUNK_TOLERANCE.
        """
        if chunk_value is None:
            return False
        with self._lock:
            target = self._target_values.get(source)
        if target is None:
            return False
        if tolerance is None:
            tolerance = self.DEFAULT_CHUNK_TOLERANCE.get(source, 0.0)
        return abs(float(chunk_value) - target) <= tolerance

    @property
    def is_valid(self) -> bool:
        """True if all pending state changes have settled."""
        with self._lock:
            return all(self._is_source_settled_unlocked(s, self._changes[s]) for s in self._pending)

    def is_valid_for(self, exclude_sources: tuple[FrameValiditySource, ...] = ()) -> bool:
        """True if valid, ignoring specified sources.

        The frame can predate an excluded change: see the module docstring.
            fv.is_valid_for(exclude_sources=(FrameValiditySource.Z_MOVE,))
        """
        with self._lock:
            return all(
                self._is_source_settled_unlocked(s, self._changes[s])
                for s in self._pending
                if s not in exclude_sources
            )

    def frames_until_valid(self, exclude_sources: tuple[FrameValiditySource, ...] = ()) -> int:
        """Number of frames that must be grabbed before the next valid frame.

        Returns 0 if already valid. A move counts its frames from its stop,
        so while its axis moves it returns the move's whole count; a move
        whose count is met while the settle check says moving returns 1
        (keep draining).
        """
        with self._lock:
            max_remaining = 0
            for source in self._pending:
                if source in exclude_sources:
                    continue
                pending = self._changes[source]
                if pending.remaining > 0:
                    max_remaining = max(max_remaining, pending.remaining)
                elif source in self.MOTION_SOURCES and self._settle_check_fn is not None:  # noqa: SIM102
                    # Frame count met but axis still moving -- keep draining
                    if not self._settle_check_fn(source):
                        max_remaining = max(max_remaining, 1)
            return max(0, max_remaining)

    def owed(
        self, exclude_sources: tuple[FrameValiditySource, ...] = (), uncertainty_ns: int = 0
    ) -> Owed:
        """What a capture waits on, each change by its own rule.

        A change judged by the credit rule is owed its count, as
        ``frames_until_valid`` counts it: with every source on the credit
        rule the two are the same number. A change judged by the clock rule
        is owed time instead -- its anchor plus its delay plus
        ``uncertainty_ns``, on the host's clock -- whether or not frames have
        been counted toward it, since a frame counts only once it started
        after that instant. A move under either rule is also owed a frame
        while its axis has not settled, and a clock move until its in-force
        time is stamped: its anchor is not known before.

        Args:
            exclude_sources: Sources a frame may predate, as for
                ``frames_until_valid``.
            uncertainty_ns: The clock mapping's uncertainty, which the clock
                rule adds (``ClockMapping.uncertainty_ns``); 0 without one.
        """
        with self._lock:
            frames = 0
            until_ns = None
            for source, change in self._changes.items():
                if source in exclude_sources:
                    continue
                moving = (
                    source in self.MOTION_SOURCES
                    and self._settle_check_fn is not None
                    and not self._settle_check_fn(source)
                )
                if change.rule.rule is Rule.CLOCK:
                    anchor = change.in_force_ns if change.needs_in_force else change.written_ns
                    if moving or anchor is None:
                        frames = max(frames, 1)
                        continue
                    qualifies_ns = anchor + math.ceil(change.rule.delay_us * 1000) + uncertainty_ns
                    until_ns = qualifies_ns if until_ns is None else max(until_ns, qualifies_ns)
                elif source in self._pending:
                    if change.remaining > 0:
                        frames = max(frames, change.remaining)
                    elif moving:
                        frames = max(frames, 1)
            return Owed(frames=frames, until_ns=until_ns)

    @property
    def pending_sources(self) -> dict[FrameValiditySource, int]:
        """Current pending sources and the frames each still needs (for debugging).

        Zero means the frame count is met, NOT that the source has settled:
        a motion source's count starts at its stop, and the settle check is
        not read here. Read
        is_valid, frames_until_valid() or unsettled_motion_sources() for
        settledness.
        """
        with self._lock:
            return {s: self._changes[s].remaining for s in self._pending}

    @property
    def frame_counter(self) -> int:
        """Current frame counter value (for debugging)."""
        with self._lock:
            return self._frame_counter

    @property
    def invalidation_counts(self) -> dict[FrameValiditySource, int]:
        """Per-source count of every invalidate() call, monotone for the
        instance's lifetime.

        Snapshot before a grab and compare (!=) after it to detect a
        mid-window invalidation regardless of whether frames have since
        settled it. Compare full dicts, not shared keys only: a source's
        first-ever invalidation adds a key the snapshot lacks.
        """
        with self._lock:
            return dict(self._invalidation_counts)

    def changes(self) -> dict[FrameValiditySource, Change]:
        """Each source's last recorded change, settled or not, by source."""
        with self._lock:
            return dict(self._changes)

    @staticmethod
    def judge(frame: Frame, change: Change, mapping: ClockMapping | None) -> Judgment:
        """Whether ``frame`` counts for ``change``: the one place that decides it.

        A function of facts the frame, the change and the clock mapping
        carry, so a capture, the characterization tool and a test ask the
        same question of any frame and get the gate's own answer, by the
        rule the change was recorded under (``change.rule``):

        - Credit: the frame counts when it arrived after the change was
          recorded, the change's count was met by frames that arrived before
          it, and, for a change that waits on an axis, the axis had settled
          before it arrived.
        - Clock: the frame counts when it started exposing, on the camera's
          clock, after the change was in force (its write; for a move, its
          settle) plus the source's measured delay plus the mapping's
          uncertainty. A frame whose start, or whose qualifying tick, cannot
          be known -- no mapping, no ``Timestamp`` chunk, a move not yet in
          force -- does not count: nothing shows it started late enough.

        Args:
            frame: The camera's frame record; its ``seq``, ``arrived_ns`` and
                ``Timestamp`` chunk are read.
            change: The change to judge it against (``changes()``, or what
                ``invalidate`` returned).
            mapping: The camera's clock mapped onto the host's, for the
                clock rule; None for a camera without a latch.
        """
        in_force = change.in_force_ns
        start = None
        qualifies_after = None
        if change.rule.rule is Rule.CLOCK:
            start = frame.chunks.get('Timestamp') if frame.chunks else None
            anchor = in_force if change.needs_in_force else change.written_ns
            if mapping is not None and anchor is not None:
                margin_ns = change.rule.delay_us * 1000 + mapping.uncertainty_ns
                qualifies_after = mapping.camera_ticks_at(anchor) + math.ceil(
                    margin_ns * mapping.tick_hz / 1e9
                )
            counts = start is not None and qualifies_after is not None and start > qualifies_after
        else:
            counted_before = sum(1 for credit in change.credits if credit.seq < frame.seq)
            counts = (
                frame.seq > change.at_seq
                and counted_before >= change.skip
                and (
                    not change.needs_in_force
                    or (in_force is not None and in_force <= frame.arrived_ns)
                )
            )
        return Judgment(
            change=change,
            frame_seq=frame.seq,
            frames_after=frame.seq - change.at_seq,
            after_write_ms=(frame.arrived_ns - change.written_ns) / 1e6,
            after_in_force_ms=(frame.arrived_ns - in_force) / 1e6 if in_force is not None else None,
            frame_start_ticks=int(start) if start is not None else None,
            qualifies_after_ticks=qualifies_after,
            counts=counts,
        )

    def judgments(self, frame: Frame, mapping: ClockMapping | None) -> tuple[Judgment, ...]:
        """``frame`` judged against every source's last change, by source name."""
        changes = self.changes()
        return tuple(self.judge(frame, changes[source], mapping) for source in sorted(changes))

    def motion_settled(
        self, source: FrameValiditySource, at_ns: int, *, moving_seen_ns: int | None
    ) -> None:
        """Stamp ``at_ns`` as the in-force time of ``source``'s change, if the settle check now passes.

        The motion API calls this after every write of an axis state the
        settle check reads, naming the source that axis belongs to, with the
        time it took in that write's hold, once that hold is released: the
        settle check takes the axis-state lock inside this class's, so
        calling in from under the axis-state lock would take the two in the
        opposite order. A change keeps the latest stamp it is given, since a
        change waiting on two axes (X and Y) is in force only once the later
        of them settles, and each axis's stamp can land after the other's
        write; a stamp older than the change is not its own.

        The axis stopped somewhere between ``moving_seen_ns`` and ``at_ns``,
        so the change's ``earliest_ns`` moves up to ``moving_seen_ns`` with
        the stamp that sets its in-force time.

        The stamp places the change in the frame stream, as ``invalidate``
        places a write: its ``at_seq`` becomes the frames the camera has
        delivered now, its credits are cleared, and the source is pending
        again, so its frame count is met only by frames that arrived after
        the stop. A frame that arrived during the move was exposed while the
        axis could still move. A later stamp, from a later write of the axis
        settled, starts the count again from there.

        Args:
            source: The motion source of the axis whose state was written.
            at_ns: Host ``time.monotonic_ns()`` taken where it was written.
            moving_seen_ns: Host ``time.monotonic_ns()`` at the start of the
                last read that saw the axis still moving; None when no read
                did, so the stop could be as early as the change's record.
        """
        with self._lock:
            change = self._changes.get(source)
            if (
                change is None
                or not change.needs_in_force
                or at_ns < change.written_ns
                or (change.in_force_ns is not None and change.in_force_ns >= at_ns)
                or not self._settle_check_fn(source)
            ):
                return
            # Of two axes (X and Y) the later to stop sets it, as it sets
            # the in-force time.
            earliest_ns = change.earliest_ns
            if moving_seen_ns is not None:
                earliest_ns = max(earliest_ns, moving_seen_ns)
            change = dataclasses.replace(
                change,
                at_seq=self._frames_delivered(),
                credits=(),
                in_force_ns=at_ns,
                earliest_ns=earliest_ns,
            )
            self._changes[source] = change
            self._pending.add(source)
            at_seq = change.at_seq
            pending_count = len(self._pending)
        if profile_trace.ENABLE_PROFILE_TRACE:
            self._trace_rows([('in_force', source, '', at_seq, change.skip, pending_count)])

    def unsettled_motion_sources(
        self, exclude_sources: tuple[FrameValiditySource, ...] = ()
    ) -> tuple[FrameValiditySource, ...]:
        """Pending motion sources whose settle-check says still moving.

        The capture deadline suspends while this is non-empty: motion has
        an authoritative completion signal, and a frame-budget clock must
        not out-vote it. Calling the settle-check under the lock follows
        the ordering frames_until_valid already established (validity lock,
        then the axis-state lock inside the callback).
        """
        with self._lock:
            if self._settle_check_fn is None:
                return ()
            return tuple(
                s
                for s in self._pending
                if s in self.MOTION_SOURCES
                and s not in exclude_sources
                and not self._settle_check_fn(s)
            )

    @classmethod
    def check_timing_record(cls, record: dict) -> None:
        """Refuse a camera timing record this gate cannot take as measured.

        A record (``data/camera_timing/<model>.json``) states, in its
        ``validity`` group, how late each change reaches a frame: per source
        and direction, either ``apply_delay_us``, the least start over the
        ``trials``, on the camera's clock and after the change was in force,
        of the first frame showing it -- a bound the sensor's true delay never
        exceeds, so a frame starting after it shows the change, where the
        latest start of a frame still showing the old state is a bound from
        below that lets a stale frame through -- or, for a camera without a
        clock, the frames delivered after the change's arrival position before
        the first that shows it (``delivered_frames``) as
        ``{'p50', 'p95', 'max'}`` over its ``trials``; each with the
        ``provenance`` of the run that measured it. A value with no run
        behind it is not a measurement. The gate's own counts are
        ``SKIP_FRAMES``; nothing here changes them.

        Raises:
            ValueError: the record is not ``TIMING_RECORD_SCHEMA``, names a
                source this gate has no change for, or holds a value without
                exactly one measure, its trials or its provenance.
        """
        if record.get('schema') != TIMING_RECORD_SCHEMA:
            raise ValueError(
                f'schema {record.get("schema")!r}, not {TIMING_RECORD_SCHEMA}: '
                'a record this release cannot read'
            )
        validity = record.get('validity')
        if not isinstance(validity, dict):
            raise ValueError('no validity group')
        for source, directions in validity.items():
            if source not in cls.SKIP_FRAMES:
                raise ValueError(f'validity names {source!r}, a source with no change')
            if not isinstance(directions, dict):
                raise ValueError(f'validity.{source} is not a group of directions')
            for direction, value in directions.items():
                where = f'validity.{source}.{direction}'
                if not isinstance(value, dict):
                    raise ValueError(f'{where} is not a measured value')
                measures = [m for m in TIMING_MEASURES if m in value]
                if len(measures) != 1:
                    raise ValueError(
                        f'{where} holds {measures or "no measure"}, not one of {TIMING_MEASURES}'
                    )
                measured = value[measures[0]]
                if measures[0] == 'apply_delay_us':
                    if not _is_amount(measured):
                        raise ValueError(f'{where}.apply_delay_us is not one number, 0 or more')
                elif not (
                    isinstance(measured, dict)
                    and set(measured) == {'p50', 'p95', 'max'}
                    and all(_is_amount(v) for v in measured.values())
                ):
                    raise ValueError(
                        f'{where}.{measures[0]} is not p50, p95 and max, each 0 or more'
                    )
                trials = value.get('trials')
                if not (isinstance(trials, int) and not isinstance(trials, bool) and trials > 0):
                    raise ValueError(f'{where} states no trials')
                provenance = value.get('provenance')
                if not isinstance(provenance, dict):
                    raise ValueError(f'{where} has no provenance')
                missing = [k for k in TIMING_PROVENANCE if not provenance.get(k)]
                if missing:
                    raise ValueError(f'{where} has no provenance for {missing}')

    def reset(self) -> None:
        """Clear all pending invalidations and reset frame counter.

        Deliberately leaves invalidation_counts and each source's last change
        untouched: the change is what a later frame is judged against, and
        the counts are
        a monotone history, not pending state, and a capture comparing
        snapshots across a reset must still see any invalidation the reset
        would otherwise erase. Clearing them here would let an
        invalidate-then-reset sequence hide a real state change from an
        in-flight capture.
        """
        with self._lock:
            self._pending.clear()
            self._frame_counter = 0
            self._target_values.clear()
            self._last_counted_seq = None
