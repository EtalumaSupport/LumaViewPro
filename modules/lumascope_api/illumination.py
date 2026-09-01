# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""IlluminationAPI -- sub-API for LED / illuminator control.

IlluminationAPI owns _led_state (single SoT), _led_owners,
_led_listeners, and the three locks that serialize their access plus
LED-driver I/O.

Channel-spec widening (set_channel / clear_channel) is a future
extension.
"""

from __future__ import annotations

import enum
import logging as _logging
import os
import threading
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lib import profile_trace
from lvp_logger import logger
from modules.exceptions import ConfigError, HardwareCommandRefusedError
from modules.sequential_io_executor import ENQUEUED, IOTask

if TYPE_CHECKING:
    from modules.lumascope_api._lumascope import Lumascope
    from drivers.protocols import LEDBoardProtocol

_api_log = _logging.getLogger('LVP.api')


def _read_fx2_wire_setting() -> bool:
    """Read fx2_debug_wire_enabled from settings.json at module import.

    Replaces the prior LVP_FX2_DEBUG_WIRE environment-variable gate.
    """
    from modules.settings_init import load_fx2_debug_wire_setting

    try:
        import lvp_logger

        base_dir = lvp_logger.lvp_appdata
    except (ImportError, AttributeError):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return load_fx2_debug_wire_setting(base_dir)


_FX2_WIRE_SETTING = _read_fx2_wire_setting()

# How long a dispatched LED write waits on the io worker before giving up.
# An LED command is a short serial write behind at most one other queued
# command, so this is a liveness bound rather than a budget -- if it ever
# expires, the worker is wedged and the caller should hear about it instead
# of blocking forever. Deliberately not a public parameter: an external
# caller has no way to know what value would be right.
_LED_WRITE_TIMEOUT_S = 5.0


class LedTransition(enum.Enum):
    """The LED-relevant moments in a run / autofocus / manual-nav lifecycle.

    The LED authority decides one target illumination set per transition. Naming
    the moments as an enum (rather than threading booleans through call sites)
    means every decider routes through a single decision function, and a caller
    cannot ask for a transition the authority does not handle.
    """

    STEP_LIGHT = enum.auto()
    AF_ENTER = enum.auto()
    AF_TO_CAPTURE = enum.auto()
    STEP_BOUNDARY = enum.auto()
    SCAN_IDLE = enum.auto()
    RUN_END = enum.auto()
    MANUAL_STEP = enum.auto()


# Transitions whose illumination must be confirmed lit before the caller moves
# on: the LED has to be on before the camera grabs a protocol step's frame
# (STEP_LIGHT) or autofocus scans for focus (AF_ENTER), or the frame captures
# dark / the focus metric reads an unlit field. apply() blocks the on-command
# for these so the confirm-before-acquire is a property of the transition, not a
# flag each caller must remember to pass.
_CONFIRM_ON_TRANSITIONS = frozenset({LedTransition.STEP_LIGHT, LedTransition.AF_ENTER})

# Transitions the submitter must NOT wait on: SCAN_IDLE is applied by the run
# loop on the failure-retry path, where the executor carrying the LED command
# may be the very thing that wedged the scan -- waiting there stalls the
# run-loop thread (delaying a user Stop and the consecutive-failure abort) for
# the full result timeout while the off cannot execute anyway until the queue
# drains. FIFO ordering on the protocol IO queue already places the off before
# any later move or step light, so nothing downstream needs the completion.
# Same idiom as _CONFIRM_ON_TRANSITIONS: waiting semantics are a property of
# the transition, not a flag each caller must remember.
FIRE_AND_FORGET_TRANSITIONS = frozenset({LedTransition.SCAN_IDLE})


class LedEndPolicy(enum.Enum):
    """What the LEDs do when a run ends: go dark, or return to the pre-run state."""

    OFF = enum.auto()
    RETURN_TO_ORIGINAL = enum.auto()


@dataclass(frozen=True)
class LedTransitionCtx:
    """Primitives the LED authority needs to decide a transition's target set.

    Every field is a channel number, a current, a boolean, or a set of
    (channel, illumination_ma) pairs -- never a protocol Step. The protocol-layer caller reads
    the Step and precomputes the booleans (same color as the next step, same
    z-stack group, the resolved across-move setting), then calls down with this
    context. Keeping the illumination layer free of Step parsing keeps it
    independent of the protocol schema, and a frozen primitives-only dataclass
    makes passing a Step dict a type error rather than a thing to remember not
    to do.

    Fields:
        channel: The transition's primary channel (the step / AF / preview
            color), or None when the transition lights nothing.
        illumination_ma: The primary channel's current, paired with ``channel``.
        same_zstack_group: This step and the next are in one z-stack group, so
            illumination is held unconditionally across the boundary.
        same_color: This step and the next request the same color.
        keep_led_across_moves: The resolved opt-in that keeps a same-color
            channel lit across a stage move (default off; brightfield speed).
        keep_led_on: Autofocus holds its channel for the following capture
            instead of restoring the pre-autofocus state.
        preview_on: Manual-nav preview is enabled, so stepping lights the step
            channel.
        is_scan_boundary: This boundary crosses into the inter-scan idle (the
            last step of a non-final scan). The LED goes dark across it
            regardless of the hold flags, so the sample is not lit during the
            wait between scans.
        is_run_end_boundary: This is the final step of the run. The boundary
            holds this channel only if the run-end target (end_policy +
            snapshot_lit) re-lights it, so the boundary off plus run-end on do
            not produce a visible end-of-acquire flicker. The hold derives from
            the run-end target itself, not a separately-computed flag.
        end_policy: The run's end-state when the run finishes.
        snapshot_lit: The (channel, illumination_ma) pairs lit at the moment a snapshot was
            taken -- the pre-run / pre-autofocus live state to restore.
    """

    channel: int | None = None
    illumination_ma: float | None = None
    same_zstack_group: bool = False
    same_color: bool = False
    keep_led_across_moves: bool = False
    keep_led_on: bool = False
    preview_on: bool = False
    is_scan_boundary: bool = False
    is_run_end_boundary: bool = False
    end_policy: LedEndPolicy = LedEndPolicy.OFF
    snapshot_lit: frozenset[tuple[int, float]] = frozenset()


class LedLease:
    """Opaque LED-ownership token handed out by IlluminationAPI.

    Holding the lease grants exclusive LED control: while it is held the
    illumination API refuses writes from anyone else. The refusal is enforced
    inside ``led_on`` / ``led_off`` themselves -- there is no separate
    permission check for a caller to consult first, and adding one would just
    create a window between the check and the write.
    Release it at the end of the run -- which turns the owner's channels
    off by default so the end-state is decided by the release, not
    reconstructed at each call site -- or use it as a context manager.

    The token cannot be forged: only the API constructs one. That is what
    makes ownership enforceable rather than the older advisory tags, where
    any caller could claim to be 'protocol'.
    """

    def __init__(
        self,
        api: IlluminationAPI,
        owner_name: str,
        alive: typing.Callable[[], bool],
        parent: LedLease | None = None,
    ) -> None:
        self._api = api
        self.owner_name = owner_name
        # The owner's authoritative in-flight fact (a run's generation-
        # scoped in-progress probe, AF's in-progress flag). This is the
        # ONLY stranded-holder evidence: thread identity was rejected as
        # an anchor because leases are acquired on caller threads (UI,
        # scripts) while the work executes on persistent worker threads,
        # so thread death proves nothing about the operation either way.
        self._alive = alive
        self._parent = parent
        self._released = False

    def release(self, *, leave_on: bool = False) -> None:
        """Release the lease (idempotent).

        Internal lease mechanics -- not part of the L2 API surface.

        Args:
            leave_on: Keep the owner's LEDs lit instead of turning them off.
                Used when a run's declared end-state keeps illumination on.
        """
        self._api._release_led_lease(self, leave_on=leave_on)

    def acquire_child(
        self, owner_name: str, *, alive: typing.Callable[[], bool]
    ) -> LedLease | None:
        """Take a nested lease under this one.

        Internal lease mechanics -- not part of the L2 API surface.

        The one nesting case is autofocus running inside a protocol step:
        the step holds the lease and lets autofocus drive the LED through a
        child it must outlive. Returns None if this lease is no longer held.

        Args:
            alive: The child owner's own in-flight probe (see
                acquire_led_lease).
        """
        return self._api.acquire_led_lease(owner_name, alive=alive, parent=self)

    @property
    def held(self) -> bool:
        """True until this lease (or an ancestor) has been released.

        Internal lease mechanics -- not part of the L2 API surface.
        """
        return not self._released

    def __enter__(self) -> LedLease:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    @staticmethod
    def target_leds(
        transition: LedTransition, ctx: LedTransitionCtx
    ) -> frozenset[tuple[int, float]]:
        """The single LED decision: which channels should be lit after a transition.

        Internal lease mechanics -- not part of the L2 API surface.

        Pure function of the transition and its context -- it reads no hardware
        and holds no state, so the policy is testable in isolation and identical
        for every caller. An empty set means "all channels dark."

        Args:
            transition: The lifecycle moment being decided.
            ctx: The precomputed primitives for this transition.

        Returns:
            The set of (channel, illumination_ma) pairs that should be lit afterward.

        Raises:
            ValueError: If the transition is not one the authority handles.
        """
        primary: frozenset[tuple[int, float]] = (
            frozenset({(ctx.channel, ctx.illumination_ma)})
            if ctx.channel is not None and ctx.illumination_ma is not None
            else frozenset()
        )
        if transition is LedTransition.STEP_LIGHT:
            return primary
        if transition is LedTransition.AF_ENTER:
            return primary
        if transition is LedTransition.AF_TO_CAPTURE:
            return primary if ctx.keep_led_on else ctx.snapshot_lit
        if transition is LedTransition.STEP_BOUNDARY:
            # A scan boundary always goes dark -- the sample must not stay lit
            # through the inter-scan idle, whatever the hold flags say.
            if ctx.is_scan_boundary:
                return frozenset()
            # Final step of the run: hold this channel only if the run-end
            # policy is about to re-light it, so the boundary off plus run-end
            # on do not blink the sample. The decision IS the run-end target,
            # so the boundary and the cleanup never derive the end state from
            # the same inputs in two places.
            if ctx.is_run_end_boundary:
                run_end_lit = {ch for ch, _ in LedLease.target_leds(LedTransition.RUN_END, ctx)}
                return primary if ctx.channel in run_end_lit else frozenset()
            # Otherwise hold within a z-stack group always, and hold across a
            # stage move only for a same-color step when the opt-in is on. Else
            # extinguish, so the default never leaves a channel lit across a move.
            hold = ctx.same_zstack_group or (ctx.same_color and ctx.keep_led_across_moves)
            return primary if hold else frozenset()
        if transition is LedTransition.SCAN_IDLE:
            # The run is between scans: everything dark, unconditionally. The
            # sample must not be lit through an inter-scan idle (an hour in an
            # hourly timelapse), whatever path the scan took to get here --
            # including a dropped final write or a transient scan failure that
            # skipped the last step's boundary decision. No ctx field is read,
            # so no caller can parameterize this wrong.
            return frozenset()
        if transition is LedTransition.RUN_END:
            return (
                ctx.snapshot_lit
                if ctx.end_policy is LedEndPolicy.RETURN_TO_ORIGINAL
                else frozenset()
            )
        if transition is LedTransition.MANUAL_STEP:
            return primary if ctx.preview_on else frozenset()
        raise ValueError(f'unhandled LED transition: {transition!r}')

    def apply(self, transition: LedTransition, ctx: LedTransitionCtx) -> None:
        """Drive the LEDs to the transition's target set.

        Internal lease mechanics -- not part of the L2 API surface.

        Diffs the target against the cached state -- the single source of truth
        for LED state -- and emits only the channels that changed. A channel
        already at its target is left untouched, so re-asserting a correct state
        produces no off-then-on blink.
        """
        if not self.held:
            # A released lease must not still drive the LEDs. By the time a
            # queued transition runs the run may be over, or a new run may hold
            # the lease under the same owner name; acting now would light or
            # extinguish a channel out of turn. Refuse loudly rather than write.
            _api_log.warning(
                'LED transition %s ignored: lease %r already released',
                transition.name,
                self.owner_name,
            )
            return
        # A held lease is authoritative over the children it spawned: a child
        # holds only delegated authority, so one still on the stack when its
        # parent acts is orphaned (its operation died without releasing in
        # order). Reclaim the top before emitting, else the diff's writes --
        # checked against the stack top -- would be silently refused by the
        # dead child and the transition would no-op.
        self._api._reclaim_lease(self)
        self._emit_diff(
            self.target_leds(transition, ctx),
            block=transition in _CONFIRM_ON_TRANSITIONS,
        )

    def _emit_diff(self, target: frozenset[tuple[int, float]], *, block: bool) -> None:
        """Drive this lease's target set through the API's canonical diff.

        Tags the emit with this lease's owner so the writes are permitted while
        the lease is held. The diff itself lives on the API (``_emit_led_diff``)
        so the unleased live-UI callers share the exact same diff-and-emit.
        ``block`` waits for the LED board to confirm an illuminate before
        returning, so a confirm-before-grab transition cannot proceed dark; the
        caller derives it from the transition, never defaults it.
        """
        self._api._emit_led_diff(target, owner=self.owner_name, block=block)


def _lit_channel_pairs(
    led_states: dict, color2ch, *, drop_nonpositive: bool
) -> frozenset[tuple[int, float]]:
    """Build the lit (channel, illumination_ma) set from a color -> state mapping.

    The single definition of "which channels count as lit" shared by the pre-run
    snapshot and the run-end restore, so the two cannot drift: if the lit
    criterion were tightened in one loop but not the other, the run-end restore
    could re-light a channel the snapshot treated as dark (or omit one it treated
    as lit). An enabled channel whose color maps to a real LED channel
    contributes its (channel, illumination_ma) pair.

    Args:
        led_states: color -> {'enabled': bool, 'illumination_ma': float | None}.
        color2ch: Callable mapping a color name to a channel number (or None).
        drop_nonpositive: When True, a channel must also carry a strictly
            positive current to count as lit, and a missing/None current reads as
            zero (so an enabled zero-current channel is treated as dark). When
            False, every enabled mapped channel is taken with its stored current
            verbatim.

    Returns:
        The (channel, illumination_ma) set of channels that should be lit.
    """
    pairs = []
    for color, state in (led_states or {}).items():
        if not state.get('enabled'):
            continue
        ch = color2ch(color)
        if ch is None:
            continue
        if drop_nonpositive:
            illumination_ma = state.get('illumination_ma') or 0
            if illumination_ma <= 0:
                continue
        else:
            illumination_ma = state['illumination_ma']
        pairs.append((ch, illumination_ma))
    return frozenset(pairs)


def live_lit_pairs(illumination: IlluminationAPI) -> frozenset[tuple[int, float]]:
    """The (channel, illumination_ma) set of channels commanded lit RIGHT NOW.

    The live-state counterpart of snapshot_lit_pairs: reads the
    illumination API's own state and channel mapping so no caller
    composes the color-to-channel direction by hand. Handing the inverse
    mapping in produces an always-empty set with no error -- the channel
    lookup misses silently on every color name -- which is why this
    composition lives here once instead of at each consumer.
    """
    return snapshot_lit_pairs(illumination.get_led_states(), illumination.state_color2ch)


def snapshot_lit_pairs(
    led_states: dict, color2ch: typing.Callable[[str], int | None]
) -> frozenset[tuple[int, float]]:
    """Convert a saved LED-state mapping to the authority's lit (channel, illumination_ma) set.

    Mirrors the filter restore_led_state uses for its restore target: a channel
    counts as lit only if it is enabled with a positive current. Used to feed a
    save_led_state snapshot into apply() as snapshot_lit.

    Args:
        led_states: color -> {'enabled': bool, 'illumination_ma': float | None}.
        color2ch: Callable mapping a color name to a channel number (or None).

    Returns:
        The (channel, illumination_ma) set of channels that should be lit.
    """
    return _lit_channel_pairs(led_states, color2ch, drop_nonpositive=True)


def resolve_end_state(
    leds_state_at_end: str,
    original_led_states: dict,
    color2ch: typing.Callable[[str], int | None],
) -> tuple[LedEndPolicy | None, frozenset[tuple[int, float]]]:
    """Map a run's end-state policy and pre-run snapshot to the LED authority's
    (end_policy, snapshot_lit).

    The single derivation of a run's end LED state, shared by run cleanup (the
    RUN_END transition) and the final-step boundary (which holds a channel only
    if run-end will re-light it). Deriving it in one place is what lets the
    boundary and the cleanup agree by construction instead of computing the same
    answer from the same inputs in two places.

    Args:
        leds_state_at_end: The run's end policy -- 'off' or 'return_to_original'.
        original_led_states: The pre-run snapshot, color -> {'enabled': bool,
            'illumination_ma': float}.
        color2ch: Callable mapping a color name to a channel number (or None
            when no LED board maps it).

    Returns:
        (end_policy, snapshot_lit). end_policy is None for an unrecognized
        policy string -- the caller decides how to surface that. snapshot_lit is
        the (channel, illumination_ma) set to restore, empty for the OFF policy.
    """
    if leds_state_at_end == 'off':
        return LedEndPolicy.OFF, frozenset()
    if leds_state_at_end == 'return_to_original':
        return LedEndPolicy.RETURN_TO_ORIGINAL, _lit_channel_pairs(
            original_led_states, color2ch, drop_nonpositive=False
        )
    return None, frozenset()


class IlluminationAPI:
    """Illumination sub-API. Owns LED state, ownership tracking, and
    listener registry. Stateful bodies live here post-Phase 3d.
    """

    # Marker returned by _resolve_channel for an off-request naming a colour
    # this scope cannot drive: the request is complete without a command.
    _OFF_NOOP = object()

    def _resolve_channel(self, channel, *, missing_off_ok: bool):
        """Map a colour-name channel argument to a numeric channel.

        The single seam where a colour string becomes a channel number.
        Drivers return None for a colour the scope cannot drive; this is
        where that None becomes behavior: on-paths raise a named error (the
        COLOUR reaches the user, never a sentinel channel number), off-paths
        (``missing_off_ok=True``) return ``_OFF_NOOP`` -- a channel the
        scope does not have is definitionally off, and the caller must
        no-op BEFORE any numeric range check. Numeric input (including a
        literal None) passes through untouched so the range checks keep
        rejecting it.

        Raises:
            ConfigError: On-path request for a colour this scope has no LED
                channel for. Typed so async task-failure popups show the
                named colour rather than a generic body.
        """
        if not isinstance(channel, str):
            return channel
        mapped = self.color2ch(color=channel)
        if mapped is not None:
            # The fusion of identity and drivability: identity says what
            # the layer IS, the driver says what the board can DRIVE. A
            # layer whose record names a channel the attached board lacks
            # must fail by name here -- driving it anyway would light
            # whatever occupies that address on this board.
            drivable = tuple(self._driver.available_channels()) if self._driver else ()
            if mapped in drivable:
                return mapped
            if missing_off_ok:
                _api_log.debug(f"led_off no-op: board cannot drive '{channel}' (ch {mapped})")
                return self._OFF_NOOP
            raise ConfigError(
                f"The attached LED board cannot drive the '{channel}' layer "
                f'(channel {mapped}; board channels: {drivable}).'
            )
        if missing_off_ok:
            _api_log.debug(f"led_off no-op: scope has no '{channel}' LED channel")
            return self._OFF_NOOP
        available = tuple(r.key_name for r in self._scope.layer_identity.layers if r.led_channel)
        raise ConfigError(f"This scope has no '{channel}' LED channel; available: {available}")

    def __init__(self, scope: Lumascope, driver: LEDBoardProtocol) -> None:
        self._scope = scope
        # driver argument kept for API compatibility but unused; `_driver`
        # is a @property that re-resolves `self._scope._led_driver` so
        # disconnect / reconnect / test hot-swap propagate without
        # rebinding IlluminationAPI. Same pattern as MotionAPI._driver.
        del driver  # intentionally unused, kept for backward call sites

        # LED change listeners -- push-based UI update mechanism. Each
        # listener is called with (channel, enabled, illumination_ma, owner) whenever
        # any LED channel changes state. Fires from the thread that
        # caused the change, so listeners MUST schedule UI work via
        # Clock.schedule_once.
        self._led_listeners_lock = threading.Lock()
        self._led_listeners: list = []

        # LED state -- API-level source of truth. The API was always
        # supposed to own LED state, but the implementation initially
        # only got as far as ownership + observers + save/restore.
        # State queries (get_led_ma, led_enabled, etc.) still delegated
        # to the driver -- which worked for LEDBoard (has an internal
        # led_ma dict) but broke for FX2LEDController (thin translator,
        # returns sentinels). This dict is the primary store, analogous
        # to _pos_cache for motor position. Updated inside led_on /
        # led_off / leds_off; read by all state-query methods.
        # Each entry: color -> {'enabled': True, 'illumination_ma': float, 'owner': str}
        self._led_state: dict[str, dict] = {}

        # LED ownership tracking -- prevents subsystems from turning
        # off LEDs they did not turn on. Each led_on with an owner
        # records who claimed the channel. led_off with a non-matching
        # owner is a no-op. leds_off() without owner is the "nuclear"
        # option (shutdown only).
        self._led_owner_lock = threading.Lock()
        self._led_owners: dict[str, str] = {}  # color -> owner tag

        # Per-device LED I/O serialization, so LED stim pulses can
        # interleave with camera grabs and motor moves on their own
        # per-device locks. Wrapped with TimedLock for contention
        # tracing.
        self._led_lock = profile_trace.TimedLock(threading.RLock(), name='illumination._led_lock')

        # LED ownership lease -- the enforced layer above the advisory
        # owner tags. One lease is held at a time (one logical owner:
        # protocol or autofocus); a second owner's acquire is refused.
        # The exception is a child lease spawned by the current holder
        # (autofocus running inside a protocol step). The stack top is the
        # active owner, and only the active owner may drive the LEDs.
        # This lock guards the stack only -- it is taken briefly and never
        # held across LED I/O, so it cannot tangle with the I/O locks.
        self._led_lease_lock = threading.Lock()
        self._led_lease_stack: list[LedLease] = []

    @property
    def _driver(self) -> LEDBoardProtocol:
        """Resolve the LED driver via the composition root each access.

        Lumascope's `_led_driver` slot is reassigned on disconnect /
        reconnect and during tests that hot-swap drivers. Re-resolving
        here keeps IlluminationAPI in sync without rebinding.
        """
        return self._scope._led_driver

    # --- Sync control ---
    def _led_on_impl(
        self,
        channel,
        illumination_ma,
        block: bool = False,
        owner: str = '',
        _lease_owner: str | None = None,
    ) -> None:
        """Turn on an LED channel at the specified current.

        Args:
            channel: Channel number (0-5) or color name string.
            illumination_ma: Illumination current in milliamps.
            block: If True, wait for confirmation from the LED board.
            owner: Optional ownership tag (e.g. 'autofocus', 'protocol').
                If set, only ``led_off`` / ``leds_off_owned`` with the same
                owner can turn this channel off.  Empty string (default) means
                no ownership tracking.
            _lease_owner: Owner to use for the LED-lease check when this
                write is an internal recomposition done on behalf of a lease
                holder (e.g. a transition diff clearing other channels on
                behalf of the run). Defaults to ``owner``; external callers
                leave it unset.

        Raises:
            ValueError: If channel or illumination_ma is out of range.
            ConfigError: If a colour name this scope cannot drive is given.
        """
        if not self._driver:
            return

        channel = self._resolve_channel(channel, missing_off_ok=False)

        valid_channels = self._driver.available_channels()
        if channel not in valid_channels:
            raise ValueError(f'LED channel must be one of {valid_channels}, got {channel}')
        led_max_ma = self._scope.capabilities.led_max_ma
        if (
            not isinstance(illumination_ma, (int, float))
            or illumination_ma < 0
            or illumination_ma > led_max_ma
        ):
            raise ValueError(f'LED current must be 0-{led_max_ma} mA, got {illumination_ma}')

        # Skip redundant command if channel is already on at the same current
        color_name = self.state_ch2color(channel)
        if color_name:
            current_ma = self.get_led_ma(color_name)
            # _led_state cache-equality trace for the slider > ~150 mA
            # silent-fail bench investigation. Gated by
            # fx2_debug_wire_enabled in settings.json to match
            # drivers/fx2driver.py.
            if _FX2_WIRE_SETTING:
                cached_entry = self._led_state.get(color_name)
                is_enabled = self.led_enabled(color_name)
                try:
                    delta = (
                        None
                        if current_ma is None
                        else abs(float(illumination_ma) - float(current_ma))
                    )
                except Exception:
                    delta = 'ERR'
                _api_log.info(
                    '[FX2 LED diag] led_on cache-check color=%s '
                    'new_mA=%r (type=%s) cached_mA=%r (type=%s) '
                    'delta=%r enabled=%s cache_entry=%r',
                    color_name,
                    illumination_ma,
                    type(illumination_ma).__name__,
                    current_ma,
                    type(current_ma).__name__,
                    delta,
                    is_enabled,
                    cached_entry,
                )
            if (
                current_ma is not None
                and abs(float(illumination_ma) - float(current_ma)) < 0.01
                and self.led_enabled(color_name)
            ):
                return

        # While a run owns the LEDs, a write from any other owner is refused
        # so a live UI change cannot disturb a protocol's or autofocus's
        # channels. Emergency / shutdown paths use force_off / leds_off,
        # which bypass this on purpose.
        violator = self._lease_violation(owner if _lease_owner is None else _lease_owner)
        if violator is not None:
            _api_log.warning(
                'LED on by %r refused: %r owns the LED lease',
                owner if _lease_owner is None else _lease_owner,
                violator,
            )
            return

        with self._led_lock:
            self._driver.led_on(channel, illumination_ma, block=block)
        self._notify_if_led_command_failed()
        self._scope.imaging.frame_validity.invalidate('led')
        _api_log.info(f'led_on ch={channel} illumination_ma={illumination_ma} owner={owner!r}')

        # Update API-level state cache + ownership. Unconditional --
        # empty owner ('') is recorded too, so UI clicks (which arrive
        # without an owner tag) are tracked the same as named owners.
        color_name = self.state_ch2color(channel)
        if color_name:
            with self._led_owner_lock:
                self._led_state[color_name] = {
                    'enabled': True,
                    'illumination_ma': float(illumination_ma),
                    'owner': owner,
                }
                self._led_owners[color_name] = owner
            self._fire_led_listeners(color_name, True, float(illumination_ma), owner)

    def _led_off_impl(self, channel, owner: str = '', _lease_owner: str | None = None) -> None:
        """Turn off an LED channel.

        Args:
            channel: Channel number (0-5) or color name string.
            owner: If set, only turn off if this owner currently owns
                the channel.  A non-matching owner is a no-op (logged).
                Empty string (default) turns off unconditionally.
            _lease_owner: Owner to use for the LED-lease check when this off
                is an internal recomposition on behalf of a lease holder.
                Defaults to ``owner``; external callers leave it unset.

        Raises:
            ValueError: If channel is out of range.
        """
        if not self._driver:
            return

        channel = self._resolve_channel(channel, missing_off_ok=True)
        if channel is self._OFF_NOOP:
            return

        valid_channels = self._driver.available_channels()
        if channel not in valid_channels:
            raise ValueError(f'LED channel must be one of {valid_channels}, got {channel}')

        # Skip if channel is already off. Reads from the API-level
        # _led_state cache, which is correct for both LEDBoard and FX2.
        # Prior behavior delegated to the driver's get_led_state, which
        # for FX2 always returned False -- making led_off a complete
        # no-op.
        color_name = self.state_ch2color(channel)
        if color_name and not self.led_enabled(color_name):
            return

        # Check ownership -- if caller specifies an owner, only allow if it matches
        if owner and color_name:
            with self._led_owner_lock:
                entry = self._led_state.get(color_name, {})
                current_owner = entry.get('owner', '')
                if current_owner and current_owner != owner:
                    _api_log.debug(
                        f'led_off blocked: ch={channel} owner={owner!r} '
                        f'but owned by {current_owner!r}'
                    )
                    return

        # Refused for the same reason as led_on (see above): an empty-owner
        # off from the live UI while a run owns the channel is the shape
        # behind the autofocus-LED-killed reports, so it is rejected here.
        violator = self._lease_violation(owner if _lease_owner is None else _lease_owner)
        if violator is not None:
            _api_log.warning(
                'LED off by %r refused: %r owns the LED lease',
                owner if _lease_owner is None else _lease_owner,
                violator,
            )
            return

        with self._led_lock:
            self._driver.led_off(channel)
        self._notify_if_led_command_failed()
        self._scope.imaging.frame_validity.invalidate('led')
        _api_log.info(f'led_off ch={channel} owner={owner!r}')

        # Clear from API-level state cache + ownership
        if color_name:
            with self._led_owner_lock:
                self._led_state.pop(color_name, None)
                self._led_owners.pop(color_name, None)
            self._fire_led_listeners(color_name, False, 0.0, owner)

    def _leds_off_impl(self) -> None:
        """Turn off all LEDs (nuclear -- ignores ownership, clears all owners)."""
        if not self._driver:
            return
        with self._led_lock:
            self._driver.leds_off()
        self._notify_if_led_command_failed()
        with self._led_owner_lock:
            self._led_owners.clear()
            self._led_state.clear()
        self._scope.imaging.frame_validity.invalidate('led')
        _api_log.info('leds_off')
        for color in self._driver.available_colors():
            self._fire_led_listeners(color, False, 0.0, '')

    # --- Public dispatch ---
    # These four are what an external caller reaches: an SDK script, a REST
    # handler, the GUI. Every internal caller and both async tiers bind the
    # matching `_impl` instead, so nothing already running on an executor
    # worker or on protocol_thread ever arrives here.

    def _dispatch_led(self, impl, name, args=(), kwargs=None):
        """Run one LED command for an external caller, on the right thread.

        Three outcomes. With no executor registered the body runs on the
        calling thread -- a bare `Lumascope()` in a script or an example has
        no executors and still has to drive hardware. With a live executor
        the body runs on the io worker, serialized against every other
        hardware write, and this blocks until it has. With an executor that
        will not accept work the caller is told so, because the alternative
        is `put` returning None and the command disappearing with nothing
        raised and nothing logged.

        The refusal asks only WHETHER work is accepted. A run disables the
        camera executor while io and file are fenced instead, and `put`
        reports both the same way, so a branch that asked WHY would need a
        list of executor states kept in sync with the executor.
        """
        kwargs = kwargs or {}
        # The board check has to live here, not be left to the body. Each
        # `_impl` opens with `if not self._driver: return`, which never fires:
        # the composition root installs a NullLEDBoard rather than None when
        # no board is present, and that object is truthy. So the body would
        # run, the Null driver would swallow the command, and the state cache
        # would go on to record the channel as lit -- the API reporting an LED
        # on with no board attached. `led_connected` is the check that
        # distinguishes a Null board from a real one.
        if not self._scope.led_connected:
            logger.warning('[SCOPE API ] LED controller not available.')
            return None
        ex = self._scope._io_executor
        if ex is None:
            return impl(*args, **kwargs)
        if not ex.accepts_work():
            raise HardwareCommandRefusedError('exclusive_activity_running', name)
        fut = ex.put(IOTask(action=impl, args=args, kwargs=kwargs), return_future=True)
        if fut is None:
            # A protocol fence can land between the check above and the
            # submit; without this the race surfaces as an AttributeError on
            # the missing future instead of the typed refusal.
            raise HardwareCommandRefusedError('exclusive_activity_running', name)
        return fut.result(timeout=_LED_WRITE_TIMEOUT_S)

    def led_on(
        self,
        channel: int | str,
        illumination_ma: float,
        block: bool = False,
        owner: str = '',
        _lease_owner: str | None = None,
    ) -> None:
        """Turn on an LED channel at the specified current, and wait for it.

        See ``_led_on_impl`` for the argument contract and the errors it
        raises; this adds only the dispatch described on ``_dispatch_led``.
        """
        return self._dispatch_led(
            self._led_on_impl,
            'led_on',
            args=(channel, illumination_ma, block, owner, _lease_owner),
        )

    def led_off(self, channel, owner: str = '', _lease_owner: str | None = None) -> None:
        """Turn off an LED channel, and wait for it.

        See ``_led_off_impl`` for the argument contract.
        """
        return self._dispatch_led(
            self._led_off_impl, 'led_off', args=(channel, owner, _lease_owner)
        )

    def leds_off(self) -> None:
        """Turn off all LEDs, and wait for it.

        Nuclear -- ignores ownership, clears all owners. See
        ``_leds_off_impl``.
        """
        return self._dispatch_led(self._leds_off_impl, 'leds_off')

    def apply_transition(
        self, transition: LedTransition, ctx: LedTransitionCtx, *, owner: str = ''
    ) -> None:
        """Drive an unleased LED transition through the authority, and wait.

        Internal LED-transition entry -- not part of the L2 API surface
        (clients use ``led_on``/``led_off``).

        See ``_apply_transition_impl`` for what a transition is and when it
        is refused for lease reasons, which is separate from the dispatch
        refusal here.
        """
        return self._dispatch_led(
            self._apply_transition_impl,
            'apply_transition',
            args=(transition, ctx),
            kwargs={'owner': owner},
        )

    def _leds_off_emergency(self, *, timeout_s: float = 2.0) -> None:
        """Bounded leds-off for atexit / abnormal-exit paths only.

        Normal `leds_off` blocks on `_led_lock` indefinitely. atexit hooks
        run on the main thread and cannot honor timeouts, so an in-flight
        LED command holding `_led_lock` would deadlock interpreter
        teardown.

        This variant uses `_led_lock.acquire(timeout=timeout_s)` with a
        log-and-skip fallback. The post-call notification / owner-clear /
        listener-fire paths are also skipped -- by the time atexit fires,
        the notification stack, state cache, and listener bus may already
        be torn down. Don't call from normal code paths; use `leds_off`
        instead.
        """
        if not self._driver:
            return
        acquired = self._led_lock.acquire(timeout=timeout_s)
        if not acquired:
            try:
                _api_log.warning(
                    f'leds_off_emergency: _led_lock held past {timeout_s}s; '
                    'skipping LED-off to avoid atexit deadlock'
                )
            except Exception:
                pass
            return
        try:
            self._driver.leds_off()
        finally:
            self._led_lock.release()

    def _notify_if_led_command_failed(self) -> None:
        """Read driver.last_command_error and fire a sample-safety
        notification if the most recent LED command did not confirm.

        Called after every LED driver call (leds_off, led_on, led_off).
        Drivers that don't expose the field
        (NullLEDBoard, SimulatedLEDBoard, FX2 -- pre-migration) are
        silently skipped via getattr-default. notification_center
        dedups by (category, title) over a 5s window so a stream of
        protocol-driven led_on failures yields one popup, not thirty.
        """
        err = getattr(self._driver, 'last_command_error', None)
        if not err:
            return
        op = err.get('op', '<unknown LED command>')
        reason = err.get('reason', 'unknown')
        _api_log.warning(f'LED command did not confirm: {op} ({reason})')
        from modules.notification_center import notifications

        notifications.warning(
            'LED Safety',
            'LED command did not confirm',
            'The illumination did not confirm a command, so it may not '
            'match what the display shows. Confirm the light is off '
            'before placing a sample. If illumination keeps misbehaving, '
            'check the USB cable and power connections, then restart '
            'LumaViewPro.',
        )
        # Clear so subsequent successful calls reset the surface.
        # Drivers that set the field will overwrite again on next failure.
        try:
            self._driver.last_command_error = None
        except Exception as e:
            logger.debug(
                '[SCOPE API ] illumination: clear last_command_error '
                'failed; driver may not implement the attribute: %s: %s',
                type(e).__name__,
                e,
            )

    # --- Async control ---
    def _submit_io(
        self,
        action,
        name,
        *,
        args=None,
        kwargs=None,
        callback=None,
        cb_kwargs=None,
    ):
        """Guard LED connectivity, then queue an IOTask on the io_executor.

        The shared connectivity-guard + executor-resolve + enqueue path behind
        the async LED wrappers, so a disconnected board no-ops identically
        everywhere instead of each wrapper re-deriving the guard.

        Returns True when the task was enqueued (or ran inline, when there is
        no executor), False when it did not: the controller is absent, or the
        executor refused the task. Every False is logged, so a caller that
        cannot act on the result still leaves a trace.

        Args:
            action: The bound method the IOTask runs.
            name: Caller name for the executor-required diagnostic.
            args: Positional args for ``action``.
            kwargs: Keyword args for ``action``.
            callback: Optional completion callback.
            cb_kwargs: Optional kwargs passed to the callback.
        """
        if not self._scope.led_connected:
            logger.warning('[SCOPE API ] LED controller not available.')
            return False
        task = IOTask(
            action=action,
            args=args,
            kwargs=kwargs,
            callback=callback,
            cb_kwargs=cb_kwargs,
        )
        ex = self._scope._io_executor
        if ex is None:
            # Nothing to defer to, so the work happens on this thread -- the
            # same rule the public dispatcher follows, so the surface has one
            # answer for "no executor" rather than one per tier. Running the
            # task rather than the bare action keeps the callback and the
            # error reporting on the production path; IOTask already falls
            # back to a direct dispatch when there is no UI dispatcher.
            # run() renames the current thread to the task's name (normally
            # the worker's); an unnamed task would blank the CALLING thread's
            # name here, so hand it the name it already has.
            task.set_name(threading.current_thread().name)
            result, exception = task.run()
            task.on_complete(result, exception)
            if exception is not None:
                raise exception
            return True
        # Success is the ENQUEUED identity rather than "not None": put() has
        # more than one way to decline a task, and the other one
        # (LIVE_FRAME_DROPPED) is truthy, so a negative test would eventually
        # read a refusal as a success.
        if ex.put(task) is not ENQUEUED:
            # Fire-and-forget callers cannot be handed an exception -- every
            # UI callsite would need a handler for a state it cannot prevent --
            # so the refusal is recorded instead of vanishing.
            logger.warning(
                f'[SCOPE API ] {name} dropped: the io executor is not accepting '
                f'work (disabled, or fenced by a running protocol)'
            )
            return False
        return True

    def leds_off_async(self, *, callback=None) -> None:
        """Submit ``leds_off`` to the io_executor.

        No-op if LED disconnected.

        Args:
            callback: Optional completion callback.
        """
        if self._submit_io(self._leds_off_impl, 'leds_off_async', callback=callback):
            logger.info('[SCOPE API ] leds_off_async()')

    def led_on_async(
        self,
        channel: int | str,
        illumination_ma: float,
        *,
        callback: typing.Callable | None = None,
        cb_kwargs: dict | None = None,
        owner: str = '',
    ) -> None:
        """Submit ``led_on(channel, illumination_ma)`` to the io_executor.

        Args:
            channel: Channel number or color name.
            illumination_ma: LED current in milliamps.
            callback: Optional completion callback.
            cb_kwargs: Optional kwargs passed to the callback.
            owner: Optional ownership tag for the LED state.
        """
        kwargs = {'owner': owner} if owner else None
        self._submit_io(
            self._led_on_impl,
            'led_on_async',
            args=(channel, illumination_ma),
            kwargs=kwargs,
            callback=callback,
            cb_kwargs=cb_kwargs,
        )

    def led_off_async(self, channel, *, callback=None, cb_kwargs=None, owner: str = '') -> None:
        """Submit ``led_off(channel)`` to the io_executor.

        Args:
            channel: Channel number or color name.
            callback: Optional completion callback.
            cb_kwargs: Optional kwargs passed to the callback.
            owner: Optional ownership tag; only matching owner can turn
                off the channel.
        """
        kwargs = {'channel': channel}
        if owner:
            kwargs['owner'] = owner
        self._submit_io(
            self._led_off_impl,
            'led_off_async',
            kwargs=kwargs,
            callback=callback,
            cb_kwargs=cb_kwargs,
        )

    # --- State ---
    def get_led_ma(self, channel: str) -> float | None:
        """Get the current illumination level for an LED channel.

        Reads from the API-level _led_state cache. Does NOT delegate
        to the driver -- the API layer is the single source of truth.

        Args:
            channel: Channel name (e.g. "Blue", "Green", "Red", "BF").

        Returns:
            Illumination in milliamps when the channel has an active
            value set; None when the LED board is absent or the channel
            is off / never set. Use ``led_enabled(channel)`` to distinguish
            "off but reachable" from "no LED board."
        """
        if not self._driver:
            return None
        with self._led_owner_lock:
            entry = self._led_state.get(channel)
            return entry['illumination_ma'] if entry else None

    def led_enabled(self, channel: str) -> bool:
        """Whether a specific LED channel is currently on.

        Reads from the API-level _led_state cache. Prior behavior
        delegated to the driver's get_led_state, which for
        FX2LEDController always returned False -- making led_off a
        complete no-op on FX2 cameras.

        Args:
            channel: Channel name (e.g. "Blue", "Green", "Red", "BF").

        Returns:
            True if the channel is currently on.
        """
        if not self._driver:
            return False
        with self._led_owner_lock:
            return self._led_state.get(channel) is not None

    def get_led_state(self, channel: str) -> dict:
        """Get the on/off state, illumination, and owner for an LED channel.

        Reads from the API-level _led_state cache.

        Args:
            channel: Channel name (e.g. "Blue", "Green", "Red", "BF").

        Returns:
            {'enabled': bool, 'illumination_ma': float | None, 'owner': str}.
            illumination_ma is None when off / no LED board (matches the
            None sentinel contract on get_led_ma).
            owner is '' when off / no LED board.
        """
        if not self._driver:
            return {'enabled': False, 'illumination_ma': None, 'owner': ''}
        with self._led_owner_lock:
            entry = self._led_state.get(channel)
            if entry is None:
                return {'enabled': False, 'illumination_ma': None, 'owner': ''}
            return {
                'enabled': True,
                'illumination_ma': entry['illumination_ma'],
                'owner': entry.get('owner', ''),
            }

    def get_led_states(self) -> dict:
        """Get state, illumination, and owner for all LED channels.

        Returns states for ALL channels the driver supports (not just
        currently-on channels).

        Returns:
            Mapping of color -> {'enabled': bool, 'illumination_ma': float | None,
            'owner': str} for every channel the driver supports.
            illumination_ma is None and owner is '' when the channel
            is off. Empty if no LED board is connected.
        """
        if not self._driver:
            return {}
        all_colors = self._driver.available_colors()
        with self._led_owner_lock:
            return {
                color: (
                    {
                        'enabled': True,
                        'illumination_ma': self._led_state[color]['illumination_ma'],
                        'owner': self._led_state[color].get('owner', ''),
                    }
                    if color in self._led_state
                    else {'enabled': False, 'illumination_ma': None, 'owner': ''}
                )
                for color in all_colors
            }

    # --- Save / restore ---
    def save_led_state(self, tag: str) -> dict:
        """Snapshot the current LED state for later restoration.

        Args:
            tag: Descriptive name for the snapshot (for logging).

        Returns:
            Snapshot suitable for passing to ``restore_led_state``.
        """
        states = self.get_led_states()
        with self._led_owner_lock:
            owners = dict(self._led_owners)
        snapshot = {'tag': tag, 'states': states, 'owners': owners}
        _api_log.info(
            f'save_led_state tag={tag}: {[c for c, s in states.items() if s.get("enabled")]}'
        )
        return snapshot

    def restore_led_state(self, snapshot: dict, owner: str = '') -> None:
        """Restore LEDs to a previously saved state.

        Turns off channels owned by *owner* (or all if owner is empty),
        then re-enables channels that were on in the snapshot.

        Args:
            snapshot: Return value from ``save_led_state``.
            owner: If set, only turn off channels currently owned by
                this owner before restoring.
        """
        if not snapshot:
            return
        tag = snapshot.get('tag', '?')
        saved_states = snapshot.get('states', {})
        _api_log.info(f'restore_led_state tag={tag}')

        # Channels that should be ON after restore, with their target mA.
        target_on = {
            color: state.get('illumination_ma', 0)
            for color, state in saved_states.items()
            if state.get('enabled', False) and (state.get('illumination_ma') or 0) > 0
        }

        # Turn off only channels that should NOT be on after restore, so a
        # channel already lit at its target is left untouched (no off-then-on
        # blink). With an owner, restrict the turn-off to that owner's channels
        # and leave other subsystems' channels alone; without an owner, clear
        # every currently-lit channel that is not part of the restore target.
        if owner:
            with self._led_owner_lock:
                owned = [c for c, own in self._led_owners.items() if own == owner]
            for color in owned:
                if color not in target_on:
                    self._led_off_impl(channel=color, owner=owner)
        else:
            for color in list(self.get_led_states()):
                if color not in target_on and self.led_enabled(color):
                    self._led_off_impl(channel=color, _lease_owner=owner)

        # Re-assert the target channels; led_on self-skips channels already at
        # their target mA, so this does not blink an already-correct channel.
        for color, illumination_ma in target_on.items():
            ch = self.state_color2ch(color)
            if ch is not None:
                saved_owner = snapshot.get('owners', {}).get(color, '')
                # The restored owner tag is the channel's original owner, but
                # the lease check is on behalf of the restorer (e.g. AF
                # re-asserting a pre-run UI channel).
                self._led_on_impl(
                    channel=ch,
                    illumination_ma=illumination_ma,
                    owner=saved_owner,
                    _lease_owner=owner,
                )

    def leds_off_owned(self, owner: str) -> None:
        """Turn off only the LED channels owned by *owner*.

        Channels owned by other subsystems are left alone. See
        ``_leds_off_owned_impl`` for the body; this adds only the dispatch
        described on ``_dispatch_led``.

        Args:
            owner: The owner tag whose channels should be turned off.
        """
        return self._dispatch_led(self._leds_off_owned_impl, 'leds_off_owned', args=(owner,))

    def _leds_off_owned_impl(self, owner: str) -> None:
        """Turn off only the LED channels owned by *owner*.

        Lease release binds this directly rather than the dispatcher:
        teardown runs while a protocol fence is up, where the dispatcher
        rightly refuses external work.
        """
        if not self._driver or not owner:
            return
        with self._led_owner_lock:
            channels_to_off = [color for color, own in self._led_owners.items() if own == owner]
            for color in channels_to_off:
                self._led_owners.pop(color, None)
                self._led_state.pop(color, None)
        for color in channels_to_off:
            ch = self.state_color2ch(color)
            if ch is not None:
                with self._led_lock:
                    self._driver.led_off(ch)
                self._scope.imaging.frame_validity.invalidate('led')
                _api_log.info(f'led_off ch={ch} (owned release by {owner})')
                self._fire_led_listeners(color, False, 0.0, owner=owner)

    # --- Ownership lease ---
    def _holder_is_stranded(self, lease: LedLease) -> str | None:
        """The evidence that *lease*'s owner is dead, or None if it is live.

        A holder is stranded only when that is PROVABLE: its own liveness
        probe answers False (the run/AF that acquired it is no longer in
        flight). Anything else is a live holder, however inconvenient for
        the contender. The probe is the sole evidence -- thread identity
        was rejected as an anchor (acquiring threads are callers, not the
        executing workers, so thread death proves nothing).
        """
        try:
            if not lease._alive():
                return 'liveness probe returned False'
        except Exception as ex:
            return f'liveness probe raised {type(ex).__name__}: {ex}'
        return None

    def acquire_led_lease(
        self,
        owner_name: str,
        *,
        alive: typing.Callable[[], bool],
        parent: LedLease | None = None,
    ) -> LedLease | None:
        """Acquire the exclusive LED-ownership lease.

        Internal run-exclusivity machinery -- not part of the L2 API
        surface (clients drive ``led_on``/``led_off``; a refusal names the
        holder).

        While a lease is held, only its owner may drive the LEDs.
        Contention is arbitrated HERE, on the resource, not at call
        sites: a holder whose owner is provably dead (its liveness probe
        answers False) is reclaimed with
        the evidence logged; a LIVE holder refuses the requester, and a
        refused requester must refuse its own operation -- no caller may
        reset the stack out from under a live owner. It never raises on
        contention, so a contended acquire cannot crash a protocol or
        autofocus run.

        Args:
            owner_name: Human-readable owner for logs ('protocol',
                'autofocus').
            alive: The owner's authoritative in-flight fact (e.g. the
                run's in-progress event's is_set, AF's in-progress flag).
                Must already answer True at acquire time; this is what
                lets a LATER contender distinguish this holder's death
                from its mere inconvenience.
            parent: The caller's own lease when requesting a nested child;
                only the current holder may spawn a child.

        Returns:
            A LedLease token, or None if a live owner already holds the
            lease (or a stale parent was supplied).

        Raises:
            ValueError: alive() did not answer True at acquire time -- a
                misordered probe would silently create a window in which
                this holder looks stranded and can be reclaimed.
        """
        if not alive():
            raise ValueError(
                f'LED lease acquire for {owner_name!r}: the alive probe must '
                'answer True at acquire time (set the in-flight fact before '
                'acquiring)'
            )
        with self._led_lease_lock:
            active = self._led_lease_stack[-1] if self._led_lease_stack else None
            if active is not None and parent is not active:
                # Arbitrate against the stack ROOT: descendants stand and
                # fall with the top-level owner that spawned them.
                root = self._led_lease_stack[0]
                evidence = self._holder_is_stranded(root)
                if evidence is None:
                    _api_log.warning(
                        'LED lease acquire refused: %r requested but %r holds it and is live',
                        owner_name,
                        active.owner_name,
                    )
                    return None
                dropped = [held.owner_name for held in self._led_lease_stack]
                for held in self._led_lease_stack:
                    held._released = True
                self._led_lease_stack.clear()
                _api_log.warning(
                    'LED lease stack reclaimed from stranded owner %r (%s); '
                    'dropped: %s; granting to %r',
                    root.owner_name,
                    evidence,
                    dropped,
                    owner_name,
                )
                if parent is not None:
                    # The requester wanted a child of a lease that just fell
                    # with the reclaimed stack; a child cannot outlive its
                    # parent, so the grant below would dangle.
                    _api_log.warning(
                        'LED lease child acquire refused for %r: parent fell '
                        'with the reclaimed stack',
                        owner_name,
                    )
                    return None
            elif active is None and parent is not None:
                # A parent was supplied but nothing is held -- the parent
                # already released. Refuse rather than silently promote the
                # child to a top-level lease.
                _api_log.warning(
                    'LED lease child acquire refused for %r: parent lease not held',
                    owner_name,
                )
                return None
            lease = LedLease(self, owner_name, alive=alive, parent=parent)
            self._led_lease_stack.append(lease)
            _api_log.info(
                'LED lease acquired by %r (depth=%d)', owner_name, len(self._led_lease_stack)
            )
            return lease

    def _release_led_lease(self, lease: LedLease, *, leave_on: bool = False) -> None:
        """Release a lease (called via LedLease.release). Idempotent.

        By default the owner's channels are turned off, so the LED
        end-state is a property of the release. An owner whose declared
        end-state keeps illumination on passes leave_on=True.
        """
        with self._led_lease_lock:
            if lease._released:
                return
            if lease not in self._led_lease_stack:
                lease._released = True
                return
            # Normal use is last-in-first-out (a child releases before its
            # parent). An out-of-order release means a child outlived its
            # parent; drop the whole tail above this lease so the stack
            # cannot wedge and lock out the next run.
            idx = self._led_lease_stack.index(lease)
            for stranded in self._led_lease_stack[idx:]:
                stranded._released = True
            del self._led_lease_stack[idx:]
            owner_name = lease.owner_name
        if not leave_on:
            # The impl, not the dispatcher: release runs in fenced run-teardown
            # contexts where the dispatcher rightly refuses external work.
            self._leds_off_owned_impl(owner_name)
        _api_log.info('LED lease released by %r%s', owner_name, ' (leave_on)' if leave_on else '')

    def _reclaim_lease(self, lease: LedLease) -> None:
        """Make *lease* the active (top) owner, releasing any descendants above it.

        The symmetric twin of the out-of-order tail-drop in
        ``_release_led_lease``: there a child outliving its parent's release
        drops the stranded tail; here a held parent that needs to act reclaims
        from descendants that never released. A held lease is authoritative
        over what it spawned, so a child still stacked above it has been
        orphaned (e.g. an autofocus run wedged past its abort wait and never
        ran its release). No-op when *lease* is already the top or has been
        released. Only the held lease's own write paths call this, so it cannot
        steal authority from a live sibling -- there are no siblings, only a
        single ownership stack.
        """
        with self._led_lease_lock:
            if lease._released or lease not in self._led_lease_stack:
                return
            idx = self._led_lease_stack.index(lease)
            stranded = self._led_lease_stack[idx + 1 :]
            for held in stranded:
                held._released = True
            del self._led_lease_stack[idx + 1 :]
        if stranded:
            _api_log.warning(
                'LED lease %r reclaimed top from orphaned descendants: %s',
                lease.owner_name,
                [held.owner_name for held in stranded],
            )

    def _lease_violation(self, owner: str) -> str | None:
        """The active lease owner if *owner* may NOT write right now, else None.

        One lock acquisition for the LED-write paths to surface out-of-turn
        writes. Returns None when the write is permitted (no lease held, or
        owner matches the active holder).
        """
        with self._led_lease_lock:
            if not self._led_lease_stack:
                return None
            active = self._led_lease_stack[-1].owner_name
            return None if owner == active else active

    @property
    def led_lease_owner(self) -> str | None:
        """The active LED-lease owner name, or None if the LEDs are unleased.

        Internal lease introspection -- not part of the L2 API surface
        (refusals carry the holder).
        """
        with self._led_lease_lock:
            return self._led_lease_stack[-1].owner_name if self._led_lease_stack else None

    def _emit_led_diff(
        self, target: frozenset[tuple[int, float]], *, owner: str, block: bool
    ) -> None:
        """Turn off lit channels not in the target, then assert the target.

        The single diff-and-emit every transition drives through -- both the
        leased run callers (via LedLease) and the unleased live-UI callers (via
        apply_transition). Trusts the state cache to skip already-correct
        channels: led_on self-skips a channel already at its current and led_off
        self-skips a dark one, so re-asserting a correct target emits nothing (no
        off-then-on blink). The off clears the channel regardless of who lit it
        but checks the lease as ``owner`` so it is permitted while that owner
        holds the lease (or while no lease is held, for an unleased UI write).
        ``block`` waits for the board to confirm each illuminate before
        returning -- set for a transition whose LED must be on before the
        camera grabs; the off does not block (clearing a channel never gates a
        grab). ``restore_led_state`` is an owner-scoped variant of this same
        off-non-target-then-reassert diff (it offs only the restoring owner's
        channels before re-asserting the snapshot); keep its blink-avoidance
        consistent with this primitive.
        """
        target_channels = {ch for ch, _ in target}
        # Snapshot under the lock, then release before emitting: the off/on
        # primitives re-acquire _led_owner_lock (not reentrant) and _led_off_impl
        # pops from _led_state as it clears each channel.
        with self._led_owner_lock:
            lit_colors = list(self._led_state)
        for color in lit_colors:
            ch = self.state_color2ch(color)
            if ch is not None and ch not in target_channels:
                self._led_off_impl(channel=ch, _lease_owner=owner)
        for ch, illumination_ma in target:
            self._led_on_impl(channel=ch, illumination_ma=illumination_ma, owner=owner, block=block)

    def _apply_transition_impl(
        self, transition: LedTransition, ctx: LedTransitionCtx, *, owner: str = ''
    ) -> None:
        """Drive an unleased LED transition through the authority.

        The lease-free counterpart to LedLease.apply, for live-UI writers that
        hold no lease -- LED control is open season while nothing is leased, but
        the decision still routes through target_leds and the emit through the
        one diff, so a no-op transition (a same-color re-navigation) blinks
        nothing. Refuses while a lease is held: a run owns the LEDs, and a stray
        UI write must not cut in mid-run rather than emit a partial diff the
        per-channel lease check would reject anyway.
        """
        violator = self._lease_violation(owner)
        if violator is not None:
            _api_log.warning(
                'LED transition %s ignored: LEDs leased by %r, not the unleased UI',
                transition.name,
                violator,
            )
            return
        self._emit_led_diff(
            LedLease.target_leds(transition, ctx),
            owner=owner,
            block=transition in _CONFIRM_ON_TRANSITIONS,
        )

    def apply_transition_async(
        self,
        transition: LedTransition,
        ctx: LedTransitionCtx,
        *,
        owner: str = '',
        callback=None,
        cb_kwargs=None,
    ) -> None:
        """Submit ``apply_transition`` to the io_executor.

        Internal -- the GUI's non-blocking submission of
        ``apply_transition``; not part of the L2 API surface.

        Manual step navigation runs the LED transition here so it serializes on
        the same io_executor as the stage moves (no move racing the LEDs) and
        does not block the UI thread.
        """
        kwargs = {'owner': owner} if owner else None
        self._submit_io(
            self._apply_transition_impl,
            'apply_transition_async',
            args=(transition, ctx),
            kwargs=kwargs,
            callback=callback,
            cb_kwargs=cb_kwargs,
        )

    def force_off(self) -> None:
        """Turn off all LEDs unconditionally, bypassing any held lease.

        Internal lease-bypassing safety off for the writer path -- not
        part of the L2 API surface (clients use ``leds_off``).

        The unblockable safety path for emergency / error / shutdown
        callers: an idempotent off must never be refused because another
        subsystem holds the lease. When a lease is held this logs loudly so
        the bypass is visible in post-mortem; the lease itself is left
        intact, so its holder still releases normally.
        """
        held = self.led_lease_owner
        if held is not None:
            _api_log.warning('force_off bypassing held LED lease owned by %r', held)
        self._leds_off_impl()

    # --- Enable / disable ---
    # --- Wait ---
    # --- Channel mapping ---
    # Two mappings, deliberately separate. IDENTITY (color2ch/ch2color)
    # answers from the unit's resolved layer records: what a layer IS on
    # this unit. STATE (state_color2ch/state_ch2color) answers from the
    # driver's own table: what the board can physically address. Cache
    # bookkeeping, restore snapshots, and extinguish sweeps ride the
    # STATE mapping so a channel that is physically lit is always
    # recordable and extinguishable even when the current identity
    # cannot name it (a mid-session identity change must never strand a
    # lit LED or drop it from a restore set).
    def ch2color(self, channel: int) -> str | None:
        """The stable layer name whose record drives *channel*, else None.

        Answers from the unit's resolved layer identity -- the same
        records `scope.layer_identity` exposes.
        """
        for record in self._scope.layer_identity.layers:
            if channel in record.led_channel:
                return record.key_name
        return None

    def color2ch(self, color: str) -> int | None:
        """The LED board address of the layer named *color*, else None.

        Answers from the unit's resolved layer identity by stable
        `key_name`. None means the layer is unknown to this unit's
        identity OR drives no LED (luminescence): on-paths turn that
        into a named error at `_resolve_channel`, off-paths no-op.
        """
        record = self._scope.layer_identity.find(color)
        if record is None or not record.led_channel:
            return None
        return record.led_channel[0]

    def state_ch2color(self, channel: int) -> str | None:
        """The DRIVER's name for *channel* -- state bookkeeping only,
        not part of the L2 API surface. Identity questions use
        `ch2color`; this exists so state records and extinguish paths
        follow the board's own table.
        """
        if not self._driver:
            return None
        return self._driver.ch2color(channel)

    def state_color2ch(self, color: str) -> int | None:
        """The DRIVER's channel for *color* -- state bookkeeping only,
        not part of the L2 API surface. Restore snapshots are keyed by
        the names the state store recorded at light time; replaying them
        through the driver's table guarantees a lit channel can always
        be re-addressed, whatever the current identity says.
        """
        if not self._driver:
            return None
        return self._driver.color2ch(color)

    # --- Listeners ---
    def add_led_listener(self, listener: typing.Callable) -> None:
        """Register a callback for LED state changes.

        The listener is called with ``(channel, enabled, illumination_ma, owner)`` whenever
        any LED channel changes state.  It fires from the thread that caused
        the change, so listeners **must** schedule UI work via
        ``Clock.schedule_once``.

        Args:
            listener: ``callable(channel: str, enabled: bool, illumination_ma: float, owner: str)``
        """
        with self._led_listeners_lock:
            self._led_listeners.append(listener)

    def remove_led_listener(self, listener) -> None:
        """Unregister an LED listener.

        Args:
            listener: A callable previously passed to ``add_led_listener``.
                Silently ignores listeners that are not currently registered.
        """
        with self._led_listeners_lock:
            try:
                self._led_listeners.remove(listener)
            except ValueError:
                pass

    def _fire_led_listeners(
        self, channel: str, enabled: bool, illumination_ma: float, owner: str = ''
    ) -> None:
        """Notify all LED listeners of a state change on *channel*."""
        with self._led_listeners_lock:
            listeners = list(self._led_listeners)
        for fn in listeners:
            try:
                fn(channel, enabled, illumination_ma, owner)
            except Exception as ex:
                _api_log.debug(f'led listener error: {ex}')
