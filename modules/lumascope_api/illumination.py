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
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lib import profile_trace
from lvp_logger import logger
from modules.sequential_io_executor import IOTask

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
    RUN_END = enum.auto()
    MANUAL_STEP = enum.auto()


# Transitions whose illumination must be confirmed lit before the caller moves
# on: the LED has to be on before the camera grabs a protocol step's frame
# (STEP_LIGHT) or autofocus scans for focus (AF_ENTER), or the frame captures
# dark / the focus metric reads an unlit field. apply() blocks the on-command
# for these so the confirm-before-acquire is a property of the transition, not a
# flag each caller must remember to pass.
_CONFIRM_ON_TRANSITIONS = frozenset({LedTransition.STEP_LIGHT, LedTransition.AF_ENTER})


class LedEndPolicy(enum.Enum):
    """What the LEDs do when a run ends: go dark, or return to the pre-run state."""

    OFF = enum.auto()
    RETURN_TO_ORIGINAL = enum.auto()


@dataclass(frozen=True)
class LedTransitionCtx:
    """Primitives the LED authority needs to decide a transition's target set.

    Every field is a channel number, a current, a boolean, or a set of
    (channel, mA) pairs -- never a protocol Step. The protocol-layer caller reads
    the Step and precomputes the booleans (same color as the next step, same
    z-stack group, the resolved across-move setting), then calls down with this
    context. Keeping the illumination layer free of Step parsing keeps it
    independent of the protocol schema, and a frozen primitives-only dataclass
    makes passing a Step dict a type error rather than a thing to remember not
    to do.

    Fields:
        channel: The transition's primary channel (the step / AF / preview
            color), or None when the transition lights nothing.
        mA: The primary channel's current, paired with ``channel``.
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
        snapshot_lit: The (channel, mA) pairs lit at the moment a snapshot was
            taken -- the pre-run / pre-autofocus live state to restore.
    """

    channel: int | None = None
    mA: float | None = None
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
    illumination API refuses writes from anyone else (the run-boundary
    callers consult ``led_write_allowed`` before driving the LEDs).
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
        parent: LedLease | None = None,
    ) -> None:
        self._api = api
        self.owner_name = owner_name
        self._parent = parent
        self._released = False

    def release(self, *, leave_on: bool = False) -> None:
        """Release the lease (idempotent).

        Args:
            leave_on: Keep the owner's LEDs lit instead of turning them off.
                Used when a run's declared end-state keeps illumination on.
        """
        self._api._release_led_lease(self, leave_on=leave_on)

    def acquire_child(self, owner_name: str) -> LedLease | None:
        """Take a nested lease under this one.

        The one nesting case is autofocus running inside a protocol step:
        the step holds the lease and lets autofocus drive the LED through a
        child it must outlive. Returns None if this lease is no longer held.
        """
        return self._api.acquire_led_lease(owner_name, parent=self)

    @property
    def held(self) -> bool:
        """True until this lease (or an ancestor) has been released."""
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

        Pure function of the transition and its context -- it reads no hardware
        and holds no state, so the policy is testable in isolation and identical
        for every caller. An empty set means "all channels dark."

        Args:
            transition: The lifecycle moment being decided.
            ctx: The precomputed primitives for this transition.

        Returns:
            The set of (channel, mA) pairs that should be lit afterward.

        Raises:
            ValueError: If the transition is not one the authority handles.
        """
        primary: frozenset[tuple[int, float]] = (
            frozenset({(ctx.channel, ctx.mA)})
            if ctx.channel is not None and ctx.mA is not None
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


def snapshot_lit_pairs(led_states: dict, color2ch) -> frozenset[tuple[int, float]]:
    """Convert a saved LED-state mapping to the authority's lit (channel, mA) set.

    Mirrors the filter restore_led_state uses for its restore target: a channel
    counts as lit only if it is enabled with a positive current. Used to feed a
    save_led_state snapshot into apply() as snapshot_lit.

    Args:
        led_states: color -> {'enabled': bool, 'illumination_ma': float | None}.
        color2ch: Callable mapping a color name to a channel number (or None).

    Returns:
        The (channel, mA) set of channels that should be lit.
    """
    pairs = []
    for color, state in (led_states or {}).items():
        if not state.get('enabled'):
            continue
        mA = state.get('illumination_ma') or 0
        if mA <= 0:
            continue
        ch = color2ch(color)
        if ch is not None:
            pairs.append((ch, mA))
    return frozenset(pairs)


def resolve_end_state(
    leds_state_at_end: str,
    original_led_states: dict,
    color2ch,
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
        the (channel, mA) set to restore, empty for the OFF policy.
    """
    if leds_state_at_end == 'off':
        return LedEndPolicy.OFF, frozenset()
    if leds_state_at_end == 'return_to_original':
        pairs = []
        for color, color_data in (original_led_states or {}).items():
            if not color_data.get('enabled'):
                continue
            ch = color2ch(color)
            if ch is not None:
                pairs.append((ch, color_data['illumination_ma']))
        return LedEndPolicy.RETURN_TO_ORIGINAL, frozenset(pairs)
    return None, frozenset()


class IlluminationAPI:
    """Illumination sub-API. Owns LED state, ownership tracking, and
    listener registry. Stateful bodies live here post-Phase 3d.
    """

    def __init__(self, scope: Lumascope, driver: LEDBoardProtocol) -> None:
        self._scope = scope
        # driver argument kept for API compatibility but unused; `_driver`
        # is a @property that re-resolves `self._scope._led_driver` so
        # disconnect / reconnect / test hot-swap propagate without
        # rebinding IlluminationAPI. Same pattern as MotionAPI._driver.
        del driver  # intentionally unused, kept for backward call sites

        # LED change listeners -- push-based UI update mechanism. Each
        # listener is called with (color, enabled, mA, owner) whenever
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
    def led_on(
        self, channel, mA, block: bool = False, owner: str = '', _lease_owner: str | None = None
    ) -> None:
        """Turn on an LED channel at the specified current.

        Args:
            channel: Channel number (0-5) or color name string.
            mA: Illumination current in milliamps.
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
            ValueError: If channel or mA is out of range.
        """
        if not self._driver:
            return

        if isinstance(channel, str):
            channel = self.color2ch(color=channel)

        valid_channels = self._driver.available_channels()
        if channel not in valid_channels:
            raise ValueError(f'LED channel must be one of {valid_channels}, got {channel}')
        led_max_ma = self._scope.capabilities.led_max_ma
        if not isinstance(mA, (int, float)) or mA < 0 or mA > led_max_ma:
            raise ValueError(f'LED current must be 0-{led_max_ma} mA, got {mA}')

        # Skip redundant command if channel is already on at the same current
        color_name = self.ch2color(channel)
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
                    delta = None if current_ma is None else abs(float(mA) - float(current_ma))
                except Exception:
                    delta = 'ERR'
                _api_log.info(
                    '[FX2 LED diag] led_on cache-check color=%s '
                    'new_mA=%r (type=%s) cached_mA=%r (type=%s) '
                    'delta=%r enabled=%s cache_entry=%r',
                    color_name,
                    mA,
                    type(mA).__name__,
                    current_ma,
                    type(current_ma).__name__,
                    delta,
                    is_enabled,
                    cached_entry,
                )
            if (
                current_ma is not None
                and abs(float(mA) - float(current_ma)) < 0.01
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
            self._driver.led_on(channel, mA, block=block)
        self._notify_if_led_command_failed()
        self._scope.imaging.frame_validity.invalidate('led')
        _api_log.info(f'led_on ch={channel} mA={mA} owner={owner!r}')

        # Update API-level state cache + ownership. Unconditional --
        # empty owner ('') is recorded too, so UI clicks (which arrive
        # without an owner tag) are tracked the same as named owners.
        color_name = self.ch2color(channel)
        if color_name:
            with self._led_owner_lock:
                self._led_state[color_name] = {
                    'enabled': True,
                    'illumination_ma': float(mA),
                    'owner': owner,
                }
                self._led_owners[color_name] = owner
            self._fire_led_listeners(color_name, True, float(mA), owner)

    def led_off(self, channel, owner: str = '', _lease_owner: str | None = None) -> None:
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

        if isinstance(channel, str):
            channel = self.color2ch(color=channel)

        valid_channels = self._driver.available_channels()
        if channel not in valid_channels:
            raise ValueError(f'LED channel must be one of {valid_channels}, got {channel}')

        # Skip if channel is already off. Reads from the API-level
        # _led_state cache, which is correct for both LEDBoard and FX2.
        # Prior behavior delegated to the driver's get_led_state, which
        # for FX2 always returned False -- making led_off a complete
        # no-op.
        color_name = self.ch2color(channel)
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

    def leds_off(self) -> None:
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

    def leds_off_emergency(self, *, timeout_s: float = 2.0) -> None:
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

        Called after every LED driver call (leds_off, led_on, led_off,
        leds_enable, leds_disable). Drivers that don't expose the field
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
        from modules.notification_center import notifications

        notifications.warning(
            'LED Safety',
            'LED command did not confirm',
            f'LED board did not acknowledge {op} ({reason}). If illumination '
            f'is not behaving as expected, check that the LED board is '
            f'powered and connected; turn off illumination manually '
            f'before placing a sample.',
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

    def led_on_fast(self, channel, mA) -> None:
        """Turn on an LED with write-only (no read-back) for time-critical pulses.

        Args:
            channel: Channel number (0-5) or color name string.
            mA: Illumination current in milliamps.

        Raises:
            ValueError: If channel or mA is out of range.
        """
        if not self._driver:
            return
        if isinstance(channel, str):
            channel = self.color2ch(color=channel)
        valid_channels = self._driver.available_channels()
        if channel not in valid_channels:
            raise ValueError(f'LED channel must be one of {valid_channels}, got {channel}')
        led_max_ma = self._scope.capabilities.led_max_ma
        if not isinstance(mA, (int, float)) or mA < 0 or mA > led_max_ma:
            raise ValueError(f'LED current must be 0-{led_max_ma} mA, got {mA}')
        with self._led_lock:
            self._driver.led_on_fast(channel, mA)
        self._scope.imaging.frame_validity.invalidate('led')
        color_name = self.ch2color(channel)
        if color_name:
            self._fire_led_listeners(color_name, True, float(mA), '')

    def led_off_fast(self, channel) -> None:
        """Turn off an LED with write-only (no read-back) for time-critical pulses.

        Args:
            channel: Channel number (0-5) or color name string.

        Raises:
            ValueError: If channel is out of range.
        """
        if not self._driver:
            return
        if isinstance(channel, str):
            channel = self.color2ch(color=channel)
        valid_channels = self._driver.available_channels()
        if channel not in valid_channels:
            raise ValueError(f'LED channel must be one of {valid_channels}, got {channel}')
        with self._led_lock:
            self._driver.led_off_fast(channel)
        self._scope.imaging.frame_validity.invalidate('led')
        color_name = self.ch2color(channel)
        if color_name:
            self._fire_led_listeners(color_name, False, 0.0, '')

    def leds_off_fast(self) -> None:
        """Turn off all LEDs with write-only (no read-back) for time-critical pulses."""
        if not self._driver:
            return
        with self._led_lock:
            self._driver.leds_off_fast()
        self._scope.imaging.frame_validity.invalidate('led')
        with self._led_owner_lock:
            self._led_state.clear()
        for color in self._driver.available_colors():
            self._fire_led_listeners(color, False, 0.0, '')

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
        return_future=False,
    ):
        """Guard LED connectivity, then queue an IOTask on the io_executor.

        The shared connectivity-guard + executor-resolve + enqueue path behind
        both the async LED wrappers and the blocking ``*_sync`` wrappers, so a
        disconnected board no-ops identically everywhere instead of each wrapper
        re-deriving the guard. Returns False (warning logged) when the
        controller is absent so callers can no-op uniformly; otherwise True for
        a fire-and-forget submit, or the task waiter when ``return_future`` is
        set (the ``*_sync`` wrappers block on it; it can be None when the
        executor declines the task, e.g. a protocol is running).

        Args:
            action: The bound method the IOTask runs.
            name: Caller name for the executor-required diagnostic.
            args: Positional args for ``action``.
            kwargs: Keyword args for ``action``.
            callback: Optional completion callback.
            cb_kwargs: Optional kwargs passed to the callback.
            return_future: When True, return the executor waiter to block on.
        """
        if not self._scope.led_connected:
            logger.warning('[SCOPE API ] LED controller not available.')
            return False
        ex = self._scope._require_executor(self._scope._io_executor, name)
        result = ex.put(
            IOTask(
                action=action,
                args=args,
                kwargs=kwargs,
                callback=callback,
                cb_kwargs=cb_kwargs,
            ),
            return_future=return_future,
        )
        return result if return_future else True

    def leds_off_async(self, *, callback=None) -> None:
        """Submit ``leds_off`` to the io_executor.

        No-op if LED disconnected.

        Args:
            callback: Optional completion callback.
        """
        if self._submit_io(self.leds_off, 'leds_off_async', callback=callback):
            logger.info('[SCOPE API ] leds_off_async()')

    def led_on_async(self, channel, mA, *, callback=None, cb_kwargs=None, owner: str = '') -> None:
        """Submit ``led_on(channel, mA)`` to the io_executor.

        Args:
            channel: Channel number or color name.
            mA: LED current in milliamps.
            callback: Optional completion callback.
            cb_kwargs: Optional kwargs passed to the callback.
            owner: Optional ownership tag for the LED state.
        """
        kwargs = {'owner': owner} if owner else None
        self._submit_io(
            self.led_on,
            'led_on_async',
            args=(channel, mA),
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
            self.led_off,
            'led_off_async',
            kwargs=kwargs,
            callback=callback,
            cb_kwargs=cb_kwargs,
        )

    def led_on_sync(self, channel, mA, *, timeout_s=5, owner: str = '') -> None:
        """Run ``led_on`` through the io_executor and block until done.

        Args:
            channel: Channel number or color name.
            mA: LED current in milliamps.
            timeout_s: Max seconds to wait for completion.
            owner: Optional ownership tag for the LED state.
        """
        kwargs = {'owner': owner} if owner else None
        fut = self._submit_io(
            self.led_on,
            'led_on_sync',
            args=(channel, mA),
            kwargs=kwargs,
            return_future=True,
        )
        if fut:
            fut.result(timeout=timeout_s)

    def leds_off_sync(self, *, timeout_s=5) -> None:
        """Run ``leds_off`` through the io_executor and block until done.

        Args:
            timeout_s: Max seconds to wait for completion.
        """
        fut = self._submit_io(self.leds_off, 'leds_off_sync', return_future=True)
        if fut:
            fut.result(timeout=timeout_s)

    # --- State ---
    def get_led_ma(self, color: str) -> float | None:
        """Get the current illumination level for an LED channel.

        Reads from the API-level _led_state cache. Does NOT delegate
        to the driver -- the API layer is the single source of truth.

        Args:
            color: Channel color name (e.g. "Blue", "Green", "Red", "BF").

        Returns:
            Illumination in milliamps when the channel has an active
            value set; None when the LED board is absent or the channel
            is off / never set. Use ``led_enabled(color)`` to distinguish
            "off but reachable" from "no LED board."
        """
        if not self._driver:
            return None
        with self._led_owner_lock:
            entry = self._led_state.get(color)
            return entry['illumination_ma'] if entry else None

    def led_enabled(self, color: str) -> bool:
        """Whether a specific LED channel is currently on.

        Reads from the API-level _led_state cache. Prior behavior
        delegated to the driver's get_led_state, which for
        FX2LEDController always returned False -- making led_off a
        complete no-op on FX2 cameras.

        Args:
            color: Channel color name (e.g. "Blue", "Green", "Red", "BF").

        Returns:
            True if the channel is currently on.
        """
        if not self._driver:
            return False
        with self._led_owner_lock:
            return self._led_state.get(color) is not None

    def led_illumination(self, color: str) -> float | None:
        """Current mA for an LED channel, or None if off / unavailable.

        Thin wrapper over ``get_led_ma``; see that method for the
        None-vs-float contract.

        Args:
            color: Channel color name (e.g. "Blue", "Green", "Red", "BF").

        Returns:
            Illumination in milliamps when set; None otherwise.
        """
        return self.get_led_ma(color)

    @property
    def led_states(self) -> dict:
        """Snapshot of all LED states {color: {enabled, illumination}}.

        Returns:
            Mapping of color -> {'enabled': bool, 'illumination_ma': float}.
            Empty if no LED board is connected.
        """
        if not self._driver:
            return {}
        with self._led_owner_lock:
            return {
                color: {'enabled': True, 'illumination_ma': entry['illumination_ma']}
                for color, entry in self._led_state.items()
            }

    def get_led_state(self, color: str) -> dict:
        """Get the on/off state, illumination, and owner for an LED channel.

        Reads from the API-level _led_state cache.

        Args:
            color: Channel color name (e.g. "Blue", "Green", "Red", "BF").

        Returns:
            {'enabled': bool, 'illumination_ma': float | None, 'owner': str}.
            illumination_ma is None when off / no LED board (matches the
            None sentinel contract on get_led_ma and led_illumination).
            owner is '' when off / no LED board.
        """
        if not self._driver:
            return {'enabled': False, 'illumination_ma': None, 'owner': ''}
        with self._led_owner_lock:
            entry = self._led_state.get(color)
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

    def get_led_status(self) -> int | None:
        """Get the LED board status register.

        Returns:
            Driver-defined status object (typically int bitfield), or
            None if no LED board is connected.
        """
        if not self._driver:
            return None
        return self._driver.get_status()

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
                    self.led_off(channel=color, owner=owner)
        else:
            for color in list(self.get_led_states()):
                if color not in target_on and self.led_enabled(color):
                    self.led_off(channel=color, _lease_owner=owner)

        # Re-assert the target channels; led_on self-skips channels already at
        # their target mA, so this does not blink an already-correct channel.
        for color, mA in target_on.items():
            ch = self.color2ch(color)
            if ch is not None:
                saved_owner = snapshot.get('owners', {}).get(color, '')
                # The restored owner tag is the channel's original owner, but
                # the lease check is on behalf of the restorer (e.g. AF
                # re-asserting a pre-run UI channel).
                self.led_on(channel=ch, mA=mA, owner=saved_owner, _lease_owner=owner)

    def leds_off_owned(self, owner: str) -> None:
        """Turn off only the LED channels owned by *owner*.

        Channels owned by other subsystems are left alone.

        Args:
            owner: The owner tag whose channels should be turned off.
        """
        if not self._driver or not owner:
            return
        with self._led_owner_lock:
            channels_to_off = [color for color, own in self._led_owners.items() if own == owner]
            for color in channels_to_off:
                self._led_owners.pop(color, None)
                self._led_state.pop(color, None)
        for color in channels_to_off:
            ch = self.color2ch(color)
            if ch is not None:
                with self._led_lock:
                    self._driver.led_off(ch)
                self._scope.imaging.frame_validity.invalidate('led')
                _api_log.info(f'led_off ch={ch} (owned release by {owner})')
                self._fire_led_listeners(color, False, 0.0, owner=owner)

    # --- Ownership lease ---
    def acquire_led_lease(
        self, owner_name: str, *, parent: LedLease | None = None
    ) -> LedLease | None:
        """Acquire the exclusive LED-ownership lease.

        While a lease is held, only its owner may drive the LEDs. A second
        owner's request is refused and returns None -- the caller must cope.
        It never raises, so a contended acquire cannot crash a protocol or
        autofocus run.

        Args:
            owner_name: Human-readable owner for logs ('protocol',
                'autofocus').
            parent: The caller's own lease when requesting a nested child;
                only the current holder may spawn a child.

        Returns:
            A LedLease token, or None if another owner already holds the
            lease (or a stale parent was supplied).
        """
        with self._led_lease_lock:
            active = self._led_lease_stack[-1] if self._led_lease_stack else None
            if active is not None:
                if parent is not active:
                    _api_log.warning(
                        'LED lease acquire refused: %r requested but %r holds it',
                        owner_name,
                        active.owner_name,
                    )
                    return None
            elif parent is not None:
                # A parent was supplied but nothing is held -- the parent
                # already released. Refuse rather than silently promote the
                # child to a top-level lease.
                _api_log.warning(
                    'LED lease child acquire refused for %r: parent lease not held',
                    owner_name,
                )
                return None
            lease = LedLease(self, owner_name, parent=parent)
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
            self.leds_off_owned(owner_name)
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

    def led_write_allowed(self, owner_name: str) -> bool:
        """Whether *owner_name* may drive the LEDs right now.

        True when no lease is held (live UI control is open season) or when
        owner_name matches the active (innermost) holder. An empty owner --
        a bare UI click -- is therefore allowed only while the LEDs are
        unleased.
        """
        with self._led_lease_lock:
            if not self._led_lease_stack:
                return True
            return owner_name == self._led_lease_stack[-1].owner_name

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
        """The active LED-lease owner name, or None if the LEDs are unleased."""
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
        for color in self.led_states:
            ch = self.color2ch(color)
            if ch is not None and ch not in target_channels:
                self.led_off(channel=ch, _lease_owner=owner)
        for ch, mA in target:
            self.led_on(channel=ch, mA=mA, owner=owner, block=block)

    def apply_transition(
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

        Manual step navigation runs the LED transition here so it serializes on
        the same io_executor as the stage moves (no move racing the LEDs) and
        does not block the UI thread.
        """
        kwargs = {'owner': owner} if owner else None
        self._submit_io(
            self.apply_transition,
            'apply_transition_async',
            args=(transition, ctx),
            kwargs=kwargs,
            callback=callback,
            cb_kwargs=cb_kwargs,
        )

    def force_off(self) -> None:
        """Turn off all LEDs unconditionally, bypassing any held lease.

        The unblockable safety path for emergency / error / shutdown
        callers: an idempotent off must never be refused because another
        subsystem holds the lease. When a lease is held this logs loudly so
        the bypass is visible in post-mortem; the lease itself is left
        intact, so its holder still releases normally.
        """
        held = self.led_lease_owner
        if held is not None:
            _api_log.warning('force_off bypassing held LED lease owned by %r', held)
        self.leds_off()

    def reset_led_leases(self) -> None:
        """Drop all held leases without touching the LEDs.

        The teardown path for an aborted run: free the lease so the next
        run can acquire, regardless of which (possibly dead) thread held
        it. Turning the LEDs off is the caller's separate decision via
        force_off; this only clears the ownership bookkeeping.
        """
        with self._led_lease_lock:
            if not self._led_lease_stack:
                return
            owners = [held.owner_name for held in self._led_lease_stack]
            for held in self._led_lease_stack:
                held._released = True
            self._led_lease_stack.clear()
        _api_log.warning('LED leases reset (teardown), dropped: %s', owners)

    # --- Enable / disable ---
    def leds_enable(self) -> None:
        """Enable all LED channels (allows them to be turned on)."""
        if not self._driver:
            return
        self._driver.leds_enable()
        self._notify_if_led_command_failed()

    def leds_disable(self) -> None:
        """Disable all LED channels (prevents them from turning on)."""
        if not self._driver:
            return
        self._driver.leds_disable()
        self._notify_if_led_command_failed()

    # --- Wait ---
    def wait_until_led_on(self, timeout_s: float = 5.0) -> bool:
        """Block until the LED board confirms an LED is on.

        Mirrors motion.wait_until_finished_moving in shape.

        Args:
            timeout_s: Maximum seconds to wait (default 5s).

        Returns:
            bool: True if confirmed on, False on timeout / no driver /
            firmware lacks STATUS (current state until v3.1 firmware).
        """
        if not self._driver:
            return False
        return self._driver.wait_until_on(timeout_s)

    # --- Channel mapping ---
    def ch2color(self, channel: int) -> str | None:
        """Convert a channel number to its color name string.

        Args:
            channel: Channel number (0=Blue, 1=Green, 2=Red, 3=BF, 4=PC, 5=DF).

        Returns:
            Color name (e.g. "Blue", "BF"), or None if LED board unavailable.
        """
        if not self._driver:
            return None
        return self._driver.ch2color(channel)

    def color2ch(self, color: str) -> int | None:
        """Convert a color name string to its channel number.

        Args:
            color: Color name ("Blue", "Green", "Red", "BF", "PC", "DF").

        Returns:
            Channel number (0-5), or None if LED board unavailable.
        """
        if not self._driver:
            return None
        return self._driver.color2ch(color)

    # --- Listeners ---
    def add_led_listener(self, listener) -> None:
        """Register a callback for LED state changes.

        The listener is called with ``(color, enabled, mA, owner)`` whenever
        any LED channel changes state.  It fires from the thread that caused
        the change, so listeners **must** schedule UI work via
        ``Clock.schedule_once``.

        Args:
            listener: ``callable(color: str, enabled: bool, mA: float, owner: str)``
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

    def _fire_led_listeners(self, color: str, enabled: bool, mA: float, owner: str = '') -> None:
        """Notify all LED listeners of a state change on *color*."""
        with self._led_listeners_lock:
            listeners = list(self._led_listeners)
        for fn in listeners:
            try:
                fn(color, enabled, mA, owner)
            except Exception as ex:
                _api_log.debug(f'led listener error: {ex}')
