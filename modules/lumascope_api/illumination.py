# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""IlluminationAPI -- sub-API for LED / illuminator control.

IlluminationAPI owns _led_state (single SoT, by LED address), _lit_by,
_led_listeners, and the three locks that serialize their access plus
LED-driver I/O.

Channel-spec widening (set_channel / clear_channel) is a future
extension.
"""

from __future__ import annotations

import enum
import functools
import logging as _logging
import threading
from dataclasses import dataclass
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from drivers import host_clock
from drivers.exceptions import BoardNoReplyError
from drivers.null_ledboard import NullLEDBoard
from lib import profile_trace
from lib.lumascope_error import Fault
from modules.finite_number import refuse_unless_finite_number
from modules.exceptions import (
    ArgumentRefusedError,
    HardwareCommandRefusedError,
    MissingPart,
)
from modules.layer_record import IdentitySource, refuse_unknown_layer
from modules.sequential_io_executor import IOTask
from modules.api_surface import Admission, api, api_fields
from modules.frame_validity import FrameValiditySource

if TYPE_CHECKING:
    from modules.activity_claim import Taking
    from modules.lumascope_api._lumascope import Lumascope
    from drivers.protocols import LEDBoardProtocol

_api_log = _logging.getLogger('LVP.api')


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
    Release it at the end of the run -- which turns off the channels this
    lease lit, by default, so the end-state is decided by the release, not
    reconstructed at each call site -- or use it as a context manager.

    The token cannot be forged: only the API constructs one, and a write is
    permitted by holding the token, never by naming it. ``purpose`` labels
    the lease in logs and refusal text and is never compared.
    """

    def __init__(
        self,
        api: IlluminationAPI,
        purpose: str,
        *,
        claim: Taking | None,
        parent: LedLease | None,
    ) -> None:
        self._api = api
        self.purpose = purpose
        # A top-level lease is taken under a held claim, and that claim is
        # the ONLY stranded-holder evidence: once its taking no longer holds
        # the claim, the activity that took the lease is over. Thread
        # identity was rejected as an anchor because leases are acquired on
        # caller threads (UI, scripts) while the work executes on persistent
        # worker threads, so thread death proves nothing either way. A child
        # carries no claim: it stands and falls with its root.
        self._claim = claim
        self._parent = parent
        self._released = False

    def release(self, *, leave_on: bool = False) -> None:
        """Release the lease (idempotent).

        Internal lease mechanics -- not part of the L2 API surface.

        Args:
            leave_on: Keep this lease's LEDs lit instead of turning them off.
                Used when a run's declared end-state keeps illumination on.
        """
        self._api._release_led_lease(self, leave_on=leave_on)

    def acquire_child(self, purpose: str) -> LedLease | None:
        """Take a nested lease under this one.

        Internal lease mechanics -- not part of the L2 API surface.

        The one nesting case is autofocus running inside a protocol step:
        the step holds the lease and lets autofocus drive the LED through a
        child it must outlive. Returns None if this lease is no longer held.
        A child left behind by a dead autofocus is dropped by this lease's
        next write (``_reclaim_lease``), so it needs no liveness of its own.
        """
        return self._api._acquire_led_lease(purpose, claim=None, parent=self)

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
        """Drive the LEDs to the transition's target set, on the io lane.

        Internal lease mechanics -- not part of the L2 API surface.

        Diffs the target against the cached state -- the single source of truth
        for LED state -- and emits only the channels that changed. A channel
        already at its target is left untouched, so re-asserting a correct state
        produces no off-then-on blink. Dispatched like every other LED write,
        so it runs on the io worker under the caller's taking: inline when the
        caller is already a task there, through the run's door otherwise.
        """
        return self._api._dispatch_led(self._apply_impl, 'LedLease.apply', args=(transition, ctx))

    def _apply_impl(self, transition: LedTransition, ctx: LedTransitionCtx) -> None:
        if not self.held:
            # A released lease must not still drive the LEDs. By the time a
            # queued transition runs the run may be over, or a new run may hold
            # a lease of the same purpose; acting now would light or
            # extinguish a channel out of turn. Refuse loudly rather than write.
            _api_log.warning(
                'LED transition %s ignored: lease %r already released',
                transition.name,
                self.purpose,
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

        Makes the writes as this lease, so they are permitted while it is the
        active holder. The diff itself lives on the API (``_emit_led_diff``)
        so the unleased live-UI callers share the exact same diff-and-emit.
        ``block`` waits for the LED board to confirm an illuminate before
        returning, so a confirm-before-grab transition cannot proceed dark; the
        caller derives it from the transition, never defaults it.
        """
        self._api._emit_led_diff(target, lease=self, block=block, member='LedLease.apply')


def _lit_channel_pairs(
    readings: tuple[LedReading, ...] | None, *, drop_nonpositive: bool
) -> frozenset[tuple[int, float]]:
    """Build the lit (address, illumination_ma) set from LED readings.

    The single definition of "which addresses count as lit" shared by the
    pre-run snapshot and the run-end restore, so the two cannot drift: if the
    lit criterion were tightened in one loop but not the other, the run-end
    restore could re-light an address the snapshot treated as dark (or omit one
    it treated as lit). A lit reading contributes its (address,
    illumination_ma) pair, whether or not the identity names it.

    Args:
        readings: The readings, as ``get_led_states`` answers them; None (no
            LED board installed) is nothing lit.
        drop_nonpositive: When True, an address must also carry a strictly
            positive current to count as lit (an address lit at 0 mA is
            treated as dark). When False, every lit address is taken with
            its current.

    Returns:
        The (address, illumination_ma) set of addresses that should be lit.
    """
    return frozenset(
        (r.address, r.illumination_ma)
        for r in readings or ()
        if r.state == LedState.LIT and (r.illumination_ma > 0 or not drop_nonpositive)
    )


def live_lit_pairs(illumination: IlluminationAPI) -> frozenset[tuple[int, float]]:
    """The (address, illumination_ma) set of addresses commanded lit RIGHT NOW.

    The live-state counterpart of snapshot_lit_pairs, read from the
    illumination API's own store.
    """
    return snapshot_lit_pairs(illumination.get_led_states())


def snapshot_lit_pairs(readings: tuple[LedReading, ...] | None) -> frozenset[tuple[int, float]]:
    """Convert saved LED readings to the authority's lit (address, illumination_ma) set.

    Mirrors the filter restore_led_state uses for its restore target: an
    address counts as lit only if it is lit at a positive current. Used to
    feed a save_led_state snapshot into apply() as snapshot_lit.
    """
    return _lit_channel_pairs(readings, drop_nonpositive=True)


def lit_other_than(
    illumination: IlluminationAPI, layer: str, lit: Mapping[int, float] | None = None
) -> list[str]:
    """Every LED lit but ``layer``'s, by channel name, else by board address.

    The one answer to whether a frame named for ``layer`` is lit by another
    light: a capture held at a layer (``ImagingAPI._capture_holding_layer_impl``),
    the still and a manual recording's start each refuse when it is not empty.

    Args:
        illumination: The LED store, read for which channel each address is.
        layer: The layer the frame is named for.
        lit: The addresses lit, from a frame's record (``FrameRecord.illumination_ma``);
            None reads them from the store now, ``snapshot_lit_pairs``'s meaning.
    """
    readings = illumination.get_led_states() or ()
    if lit is None:
        lit = dict(snapshot_lit_pairs(readings))
    return [
        r.name or f'LED address {r.address}'
        for r in readings
        if r.address in lit and r.name != layer
    ]


def resolve_end_state(
    end_policy: LedEndPolicy,
    original_led_states: tuple[LedReading, ...] | None,
) -> frozenset[tuple[int, float]]:
    """The channels a run's end lights, from its end policy and pre-run snapshot.

    The single derivation of a run's end LED state, shared by run cleanup (the
    RUN_END transition) and the final-step boundary (which holds a channel only
    if run-end will re-light it). Deriving it in one place is what lets the
    boundary and the cleanup agree by construction instead of computing the same
    answer from the same inputs in two places.

    Args:
        end_policy: The run's end policy (``SequencedCaptureRunMode.led_end_policy``).
        original_led_states: The pre-run readings (``get_led_states``).

    Returns:
        The (channel, illumination_ma) set to restore, empty for the OFF policy.
    """
    if end_policy is LedEndPolicy.OFF:
        return frozenset()
    return _lit_channel_pairs(original_led_states, drop_nonpositive=False)


class LedState(enum.StrEnum):
    """What an LED address is doing, as far as the API can vouch for it."""

    LIT = 'lit'
    DARK = 'dark'
    UNKNOWN = 'unknown'


@api_fields('address', 'name', 'state', 'illumination_ma')
@dataclass(frozen=True)
class LedReading:
    """One LED address: the layer it lights, what it is doing, and its current.

    The answer of every LED read and the ``led`` event.

    Attributes:
        address: The LED board address.
        name: The layer the unit's identity names for the address, read at
            the moment of the reading; None when the identity names none,
            which never hides the address.
        state: Lit, dark, or unknown.
        illumination_ma: The commanded current in mA when lit; None
            otherwise.
    """

    address: int
    name: str | None
    state: LedState
    illumination_ma: float | None

    def __post_init__(self) -> None:
        if (self.illumination_ma is not None) != (self.state == LedState.LIT):
            raise ValueError(
                f'a {self.state} LED reading with illumination_ma={self.illumination_ma!r}: '
                'only a lit address carries a current'
            )


@api_fields('tag', 'readings')
@dataclass(frozen=True)
class LedSnapshot:
    """Every LED address as it read at one moment, for ``restore_led_state``.

    Attributes:
        tag: A label for logs.
        readings: One reading per address of the board.
    """

    tag: str
    readings: tuple[LedReading, ...]


@dataclass(frozen=True)
class _Held:
    """What the store holds for an address it does not hold dark.

    Lit at a current, or unknown. Dark is the address's absence from the
    store, so the store cannot hold an address both dark and lit.
    """

    state: LedState
    illumination_ma: float | None = None

    def __post_init__(self) -> None:
        if self.state == LedState.DARK:
            raise ValueError('a dark LED address is absent from the store, never held')
        if (self.illumination_ma is not None) != (self.state == LedState.LIT):
            raise ValueError(
                f'a held {self.state} LED address with illumination_ma={self.illumination_ma!r}'
            )


class IlluminationAPI:
    """Illumination sub-API. Owns LED state, ownership tracking, and
    listener registry.
    """

    def _refuse_not_a_board_address(self, address: int, argument: str) -> None:
        """Refuse an address that is not on the attached board's table.

        Raises:
            ArgumentRefusedError: ``'led_address_unknown'``, naming
                ``argument`` and offering the board's addresses.
        """
        board_addresses = tuple(self._driver.available_channels())
        if address not in board_addresses:
            raise ArgumentRefusedError(
                'led_address_unknown', argument=argument, value=address, offered=board_addresses
            )

    def _refuse_unless_lightable(self, address: int, member: str) -> None:
        """Refuse a light of ``address`` that this scope cannot make.

        Asked on the lane by every path that lights an address, in one
        order: the board, then the model. A scope with no LED controller
        connected refuses (``not_connected``); an address off the attached
        board's table is ``led_address_unknown``; an address no channel of
        this model drives (an LS560's 2) is ``not_on_this_scope``. A model this
        release does not know (identity unresolved) has no channels to
        judge an address by, so its addresses are the board's.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'not_on_this_scope'``, naming the missing part.
            ArgumentRefusedError: ``'led_address_unknown'``, offering the
                board's addresses.
        """
        self.refuse_controller_not_connected(member)
        self._refuse_not_a_board_address(address, 'address')
        identity = self._scope.channel_identity
        if identity.source != IdentitySource.UNRESOLVED and self._identity_name(address) is None:
            part = MissingPart.led_address(address)
            raise HardwareCommandRefusedError(part.reason, member, missing=part)

    def _drivable_address(self, channel: str) -> int | None:
        """The board address of ``channel``'s LED; None when this scope cannot light it.

        The fusion of identity and drivability: identity says what the
        channel IS, the driver says what the board can DRIVE. None for a
        channel of the model that drives no LED (Lumi), a channel the model
        lacks (an LS560's Ex589), and a channel whose record names an address
        the attached board lacks -- lighting that address anyway would light
        whatever occupies it on this board.
        """
        address = self.color2ch(color=channel)
        if address is None or address not in self._driver.available_channels():
            return None
        return address

    def _address_to_light(self, channel: str, member: str) -> int:
        """The board address a light of ``channel`` writes, or the refusal that says why not.

        Asked on the lane of a name the door has judged
        (``refuse_unknown_layer``), in the address lane's order: the
        board, then the model, so an LED this model does not have is
        ``not_on_this_scope`` named as the command named it.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'not_on_this_scope'``, naming the missing part.
        """
        self.refuse_controller_not_connected(member)
        address = self._drivable_address(channel)
        if address is None:
            part = MissingPart.led(channel)
            raise HardwareCommandRefusedError(part.reason, member, missing=part)
        return address

    def refuse_controller_not_connected(self, member: str) -> None:
        """Refuse when no LED controller is connected.

        The LED's presence question, asked by every command on the io lane
        before it writes: none came up at bring-up, its cable was pulled,
        or the scope was disconnected (then the lane itself refuses first,
        ``scope_disconnected``). Offered alone to a caller that must refuse
        before work of its own starts -- Go To Step asks it before moving
        when the step's preview would light.

        A consult seam, not part of the L2 API surface: an L2 caller's LED
        command asks it itself.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, naming the LED
                controller. Nothing was sent.
        """
        if not self._scope.led_connected:
            part = MissingPart.LED_CONTROLLER
            raise HardwareCommandRefusedError(part.reason, member, missing=part)

    def _has_board(self) -> bool:
        """Whether an LED board is installed: not the null board.

        The null board stands in when none came up at bring-up and after
        ``disconnect()``; with it the state reads have no board to describe
        and answer None. A board whose cable was pulled stays installed
        until the scope is disconnected.
        """
        return not isinstance(self._driver, NullLEDBoard)

    def _believed_lit(self, address: int) -> bool:
        """Whether the store holds ``address`` other than dark -- the store, not the board."""
        with self._led_state_lock:
            return address in self._led_state

    def __init__(self, scope: Lumascope, driver: LEDBoardProtocol) -> None:
        self._scope = scope
        # driver argument kept for API compatibility but unused; `_driver`
        # is a @property that re-resolves `self._scope._led_driver` so
        # disconnect / reconnect / test hot-swap propagate without
        # rebinding IlluminationAPI. Same pattern as MotionAPI._driver.
        del driver  # intentionally unused, kept for backward call sites

        # LED change listeners -- push-based UI update mechanism. Each
        # listener is called with a reading's fields (address, name, state,
        # illumination_ma) whenever any LED address changes state. Fires from the thread that
        # caused the change, so listeners MUST schedule UI work via
        # Clock.schedule_once.
        self._led_listeners_lock = threading.Lock()
        self._led_listeners: list = []

        # LED state -- API-level source of truth. The API was always
        # supposed to own LED state, but the implementation initially
        # only got as far as observers + save/restore.
        # State queries (get_led_state and its kin) still delegated
        # to the driver -- which worked for LEDBoard (has an internal
        # led_ma dict) but broke for FX2LEDController (thin translator,
        # returns sentinels). This dict is the primary store, analogous
        # to _pos_cache for motor position. Updated inside led_on /
        # led_off / leds_off; read by all state-query methods.
        # Keyed by LED address, never by a name: an address the identity
        # cannot name is still held, so it can still be seen and darkened.
        # An address absent from the store is dark.
        self._led_state: dict[int, _Held] = {}

        # Which lease lit each lit address (None for a write made while
        # nothing was leased), so a lease's release turns off what it lit
        # and leaves an address another lease or the live UI lit alone.
        # Guarded with _led_state by one lock.
        self._led_state_lock = threading.Lock()
        self._lit_by: dict[int, LedLease | None] = {}

        # Per-device LED I/O serialization, so LED stim pulses can
        # interleave with camera grabs and motor moves on their own
        # per-device locks. Wrapped with TimedLock for contention
        # tracing.
        self._led_lock = profile_trace.TimedLock(threading.RLock(), name='illumination._led_lock')

        # LED ownership lease. One lease is held at a time (one logical
        # holder: protocol or autofocus); a second acquire is refused.
        # The exception is a child lease spawned by the current holder
        # (autofocus running inside a protocol step). The stack top is the
        # active holder, and only the active holder may drive the LEDs.
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
        address: int,
        illumination_ma,
        block: bool = False,
        *,
        member: str,
        _lease: LedLease | None = None,
    ) -> float:
        """Turn on an LED address at the specified current.

        The board lights at the nearest current its step can deliver, and a
        request above zero but below one step at one step; that commanded
        current, not the request, is what is skipped on, sent, stored and
        returned. 0 mA leaves the address on at 0 mA, which the driver sends
        as the board's off.

        Args:
            address: The LED board address.
            illumination_ma: Illumination current in milliamps.
            block: If True, wait for confirmation from the LED board.
            member: The member whose call this is, named by its refusals.
            _lease: The lease making this write, from its own diff; None for
                a caller holding no lease. Refused while a different lease is
                the active holder.

        Returns:
            The commanded current in mA.

        Raises:
            ArgumentRefusedError: ``'not_a_number'``, ``illumination_ma`` is
                not a finite number; ``'illumination_out_of_range'``, it is
                outside 0 to the attached board's maximum (``limits``);
                ``'led_address_unknown'``, ``address`` is not on the attached
                board's table.
            HardwareCommandRefusedError: ``'not_connected'`` with no LED
                controller connected; ``'not_on_this_scope'`` for an address this
                model does not drive; ``'exclusive_activity_running'`` while
                a lease other than ``_lease`` holds the LEDs. Nothing was
                sent.
            BoardNoReplyError: the board did not answer; the address is
                unknown (``_write``).
        """
        commanded_ma = self._judge_light(address, illumination_ma, member)

        # Skip redundant command if the address is already lit at the same
        # current; an unknown address carries no current, so it is written.
        with self._led_state_lock:
            cached_entry = self._led_state.get(address)
        current_ma = None if cached_entry is None else cached_entry.illumination_ma
        # _led_state cache-equality trace for the slider > ~150 mA
        # silent-fail bench investigation. Gated by the scope's
        # fx2_debug_wire, the same flag the FX2 LED driver traces with.
        if self._scope._fx2_debug_wire:
            is_enabled = cached_entry is not None
            try:
                delta = (
                    None if current_ma is None else abs(float(illumination_ma) - float(current_ma))
                )
            except Exception:
                delta = 'ERR'
            _api_log.info(
                '[FX2 LED diag] led_on cache-check ch=%s '
                'new_mA=%r (type=%s) cached_mA=%r (type=%s) '
                'delta=%r enabled=%s cache_entry=%r',
                address,
                illumination_ma,
                type(illumination_ma).__name__,
                current_ma,
                type(current_ma).__name__,
                delta,
                is_enabled,
                cached_entry,
            )
        if current_ma is not None and abs(commanded_ma - float(current_ma)) < 0.01:
            return commanded_ma

        # While a run holds the LEDs, a write by anyone but the active lease
        # is refused so a live UI change cannot disturb a protocol's or
        # autofocus's channels. Emergency / shutdown paths use force_off /
        # leds_off, which bypass this on purpose.
        self._refuse_unless_lease_holds(_lease, member)

        began_ns = self._write(
            (address,), lambda: self._driver.led_on(address, commanded_ma, block=block), _lease
        )
        requested = (
            f' (requested {illumination_ma})' if abs(commanded_ma - illumination_ma) >= 0.01 else ''
        )
        _api_log.info(
            f'led_on ch={address} illumination_ma={commanded_ma:g}{requested}'
            f'{f" lease={_lease.purpose!r}" if _lease is not None else ""}'
        )

        # Update the API-level state store and which lease lit the address.
        # Unconditional -- a write with no lease (a UI click) is recorded too.
        with self._led_state_lock:
            self._led_state[address] = _Held(LedState.LIT, commanded_ma)
            self._lit_by[address] = _lease
            reading = self._reading(address)
        # The change is recorded once the store holds it: a frame counts for
        # it by when it began against the record, and the light it is made
        # under is the store's (the simulated sensor reads it).
        self._scope.frame_validity.invalidate(FrameValiditySource.LED, began_ns=began_ns)
        self._fire_led_listeners(reading)
        return commanded_ma

    def _judge_light(self, address: int, illumination_ma: float, member: str) -> float:
        """The current a light of ``address`` commands, or the refusal.

        Judged on every path that lights an LED, not at the door: a
        snapshot or a diff's target hands this a caller's addresses and
        currents.

        Raises:
            ArgumentRefusedError: ``'led_address_unknown'``, ``'not_a_number'``
                or ``'illumination_out_of_range'``, as ``_led_on_impl`` says.
            HardwareCommandRefusedError: ``'not_connected'`` or
                ``'not_on_this_scope'``, as ``_led_on_impl`` says.
        """
        self._refuse_unless_lightable(address, member)
        refuse_unless_finite_number(illumination_ma, 'illumination_ma')
        led_max_ma = self._scope.capabilities.led_max_ma
        if illumination_ma < 0 or illumination_ma > led_max_ma:
            raise ArgumentRefusedError(
                'illumination_out_of_range',
                argument='illumination_ma',
                value=illumination_ma,
                limits=(0, led_max_ma),
            )
        return self._driver.commanded_ma(illumination_ma)

    def _led_off_impl(self, address: int, *, member: str, _lease: LedLease | None = None) -> None:
        """Turn off an LED address.

        Args:
            address: The LED board address.
            member: The member whose call this is, named by its refusals.
            _lease: The lease making this write, from its own diff; None for
                a caller holding no lease. Refused while a different lease is
                the active holder.

        An off whose end state already holds is satisfied with no write and
        no log: an address the API holds dark, any off with no LED
        controller and nothing believed lit. Only an off of an address
        believed lit asks for the controller.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, an address
                believed lit and no LED controller connected to darken it;
                ``'exclusive_activity_running'``, an address believed lit
                while a lease other than ``_lease`` holds the LEDs.
            BoardNoReplyError: the board did not answer; the address is
                unknown (``_write``).
        """
        # Skip if the address is already dark. Reads from the API-level
        # _led_state store, which is correct for both LEDBoard and FX2.
        # Prior behavior delegated to the driver's get_led_state, which
        # for FX2 always returned False -- making led_off a complete
        # no-op.
        if not self._believed_lit(address):
            return
        self.refuse_controller_not_connected(member)

        # Refused for the same reason as led_on (see above): an off from the
        # live UI while a run holds the address is the shape behind the
        # autofocus-LED-killed reports, so it is rejected here.
        self._refuse_unless_lease_holds(_lease, member)

        began_ns = self._write((address,), lambda: self._driver.led_off(address), _lease)
        _api_log.info(
            f'led_off ch={address}{f" lease={_lease.purpose!r}" if _lease is not None else ""}'
        )

        # Clear from the API-level state store, then record the change (see
        # _led_on_impl).
        with self._led_state_lock:
            self._led_state.pop(address, None)
            self._lit_by.pop(address, None)
            reading = self._reading(address)
        self._scope.frame_validity.invalidate(FrameValiditySource.LED, began_ns=began_ns)
        self._fire_led_listeners(reading)

    def _leds_off_impl(self) -> None:
        """Turn off all LEDs (nuclear -- ignores any held lease).

        With an LED controller connected the board's all-off is written
        whatever the API believes lit. Without one, it is satisfied when
        nothing is believed lit.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, a channel
                believed lit and no LED controller connected to darken it.
            BoardNoReplyError: the board did not answer; every address is
                unknown (``_write_leds_off``).
        """
        if not self._scope.led_connected:
            with self._led_state_lock:
                lit = bool(self._led_state)
            if lit:
                self.refuse_controller_not_connected('leds_off')
            return
        self._write_leds_off()

    def _leds_off_if_present(self) -> bool:
        """The scope's own all-off: bring-up, shutdown and the safety off.

        Asks presence first and writes nothing when no LED controller is
        connected: there is nothing to darken, and a write to a board
        unplugged since bring-up only fails. Reads ``led_connected`` rather
        than the driver's truthiness because with no board the composition
        root installs a NullLEDBoard, which is truthy.

        Returns:
            True when the all-off was written; False when no LED controller
            was connected to write it to, or the write found it gone.
        """
        return self._asked_again_after_a_fault(self._leds_off_if_connected)()

    def _leds_off_if_connected(self) -> bool:
        if not self._scope.led_connected:
            return False
        self._write_leds_off()
        return True

    def _asked_again_after_a_fault(self, impl):
        """``impl``, asked for the LED controller again when it raises a fault.

        Every LED body asks for the controller before it writes, so one
        already gone is answered with nothing sent. A body whose write is
        what finds it gone (the Classic's LED shares the camera's USB
        device, which tells nothing when unplugged) raises the device's
        fault instead. A fault raised with the controller now disconnected
        runs the body once more: its presence question answers as it would
        have at the door, refused ``not_connected`` (chained to the fault)
        or an off satisfied, and nothing is written, since every body asks
        before it changes anything. A fault with the controller still
        connected, or a refusal or defect of any other kind, keeps its type.
        """

        @functools.wraps(impl)
        def ask_again_after_a_fault(*args, **kwargs):
            try:
                return impl(*args, **kwargs)
            except Fault as fault:
                if self._scope.led_connected:
                    raise
                try:
                    return impl(*args, **kwargs)
                except HardwareCommandRefusedError as refusal:
                    raise refusal from fault

        return ask_again_after_a_fault

    def _write(
        self,
        addresses: tuple[int, ...],
        write: Callable[[], None],
        lease: LedLease | None,
    ) -> int:
        """Make one board write of ``addresses`` under the LED lock.

        A board that did not answer may have done what it was told or not,
        so every address written is held unknown, believed neither lit nor
        dark: an off writes it and a light is never skipped for it. It is
        held as lit by ``lease``, whose release then darkens it. No frame
        is judged against a change that may not have happened; the caller
        that made it fails instead.

        Returns:
            The scope's steady clock (``host_clock.steady_ns``) when the write
            began, the earliest its change could have lit the sample.

        Raises:
            BoardNoReplyError: the board did not answer; the listeners
                have been told each address is unknown.
        """
        try:
            with self._led_lock:
                began_ns = host_clock.steady_ns()
                write()
        except BoardNoReplyError:
            with self._led_state_lock:
                for address in addresses:
                    self._led_state[address] = _Held(LedState.UNKNOWN)
                    self._lit_by[address] = lease
                readings = [self._reading(address) for address in addresses]
            _api_log.warning(f'LED write unanswered: addresses {list(addresses)} unknown')
            for reading in readings:
                self._fire_led_listeners(reading)
            raise
        return began_ns

    def _write_leds_off(self) -> None:
        """Write the board's all-off and record every address dark.

        Raises:
            BoardNoReplyError: the board did not answer; every address of
                the board is unknown (``_write``).
        """
        began_ns = self._write(self._driver.available_channels(), self._driver.leds_off, None)
        with self._led_state_lock:
            self._lit_by.clear()
            self._led_state.clear()
            readings = self._readings()
        self._scope.frame_validity.invalidate(FrameValiditySource.LED, began_ns=began_ns)
        _api_log.info('leds_off')
        for reading in readings:
            self._fire_led_listeners(reading)

    # --- Public dispatch ---
    # These four are what every caller reaches: an SDK script, a REST
    # handler, the GUI -- and the run, the autofocus sweep and the diagnostics,
    # which call them under their taking so the lane admits their work while
    # they hold the scope. From a task already on the lane's worker the lane
    # runs the body inline.

    def _dispatch_led(self, impl, name, args=(), kwargs=None, *, lights=False):
        """Run one LED command for an external caller, on the right thread.

        The body runs on the scope's io lane, serialized against every other
        hardware write, and this blocks until it has. A lane that will not
        accept work tells the caller so, because the alternative is `put`
        returning None and the command disappearing with nothing raised and
        nothing logged.

        The lane's ``call`` decides a refusal and raises it to the caller:
        the lane is closed, or a run, a diagnostic or a home holds the scope and this
        call is not made under its taking, or ``lights`` is set and a
        recording holds the scope: a recording keeps the one light it was
        started under, so any write that would light an LED would falsify
        its frames, and an off never does. The body asks for the LED
        controller on the lane (``refuse_controller_not_connected``), so a
        lane refusal is never hidden behind the presence question, and asks
        again when its write finds the controller gone
        (``_asked_again_after_a_fault``).
        """
        kwargs = kwargs or {}
        return self._scope._io_executor.call(
            IOTask(
                action=self._asked_again_after_a_fault(impl),
                args=args,
                kwargs=kwargs,
                falsifies_recording=lights,
            ),
            name,
            _LED_WRITE_TIMEOUT_S,
        )

    @api(admitted_by=(Admission.LANE_CALL,))
    def led_on(
        self,
        channel: str,
        illumination_ma: float,
    ) -> float:
        """Turn on a channel's LED at the specified current, and wait for it.

        The channel is lit at its LED's address, as one command on the LED
        board's lane, waited for. The board lights at the nearest current its step can
        deliver, and a request above zero but below one step at one step;
        that commanded current, not the request, is what is stored and
        returned. 0 mA leaves the channel on at 0 mA, which the board treats
        as off.

        Args:
            channel: The channel to light, by its name in this release.
            illumination_ma: The current in mA, from 0 to the attached
                board's maximum (``limits``).

        Returns:
            The commanded current in mA.

        Raises:
            ArgumentRefusedError: ``'channel_unknown'``, ``channel`` is not a
                channel of this release, refused at the door whatever the
                scope is doing; ``'not_a_number'``, ``illumination_ma`` is
                not a finite number; ``'illumination_out_of_range'``, it is
                outside 0 to the attached board's maximum. Nothing was sent.
            HardwareCommandRefusedError: ``'not_connected'`` with no LED
                controller connected; ``'not_on_this_scope'`` for a channel
                whose LED this model does not drive;
                ``'exclusive_activity_running'``, another activity holds the
                scope or the LEDs, or a manual recording runs and
                ``illumination_ma`` is above 0 mA. Nothing was sent.
            BoardNoReplyError: the board did not answer; whether the LED
                is lit is unknown.
        """
        refuse_unknown_layer(channel, argument='channel')
        return self._dispatch_led(
            self._led_on_by_name_impl,
            'led_on',
            args=(channel, illumination_ma),
            lights=illumination_ma > 0,
        )

    def _led_on_by_name_impl(self, channel: str, illumination_ma: float) -> float:
        """``led_on``'s body for a name: its address, then the address lane."""
        address = self._address_to_light(channel, 'led_on')
        return self._led_on_impl(address, illumination_ma, member='led_on')

    @api(admitted_by=(Admission.LANE_CALL,))
    def led_off(self, channel: str) -> None:
        """Turn off a channel's LED, and wait for it.

        The channel is darkened at its LED's address. An off whose end state
        already holds is satisfied with no write: a channel the API holds
        dark, or any off with no LED controller and nothing believed lit. Only an off of a
        channel believed lit asks for the controller.

        Args:
            channel: The channel to darken, by its name in this release.

        Raises:
            ArgumentRefusedError: ``'channel_unknown'``, ``channel`` is not a
                channel of this release, refused at the door whatever the
                scope is doing; nothing was sent.
            HardwareCommandRefusedError: ``'not_connected'``, the channel is
                believed lit and no LED controller is connected to darken it;
                ``'exclusive_activity_running'``, another activity holds the
                scope or the LEDs. Nothing was sent.
            BoardNoReplyError: the board did not answer; whether the LED
                is lit is unknown.
        """
        refuse_unknown_layer(channel, argument='channel')
        return self._dispatch_led(self._led_off_by_name_impl, 'led_off', args=(channel,))

    def _led_off_by_name_impl(self, channel: str) -> None:
        """``led_off``'s body for a name: its address, then the address lane.

        A channel the model has no LED for -- one the model lacks, one that
        drives no LED -- is definitionally dark, so the off is complete with
        no command.
        """
        address = self._drivable_address(channel)
        if address is None:
            _api_log.debug(f"led_off no-op: scope has no '{channel}' LED")
            return
        self._led_off_impl(address, member='led_off')

    @api(in_process=True, engineering=True)
    def led_on_by_address(self, address: int, illumination_ma: float) -> float:
        """Light an LED board address at the specified current, and wait for it.

        For the bench and for a model this release does not know; every
        other caller lights a channel by its name (``led_on``). The address
        is lit on the lane ``led_on`` lights a name's address on, judged in
        the same order, and read back and reported by its address as a
        name's is. The current is commanded on the board's step, as
        ``led_on`` says.

        Args:
            address: The LED board address.
            illumination_ma: The current in mA, from 0 to the attached
                board's maximum (``limits``).

        Returns:
            The commanded current in mA.

        Raises:
            ArgumentRefusedError: ``'led_address_unknown'``, ``address`` is
                not on the attached board's table, offering its addresses;
                ``'not_a_number'`` or ``'illumination_out_of_range'``, as
                ``led_on`` says. Nothing was sent.
            HardwareCommandRefusedError: ``'not_connected'`` with no LED
                controller connected; ``'not_on_this_scope'``
                (``MissingPart.led_address``) for an address this model does
                not drive; ``'exclusive_activity_running'``, another activity
                holds the scope or the LEDs, or a manual recording runs and
                ``illumination_ma`` is above 0 mA. Nothing was sent.
            BoardNoReplyError: the board did not answer; whether the LED
                is lit is unknown.
        """
        self._refuse_not_a_board_address(address, 'address')
        return self._dispatch_led(
            self._led_on_impl,
            'led_on_by_address',
            args=(address, illumination_ma),
            kwargs={'member': 'led_on_by_address'},
            lights=illumination_ma > 0,
        )

    @api(in_process=True, engineering=True)
    def led_off_by_address(self, address: int) -> None:
        """Darken an LED board address, and wait for it.

        The engineering counterpart of ``led_off``, as ``led_on_by_address``
        is of ``led_on``; an off whose end state already holds is satisfied
        with no write, as ``led_off`` says.

        Args:
            address: The LED board address.

        Raises:
            ArgumentRefusedError: ``'led_address_unknown'``, ``address`` is
                not on the attached board's table, offering its addresses;
                nothing was sent.
            HardwareCommandRefusedError: ``'not_connected'``, the address is
                believed lit and no LED controller is connected to darken it;
                ``'exclusive_activity_running'``, another activity holds the
                scope or the LEDs. Nothing was sent.
            BoardNoReplyError: the board did not answer; whether the LED
                is lit is unknown.
        """
        self._refuse_not_a_board_address(address, 'address')
        return self._dispatch_led(
            self._led_off_impl,
            'led_off_by_address',
            args=(address,),
            kwargs={'member': 'led_off_by_address'},
        )

    @api(admitted_by=(Admission.LANE_CALL,))
    def leds_off(self) -> None:
        """Turn off all LEDs, and wait for it.

        Whatever holds the LEDs, every one is turned off: any lease is
        ignored. With an LED controller connected the board's all-off is
        written whatever the API believes lit; without one, it is satisfied
        when nothing is believed lit.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, a channel
                believed lit and no LED controller connected to darken it.
            BoardNoReplyError: the board did not answer; whether each LED
                is lit is unknown.
        """
        return self._dispatch_led(self._leds_off_impl, 'leds_off')

    def apply_transition(self, transition: LedTransition, ctx: LedTransitionCtx) -> None:
        """Drive an unleased LED transition through the authority, and wait.

        Internal LED-transition entry -- not part of the L2 API surface
        (clients use ``led_on``/``led_off``).

        See ``_apply_transition_impl`` for what a transition is and when it
        is refused for lease reasons, which is separate from the dispatch
        refusal here; a transition whose target lights anything is refused
        whole while a manual recording runs.
        """
        return self._dispatch_led(
            self._apply_transition_impl,
            'apply_transition',
            args=(transition, ctx),
            lights=bool(LedLease.target_leds(transition, ctx)),
        )

    def _leds_off_emergency(self, *, timeout_s: float = 2.0) -> None:
        """Bounded leds-off for atexit / abnormal-exit paths only.

        Normal `leds_off` blocks on `_led_lock` indefinitely. atexit hooks
        run on the main thread and cannot honor timeouts, so an in-flight
        LED command holding `_led_lock` would deadlock interpreter
        teardown.

        This variant uses `_led_lock.acquire(timeout=timeout_s)` with a
        log-and-skip fallback. The post-call notification / state-clear /
        listener-fire paths are also skipped -- by the time atexit fires,
        the notification stack, state cache, and listener bus may already
        be torn down. Don't call from normal code paths; use `leds_off`
        instead. Writes nothing when no LED board is connected, for the
        reason ``_leds_off_if_present`` gives, and returns quietly when its
        write is what finds the board gone (``_asked_again_after_a_fault``).
        """
        self._asked_again_after_a_fault(self._leds_off_bounded)(timeout_s)

    def _leds_off_bounded(self, timeout_s: float) -> None:
        if not self._scope.led_connected:
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

    # --- State ---
    def _identity_name(self, address: int) -> str | None:
        """The layer the unit's identity names for *address*, else None."""
        return self._scope.channel_identity.key_at(address)

    def _reading(self, address: int) -> LedReading:
        """*address* as the store holds it, named through the identity now.

        The caller holds ``_led_state_lock``.
        """
        held = self._led_state.get(address)
        if held is None:
            return LedReading(address, self._identity_name(address), LedState.DARK, None)
        return LedReading(address, self._identity_name(address), held.state, held.illumination_ma)

    def _readings(self) -> tuple[LedReading, ...]:
        """Every address of the board and every address the store holds, by address.

        The caller holds ``_led_state_lock``.
        """
        addresses = sorted(set(self._driver.available_channels()) | set(self._led_state))
        return tuple(self._reading(address) for address in addresses)

    @api(reads=True)
    def get_led_states(self) -> tuple[LedReading, ...] | None:
        """Read every LED address of the attached board.

        Every address is reported, whether or not the unit's identity
        names it, so a lit address is never hidden by its missing name.

        Returns:
            One reading per address, in address order, each named through
            the unit's identity. None when no LED board is installed (none
            came up, or after ``disconnect()``).

        Raises:
            Nothing.
        """
        if not self._has_board():
            return None
        with self._led_state_lock:
            return self._readings()

    # --- Save / restore ---
    @api(reads=True)
    def save_led_state(self, tag: str) -> LedSnapshot | None:
        """Snapshot every LED address for later restoration.

        Args:
            tag: Descriptive name for the snapshot (for logging).

        Returns:
            Snapshot suitable for passing to ``restore_led_state``; None
            when no LED board is installed, so there is no state to keep.

        Raises:
            Nothing.
        """
        readings = self.get_led_states()
        if readings is None:
            return None
        _api_log.info(
            f'save_led_state tag={tag}: {[r.address for r in readings if r.state == LedState.LIT]}'
        )
        return LedSnapshot(tag, readings)

    @api(admitted_by=(Admission.LANE_CALL,))
    def restore_led_state(self, snapshot: LedSnapshot) -> None:
        """Restore LEDs to a previously saved state, as one task on the IO lane, and wait.

        Turns off the addresses the snapshot does not have lit, then lights
        the addresses it has lit at a positive current.

        Args:
            snapshot: Return value from ``save_led_state``.

        Raises:
            HardwareCommandRefusedError: ``'exclusive_activity_running'``,
                another activity holds the scope or the LEDs, raised at the
                first channel it would change; or a manual recording runs
                and the snapshot has an LED lit, raised before any change.
                ``'not_connected'``, no LED controller is connected, or
                ``'not_on_this_scope'``, before anything is written.
            ArgumentRefusedError: ``'led_address_unknown'``,
                ``'not_a_number'`` or ``'illumination_out_of_range'`` for a
                light in the snapshot, before anything is written.
            BoardNoReplyError: the board did not answer a write; that
                channel is unknown and the writes after it are not made.
        """
        return self._dispatch_led(
            self._restore_led_state_impl,
            'restore_led_state',
            args=(snapshot,),
            lights=bool(snapshot_lit_pairs(snapshot.readings)),
        )

    def _restore_led_state_impl(self, snapshot: LedSnapshot) -> None:
        _api_log.info(f'restore_led_state tag={snapshot.tag}')
        self._emit_led_diff(
            snapshot_lit_pairs(snapshot.readings),
            lease=None,
            block=False,
            member='restore_led_state',
        )

    def _leds_off_lit_by(self, lease: LedLease) -> None:
        """Turn off only the LED channels *lease* lit.

        Channels lit by another lease or by an unleased write are left
        alone.

        Raises:
            HardwareCommandRefusedError: ``'not_connected'``, the lease lit
                a channel and no LED controller is connected to darken it;
                the channel stays believed lit, or unknown when the write
                that found the controller gone went unanswered.
            BoardNoReplyError: the board did not answer an off; that
                channel is unknown and still the lease's (``_write``), and
                the channels after it are left as they were.
        """
        with self._led_state_lock:
            channels_to_off = [address for address, by in self._lit_by.items() if by is lease]
        if not channels_to_off:
            return
        self.refuse_controller_not_connected('LedLease.release')
        for ch in channels_to_off:
            began_ns = self._write((ch,), functools.partial(self._driver.led_off, ch), lease)
            # Dark once written, never before: a write that finds the
            # controller gone leaves the channel believed lit, or unknown
            # when the board did not answer it, and the release asked
            # again is refused for it.
            with self._led_state_lock:
                self._lit_by.pop(ch, None)
                self._led_state.pop(ch, None)
            self._scope.frame_validity.invalidate(FrameValiditySource.LED, began_ns=began_ns)
            _api_log.info(f'led_off ch={ch} (release of the {lease.purpose!r} lease)')
            with self._led_state_lock:
                reading = self._reading(ch)
            self._fire_led_listeners(reading)

    def _forget_led_state(self) -> None:
        """Hold nothing lit: ``disconnect()`` darkened the board and removed it.

        The emergency off ``disconnect()`` writes leaves the state store as
        it was, so the store is cleared here once the board is gone, and
        the listeners are told each address that was held went dark.
        """
        with self._led_state_lock:
            held = list(self._led_state)
            self._led_state.clear()
            self._lit_by.clear()
            readings = [self._reading(address) for address in held]
        for reading in readings:
            self._fire_led_listeners(reading)

    # --- Ownership lease ---
    def _holder_is_stranded(self, lease: LedLease) -> str | None:
        """The evidence that *lease*'s activity is over, or None if it is live.

        *lease* is a stack root, so it was taken under a held claim. It is
        stranded only when that is PROVABLE: its taking no longer holds the
        claim (the run that took it ended or was killed). Anything else is a
        live holder, however inconvenient for the contender.
        """
        if not lease._claim.holds:
            return 'its activity claim is no longer held'
        return None

    def acquire_led_lease(self, purpose: str, *, claim: Taking) -> LedLease:
        """Acquire the exclusive LED-ownership lease under a held claim.

        Internal run-exclusivity machinery -- not part of the L2 API
        surface (clients drive ``led_on``/``led_off``).

        While a lease is held, only it may drive the LEDs. A top-level
        lease is only ever taken under the session's one activity claim,
        and the caller holds it, so any lease already on the stack was
        taken under a taking that no longer holds: it is stranded, and is
        reclaimed here with the evidence logged. A second LIVE top-level
        holder would mean two activities hold the one claim; that is
        raised, never answered as a refusal.

        Args:
            purpose: A label for logs ('protocol'); never compared, so it
                grants nothing.
            claim: The caller's held activity claim. The lease lives while
                this taking holds the claim, which is what lets a LATER
                contender tell this holder's death from its inconvenience.

        Returns:
            The LedLease token.

        Raises:
            ValueError: *claim* no longer holds at acquire time -- a lease
                taken under it would be stranded from its first moment.
            RuntimeError: a lease taken under a claim that still holds is
                on the stack -- two live activities.
        """
        if not claim.holds:
            raise ValueError(
                f'LED lease acquire for {purpose!r}: the claim must be held at '
                'acquire time (take the claim before the lease)'
            )
        lease = self._acquire_led_lease(purpose, claim=claim, parent=None)
        if lease is None:
            raise RuntimeError(
                f'LED lease acquire for {purpose!r}: the {self.led_lease_purpose!r} lease '
                'is held under a claim that still holds -- two live activities'
            )
        return lease

    def _acquire_led_lease(
        self, purpose: str, *, claim: Taking | None, parent: LedLease | None
    ) -> LedLease | None:
        """Push a lease onto the stack: a root under *claim*, or a child of *parent*."""
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
                        purpose,
                        active.purpose,
                    )
                    return None
                dropped = [held.purpose for held in self._led_lease_stack]
                for held in self._led_lease_stack:
                    held._released = True
                self._led_lease_stack.clear()
                _api_log.warning(
                    'LED lease stack reclaimed from stranded owner %r (%s); '
                    'dropped: %s; granting to %r',
                    root.purpose,
                    evidence,
                    dropped,
                    purpose,
                )
                if parent is not None:
                    # The requester wanted a child of a lease that just fell
                    # with the reclaimed stack; a child cannot outlive its
                    # parent, so the grant below would dangle.
                    _api_log.warning(
                        'LED lease child acquire refused for %r: parent fell '
                        'with the reclaimed stack',
                        purpose,
                    )
                    return None
            elif active is None and parent is not None:
                # A parent was supplied but nothing is held -- the parent
                # already released. Refuse rather than silently promote the
                # child to a top-level lease.
                _api_log.warning(
                    'LED lease child acquire refused for %r: parent lease not held',
                    purpose,
                )
                return None
            lease = LedLease(self, purpose, claim=claim, parent=parent)
            self._led_lease_stack.append(lease)
            _api_log.info(
                'LED lease acquired by %r (depth=%d)', purpose, len(self._led_lease_stack)
            )
            return lease

    def _release_led_lease(self, lease: LedLease, *, leave_on: bool = False) -> None:
        """Release a lease (called via LedLease.release). Idempotent.

        By default the channels this lease lit are turned off, so the LED
        end-state is a property of the release. A holder whose declared
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
        if not leave_on:
            # On the io lane like every LED write; the releasing activity's
            # taking carries it through its own door while the lane is fenced.
            self._dispatch_led(self._leds_off_lit_by, 'LedLease.release', args=(lease,))
        _api_log.info(
            'LED lease released by %r%s', lease.purpose, ' (leave_on)' if leave_on else ''
        )

    def _reclaim_lease(self, lease: LedLease) -> None:
        """Make *lease* the active (top) holder, releasing any descendants above it.

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
                lease.purpose,
                [held.purpose for held in stranded],
            )

    def _refuse_unless_lease_holds(self, lease: LedLease | None, member: str) -> None:
        """Refuse an LED write by *lease* unless it may happen now.

        A write is permitted when no lease is held, or when *lease* is the
        active holder itself -- the object, not its purpose.

        Raises:
            HardwareCommandRefusedError: ``'exclusive_activity_running'``,
                naming the activity that took the root lease's claim, the
                holder the IO lane names.
        """
        with self._led_lease_lock:
            if not self._led_lease_stack or lease is self._led_lease_stack[-1]:
                return
            holder = self._led_lease_stack[0]._claim.claim.holder
        raise HardwareCommandRefusedError(
            'exclusive_activity_running', member, holder.kind if holder is not None else None
        )

    @property
    def led_lease_purpose(self) -> str | None:
        """The active LED lease's purpose label, or None if the LEDs are unleased.

        Internal lease introspection for logs and refusal text -- not part
        of the L2 API surface, and never a credential.
        """
        with self._led_lease_lock:
            return self._led_lease_stack[-1].purpose if self._led_lease_stack else None

    def _emit_led_diff(
        self,
        target: frozenset[tuple[int, float]],
        *,
        lease: LedLease | None,
        block: bool,
        member: str,
    ) -> None:
        """Turn off lit channels not in the target, then assert the target.

        The single diff-and-emit every transition drives through -- both the
        leased run callers (via LedLease) and the unleased live-UI callers (via
        apply_transition). Trusts the state cache to skip already-correct
        channels: led_on self-skips a channel already at its current and led_off
        self-skips a dark one, so re-asserting a correct target emits nothing (no
        off-then-on blink). The off clears the channel regardless of who lit it
        but writes as ``lease`` so it is permitted while that lease is the
        active holder (or, with None, while no lease is held, for an unleased
        UI write).
        ``block`` waits for the board to confirm each illuminate before
        returning -- set for a transition whose LED must be on before the
        camera grabs; the off does not block (clearing a channel never gates a
        grab). ``restore_led_state`` drives this diff over a saved snapshot.
        ``member`` is the member whose call this is, named by its refusals.

        Raises:
            HardwareCommandRefusedError: ``'exclusive_activity_running'``,
                ``'not_connected'`` or ``'not_on_this_scope'``, before anything
                is written.
            ArgumentRefusedError: ``'led_address_unknown'``,
                ``'not_a_number'`` or ``'illumination_out_of_range'`` for a
                target light, before anything is written.
            BoardNoReplyError: the board did not answer a write; that
                channel is unknown and the writes after it are not made.
        """
        # The whole target is judged before anything is turned off, so a
        # refused light leaves every LED as it was.
        self._refuse_unless_lease_holds(lease, member)
        for ch, illumination_ma in target:
            self._judge_light(ch, illumination_ma, member)
        target_channels = {ch for ch, _ in target}
        # Snapshot under the lock, then release before emitting: the off/on
        # primitives re-acquire _led_state_lock (not reentrant) and _led_off_impl
        # pops from _led_state as it clears each address.
        with self._led_state_lock:
            held = list(self._led_state)
        for ch in held:
            if ch not in target_channels:
                self._led_off_impl(ch, member=member, _lease=lease)
        for ch, illumination_ma in target:
            self._led_on_impl(ch, illumination_ma, block=block, member=member, _lease=lease)

    def _apply_transition_impl(self, transition: LedTransition, ctx: LedTransitionCtx) -> None:
        """Drive an unleased LED transition through the authority.

        The lease-free counterpart to LedLease.apply, for live-UI writers that
        hold no lease -- LED control is open season while nothing is leased, but
        the decision still routes through target_leds and the emit through the
        one diff, so a no-op transition (a same-color re-navigation) blinks
        nothing. Refused whole while a lease is held: a run owns the LEDs, and
        a stray UI write must not cut in mid-run, so nothing of the diff is
        emitted.

        Raises:
            HardwareCommandRefusedError: ``'exclusive_activity_running'``,
                a lease holds the LEDs; nothing was sent.
        """
        self._refuse_unless_lease_holds(None, 'apply_transition')
        self._emit_led_diff(
            LedLease.target_leds(transition, ctx),
            lease=None,
            block=transition in _CONFIRM_ON_TRANSITIONS,
            member='apply_transition',
        )

    def force_off(self) -> bool:
        """Turn off all LEDs unconditionally, bypassing any held lease.

        Internal lease-bypassing safety off for the writer path -- not
        part of the L2 API surface (clients use ``leds_off``).

        The unblockable safety path for emergency / error / shutdown
        callers: an idempotent off must never be refused because another
        subsystem holds the lease. When a lease is held this logs loudly so
        the bypass is visible in post-mortem; the lease itself is left
        intact, so its holder still releases normally. With no LED
        controller connected it writes nothing and is not refused (the
        scope's own off asks presence first).

        Returns:
            True when the all-off was written; False when no LED controller
            was connected to write it to, so whatever the board holds lit
            stays lit, or when the board did not answer it, every address
            then unknown (``_write_leds_off``).
        """
        held = self.led_lease_purpose
        if held is not None:
            _api_log.warning('force_off bypassing the held %r LED lease', held)
        try:
            return self._leds_off_if_present()
        except BoardNoReplyError as e:
            _api_log.error(f'force_off unanswered: {e}')
            return False

    # --- Enable / disable ---
    # --- Wait ---
    # --- Channel mapping ---
    # One mapping, the unit's resolved layer identity: what a layer IS on
    # this unit. The state store, restore snapshots and extinguish sweeps
    # are keyed by address, so a physically lit address is always
    # recordable and extinguishable even when the identity cannot name it
    # (a mid-session identity change must never strand a lit LED or drop
    # it from a restore set).
    def ch2color(self, channel: int) -> str | None:
        """The stable layer name whose record drives *channel*, else None.

        Answers from the unit's resolved layer identity -- the same
        records `scope.channel_identity` exposes.

        A consult seam, not part of the L2 API surface: an L2 caller names
        a channel, and reads which addresses it drives from its record's
        ``led_addresses``.

        Raises:
            ArgumentRefusedError: ``'led_address_unknown'``, *channel* is
                not on the attached board's table.
        """
        self._refuse_not_a_board_address(channel, 'channel')
        return self._identity_name(channel)

    def color2ch(self, color: str) -> int | None:
        """The LED board address of the layer named *color*, else None.

        Answers from the unit's resolved layer identity by stable
        `key_name`. None means the layer is unknown to this unit's
        identity OR drives no LED (luminescence): on-paths turn that
        into a named refusal at `_address_to_light`, off-paths no-op.

        A consult seam, not part of the L2 API surface: an L2 caller names
        the channel itself.

        Raises:
            ArgumentRefusedError: ``'channel_unknown'``, *color* is not a layer.
        """
        refuse_unknown_layer(color, argument='color')
        record = self._scope.channel_identity.find(color)
        if record is None or not record.led_addresses:
            return None
        return record.led_addresses[0]

    # --- Listeners ---
    @api(in_process=True, event=LedReading)
    def add_led_listener(
        self, listener: Callable[[int, str | None, LedState, float | None], None]
    ) -> None:
        """Register a callback for LED state changes.

        The listener is called with the changed address's reading,
        ``(address, name, state, illumination_ma)`` as ``LedReading``
        holds them, whenever any LED address changes state. It fires from
        the thread that caused the change, so listeners **must** schedule
        UI work via ``Clock.schedule_once``.

        Args:
            listener: ``callable(address: int, name: str | None, state:
                LedState, illumination_ma: float | None)``

        Raises:
            Nothing.
        """
        with self._led_listeners_lock:
            self._led_listeners.append(listener)

    @api(in_process=True)
    def remove_led_listener(
        self, listener: Callable[[int, str | None, LedState, float | None], None]
    ) -> None:
        """Unregister an LED listener.

        Args:
            listener: A callable previously passed to ``add_led_listener``.
                Silently ignores listeners that are not currently registered.

        Raises:
            Nothing.
        """
        with self._led_listeners_lock:
            try:
                self._led_listeners.remove(listener)
            except ValueError:
                pass

    def _fire_led_listeners(self, reading: LedReading) -> None:
        """Notify all LED listeners of a state change, as *reading* has it."""
        from modules.notification_center import notifications

        with self._led_listeners_lock:
            listeners = list(self._led_listeners)
        notifications.tell_listeners(
            listeners,
            reading.address,
            reading.name,
            reading.state,
            reading.illumination_ma,
            category='Illumination',
        )
