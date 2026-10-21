# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Atomic compare-and-claim for the session's one exclusive activity."""

import contextlib
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from modules.api_surface import api
from modules.caller import Caller, current_caller
from modules.exceptions import (
    UNDER_CONTROL_TITLE,
    ControlSessionRefusedError,
    HardwareCommandRefusedError,
    RecordingRefusedError,
    Refusal,
    RefusalCause,
    SessionClosingError,
    the_activity_named,
    under_control_sentence,
)

# The activity kinds that hold the WHOLE scope: a run and a diagnostic both
# drive every axis, the LEDs and the camera, and a home leaves every axis
# unknown until it ends, so while one holds the claim a lane runs only work
# made under its taking, the controls lock and the objective cannot change
# under it. A recording is not here -- focusing, moving, LED, gain and
# exposure stay open to anyone during one.
SCOPE_HOLDING_KINDS = frozenset({'protocol', 'diagnostic', 'home'})

_acting = threading.local()


class FalsifyingChangeInFlightError(Refusal, Exception):
    """A recording was asked for while a write that would falsify it runs.

    Raised by ``try_claim('recording')``. A recording writes nothing to a
    lane, so unlike a run it does not queue behind a turret or frame change
    already under way: started then, its first frames would record the
    change and its file would claim the settings it read before it. The
    change ends, and the same start may then succeed.

    Attributes:
        reason: ``'falsifying_change_in_flight'``.
    """

    cause = RefusalCause.STATE
    title = RecordingRefusedError._TITLES['falsifying_change_in_flight']

    def __init__(self):
        super().__init__(
            'The microscope is changing the objective or the camera frame; '
            'a recording started now would record the change.'
        )
        self.reason = 'falsifying_change_in_flight'


@dataclass(frozen=True)
class ControlSession:
    """The one client controlling the scope: what it was taken as, and by whom.

    Attributes:
        id: What a request names to act in this session; a fresh one per
            take, so an ended session's id never names a later one.
        name: The name the client took it with, for the person at the scope.
        taken_by: The host the take came through (``modules.caller``).
        taken_at: When it was taken, as ``time.time()``.
    """

    id: str
    name: str
    taken_by: Caller
    taken_at: float


class ControlSessionHeldError(Refusal, Exception):
    """An activity was asked for, while a control session is held, by someone other than its holder.

    Raised by ``try_claim``. Each taker answers it with its own typed
    refusal, as it answers ``FalsifyingChangeInFlightError``: a None from
    ``try_claim`` means another activity holds the slot, and the slot is
    free here. Nothing was taken.

    Attributes:
        reason: ``'control_session_held'``.
        session: The control session that holds.
    """

    cause = RefusalCause.STATE
    title = UNDER_CONTROL_TITLE

    def __init__(self, session: ControlSession):
        super().__init__(under_control_sentence(session.taken_by, session.name))
        self.reason = 'control_session_held'
        self.session = session


def current_taking() -> 'Taking | ControllerTaking | None':
    """The taking this thread is acting under, or None."""
    return getattr(_acting, 'taking', None)


def activity_taking(taking: 'Taking | ControllerTaking | None') -> 'Taking | None':
    """``taking`` when it is an activity's -- a held claim or a borrowing of one -- else None.

    The controller's taking is no activity's: it never ends as an activity
    does, so a reader asking whether an activity has let go reads it as no
    taking at all.
    """
    return taking if isinstance(taking, (HeldClaim, _Borrowing)) else None


@contextlib.contextmanager
def acting(taking: 'Taking | ControllerTaking | None') -> Iterator[None]:
    """Act under ``taking`` on this thread for the length of a ``with`` block.

    A lane stamps each task with the taking its submitter acts under, and
    while the scope is held it runs only the holder's; so every thread that
    does a holder's work enters its taking here. None acts under nothing,
    which lets a caller hand on whatever taking it was given without a
    branch. Nests: the previous taking is restored on exit.
    """
    previous = current_taking()
    _acting.taking = taking
    try:
        yield
    finally:
        _acting.taking = previous


@dataclass(frozen=True)
class RunIdentity:
    """Which run: who started it, and what it is in a person's words.

    ``started_by`` is provenance -- the caller the starter's host stamped on
    its thread (``modules.caller``), handed to callers as a refusal's
    ``holder_trigger``. ``words`` is the run's kind as a sentence names it
    ('Z-stack'). One object, so the two cannot come from different runs.
    """

    started_by: Caller
    words: str


def the_run_named(run: RunIdentity | None, *, sentence_start: bool = False) -> str:
    """Name a run for a refusal sentence: 'the Z-stack run', or 'a run'.

    One phrasing, one home. 'a run' only where no run is known -- a holder
    that released between a failed take and the read of who held it.
    """
    named = f'the {run.words} run' if run is not None else 'a run'
    return named[0].upper() + named[1:] if sentence_start else named


@dataclass(frozen=True)
class ActivityHolder:
    """Who holds the session's one exclusive activity.

    The kind of activity and, when that activity is a run, which run --
    one immutable object published by a single attribute store, so a
    reader that sees the holder sees the run behind it and cannot pair
    a live claim with a name read an instant earlier or later. An
    activity that is not a run carries no run: a recording's kind IS the
    whole answer. A run holder without its run cannot be built.
    """

    kind: str
    run: RunIdentity | None = None

    def __post_init__(self) -> None:
        if (self.kind == 'protocol') != (self.run is not None):
            raise ValueError(
                f'a {self.kind!r} holder with run={self.run!r}: a run names its run, '
                'and nothing else does'
            )

    @property
    def run_started_by(self) -> Caller | None:
        """Who started the holding run; None for an activity that is not a run."""
        return self.run.started_by if self.run is not None else None


def the_holder_named(holder: ActivityHolder | None) -> str:
    """Name what holds the scope, to open a refusal sentence.

    A run by its kind ('The Z-stack run'), another activity by its kind ('A
    diagnostic', 'A home') -- an activity the user cannot name is one they
    cannot go and stop. 'Another activity' only when the holder released
    between the failed take and this read.
    """
    if holder is not None and holder.run is not None:
        return the_run_named(holder.run, sentence_start=True)
    return the_activity_named(holder.kind if holder is not None else None)


class HeldClaim:
    """The claim as held by the one caller that took it.

    A fresh object per taking, so it identifies that taking and no
    other: the next run's HeldClaim is a different object even when its
    holder description is equal. Only it releases the claim. Nothing
    else hands it out -- ``ActivityClaim.holder`` describes the holder,
    it does not return this.
    """

    __slots__ = ('_claim', '_control')

    def __init__(self, claim: 'ActivityClaim', control: ControlSession | None = None) -> None:
        self._claim = claim
        self._control = control

    @property
    def claim(self) -> 'ActivityClaim':
        """The claim this taking was taken from."""
        return self._claim

    @property
    def control(self) -> ControlSession | None:
        """The control session this activity was taken for; None when none was held."""
        return self._control

    @api
    @property
    def holds(self) -> bool:
        """Whether this taking still holds the claim.

        False once it is released, and stays False: a later taking of the
        same claim is a different object.

        Raises:
            Nothing.
        """
        return self._claim._is_held_by(self)

    def release(self) -> None:
        """Release the claim this taking holds.

        Not published: whoever took the claim releases it on its own exit
        path (a diagnostic's block, a run's cleanup), so a caller handed the
        taking never owns its release.

        Raises:
            RuntimeError: this taking no longer holds the claim -- a
                release path that runs twice or late fails loudly rather
                than freeing a claim a newer activity took.
        """
        self._claim._release(self)

    def lend(self) -> 'BorrowedClaim':
        """Lend this taking to work that runs inside its activity."""
        return BorrowedClaim(self)


class _Borrowing:
    """What a borrower holds: it acts under the lender's claim, and its
    release leaves that claim held -- the lender releases at its own end.

    It answers the rest of a held claim's questions the same way, so work
    that runs under it (a run inside a diagnostic, and a recording inside
    that run) needs no branch on whether it borrowed. It holds until it is
    released or its lender stops holding, and it lends itself: what it lent
    ends when it does, so a recording inside a borrowed run cannot outlive
    the run, and a lane stops taking an ended run's work while the lender
    still holds the scope.
    """

    __slots__ = ('_ended', '_lender')

    def __init__(self, lender: 'Taking') -> None:
        self._lender = lender
        self._ended = False

    @property
    def claim(self) -> 'ActivityClaim':
        """The claim the lender took, at the root of the lending chain."""
        return self._lender.claim

    @property
    def holds(self) -> bool:
        """Whether this borrowing has not ended and its lender still holds."""
        return not self._ended and self._lender.holds

    def release(self) -> None:
        """End this borrowing and everything lent from it; the lender keeps its claim."""
        self._ended = True
        self.claim._returned(self)

    def lend(self) -> 'BorrowedClaim':
        """Lend this borrowing to work nested inside it."""
        return BorrowedClaim(self)


class BorrowedClaim:
    """A held claim lent to work inside the holder's activity.

    A recording inside a run acts under the run's claim: its start is
    granted while the run still holds the claim, and its end cannot free
    it. The same shape as ActivityClaim to that work, so the work does
    not know whether it runs alone or inside another activity.
    """

    __slots__ = ('_lender',)

    def __init__(self, lender: 'HeldClaim | _Borrowing') -> None:
        self._lender = lender

    @property
    def holder(self) -> 'ActivityHolder | None':
        """The claim's current holder, as ActivityClaim.holder answers it."""
        return self._lender.claim.holder

    @property
    def blocking_holder(self) -> 'ActivityHolder | None':
        """The holder that would refuse a taking through this borrow.

        None while the lender holds: the lender is the activity this work
        runs inside, not one in its way. Once the lender has released, the
        claim's current holder, whoever took it since.
        """
        if self._lender.holds:
            return None
        return self.holder

    def refusing_control(
        self, taking: 'Taking | ControllerTaking | None'
    ) -> 'ControlSession | None':
        """The control session that refuses work made under ``taking``, as ActivityClaim answers it."""
        return self._lender.claim.refusing_control(taking)

    def try_claim(self, owner: str, run: RunIdentity | None = None) -> _Borrowing | None:
        """Act under the lender's taking; None once the lender no longer holds.

        A run that borrows is recorded on the claim as the run holding the
        scope, with its identity; the holder stays the lender.
        """
        return self._lender.claim._lend(self._lender, owner, run)

    def announce(self) -> None:
        """Tell the claim's listener of a change the work made, as ActivityClaim.announce.

        Work that outlives its own borrowing -- a video step's finish runs
        after its recording returned what it borrowed -- still announces
        through the claim it was lent.
        """
        self._lender.claim.announce()


# What a taker holds: its own taking, or a borrowing of someone else's.
# Both answer ``holds``, ``release`` and ``lend`` the same way, so the work
# they cover is written once for either.
Taking = HeldClaim | _Borrowing


class ControllerTaking:
    """What the holder of a control session acts under.

    Not an activity: it takes no slot, so it has nothing to release or
    lend, and it does not end as an activity does. Work made under it is
    the controller's for as long as its session is the claim's live one;
    once the session is ended it is no one's in particular, and is admitted
    or refused as any caller's is. The lanes carry it as they carry every
    taking, so the controller's lane work is known for its own.
    """

    __slots__ = ('_claim', '_session')

    def __init__(self, claim: 'ActivityClaim', session: ControlSession) -> None:
        self._claim = claim
        self._session = session

    @property
    def claim(self) -> 'ActivityClaim':
        """The claim the session was taken from."""
        return self._claim

    @property
    def session(self) -> ControlSession:
        """The control session this taking acts for."""
        return self._session


def command_refused(
    refusing: 'ActivityHolder | ControlSession | None', member: str
) -> HardwareCommandRefusedError:
    """The refusal of a hardware command for what the claim says refuses it.

    The one place a claim's answer becomes the command's refusal, so a
    control session is named the same at every door that asks: a lane, a
    write that would falsify a recording, a change of the shown layer.

    Args:
        refusing: The control session, or the activity holding the scope;
            None when the holder released before it could be read.
        member: The member or task refused, for the log.
    """
    if isinstance(refusing, ControlSession):
        return HardwareCommandRefusedError(
            'control_session_held',
            member,
            holder_trigger=refusing.taken_by,
            control_session=refusing.name,
        )
    return HardwareCommandRefusedError(
        'exclusive_activity_running',
        member,
        refusing.kind if refusing is not None else None,
        holder_trigger=refusing.run_started_by if refusing is not None else None,
    )


def _root(taking: 'Taking | ControllerTaking | None') -> 'HeldClaim | ControllerTaking | None':
    """The taking a borrowing was lent from, through every lender; any other taking itself."""
    while isinstance(taking, _Borrowing):
        taking = taking._lender
    return taking


class ActivityClaim:
    """Arbitrates the session's one exclusive activity.

    Exactly one exclusive activity (a protocol run XOR a video
    recording) may hold the claim at a time. ``try_claim`` is atomic:
    of two concurrent claimants exactly one wins. The claim is not
    reentrant -- a second ``try_claim`` fails even for the same owner,
    so a claimant that lost track of its own state cannot silently
    stack claims. Ownership is the HeldClaim the winner receives, never
    a name: anyone can spell a kind, only the taker holds the object.
    """

    def __init__(self, on_transition=None) -> None:
        """Args:
        on_transition: Optional zero-argument callable invoked after
            every successful claim or release, and on every
            ``announce`` by the activity holding it. It fires OUTSIDE this
            claim's lock but ON the transitioning thread, which may
            hold engine locks of its own -- so it must only schedule
            or notify (level-read listeners re-read state when they
            run); it must not acquire engine locks or block.
        """
        self._lock = threading.Lock()
        self._holder: ActivityHolder | None = None
        self._held: HeldClaim | None = None
        self._on_transition = on_transition
        # A run acting under a lent claim -- a run inside a diagnostic -- and
        # the borrowing it holds. The holder stays the lender, which is what
        # a refusal names; this is what says a run is in progress, and which.
        self._lent_run: tuple[_Borrowing, ActivityHolder] | None = None
        # The writes that would falsify a recording, running now. Counted
        # where each one runs -- a lane's worker, or inline inside another
        # task on it -- and under this lock, so a recording and such a
        # write can never both be admitted.
        self._falsifying = 0
        # Set by the session's close for as long as it runs, under this
        # lock: every new taking is refused meanwhile, while work lent by an
        # activity already under way is still admitted.
        self._closing = False
        # Set when the close finished, and never cleared: a start that
        # passed its own checks while the scope was still connected takes
        # the claim last, so the claim is what refuses it once the session
        # has let go of the scope.
        self._closed = False
        # The control session and its holder's taking, above the activity
        # slot: while set, only the holder's work is admitted, and the slot
        # keeps meaning what is running.
        self._control: tuple[ControlSession, ControllerTaking] | None = None
        # The guarded writes -- a setting, a file -- running now. A take
        # landing mid-write would leave its hardware and its store split, so
        # a take is refused while any runs; counted under this lock.
        self._guarded = 0

    @property
    def closing(self) -> bool:
        """True while the session's close runs."""
        return self._closing

    def begin_closing(self) -> None:
        """Refuse every new taking for the session's close: until ``end_closing``,
        and for good if it finished.

        A taking already held keeps its claim and ends through its owner;
        a borrowing lent from it is still granted, because it is that
        activity's own work. Announced, so a listener hears the close begin.
        """
        with self._lock:
            self._closing = True
        self.announce()

    def end_closing(self, *, closed: bool) -> None:
        """The session's close has ended.

        Args:
            closed: True when it finished, so every new taking stays
                refused for good; False when it raised part-way, so
                ``begin_closing``'s refusal ends with it until the close is
                called again.

        Announced, so a listener hears the close end.
        """
        with self._lock:
            self._closing = False
            self._closed = self._closed or closed
        self.announce()

    @property
    def holder(self) -> ActivityHolder | None:
        """The current holder, or None when unheld.

        One attribute read of an immutable object, so every read stays
        lock-free and a reader cannot see a half-written holder.
        """
        return self._holder

    @property
    def blocking_holder(self) -> ActivityHolder | None:
        """The holder that would refuse a taking: the current holder.

        The same question BorrowedClaim answers, so a caller that may hold
        either asks it once without knowing which it holds.
        """
        return self._holder

    @property
    def owner(self) -> str | None:
        """The current holder's kind, or None when unheld."""
        holder = self._holder
        return holder.kind if holder is not None else None

    @property
    def run_holder(self) -> ActivityHolder | None:
        """The run holding the scope, or None when no run does.

        The holder when it is a run; otherwise a run acting under the
        holder's lent claim, for as long as that borrowing holds.
        """
        holder = self._holder
        if holder is not None and holder.kind == 'protocol':
            return holder
        lent = self._lent_run
        if lent is None:
            return None
        borrowing, run = lent
        return run if borrowing.holds else None

    @property
    def control(self) -> ControlSession | None:
        """The control session held now, or None."""
        control = self._control
        return control[0] if control is not None else None

    def take_control(self, name: str) -> ControllerTaking:
        """Take the one control session, for the caller this thread acts for.

        Taken only while nothing of anyone else's is under way: no other
        session, no activity in the slot -- a run, recording, home or
        diagnostic, the operator's or a session's still unwinding -- and no
        guarded write still running, so nothing is ever put under a client's
        control half done. Announced.

        Args:
            name: What the person at the scope is told holds it.

        Returns:
            The taking the holder's calls act under (``acting``).

        Raises:
            SessionClosingError: the session's close is running or has
                finished.
            ControlSessionRefusedError: ``'control_session_held'``,
                ``'exclusive_activity_running'`` or ``'write_in_flight'``.
                Nothing was taken.
        """
        caller = current_caller()
        with self._lock:
            if self._closing or self._closed:
                raise SessionClosingError('control')
            if self._control is not None:
                held = self._control[0]
                raise ControlSessionRefusedError(
                    'control_session_held',
                    holder_trigger=held.taken_by,
                    control_session=held.name,
                )
            if self._holder is not None:
                raise ControlSessionRefusedError(
                    'exclusive_activity_running',
                    holder=self._holder.kind,
                    holder_trigger=self._holder.run_started_by,
                )
            if self._guarded:
                raise ControlSessionRefusedError('write_in_flight')
            session = ControlSession(
                id=uuid.uuid4().hex, name=name, taken_by=caller, taken_at=time.time()
            )
            controller = ControllerTaking(self, session)
            self._control = (session, controller)
        self.announce()
        return controller

    def end_control(self, session_id: str) -> ControlSession:
        """End the live control session, named by its id; anyone may. Announced.

        Ends the session only: what its holder started -- a run, a
        recording -- is the ender's to stop first, and holds the slot until
        it has unwound.

        Returns:
            The session ended.

        Raises:
            ControlSessionRefusedError: ``'control_session_ended'`` -- no
                session is held, or the live one has another id.
        """
        with self._lock:
            control = self._control
            if control is None or control[0].id != session_id:
                raise ControlSessionRefusedError('control_session_ended')
            self._control = None
        self.announce()
        return control[0]

    def enter_guarded_write(self, taking: 'Taking | ControllerTaking | None') -> None:
        """Admit a guarded write -- a setting, a file -- and count it while it runs.

        Admitted by the control rule alone: while a session is held, only
        work for its holder. An activity in the slot refuses nothing here;
        the write's own member says what a run, a diagnostic or a home
        allows. Every admission is paired with ``leave_guarded_write`` when
        the write returns or raises, and no take lands while one runs.

        Raises:
            ControlSessionRefusedError: ``'control_session_held'``. Nothing
                was counted.
        """
        with self._lock:
            session = self._control_refusing(taking)
            if session is not None:
                raise ControlSessionRefusedError(
                    'control_session_held',
                    holder_trigger=session.taken_by,
                    control_session=session.name,
                )
            self._guarded += 1

    def refusing_control(self, taking: 'Taking | ControllerTaking | None') -> ControlSession | None:
        """The control session that refuses work made under ``taking``, or None.

        The control rule alone, read without counting anything: a caller
        that only asks -- a run's ``prepare`` -- reads what a take of the
        slot or a lane would answer.
        """
        with self._lock:
            return self._control_refusing(taking)

    def leave_guarded_write(self) -> None:
        """A write admitted by ``enter_guarded_write`` has returned or raised."""
        with self._lock:
            self._guarded -= 1

    def try_claim(self, owner: str, run: RunIdentity | None = None) -> HeldClaim | None:
        """Atomically claim for ``owner``; None when already held.

        While a control session is held, only its holder takes the slot --
        acting under the controller's taking or under an activity taken for
        the session -- and the activity it takes is recorded as the
        session's.

        Args:
            owner: the kind of activity claiming -- a description for
                display and refusal text, not a credential.
            run: which run is claiming. Required when ``owner`` is
                ``'protocol'`` and refused otherwise: the holder cannot be
                built any other way.

        Returns:
            The HeldClaim that alone can release this taking, or None
            when another activity holds the claim.

        Raises:
            SessionClosingError: the session's close is running
                (``begin_closing``) or has finished. Nothing was taken.
            ControlSessionHeldError: a control session is held and this
                thread does not act for its holder. Nothing was taken.
            FalsifyingChangeInFlightError: ``owner`` is ``'recording'`` and a
                write that would falsify it is running.
            ValueError: a run claimed without its identity, or another
                activity claimed with one.
        """
        holder = ActivityHolder(kind=owner, run=run)
        taking = current_taking()
        with self._lock:
            if self._closing or self._closed:
                raise SessionClosingError(owner)
            session = self._control_refusing(taking)
            if session is not None:
                raise ControlSessionHeldError(session)
            if owner == 'recording' and self._falsifying:
                raise FalsifyingChangeInFlightError()
            if self._holder is not None:
                return None
            held = HeldClaim(self, self._control[0] if self._control is not None else None)
            self._held = held
            self._holder = holder
        self.announce()
        return held

    def refusing_holder(
        self, taking: 'Taking | ControllerTaking | None', *, falsifies_recording: bool = False
    ) -> ActivityHolder | ControlSession | None:
        """What refuses work made under ``taking`` -- the control session, or the activity -- or None.

        While a control session is held, only its holder's work is admitted:
        work under the controller's taking, or under an activity taken for
        the session; anything else is refused, and the session is named.

        Then, while a run, a diagnostic or a home holds the scope, only work
        under its taking -- the HeldClaim, or a borrowing of it that has not
        ended -- is the holder's. Anything else is refused, the controller's
        own plain work included, and the holder is named so the refusal can
        say who has the scope.

        A recording refuses only work that would falsify the file it is
        writing (``falsifies_recording``: the frame geometry, the pixel
        format, the turret), whoever makes it -- the recording itself never
        does. Everything else stays open to anyone during one.
        """
        return self._refusing(self._holder, taking, falsifies_recording)

    def enter_falsifying_change(
        self, taking: 'Taking | ControllerTaking | None'
    ) -> ActivityHolder | ControlSession | None:
        """Admit a write that would falsify a recording, counted while it runs.

        The refusal ``refusing_holder`` gives it, decided under the lock that
        a recording's taking holds, so a recording is either refused because
        this write runs or holds before it and refuses the write. Every
        admission is paired with ``leave_falsifying_change`` when the write
        returns.

        Returns:
            What refuses the write -- the control session, or the activity
            holding the scope -- or None when it is admitted and counted.
        """
        with self._lock:
            holder = self._refusing(self._holder, taking, True)
            if holder is None:
                self._falsifying += 1
            return holder

    def leave_falsifying_change(self) -> None:
        """A write admitted by ``enter_falsifying_change`` has returned."""
        with self._lock:
            self._falsifying -= 1

    def _refusing(
        self,
        holder: ActivityHolder | None,
        taking: 'Taking | ControllerTaking | None',
        falsifies_recording: bool,
    ) -> ActivityHolder | ControlSession | None:
        session = self._control_refusing(taking)
        if session is not None:
            return session
        if holder is None:
            return None
        if holder.kind == 'recording':
            return holder if falsifies_recording else None
        if holder.kind not in SCOPE_HOLDING_KINDS:
            return None
        activity = activity_taking(taking)
        if activity is not None and activity.holds and _root(activity) is self._held:
            return None
        return holder

    def _control_refusing(
        self, taking: 'Taking | ControllerTaking | None'
    ) -> ControlSession | None:
        """The control session that refuses work made under ``taking``, or None.

        Its holder's work is the controller's taking, or an activity taken
        for the session that still holds, through every borrowing of it.
        """
        control = self._control
        if control is None:
            return None
        session, controller = control
        root = _root(taking)
        if root is controller:
            return None
        if isinstance(root, HeldClaim) and root.control is session and root.holds:
            return None
        return session

    def _is_held_by(self, held: HeldClaim) -> bool:
        return self._held is held

    def _lend(self, lender: Taking, owner: str, run: RunIdentity | None) -> _Borrowing | None:
        """A borrowing of *lender*'s taking; a run's is recorded and announced.

        Decided under the lock a release takes, so a run is never recorded
        against a lender that has already released.
        """
        holder = ActivityHolder(kind=owner, run=run)
        with self._lock:
            if not lender.holds:
                return None
            borrowing = _Borrowing(lender)
            is_run = owner == 'protocol'
            if is_run:
                self._lent_run = (borrowing, holder)
        if is_run:
            self.announce()
        return borrowing

    def _returned(self, borrowing: _Borrowing) -> None:
        """A borrowing ended; if it was the lent run, the run is over and announced."""
        with self._lock:
            lent = self._lent_run
            was_the_run = lent is not None and lent[0] is borrowing
            if was_the_run:
                self._lent_run = None
        if was_the_run:
            self.announce()

    def _release(self, held: HeldClaim) -> None:
        with self._lock:
            if self._held is not held:
                held_by = self._holder.kind if self._holder is not None else None
                raise RuntimeError(
                    f'ActivityClaim: a release by a taking that does not hold the claim '
                    f'(held by {held_by!r})'
                )
            self._held = None
            self._holder = None
        self.announce()

    def announce(self) -> None:
        """Tell the listener the Session's run state changed; a raise stays here.

        The claim calls it for its own grant and release. An activity holding
        the claim calls it for a change of its own the holder does not show --
        a recording going live after its grant, its selection closing while it
        still holds, its finish ending after its release -- so a listener hears
        every edge from one place, and none is inferred by polling.

        The claim is already settled when this runs. A listener's raise
        carried out to the taker would leave the scope held by a taking the
        taker never received, which nothing can release; carried out to a
        releaser, it would skip what the releaser does next -- a run's return
        to IDLE, a recording's drained signal. So it is reported here, where
        it happened, and the transition completes.
        """
        if self._on_transition is None:
            return
        try:
            self._on_transition()
        except Exception as ex:
            from modules.notification_center import notifications

            notifications.report_outcome(ex, solicited=False, category='Run State')
