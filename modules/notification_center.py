# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""Centralized user-facing notification system.

Any thread can post a notification; UI subscribes and shows popups on the
main thread.  Replaces scattered ``show_notification_popup()`` calls with
a single bus that handles thread safety, deduplication, and severity
filtering.

Usage::

    from modules.notification_center import notifications

    # Producer (any thread):
    notifications.error("Motor", "Connection Lost", "Serial timeout on HOME")

    # Consumer (UI init, once):
    notifications.add_listener(my_callback, min_severity=Severity.WARNING)
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

from drivers.exceptions import HardwareError
from lib import profile_trace
from modules.exceptions import (
    BringUpError,
    CameraSettingRejected,
    CaptureError,
    ConfigError,
    HomingFailedError,
    MotorStopFailedError,
    MoveNotCompletedError,
    Notice,
    PluginError,
    ProtocolError,
    Quiet,
    Refusal,
    RefusalCause,
    Remedy,
    ScopeDisconnectError,
    SupportReportNotSavedError,
)
from modules.api_surface import api_fields

logger = logging.getLogger('LVP.notifications')
# The reporter's own record -- what happened, with the traceback when it is
# a fault: a fault's one line, shown or not, and the line of any outcome no
# one is shown. A shown refusal's or notice's one line is its display line,
# on the notifications logger, so an outcome is never read as two events.
_outcome_logger = logging.getLogger('LVP.outcomes')

# Faults whose message is written for the person. Any other exception's
# str() is a developer's words -- a Python class name, a repr -- so the
# person reads a generic sentence and the log carries the rest.
_TYPED_FAULTS = (
    BringUpError,
    CameraSettingRejected,
    CaptureError,
    ProtocolError,
    ConfigError,
    HardwareError,
    HomingFailedError,
    MotorStopFailedError,
    MoveNotCompletedError,
    PluginError,
    ScopeDisconnectError,
    SupportReportNotSavedError,
)

_UNTYPED_FAULT_BODY = 'The operation did not complete. Check the main log for details.'

# Set on an exception object once each half of its report is done, so the
# same object reported again -- by the lane that raised it and then by the
# caller that waited on it -- is logged once and shown at most once.
_LOGGED_MARK = '_lvp_outcome_logged'
_SHOWN_MARK = '_lvp_outcome_shown'
# The outcome id an object's every delivery carries, set on its first report.
_ID_MARK = '_lvp_outcome_id'


def _outcome_words(exception: BaseException) -> str:
    """The exception's own words, or a sentence saying they could not be written.

    An exception's ``__str__`` is its author's code and can itself raise.
    The reporter is where every outcome ends its flight, so a raise here
    would escape into whatever was reporting it -- a lane, a listener, a
    run's cleanup -- instead of being reported.
    """
    try:
        return str(exception)
    except Exception as failure:
        return (
            f'{type(exception).__name__} (its message could not be written: '
            f'{type(failure).__name__})'
        )


class Severity(IntEnum):
    """Notification severity levels (matches Python logging levels).

    NOTICE sits between INFO and WARNING: user-facing status (start/done
    of a long unattended operation) that says so without misdeclaring
    itself as a fault. WARNING stays "something didn't work". The level
    says how a post is logged, not whether it is shown: ``shown`` on the
    notification says that.
    """

    DEBUG = logging.DEBUG  # 10
    INFO = logging.INFO  # 20
    NOTICE = 25
    WARNING = logging.WARNING  # 30
    ERROR = logging.ERROR  # 40
    CRITICAL = logging.CRITICAL  # 50


# Name the custom level so log lines read 'NOTICE', not 'Level 25'.
logging.addLevelName(int(Severity.NOTICE), 'NOTICE')


class OutcomeKind(StrEnum):
    """What an outcome is, read from its type by the reporter.

    A string enum, so the value crosses a wire as itself. ``UNCLASSIFIED``
    is a post made straight to the centre rather than through the
    reporter: its kind was never declared, so the record says so rather
    than guessing from its severity. ``QUIET`` is never delivered to a
    listener, since a quiet outcome is never shown; it is the kind a caller
    that receives one as its answer reads.
    """

    REFUSAL = 'refusal'
    FAULT = 'fault'
    NOTICE = 'notice'
    QUIET = 'quiet'
    UNCLASSIFIED = 'unclassified'


@dataclass(frozen=True)
class Outcome:
    """What an exception's type says it is, read once for every host that answers it.

    Attributes:
        kind: A refusal, a notice, a quiet outcome or a fault.
        title: The heading its type declares; None for a fault whose type
            declares none, which each host names in its own way.
        words: Its own words, or a sentence saying they could not be written.
        for_person: Whether ``words`` are written for the person: a
            refusal's, a notice's, a quiet outcome's, and a fault's whose type
            writes them so. Any other fault's are a developer's.
        reason: Its machine-readable code; empty when its type declares none.
        remedy: The one action that answers it, when it has one.
        fatal: A fault that ends what was running.
        cause: Whether the same request could succeed later, as a refusal's
            or a quiet outcome's type declares (``RefusalCause``); None for a
            fault, a notice, and a by-contract cancel, which declares none.
    """

    kind: OutcomeKind
    title: str | None
    words: str
    for_person: bool
    reason: str
    remedy: Remedy | None
    fatal: bool
    cause: RefusalCause | None = None


def outcome_of(exception: BaseException) -> Outcome:
    """What *exception* is, as its type says: the one reading of an outcome's kind and words."""
    # Quiet first: a quiet outcome is never shown, whatever else its type is.
    if isinstance(exception, (Quiet, CancelledError)):
        kind = OutcomeKind.QUIET
    elif isinstance(exception, Refusal):
        kind = OutcomeKind.REFUSAL
    elif isinstance(exception, Notice):
        kind = OutcomeKind.NOTICE
    else:
        kind = OutcomeKind.FAULT
    words = _outcome_words(exception)
    fault = kind == OutcomeKind.FAULT
    return Outcome(
        kind=kind,
        title=getattr(exception, 'title', None) or None,
        words=words,
        for_person=not fault or (isinstance(exception, _TYPED_FAULTS) and bool(words)),
        reason=getattr(exception, 'reason', None) or '',
        remedy=getattr(exception, 'remedy', None),
        fatal=fault and bool(getattr(exception, 'fatal', False)),
        cause=_cause_of(exception) if kind in (OutcomeKind.REFUSAL, OutcomeKind.QUIET) else None,
    )


def _cause_of(exception: BaseException) -> RefusalCause | None:
    # A fault's ``cause`` can be something else entirely (the OSError behind
    # an unreadable file), so only a declared RefusalCause is one.
    cause = getattr(exception, 'cause', None)
    return cause if isinstance(cause, RefusalCause) else None


# One id per outcome: an outcome reported muted and later shown is one
# outcome delivered twice, and a subscriber tells the two deliveries apart
# from two outcomes by this number. next() on a count is atomic under the GIL.
_outcome_ids = itertools.count(1)


def _next_outcome_id() -> int:
    return next(_outcome_ids)


@api_fields(
    'severity',
    'category',
    'title',
    'message',
    'timestamp',
    'source',
    'fatal',
    'solicited',
    'operation_key',
    'remedy',
    'kind',
    'outcome_id',
    'reason',
    'shown',
    'wall_time',
)
@dataclass(frozen=True)
class Notification:
    """Immutable notification payload delivered to listeners.

    Every listener receives every post, shown or not: ``shown`` is the
    centre's decision whether this one is for display now (shutdown, an
    unattended run's mute, an attended run that already showed the same post,
    and the dedup window say no), so a listener that
    displays opens only the shown ones and a listener that records keeps
    them all.
    """

    severity: Severity
    category: str  # e.g. "Motor", "Camera", "FileIO", "Protocol"
    title: str  # short summary shown in popup title
    message: str  # detail shown in popup body
    timestamp: float = field(default_factory=time.monotonic)
    source: str = ''  # optional originating module/function
    fatal: bool = False  # reaches listeners even while a protocol suppresses popups
    # True when this notification ANSWERS a request that just arrived --
    # a refusal of a button press or an API call. Both suppression rules
    # below rest on a premise it falsifies: the unattended mute assumes
    # nobody is watching, and dedup assumes "already shown recently",
    # but someone asked, and asking twice is asking twice. Distinct from
    # fatal, which is about the fault's severity: a fault that ends the
    # operation must reach a watching user, yet one fault repeating is
    # still one fault and must still dedup.
    solicited: bool = False
    # Names the operation this notification is about, when it is one of a
    # sequence describing the same piece of work -- a "starting" notice and
    # the "finished" or "failed" notice that answers it. A UI listener can
    # then replace the earlier message instead of stacking a second one on
    # top of it. Empty for the ordinary standalone notification.
    operation_key: str = ''
    # The action that answers an outcome, when it has one: a UI listener shows
    # the notification as an offer to take it rather than as a plain warning.
    remedy: Remedy | None = None
    kind: OutcomeKind = OutcomeKind.UNCLASSIFIED
    # The same for both deliveries of one outcome (muted, then shown when a
    # person asks for it); different for every other outcome.
    outcome_id: int = field(default_factory=_next_outcome_id)
    # The outcome's machine-readable code, read from its type; empty for a
    # post that declares none.
    reason: str = ''
    shown: bool = True
    # Wall-clock seconds, for a client in another process; ``timestamp`` is
    # monotonic and orders two notifications within this one.
    wall_time: float = field(default_factory=time.time)


@dataclass
class _RunScope:
    """A live run, as the centre judges the posts made during it."""

    attended: bool
    # (category, title, message) of every post shown during the run: an
    # attended run shows an identical non-fatal post once. The message is in
    # the key because different refusals share a title -- a refused gain and
    # a refused exposure are both 'Camera Setting Not Applied'.
    shown: set[tuple[str, str, str]] = field(default_factory=set)


def _log_display_line(
    severity: Severity, category: str, title: str, message: str, reason: str
) -> None:
    """The line a displayed post leaves: ``[category] title (reason): message``.

    Collapsed to one physical line: message prose may span paragraphs, and
    raw continuation lines carry no level or timestamp prefix.
    """
    from modules import gui_logger

    because = f' ({reason})' if reason else ''
    logger.log(
        int(severity),
        f'[{category}] {gui_logger.one_line(title)}{because}: {gui_logger.one_line(message)}',
    )


class NotificationCenter:
    """Thread-safe notification bus.

    Producers call ``notify()`` (or convenience methods ``error()``, etc.)
    from any thread.  The call always logs via ``lvp_logger`` so file
    logging is never lost.  Registered listeners are invoked inline on the
    producer's thread -- UI listeners must wrap work in
    ``Clock.schedule_once``.

    Deduplication: a notification with the same ``(category, title)`` as
    one shown within ``dedup_window_s`` is not shown. Every listener still
    receives it, with ``shown`` False, and the full message still goes to
    the log file.
    """

    def __init__(self, dedup_window_s: float = 10.0):
        self._lock = threading.Lock()
        self._listeners: list[tuple[Severity, callable]] = []
        self._dedup: dict[tuple[str, str], float] = {}
        self._dedup_window_s = dedup_window_s
        # Shutdown suppression flag. When True, notifications still
        # get LOGGED (so post-mortem diagnostics survive) but no
        # listeners are invoked -- prevents the 30+ error-notification
        # flood during close that fires when queued IO tasks fail en
        # masse after the motor/camera disconnects. Issue #622.
        self._shutting_down = False
        # The live run, while one is in flight; None otherwise. During a run
        # nobody is watching, non-fatal notifications still LOG but raise no
        # popup -- a modal could stall the run, and transient faults would
        # pile up in front of an empty chair. During a run someone is
        # standing at, a fault that repeats every step is shown once rather
        # than once per dedup window. Fatal notifications (lost connection, a
        # run-aborting fault) reach listeners either way.
        #
        # ATTENDEDNESS, not "a run is in flight": the capture runner drives
        # short interactive operations too, and one of those suppressing its
        # own failure popup is exactly the bug this split prevents. The
        # runner decides which kind it is and says so; the scope only obeys.
        self._run_scope: _RunScope | None = None

    def set_shutting_down(self, value: bool = True) -> None:
        """Toggle suppression of listener dispatch. Call from on_stop
        BEFORE disconnecting hardware so teardown-induced task failures
        don't spam popups/toasts on their way out. Logs still capture
        everything."""
        with self._lock:
            self._shutting_down = bool(value)

    def open_run_scope(self, *, attended: bool) -> None:
        """Judge the posts that follow as a live run's, until close_run_scope().

        An unattended run mutes NON-FATAL, unsolicited posts; an attended run
        shows each identical one once. Fatal and solicited posts are judged as
        outside a run; logs always capture everything.

        The caller passes attendedness, not "am I busy": an interactive
        operation that routes through the same runner must pass True, or it
        silences its own failure popup. Opening replaces any scope left open,
        so a missed close is healed by the next run rather than muting every
        later one.
        """
        with self._lock:
            self._run_scope = _RunScope(attended=attended)

    def close_run_scope(self) -> None:
        """End the live run's judgement. Idempotent: pair it with every
        cleanup path, so the scope cannot stick on after the run ends."""
        with self._lock:
            self._run_scope = None

    # ------------------------------------------------------------------
    # Producer API (any thread)
    # ------------------------------------------------------------------

    def notify(
        self,
        severity: Severity,
        category: str,
        title: str,
        message: str,
        source: str = '',
        fatal: bool = False,
        operation_key: str = '',
        solicited: bool = False,
        reason: str = '',
        remedy: Remedy | None = None,
        kind: OutcomeKind | None = None,
        outcome_id: int | None = None,
    ) -> bool:
        """Post a notification and log its display line.  Thread-safe.

        Returns whether it was shown: False when shutdown, an unattended run's
        mute, an attended run that already showed the same post, or the dedup
        window suppressed it. Every listener receives it either way, with
        ``shown`` saying which.

        ``fatal`` notifications are shown even while a run suppresses
        non-fatal popups (its scope, ``open_run_scope``).

        ``solicited`` notifications answer a request that just arrived, so
        neither suppression rule applies to them: the caller is present by
        construction, and a repeated request is a repeated question. Set it
        at the funnel that knows the notification is an answer, never at an
        emitter that cannot tell who asked.

        ``operation_key`` marks this as one of a sequence about a single piece
        of work, so a UI listener can replace the earlier message rather than
        stack on it.

        ``reason`` is the outcome's machine-readable code, written into the
        log line and carried on the notification: two outcomes can share a
        title, and a support bundle has to tell them apart from the one line
        a shown outcome leaves, as a client has to from the record.

        ``remedy`` is the action an outcome names as its answer, carried to the
        listeners on the notification.

        ``kind`` and ``outcome_id`` are the reporter's: what the outcome is,
        read from its type, and the id its every delivery shares. A post made
        here directly declares no kind; it is a notice at NOTICE and below and
        unclassified above, and gets an id of its own.
        """
        _log_display_line(severity, category, title, message, reason)
        return self._deliver(
            severity,
            category,
            title,
            message,
            source=source,
            fatal=fatal,
            operation_key=operation_key,
            solicited=solicited,
            reason=reason,
            remedy=remedy,
            kind=kind,
            outcome_id=outcome_id,
        )

    def _deliver(
        self,
        severity: Severity,
        category: str,
        title: str,
        message: str,
        *,
        source: str = '',
        fatal: bool = False,
        operation_key: str = '',
        solicited: bool = False,
        reason: str = '',
        remedy: Remedy | None = None,
        kind: OutcomeKind | None = None,
        outcome_id: int | None = None,
    ) -> bool:
        """Deliver a post, writing no display line: its interaction record,
        whether it is shown, and every listener. Returns whether it was shown.

        ``notify()`` writes the display line first; ``report_outcome`` writes
        each outcome's one line itself, so an outcome is never logged twice.
        """
        from modules import gui_logger

        # Forensics: every notification (independent of any UI popup
        # bridge that may suppress it post-shutdown) lands in
        # gui_interactions.log so post-mortem can see what messages
        # the user was looking at. Best-effort -- gui_logger import or
        # logging stack failures don't disrupt the notify path.
        # Failure surfaces at warning level in the main log so a
        # silently-broken forensic-log subsystem is visible during
        # post-mortem; stderr-print is intentionally NOT used because
        # frozen pyinstaller builds suppress stderr from L1 users.
        try:
            from modules import gui_logger

            gui_logger.notification(
                severity.name if hasattr(severity, 'name') else str(severity),
                f'{category}/{title}',
                message,
                source=source or '',
            )
        except Exception as e:
            logger.warning(f'notification forensic write failed: {type(e).__name__}: {e}')

        # Dedup check + shutdown suppression: they decide whether the post is
        # shown, never whether a listener hears it.
        key = (category, title)
        now = time.monotonic()
        suppressed_reason = None
        with self._lock:
            listeners = list(self._listeners)
            scope = self._run_scope
            judged_by_run = scope is not None and not fatal and not solicited
            if self._shutting_down:
                suppressed_reason = 'shutdown'  # logged above; suppressed during close
            elif judged_by_run and not scope.attended:
                # logged above; non-fatal popups suppressed on an unattended run
                suppressed_reason = 'unattended_run'
            elif judged_by_run and (category, title, message) in scope.shown:
                suppressed_reason = 'shown_this_run'
            else:
                last = self._dedup.get(key, 0.0)
                # The window still advances for a solicited notification, so a
                # later unsolicited repeat of the same (category, title) is
                # measured from the answer the user actually saw.
                if not solicited and (now - last) < self._dedup_window_s:
                    suppressed_reason = 'dedup'  # already shown recently
                else:
                    self._dedup[key] = now
            if suppressed_reason is None and scope is not None:
                scope.shown.add((category, title, message))
        shown = suppressed_reason is None
        if not shown:
            # The forensic write above happens BEFORE this decision, so on its
            # own it says "posted", never "seen". Without this line a support
            # bundle cannot answer whether the user was ever shown a failure --
            # the popup is the only carrier, so a suppressed one would leave no
            # record anywhere that it happened. Unconditional, not behind the
            # profile-trace flag, because the question is asked of customer
            # logs captured long after the fact.
            logger.info(
                f'[{category}] {gui_logger.one_line(title)}: '
                f'not shown to the user (suppressed: {suppressed_reason})'
            )
            # Emitted outside the lock: the tracer takes its own module-wide
            # lock, and nesting the two would order a pair of locks for the
            # sake of a diagnostic. What the user never saw IS the
            # measurement here -- the popup is currently the only carrier for
            # these failures, so a suppressed one otherwise leaves no record
            # anywhere that it happened.
            if profile_trace.ENABLE_PROFILE_TRACE:
                profile_trace.trace(
                    'notification_suppressed_trace.csv',
                    'ts_ms,reason,severity,category,title,fatal',
                    [
                        f'{time.time() * 1000.0:.3f}',
                        suppressed_reason,
                        getattr(severity, 'name', severity),
                        category,
                        title,
                        int(bool(fatal)),
                    ],
                    recording_id=profile_trace.NO_RECORDING,
                )

        if kind is None:
            kind = OutcomeKind.NOTICE if severity <= Severity.NOTICE else OutcomeKind.UNCLASSIFIED
        n = Notification(
            severity=severity,
            category=category,
            title=title,
            message=message,
            timestamp=now,
            source=source,
            fatal=fatal,
            operation_key=operation_key,
            solicited=solicited,
            remedy=remedy,
            kind=kind,
            outcome_id=_next_outcome_id() if outcome_id is None else outcome_id,
            reason=reason,
            shown=shown,
        )
        for min_sev, cb in listeners:
            if severity >= min_sev:
                try:
                    cb(n)
                except Exception:
                    # Logged, not reported: a report is itself a post, and it
                    # would go straight back to the listener that just raised.
                    logger.exception(
                        f'[{category}] a notification listener raised on '
                        f'{gui_logger.one_line(title)!r}; the others were still told'
                    )
        return shown

    def report_outcome(
        self,
        exception: BaseException,
        *,
        solicited: bool,
        category: str,
        log_only: bool = False,
        fault_title: str = 'Operation failed',
        operation_key: str = '',
    ) -> None:
        """Log an outcome once and show it at most once, as its type says.

        The one place an outcome becomes a log record and a notification: an
        exception that ended its flight, or a notice, which is reported and
        never raised. What it is -- a refusal (``Refusal``), a notice
        (``Notice``), a quiet outcome (``Quiet``, or a by-contract cancel) or
        a fault (anything else) -- and the words, title and level all come
        from the exception's type; the caller says only whether a person just
        asked (``solicited``), which ``category`` it belongs to, and, with
        ``log_only``, that no one is to be shown it.

        An outcome is one log line, and this is where each is written. A
        fault's, shown or not: ERROR, naming its type, its reason code and its
        own words, with its traceback, so the record says what failed even
        when the person is shown the generic sentence; its display writes no
        line. A shown refusal's or notice's is its display line, the one
        ``notify()`` writes for a direct post, naming the type's reason code
        when it has one: one WARNING line for a refusal, one NOTICE line for
        a notice. One no one is shown: a quiet outcome at INFO, a refusal at
        WARNING and a notice at NOTICE, with no traceback. A refusal is
        shown as a warning and a notice as a notice, each under its
        ``title``; a fault as an error, or as critical when its type says
        ``fatal``, in its own words when its type writes them for a person
        and in a generic sentence when it does not, under its ``title`` or
        ``fault_title``. A quiet outcome is never shown. An outcome's
        ``remedy`` travels with it whatever its kind, and a fault whose type
        says ``fatal`` is shown even during an unattended run.

        ``operation_key`` names the operation this outcome answers when an
        earlier notice announced it, so the outcome replaces that notice
        instead of opening beside it. A refusal given none replaces the last
        refusal shown.

        Each half happens once per exception object, whoever reports it and
        from whichever thread. Shown once means delivered once: a post that
        shutdown, an unattended run's mute or the dedup window suppressed
        leaves the object unshown, for a later report to show. The listeners
        hear both deliveries, under the one ``outcome_id`` the object keeps.
        """
        outcome = outcome_of(exception)
        refusal = outcome.kind == OutcomeKind.REFUSAL
        notice = outcome.kind == OutcomeKind.NOTICE
        quiet = outcome.kind == OutcomeKind.QUIET
        # Check-and-mark only: notify() takes this same lock, so logging and
        # notifying happen after it is released.
        with self._lock:
            do_log = not getattr(exception, _LOGGED_MARK, False)
            do_show = not log_only and not quiet and not getattr(exception, _SHOWN_MARK, False)
            if do_log:
                setattr(exception, _LOGGED_MARK, True)
            if do_show:
                setattr(exception, _SHOWN_MARK, True)
            outcome_id = getattr(exception, _ID_MARK, None)
            if outcome_id is None:
                outcome_id = _next_outcome_id()
                setattr(exception, _ID_MARK, outcome_id)

        type_name = type(exception).__name__
        words = outcome.words
        reason = outcome.reason
        if do_log:
            if quiet:
                _outcome_logger.info(f'[{category}] {type_name}: {words}')
            elif refusal:
                if not do_show:
                    because = f', {reason}' if reason else ''
                    _outcome_logger.warning(f'[{category}] refused ({type_name}{because}): {words}')
            elif notice:
                if not do_show:
                    _outcome_logger.log(int(Severity.NOTICE), f'[{category}] {type_name}: {words}')
            else:
                because = f' ({reason})' if reason else ''
                _outcome_logger.error(
                    f'[{category}] raised {type_name}{because}: {words}', exc_info=exception
                )
        if not do_show:
            return
        remedy = outcome.remedy
        if refusal:
            _log_display_line(Severity.WARNING, category, exception.title, words, reason)
            delivered = self._deliver(
                Severity.WARNING,
                category,
                exception.title,
                words,
                solicited=solicited,
                operation_key=operation_key or REFUSAL_OPERATION_KEY,
                reason=reason,
                remedy=remedy,
                kind=OutcomeKind.REFUSAL,
                outcome_id=outcome_id,
            )
        elif notice:
            _log_display_line(Severity.NOTICE, category, exception.title, words, reason)
            delivered = self._deliver(
                Severity.NOTICE,
                category,
                exception.title,
                words,
                solicited=solicited,
                operation_key=operation_key,
                reason=reason,
                remedy=remedy,
                kind=OutcomeKind.NOTICE,
                outcome_id=outcome_id,
            )
        else:
            body = words if outcome.for_person else _UNTYPED_FAULT_BODY
            title = outcome.title or fault_title
            fatal = outcome.fatal
            # The fault's one line is the reporter's, written above.
            delivered = self._deliver(
                Severity.CRITICAL if fatal else Severity.ERROR,
                category,
                title,
                body,
                solicited=solicited,
                operation_key=operation_key,
                fatal=fatal,
                reason=reason,
                remedy=remedy,
                kind=OutcomeKind.FAULT,
                outcome_id=outcome_id,
            )
        if not delivered:
            # A suppressed post was never seen, so it has not spent the one
            # show: the person's own later request for this outcome shows it.
            with self._lock:
                setattr(exception, _SHOWN_MARK, False)

    # Convenience methods
    def debug(self, category: str, title: str, message: str, **kw) -> bool:
        return self.notify(Severity.DEBUG, category, title, message, **kw)

    def info(self, category: str, title: str, message: str, **kw) -> bool:
        return self.notify(Severity.INFO, category, title, message, **kw)

    def notice(self, category: str, title: str, message: str, **kw) -> bool:
        return self.notify(Severity.NOTICE, category, title, message, **kw)

    def warning(self, category: str, title: str, message: str, **kw) -> bool:
        return self.notify(Severity.WARNING, category, title, message, **kw)

    def error(self, category: str, title: str, message: str, **kw) -> bool:
        return self.notify(Severity.ERROR, category, title, message, **kw)

    def critical(self, category: str, title: str, message: str, **kw) -> bool:
        # App-level failures are fatal: they reach listeners even while a
        # protocol suppresses non-fatal popups, unless a caller overrides.
        kw.setdefault('fatal', True)
        return self.notify(Severity.CRITICAL, category, title, message, **kw)

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def add_listener(
        self, callback: Callable[[Notification], None], min_severity: Severity = Severity.WARNING
    ) -> None:
        """Register a listener.  Called on the producer's thread, for every
        post at or above ``min_severity``, shown or not."""
        with self._lock:
            self._listeners.append((min_severity, callback))

    def remove_listener(self, callback: Callable[[Notification], None]) -> None:
        """Unregister a listener.

        Matched by equality, not identity: a bound method is a new object on
        every attribute access, so ``remove_listener(obj.method)`` must find
        the one ``add_listener(obj.method)`` registered.
        """
        with self._lock:
            self._listeners = [(s, cb) for s, cb in self._listeners if cb != callback]

    # ------------------------------------------------------------------
    # Testing / introspection
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all state (for testing)."""
        with self._lock:
            self._listeners.clear()
            self._dedup.clear()


# The operation this key names is "the answer to the user's last refused
# request". One key for every refusal, deliberately: the newest refusal is
# the true answer, so the bridge's supersession replaces the dialog on a
# second press instead of stacking one per press -- which is what bounds the
# dialogs now that a solicited notification no longer dedups.
REFUSAL_OPERATION_KEY = 'run_refusal'


# Module-level singleton -- import this in producers and consumers.
notifications = NotificationCenter()
