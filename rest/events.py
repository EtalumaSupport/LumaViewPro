# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The event stream: everything the scope tells a listener, on one ``text/event-stream``.

``GET /api/v1/events``. The events are the listener members the API marks
with their record (``@api(event=Record)``), each named for its member
(``add_position_listener`` is ``position``), and the run events
(``RunEvents``), which the server hears by filling each run member's
``events`` itself. An event's ``data`` is its record, wire-encoded, keys its
field names; a run event adds ``run``, the handle of the run that sent it.
The frame events are the live view's, not the stream's.

The first event is ``status``, the ``Status`` record, with the current
``id:``; ``id:`` is a sequence number. A client that reconnects with
``Last-Event-ID`` is sent what it missed from a bounded buffer, or, once the
buffer has moved past it, ``reset`` and then ``status``. A comment is sent
every ``KEEPALIVE_S`` so a quiet stream is not taken for a dead one. When
the server closes, the last event is ``closing`` and the stream ends.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import inspect
import json
import threading
from collections.abc import AsyncIterator, Callable

from modules import api_surface, wire_encoding
from modules.notification_center import Notification
from modules.run_events import RunEvents
from modules.scope_session import ScopeSession

# The events a client missed that a reconnect is resent: about a minute of
# a homing stage's position events, the busiest source.
BUFFER = 1024
# Seconds between comments on a quiet stream.
KEEPALIVE_S = 15.0
# The events the live view carries, not the stream: one per frame.
_LIVE_VIEW = frozenset({'frame', 'frame_captured'})
# The outcome ids already sent, kept so a muted outcome shown later is not sent twice.
_OUTCOMES_KEPT = 4096


@dataclasses.dataclass(frozen=True)
class _Event:
    seq: int
    name: str
    data: str


class EventStream:
    """The events the server has heard, numbered, and the clients reading them.

    Heard on whatever thread sends them; numbered and kept on the server's
    loop, so the order a client reads is the order they were heard.
    """

    def __init__(
        self,
        session: ScopeSession,
        encode: Callable[[object], object],
        handed_out: frozenset[type],
    ) -> None:
        """Make a stream that hears nothing until ``open``.

        Args:
            session: The session whose events it carries.
            encode: A value's wire form: the server's encoder.
            handed_out: The live objects' classes, which an event leaves out.
        """
        self._session = session
        self._encode = encode
        self._handed_out = handed_out
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._buffer: collections.deque[_Event] = collections.deque(maxlen=BUFFER)
        self._seq = 0
        # Set when an event is appended, then replaced: a reader waits on the one it saw.
        self._wake: asyncio.Event | None = None
        self._outcomes: collections.OrderedDict[int, None] = collections.OrderedDict()
        self._removals: list[Callable[[], None]] = []
        # The id of the ``closing`` event, once the server has finished the stream.
        self._closing_seq: int | None = None

    def open(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start hearing every listener member's events, on *loop*."""
        self._loop = loop
        self._wake = asyncio.Event()
        for owner in _owners(self._session):
            for member_name, record in _listeners(type(owner)):
                name = member_name.removeprefix('add_').removesuffix('_listener')
                if name in _LIVE_VIEW:
                    continue
                listener = self._listener(name, record)
                getattr(owner, member_name)(listener)
                remove = getattr(owner, f'remove_{name}_listener', None)
                if remove is not None:
                    self._removals.append(lambda remove=remove, listener=listener: remove(listener))

    def close(self) -> None:
        """Stop hearing: a listener still called sends nothing."""
        with self._lock:
            self._loop = None
        for remove in self._removals:
            remove()
        self._removals.clear()

    def finish(self) -> None:
        """Send ``closing`` to every reader, and end each read: the server is closing.

        Called on the server's loop. A read begun afterwards is sent
        ``closing`` and ends.
        """
        self._append('closing', '{}')
        self._closing_seq = self._seq

    def run_events(self) -> RunTag:
        """The handlers a run call's ``events`` is filled with, tagged once its run is known."""
        return RunTag(self)

    def _listener(self, name: str, record: type) -> Callable[..., None]:
        if record is Notification:

            def heard_outcome(notification: Notification) -> None:
                if notification.solicited or not self._first(notification.outcome_id):
                    return
                self.send(name, notification)

            return heard_outcome

        def heard(*args: object) -> None:
            self.send(name, self._record(record, args))

        return heard

    def _first(self, outcome_id: int) -> bool:
        with self._lock:
            if outcome_id in self._outcomes:
                return False
            self._outcomes[outcome_id] = None
            if len(self._outcomes) > _OUTCOMES_KEPT:
                self._outcomes.popitem(last=False)
            return True

    def _record(self, record: type, args: tuple[object, ...]) -> object:
        """The event's record, built from its callback's arguments as the API declares."""
        if len(args) == 1 and isinstance(args[0], record):
            return args[0]
        if not args:
            return _read_of(self._session, record)
        return record(*args)

    def send(self, name: str, record: object, **added: object) -> None:
        """Send *record* as event *name*, with *added* beside its fields.

        A field holding a live object is left out: a run executes a copy of
        its protocol, which no client holds, and ``run`` names the run.
        """
        fields = {
            f: getattr(record, f)
            for f in _fields(record)
            if type(getattr(record, f)) not in self._handed_out
        }
        data = json.dumps({**self._encode(fields), **added}, allow_nan=False)
        with self._lock:
            if self._loop is None:
                return
            self._loop.call_soon_threadsafe(self._append, name, data)

    def _append(self, name: str, data: str) -> None:
        self._seq += 1
        self._buffer.append(_Event(self._seq, name, data))
        self._wake.set()
        self._wake = asyncio.Event()

    async def read(self, last_event_id: str | None) -> AsyncIterator[str]:
        """The stream one client reads, from where *last_event_id* left it."""
        sent = _resume_from(last_event_id)
        if sent is None or not self._holds_after(sent):
            if self._closing_seq is not None:
                yield _frame('closing', '{}', self._closing_seq)
                return
            if sent is not None:
                yield _frame('reset', '{}', self._seq)
            sent = self._seq
            status = await asyncio.to_thread(lambda: self._session.status)
            yield _frame('status', json.dumps(self._encode(status), allow_nan=False), sent)
        while True:
            missed = [e for e in self._buffer if e.seq > sent]
            if missed and missed[0].seq > sent + 1:
                # The buffer moved past this client while it read slowly.
                if self._closing_seq is not None:
                    yield _frame('closing', '{}', self._closing_seq)
                    return
                yield _frame('reset', '{}', self._seq)
                sent = self._seq
                status = await asyncio.to_thread(lambda: self._session.status)
                yield _frame('status', json.dumps(self._encode(status), allow_nan=False), sent)
                continue
            for event in missed:
                yield _frame(event.name, event.data, event.seq)
                sent = event.seq
                if event.seq == self._closing_seq:
                    return
            if missed:
                continue
            try:
                await asyncio.wait_for(self._wake.wait(), KEEPALIVE_S)
            except TimeoutError:
                yield ': keepalive\n\n'

    def _holds_after(self, seq: int) -> bool:
        """Whether every event after *seq* is still buffered."""
        if seq > self._seq:
            return False
        return seq == self._seq or (bool(self._buffer) and self._buffer[0].seq <= seq + 1)


class RunTag:
    """A run call's event handlers: each event names the run once the call has handed it out.

    A run's first events can come before the call that started it returns,
    since the run is dispatched before its handle is handed back; they are
    held, in order, until ``settle`` names the run, then sent ahead of any
    later one.
    """

    def __init__(self, stream: EventStream) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self._held: list[tuple[str, object]] | None = []
        self._run: object = None

    def events(self) -> RunEvents:
        """The ``RunEvents`` that send each event but the live view's."""
        handlers = {}
        for field in dataclasses.fields(RunEvents):
            record = api_surface.field_event(RunEvents, field.name)
            if record is None or field.name in _LIVE_VIEW:
                continue
            handlers[field.name] = self._handler(field.name, record)
        return RunEvents(**handlers)

    def _handler(self, name: str, record: type) -> Callable[..., None]:
        def heard(*args: object) -> None:
            built = self._stream._record(record, args)
            with self._lock:
                if self._held is not None:
                    self._held.append((name, built))
                    return
                self._stream.send(name, built, run=self._run)

        return heard

    def settle(self, run: object) -> None:
        """Name the run, ``{"handle", "type"}``, or None when the call handed out none; send what was held."""
        with self._lock:
            self._run = run
            held, self._held = self._held or [], None
            for name, built in held:
                self._stream.send(name, built, run=run)


def _frame(name: str, data: str, seq: int) -> str:
    return f'id: {seq}\nevent: {name}\ndata: {data}\n\n'


def _resume_from(last_event_id: str | None) -> int | None:
    if last_event_id is None:
        return None
    text = last_event_id.strip()
    return int(text) if text.isdigit() else -1


def _fields(record: object) -> tuple[str, ...]:
    published = api_surface.fields_of(type(record))
    if published:
        return published
    return tuple(f.name for f in dataclasses.fields(record))


def _owners(session: ScopeSession) -> list[object]:
    """The session and each object its marked reads lead to, the API's sub-objects."""
    found: list[object] = []
    pending: list[object] = [session]
    while pending:
        owner = pending.pop()
        if owner is None or any(owner is seen for seen in found):
            continue
        found.append(owner)
        for name in _sub_object_reads(type(owner)):
            pending.append(getattr(owner, name))
    return found


def _sub_object_reads(cls: type) -> list[str]:
    classes = wire_encoding.project_classes()
    aliases = wire_encoding.project_aliases()
    return [m.name for m in wire_encoding.wire_members(cls, classes, aliases) if m.segment]


def _listeners(cls: type) -> list[tuple[str, type]]:
    """Each listener member of *cls* that declares its event's record."""
    found = []
    for name, member in inspect.getmembers(cls):
        record = api_surface.event_of(member)
        if record is not None:
            found.append((name, record))
    return found


def _read_of(session: ScopeSession, record: type) -> object:
    """The session's read whose answer is *record*: the data of an event that carries no arguments."""
    for name, member in inspect.getmembers(type(session)):
        if isinstance(member, property) and api_surface.mark_of(member) is not None:
            returns = member.fget.__annotations__.get('return')
            if returns is record or returns == record.__name__:
                return getattr(session, name)
    raise TypeError(f'No session read answers {record.__name__}')
