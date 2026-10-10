# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What the API is: the members and fields marked here, and nothing else.

A member is API when it carries ``@api``; a class's published data
attributes are the names its ``@api_fields`` lists. Whatever is unmarked
stays callable from Python but is not documented as API and is never put
on a wire, whatever its name. ``docs/LumascopeSkills.md`` indexes exactly
this set, and ``tests/guards/test_the_api_is_declared.py`` holds the two
equal and the set closed: every type a marked member hands out or takes
in is itself published.

``@api(in_process=True)`` marks a member that is kept off the wire --
hosting a session, returning a function, or held back by policy -- so a
wire client skips it while an in-process client (Python, MATLAB calling
Python) uses it like any other.

A parameter that takes a file-system path is annotated ``FilePath``, so a
wire client can tell a path from a string. One that takes a progress
callback is annotated ``ProgressCallback``, so a host that carries the
call elsewhere -- the REST server's jobs -- passes its own and reports
how far the call has got.

A marked member's annotations are its contract at the call as well: an
argument of another type is refused before the member runs, for every
caller (``modules.api_arguments``).

An event declares its record where it is delivered: a listener member as
``@api(event=Record)``, a ``RunEvents`` field through
``event_metadata(Record)``.
The record is published, its fields name the callback's arguments in
order, and a host that carries the event elsewhere -- the REST server's
stream -- builds the record from the arguments, so no second table of
each signature exists. A callback with one argument that is itself a
record declares that record; a callback with no arguments declares the
record its host reads when the event is sent.

No project imports: every module that declares API imports this one.
"""

import dataclasses
import functools
import inspect
import os
from collections.abc import Callable

# The attribute the mark is recorded under, on the function itself: a
# property's getter, a classmethod's or staticmethod's function, or the
# function a decorator such as contextmanager returned.
MARK_ATTRIBUTE = '_lvp_api'
# The attribute a class's published data attributes are recorded under,
# read from the class's own namespace so a subclass publishes only what it
# names itself.
FIELDS_ATTRIBUTE = '_lvp_api_fields'
# The attribute, and the dataclass field metadata key, an event's record is
# declared under.
EVENT_ATTRIBUTE = '_lvp_api_event'
# The attribute a class that is not a dataclass or NamedTuple declares
# itself a record under.
RECORD_ATTRIBUTE = '_lvp_api_record'

API = 'api'
IN_PROCESS = 'in_process'

# A ``type`` statement, not an assignment: an assignment's alias evaluates to
# its union and loses the name a wire client and the guard read.
type FilePath = str | os.PathLike[str]
# How a long call tells its caller how far it has got: the percentage done,
# 0 to 100, and a status line when it has one to say. A callable rather than
# a surface to write to, so whoever asked -- a widget, a script, a remote
# caller -- renders it in its own way and on its own thread.
type ProgressCallback = Callable[[float, str | None], None]


def _function_of(member: object) -> object:
    """The function a mark is recorded on.

    A property refuses new attributes, and a classmethod or staticmethod
    wraps the function that callers reach, so each is marked through the
    function it holds.
    """
    if isinstance(member, property):
        return member.fget
    if isinstance(member, (classmethod, staticmethod)):
        return member.__func__
    return member


def _with_a_door(function: Callable, *, bound: bool) -> Callable:
    """``function`` behind its argument door, or ``function`` itself when it takes no argument.

    ``bound`` says the first parameter is ``self`` or ``cls``. The door is
    built at the first call (``modules.api_arguments``): its annotations
    resolve only once every class they name is loaded.
    """
    if len(inspect.signature(function).parameters) <= (1 if bound else 0):
        return function
    door = None

    @functools.wraps(function)
    def guarded(*args, **kwargs):
        nonlocal door
        if door is None:
            from modules.api_arguments import door_for

            door = door_for(function, bound=bound)
        args, kwargs = door(args, kwargs)
        return function(*args, **kwargs)

    return guarded


def _guarded(member: object) -> object:
    """``member`` with its function behind the argument door, as the same kind of descriptor."""
    if isinstance(member, property):
        return member
    if isinstance(member, (classmethod, staticmethod)):
        function = _with_a_door(member.__func__, bound=isinstance(member, classmethod))
        return member if function is member.__func__ else type(member)(function)
    return _with_a_door(member, bound=True)


def api[M](
    member: M | None = None, *, in_process: bool = False, event: type | None = None
) -> M | Callable[[M], M]:
    """Mark a member as API; used bare (``@api``) or as ``@api(in_process=True)``.

    ``event`` is the record of the event a listener member delivers.
    Goes outermost in a decorator stack. A member that takes an argument is
    returned behind its argument door, which refuses a value of another
    type than its annotation declares (``modules.api_arguments``); one that
    takes none is returned unchanged.
    """

    def mark(target: M) -> M:
        target = _guarded(target)
        function = _function_of(target)
        setattr(function, MARK_ATTRIBUTE, IN_PROCESS if in_process else API)
        if event is not None:
            setattr(function, EVENT_ATTRIBUTE, event)
        return target

    if member is None:
        return mark
    return mark(member)


def api_fields[C: type](*names: str, record: bool = False) -> Callable[[C], C]:
    """Name a class's published data attributes, one by one.

    Instance attributes, class constants, and dataclass or NamedTuple
    fields alike. A field added to the class later is published only by
    adding its name here.

    A dataclass or NamedTuple is a record: it crosses a wire as its data,
    and has no address there. Any other class that does is declared with
    ``record=True``; one that is not crosses as a handle to a live object.
    """

    def publish(cls: C) -> C:
        setattr(cls, FIELDS_ATTRIBUTE, tuple(names))
        if record:
            setattr(cls, RECORD_ATTRIBUTE, True)
        return cls

    return publish


def mark_of(member: object) -> str | None:
    """``API``, ``IN_PROCESS``, or None for an unmarked member."""
    return getattr(_function_of(member), MARK_ATTRIBUTE, None)


def fields_of(cls: type) -> tuple[str, ...]:
    """The data attributes *cls* itself publishes."""
    return vars(cls).get(FIELDS_ATTRIBUTE, ())


def event_metadata(record: type) -> dict[str, type]:
    """The field metadata declaring that a ``RunEvents`` handler field delivers ``record``'s event.

    Given as ``dataclasses.field(default=None, metadata=event_metadata(Record))``.
    """
    return {EVENT_ATTRIBUTE: record}


def event_of(member: object) -> type | None:
    """The record a listener member declares, or None."""
    return getattr(_function_of(member), EVENT_ATTRIBUTE, None)


def field_event(cls: type, name: str) -> type | None:
    """The record a dataclass's handler field declares, or None."""
    for field in dataclasses.fields(cls):
        if field.name == name:
            return field.metadata.get(EVENT_ATTRIBUTE)
    return None


def is_record(cls: type) -> bool:
    """Whether *cls* crosses a wire as its data: a dataclass, a NamedTuple, or declared one."""
    return (
        dataclasses.is_dataclass(cls)
        or (issubclass(cls, tuple) and hasattr(cls, '_fields'))
        or vars(cls).get(RECORD_ATTRIBUTE, False)
    )
