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
wire client can tell a path from a string.

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

API = 'api'
IN_PROCESS = 'in_process'

# A ``type`` statement, not an assignment: an assignment's alias evaluates to
# its union and loses the name a wire client and the guard read.
type FilePath = str | os.PathLike[str]


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


def api[M](
    member: M | None = None, *, in_process: bool = False, event: type | None = None
) -> M | Callable[[M], M]:
    """Mark a member as API; used bare (``@api``) or as ``@api(in_process=True)``.

    ``event`` is the record of the event a listener member delivers.
    Goes outermost in a decorator stack and returns the member unchanged.
    """

    def mark(target: M) -> M:
        function = _function_of(target)
        setattr(function, MARK_ATTRIBUTE, IN_PROCESS if in_process else API)
        if event is not None:
            setattr(function, EVENT_ATTRIBUTE, event)
        return target

    if member is None:
        return mark
    return mark(member)


def api_fields[C: type](*names: str) -> Callable[[C], C]:
    """Name a class's published data attributes, one by one.

    Instance attributes, class constants, and dataclass or NamedTuple
    fields alike. A field added to the class later is published only by
    adding its name here.
    """

    def publish(cls: C) -> C:
        setattr(cls, FIELDS_ATTRIBUTE, tuple(names))
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
