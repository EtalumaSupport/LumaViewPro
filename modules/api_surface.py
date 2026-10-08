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

No project imports: every module that declares API imports this one.
"""

from collections.abc import Callable

# The attribute the mark is recorded under, on the function itself: a
# property's getter, a classmethod's or staticmethod's function, or the
# function a decorator such as contextmanager returned.
MARK_ATTRIBUTE = '_lvp_api'
# The attribute a class's published data attributes are recorded under,
# read from the class's own namespace so a subclass publishes only what it
# names itself.
FIELDS_ATTRIBUTE = '_lvp_api_fields'

API = 'api'
IN_PROCESS = 'in_process'


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


def api[M](member: M | None = None, *, in_process: bool = False) -> M | Callable[[M], M]:
    """Mark a member as API; used bare (``@api``) or as ``@api(in_process=True)``.

    Goes outermost in a decorator stack and returns the member unchanged.
    """

    def mark(target: M) -> M:
        setattr(_function_of(target), MARK_ATTRIBUTE, IN_PROCESS if in_process else API)
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
