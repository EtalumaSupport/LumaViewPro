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
caller (``modules.api_arguments``), with ``ArgumentTypeRefusedError``
(``'wrong_argument_type'``). That refusal is every member's, so no
``Raises:`` section lists it; nor do the hardware lanes' own refusals of a
command on them, which are every hardware command's: ``HardwareCommandRefusedError``
``'exclusive_activity_running'`` while a run, a diagnostic or a home holds
the scope and the call is not made under it, ``'control_session_held'``
while a control session is held and the call is not its holder's, and
``'scope_disconnected'`` once the lanes are shut. Nor does the control
session's refusal of a write: every member that writes a setting, a
protocol file or a recording's pending frames is a guarded write
(``activity_claim.guarded_write``), refused with ``ControlSessionRefusedError``
``'control_session_held'`` while a control session is held and the call is
not its holder's. The display preferences (``settings_paths.PREFERENCES``)
are written by anyone; a support report and a logs zip are made by anyone,
since each writes a support artefact and changes no scope state; and a
post-processing build is open to anyone.

Four marks say what a member does, each read by its own reader below:
``reads`` that it changes nothing, so a host may answer it as a read;
``sets`` the read, on its own class, whose value a setter writes -- one
setting, read by that read and written by this member, so a host serves
the two as one resource; ``stop`` the member, ``'Class.member'``, that
stops what it starts -- its own class's, or the handle's it returns;
``engineering`` that it is a bench or support member, outside what a
researcher's client needs. What a
member may raise is its docstring's ``Raises:`` section, the one statement
of its contract: each class it raises as an entry, ``ClassName: words``,
or the one line ``Nothing.`` for a member that raises nothing a caller
handles. Every marked member has the section
(``tests/guards/test_every_member_states_its_contract.py``).

A record's field typed with a closed set (a ``StrEnum``, or one or None)
is judged as the record is built, by whoever builds it -- the API, a
Python caller handing a record back, the wire decoding one: a member's
plain string is stored as the member, and any other value is
``ArgumentRefusedError`` ``'not_offered'``, naming ``Record.field``
(``modules.closed_sets``). The field is found from the record's module
when the record is marked, so its set is imported or defined above it;
``tests/guards/test_a_records_closed_set_is_judged.py`` holds every such
field judged.

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
import enum
import functools
import inspect
import os
import re
import sys
import types
import typing
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
# The attribute a record's closed-set fields are recorded under, by name:
# the set, and whether None is one of the field's values.
CLOSED_FIELDS_ATTRIBUTE = '_lvp_api_closed_fields'
# The attributes the four marks are recorded under, each its own: a new
# value of MARK_ATTRIBUTE would be read as unmarked by every host that
# routes on it, and the member's route would go.
READS_ATTRIBUTE = '_lvp_api_reads'
SETS_ATTRIBUTE = '_lvp_api_sets'
STOP_ATTRIBUTE = '_lvp_api_stop'
ENGINEERING_ATTRIBUTE = '_lvp_api_engineering'

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
    """``member`` with its function behind the argument door, as the same kind of descriptor.

    A function defined in a class body is a method, its first parameter
    ``self``; one defined at a module's top level (``rest.app.build_app``)
    takes every parameter as an argument.
    """
    if isinstance(member, property):
        return member
    if isinstance(member, (classmethod, staticmethod)):
        function = _with_a_door(member.__func__, bound=isinstance(member, classmethod))
        return member if function is member.__func__ else type(member)(function)
    return _with_a_door(member, bound='.' in member.__qualname__)


def api[M](
    member: M | None = None,
    *,
    in_process: bool = False,
    event: type | None = None,
    reads: bool = False,
    sets: str | None = None,
    stop: str | None = None,
    engineering: bool = False,
) -> M | Callable[[M], M]:
    """Mark a member as API; used bare (``@api``) or as ``@api(in_process=True)``.

    ``event`` is the record of the event a listener member delivers.
    ``reads``, ``sets``, ``stop`` and ``engineering`` are the marks the
    module docstring describes.
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
        if reads:
            setattr(function, READS_ATTRIBUTE, True)
        if sets is not None:
            setattr(function, SETS_ATTRIBUTE, sets)
        if stop is not None:
            setattr(function, STOP_ATTRIBUTE, stop)
        if engineering:
            setattr(function, ENGINEERING_ATTRIBUTE, True)
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

    A record gains ``to_dict()``, its data as plain Python
    (``wire_encoding.plain``), for a caller that needs a dict: MATLAB's
    ``struct()`` takes a dict and not a dataclass. So it is above the
    ``@dataclass`` it marks, which must have run first.

    A dataclass's closed-set fields are judged as it is built (the module
    docstring), before its own ``__post_init__`` reads them.
    """

    def publish(cls: C) -> C:
        setattr(cls, FIELDS_ATTRIBUTE, tuple(names))
        if record:
            setattr(cls, RECORD_ATTRIBUTE, True)
        if is_record(cls):
            cls.to_dict = _to_dict
        if dataclasses.is_dataclass(cls):
            _judge_closed_fields(cls)
        return cls

    return publish


def _closed_set_of(annotation: object, names: dict[str, object]) -> tuple[type, bool] | None:
    """The closed set an annotation names, and whether it admits None; None when it names none.

    Read as written: a set, or a set or None. A string annotation's names
    are looked up in ``names``, the record's module.
    """
    if isinstance(annotation, str):
        parts = [part.strip() for part in annotation.split('|')]
        arms = [names.get(part, part) for part in parts if part != 'None']
        optional = len(arms) < len(parts)
    elif typing.get_origin(annotation) in (types.UnionType, typing.Union):
        arms = [arm for arm in typing.get_args(annotation) if arm is not type(None)]
        optional = len(arms) < len(typing.get_args(annotation))
    else:
        arms, optional = [annotation], False
    if len(arms) == 1 and isinstance(arms[0], type) and issubclass(arms[0], enum.StrEnum):
        return arms[0], optional
    return None


def _judge_closed_fields(cls: type) -> None:
    """Judge each of a dataclass's closed-set fields as it is built, before its ``__post_init__``."""
    names = vars(sys.modules[cls.__module__])
    closed = {}
    for field in dataclasses.fields(cls):
        found = _closed_set_of(field.type, names)
        if found is not None:
            closed[field.name] = found
    setattr(cls, CLOSED_FIELDS_ATTRIBUTE, closed)
    if not closed:
        return

    def judge(self: object) -> None:
        # Imported here: closed_sets raises a refusal, and the refusals'
        # module imports this one.
        from modules.closed_sets import refuse_unless_offered

        for name, (closed_set, optional) in closed.items():
            value = getattr(self, name)
            if value is None and optional:
                continue
            member = refuse_unless_offered(closed_set, value, f'{cls.__name__}.{name}')
            if member is not value:
                # A frozen record refuses setattr; the build is not done yet.
                object.__setattr__(self, name, member)

    # The generated __init__ calls __post_init__ when the class has one, so
    # the judge goes first inside it; otherwise it follows __init__.
    post_init = getattr(cls, '__post_init__', None)
    if post_init is not None:

        @functools.wraps(post_init)
        def judged_post_init(self, *args, **kwargs):
            judge(self)
            return post_init(self, *args, **kwargs)

        cls.__post_init__ = judged_post_init
    else:
        init = cls.__init__

        @functools.wraps(init)
        def judged_init(self, *args, **kwargs):
            init(self, *args, **kwargs)
            judge(self)

        cls.__init__ = judged_init


def _to_dict(self: object) -> dict:
    """This record's fields as a dict of plain Python data: a nested record a dict, an enum its string, a sequence a list.

    A path, a handle and a running call stay the live objects they are, a
    NaN stays a NaN, and a key keeps its type (``wire_encoding.plain``).
    """
    # Imported here: wire_encoding reads this module's marks.
    from modules.wire_encoding import plain

    return plain(self)


# A ``Raises:`` section, its entries (``ClassName: words``, continued
# deeper), and a reason an entry quotes (``'not_connected'``, or after the
# word: reason ``no_data``).
RAISES_SECTION = re.compile(r'^Raises:\n((?:[ \t]+.*\n?|\n)+)', re.M)
RAISES_ENTRY = re.compile(r'^ {4}(\w+):(.*(?:\n {8,}.*)*)', re.M)
QUOTED_REASON = re.compile(r"``'([a-z][a-z0-9_]*)'``|reason\s+``([a-z][a-z0-9_]*)``")


def raises_of(doc: str) -> list[tuple[str, tuple[str, ...]]]:
    """Each entry of *doc*'s ``Raises:`` section: the class it names, and the reasons it quotes.

    *doc* is a member's docstring as ``inspect.getdoc`` gives it. A section
    reading ``Nothing.``, like a missing one, declares nothing.
    """
    found = RAISES_SECTION.search(doc)
    if found is None:
        return []
    return [
        (name, tuple(dict.fromkeys(a or b for a, b in QUOTED_REASON.findall(words))))
        for name, words in RAISES_ENTRY.findall(found.group(1))
    ]


def mark_of(member: object) -> str | None:
    """``API``, ``IN_PROCESS``, or None for an unmarked member."""
    return getattr(_function_of(member), MARK_ATTRIBUTE, None)


def fields_of(cls: type) -> tuple[str, ...]:
    """The data attributes *cls* itself publishes."""
    return vars(cls).get(FIELDS_ATTRIBUTE, ())


def reads_of(member: object) -> bool:
    """Whether *member* is marked as changing nothing."""
    return getattr(_function_of(member), READS_ATTRIBUTE, False)


def sets_of(member: object) -> str | None:
    """The name of the read whose setting *member* writes, or None."""
    return getattr(_function_of(member), SETS_ATTRIBUTE, None)


def stop_of(member: object) -> str | None:
    """The ``'Class.member'`` that stops what *member* starts, or None."""
    return getattr(_function_of(member), STOP_ATTRIBUTE, None)


def is_engineering(member: object) -> bool:
    """Whether *member* is marked as a bench or support member."""
    return getattr(_function_of(member), ENGINEERING_ATTRIBUTE, False)


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


def closed_fields_of(cls: type) -> dict[str, tuple[type, bool]]:
    """Each closed-set field a dataclass record judges as it is built: its set, and whether it admits None."""
    return vars(cls).get(CLOSED_FIELDS_ATTRIBUTE, {})


def is_record(cls: type) -> bool:
    """Whether *cls* crosses a wire as its data: a dataclass, a NamedTuple, or declared one."""
    return (
        dataclasses.is_dataclass(cls)
        or (issubclass(cls, tuple) and hasattr(cls, '_fields'))
        or vars(cls).get(RECORD_ATTRIBUTE, False)
    )
