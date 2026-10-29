# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What the API is: the members and fields marked here, and nothing else.

A member is API when it carries ``@api``; a class's published data
attributes are the names its ``@api_fields`` lists. Whatever is unmarked
stays callable from Python but is not documented as API and is never put
on a wire, whatever its name. ``docs/API_REFERENCE.md`` is this set,
rendered from the marks and the docstrings (``entries_of``,
``markdown_of``), and ``tests/guards/test_the_api_is_declared.py`` holds
it current and the set closed: every type a marked member hands out or
takes in is itself published.

``@api(in_process=True)`` marks a member that is kept off the wire --
hosting a session, returning a function, or held back by policy -- so a
wire client skips it while an in-process client (Python, MATLAB calling
Python) uses it like any other. ``@api(in_process_arguments=(...))`` keeps
named parameters of a wire member off the wire the same way: a wire
client cannot send them, so each one's default stands for every wire call,
while an in-process caller passes them as it likes. Each is a parameter
with a default, refused at decoration otherwise.

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
the scope and the call is not made under it, and ``'scope_disconnected'``
once the lanes are shut. Nor does the control session's refusal,
``ControlSessionRefusedError`` ``'control_session_held'``, of a call that
is not its holder's while a control session is held: of every command on a
hardware lane, and of every member that writes a setting, a protocol file
or a recording's pending frames, each a guarded write
(``activity_claim.guarded_write``). The display preferences (``settings_paths.PREFERENCES``)
are written by anyone; a support report and a logs zip are made by anyone,
since each writes a support artefact and changes no scope state; and a
post-processing build is open to anyone. A member that enters a hardware
lane or a guarded write says so in its mark, ``@api(admitted_by=(...))``,
each an ``Admission``: the door whose own ``Raises:`` states those
refusals, which a host that describes the member -- the REST server's
route -- adds to the member's own
(``tests/guards/test_every_route_names_the_doors_it_enters.py``).

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
(``modules.closed_sets``). So are the items of a field that is a tuple,
list or frozenset of one, and the keys of a field that is a mapping
keyed by one, each named by its place (``Record.field[0]``). The field is found from the record's module
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
from collections.abc import Callable, Mapping

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
# The attribute a record's closed-set fields are recorded under, by name,
# each as its ``ClosedField``.
CLOSED_FIELDS_ATTRIBUTE = '_lvp_api_closed_fields'
# The attributes the four marks are recorded under, each its own: a new
# value of MARK_ATTRIBUTE would be read as unmarked by every host that
# routes on it, and the member's route would go.
READS_ATTRIBUTE = '_lvp_api_reads'
SETS_ATTRIBUTE = '_lvp_api_sets'
STOP_ATTRIBUTE = '_lvp_api_stop'
ENGINEERING_ATTRIBUTE = '_lvp_api_engineering'
# The attribute a member's off-wire parameters are recorded under, by name.
IN_PROCESS_ARGUMENTS_ATTRIBUTE = '_lvp_api_in_process_arguments'
# The attribute the doors a member enters are recorded under.
ADMITTED_BY_ATTRIBUTE = '_lvp_api_admitted_by'

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


class Admission(enum.StrEnum):
    """A door a member's call enters, whose refusals are the door's own ``Raises:``.

    The lanes named are those that ask the scope's activity claim, the
    IO and CAMERA lanes; a lane that asks none refuses nothing of the claim
    or the control session, and a member reaching only such a lane names
    no door.
    """

    # ``SequentialIOExecutor.call``: the blocking dispatch.
    LANE_CALL = 'lane_call'
    # ``SequentialIOExecutor.submit``: put on the lane without waiting.
    LANE_SUBMIT = 'lane_submit'
    # ``ActivityClaim.enter_guarded_write``: a setting or a file written.
    GUARDED_WRITE = 'guarded_write'


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
    or inside another function takes every parameter as an argument.
    """
    if isinstance(member, property):
        return member
    if isinstance(member, (classmethod, staticmethod)):
        function = _with_a_door(member.__func__, bound=isinstance(member, classmethod))
        return member if function is member.__func__ else type(member)(function)
    # The qualified name's last scope is the class body that defines a
    # method; a nested function's is ``<locals>``.
    scope = member.__qualname__.rpartition('.')[0]
    return _with_a_door(member, bound=bool(scope) and not scope.endswith('<locals>'))


def api[M](
    member: M | None = None,
    *,
    in_process: bool = False,
    event: type | None = None,
    reads: bool = False,
    sets: str | None = None,
    stop: str | None = None,
    engineering: bool = False,
    in_process_arguments: tuple[str, ...] = (),
    admitted_by: tuple[Admission, ...] = (),
) -> M | Callable[[M], M]:
    """Mark a member as API; used bare (``@api``) or as ``@api(in_process=True)``.

    ``event`` is the record of the event a listener member delivers.
    ``reads``, ``sets``, ``stop``, ``engineering``,
    ``in_process_arguments`` and ``admitted_by`` are the marks the module
    docstring describes.
    Goes outermost in a decorator stack. A member that takes an argument is
    returned behind its argument door, which refuses a value of another
    type than its annotation declares (``modules.api_arguments``); one that
    takes none is returned unchanged.

    Raises:
        TypeError: a name in ``in_process_arguments`` is not a parameter of
            the member, or is one with no default, which a wire call could
            then never fill.
    """

    def mark(target: M) -> M:
        if in_process_arguments:
            parameters = inspect.signature(_function_of(target)).parameters
            for name in in_process_arguments:
                parameter = parameters.get(name)
                if parameter is None or parameter.default is inspect.Parameter.empty:
                    raise TypeError(
                        f'{_function_of(target).__qualname__}: in-process argument {name!r} '
                        'is not a parameter with a default'
                    )
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
        if in_process_arguments:
            setattr(function, IN_PROCESS_ARGUMENTS_ATTRIBUTE, tuple(in_process_arguments))
        if admitted_by:
            setattr(function, ADMITTED_BY_ATTRIBUTE, tuple(Admission(a) for a in admitted_by))
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


class ClosedField(typing.NamedTuple):
    """Where a record's field holds a closed set's members."""

    closed_set: type[enum.StrEnum]
    # Whether None is one of the field's values.
    optional: bool
    # None: the field is a member; 'items': a tuple, list or frozenset of
    # members; 'keys': a mapping keyed by members.
    holds: str | None = None


def _closed_field_of(annotation: object, names: dict[str, object]) -> ClosedField | None:
    """How an annotation holds a closed set; None when it holds none.

    A string annotation is read in ``names``, the record's module, as far as
    the module has run: a set it cannot find there (imported only for type
    checking, or defined below the record) is no set here, and the guard
    names the field.
    """
    if isinstance(annotation, str):
        try:
            # An annotation is an expression; this reads it as typing does.
            annotation = eval(annotation, dict(names))
        except NameError:
            return None
    arms = (
        typing.get_args(annotation)
        if typing.get_origin(annotation) in (types.UnionType, typing.Union)
        else (annotation,)
    )
    present = [arm for arm in arms if arm is not type(None)]
    if len(present) != 1:
        return None
    optional = len(present) < len(arms)
    (arm,) = present
    if _is_closed_set(arm):
        return ClosedField(arm, optional)
    origin, args = typing.get_origin(arm), typing.get_args(arm)
    if not args or not _is_closed_set(args[0]):
        return None
    if origin in (list, frozenset) or (origin is tuple and args[1:] in ((), (Ellipsis,))):
        return ClosedField(args[0], optional, 'items')
    if origin in (dict, Mapping):
        return ClosedField(args[0], optional, 'keys')
    return None


def _is_closed_set(annotation: object) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, enum.StrEnum)


def _judge_closed_fields(cls: type) -> None:
    """Judge each of a dataclass's closed-set fields as it is built, before its ``__post_init__``."""
    names = vars(sys.modules[cls.__module__])
    closed = {}
    for field in dataclasses.fields(cls):
        found = _closed_field_of(field.type, names)
        if found is not None:
            closed[field.name] = found
    setattr(cls, CLOSED_FIELDS_ATTRIBUTE, closed)
    if not closed:
        return

    def judge(self: object) -> None:
        # Imported here: closed_sets raises a refusal, and the refusals'
        # module imports this one.
        from modules.closed_sets import refuse_unless_offered

        for name, (closed_set, optional, holds) in closed.items():
            value = getattr(self, name)
            if value is None and optional:
                continue
            where = f'{cls.__name__}.{name}'
            if holds is None:
                judged = refuse_unless_offered(closed_set, value, where)
            elif holds == 'items':
                judged = type(value)(
                    refuse_unless_offered(closed_set, item, f'{where}[{index}]')
                    for index, item in enumerate(value)
                )
            else:
                judged = {
                    refuse_unless_offered(closed_set, key, f'{where}[{key!r}]'): item
                    for key, item in value.items()
                }
            if judged is not value:
                # A frozen record refuses setattr; the build is not done yet.
                object.__setattr__(self, name, judged)

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


# A docstring's sections, each a heading line read as its own: what a
# reader of the API is told under it, entry by entry or as prose.
SECTIONS = ('Args', 'Attributes', 'Raises', 'Returns', 'Yields')
# The sections whose body is entries, ``name: words``, continued deeper.
ENTRY_SECTIONS = frozenset({'Args', 'Attributes', 'Raises'})
_SECTION_HEADING = re.compile(rf'^({"|".join(SECTIONS)}):\s*$')
_ENTRY = re.compile(r'^([\w*.]+):\s*(.*)$')
_CODE = re.compile(r'``(.+?)``')


def _section_lines(doc: str) -> list[tuple[str | None, list[str]]]:
    """*doc* split at its section headings: each part's heading (None for prose) and its lines, a section's dedented.

    A section's body is its indented lines and the blank lines between
    them; the first unindented line after it is prose again.
    """
    parts: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in doc.splitlines():
        heading = _SECTION_HEADING.match(line)
        if heading:
            parts.append((heading.group(1), []))
            continue
        if parts[-1][0] is not None and line.strip() and not line[:1].isspace():
            parts.append((None, []))
        parts[-1][1].append(line)
    return [(h, lines if h is None else _dedent(lines)) for h, lines in parts]


def _dedent(lines: list[str]) -> list[str]:
    widths = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    cut = min(widths, default=0)
    return [line[cut:] for line in lines]


def _entries(lines: list[str]) -> dict[str, str]:
    """A section body's entries, by name, each one's words joined onto one line."""
    found: dict[str, list[str]] = {}
    name = None
    for line in lines:
        entry = None if line[:1].isspace() else _ENTRY.match(line)
        if entry:
            name = entry.group(1)
            found[name] = [entry.group(2)]
        elif name is not None and line.strip():
            found[name].append(line.strip())
    return {n: ' '.join(w for w in words if w) for n, words in found.items()}


def entries_of(doc: str, section: str) -> dict[str, str]:
    """Each entry of *doc*'s *section* (``Args``, ``Attributes``, ``Raises``), by name, its words on one line.

    *doc* is a docstring as ``inspect.getdoc`` gives it. A missing section
    has no entries.
    """
    found: dict[str, str] = {}
    for heading, lines in _section_lines(doc):
        if heading == section:
            found |= _entries(lines)
    return found


def own_doc(cls: type) -> str:
    """The class's own docstring, never a base's nor the one ``dataclass`` writes from the signature."""
    doc = vars(cls).get('__doc__') or ''
    if doc.startswith(f'{cls.__name__}('):
        return ''
    return inspect.cleandoc(doc)


def field_descriptions(cls: type) -> dict[str, str]:
    """Each field's words, by name: its class's ``Attributes:`` entry, a subclass's over a base's.

    A field has no docstring of its own, so these are its words wherever
    it is described: the reference and the OpenAPI description alike.
    """
    described: dict[str, str] = {}
    for klass in reversed(cls.__mro__):
        described |= entries_of(own_doc(klass), 'Attributes')
    return described


def _inline(text: str) -> str:
    """Docstring prose as Markdown: a literal is a code span, and a ``<`` outside one is no tag."""
    return ''.join(
        f'`{piece}`' if index % 2 else piece.replace('<', '&lt;')
        for index, piece in enumerate(_CODE.split(text))
    )


def markdown_of(doc: str, *, leave_out: tuple[str, ...] = ()) -> str:
    """*doc* as Markdown: its prose as written, each section under a bold heading, an entry section as a list.

    *doc* is a docstring as ``inspect.getdoc`` gives it. A section named in
    *leave_out* is not rendered; an entry section with no entry
    (``Nothing.``) is rendered as its prose.
    """
    out: list[str] = []
    for heading, lines in _section_lines(doc):
        if heading in leave_out:
            continue
        entries = _entries(lines) if heading in ENTRY_SECTIONS else {}
        if heading is None:
            text = '\n'.join(_inline(line) for line in lines).strip('\n')
        elif entries:
            items = '\n'.join(f'- `{name}`: {_inline(words)}' for name, words in entries.items())
            text = f'**{heading}**\n\n{items}'
        else:
            body = ' '.join(line.strip() for line in lines if line.strip())
            text = f'**{heading}**\n\n{_inline(body)}'
        if text:
            out.append(text)
    return '\n\n'.join(out)


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


def in_process_arguments_of(member: object) -> tuple[str, ...]:
    """The names of *member*'s parameters a wire client does not send."""
    return getattr(_function_of(member), IN_PROCESS_ARGUMENTS_ATTRIBUTE, ())


def admitted_by_of(member: object) -> tuple[Admission, ...]:
    """The doors *member*'s mark says it enters."""
    return getattr(_function_of(member), ADMITTED_BY_ATTRIBUTE, ())


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


def closed_fields_of(cls: type) -> dict[str, ClosedField]:
    """Each closed-set field a dataclass record judges as it is built, and how it holds its set."""
    return vars(cls).get(CLOSED_FIELDS_ATTRIBUTE, {})


def is_record(cls: type) -> bool:
    """Whether *cls* crosses a wire as its data: a dataclass, a NamedTuple, or declared one."""
    return (
        dataclasses.is_dataclass(cls)
        or (issubclass(cls, tuple) and hasattr(cls, '_fields'))
        or vars(cls).get(RECORD_ATTRIBUTE, False)
    )
