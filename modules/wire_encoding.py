# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""How each value an API member hands out or takes in crosses a wire.

The one declaration of the wire forms, read two ways: ``wire_gaps`` walks
the API from the Session, edge by edge, and names every edge whose type
has no form, so a member that would hand a wire client something it
cannot carry fails the build (``tests/guards/test_every_wire_edge_has_a_form``);
``encode`` turns a value into JSON-ready data for a host that carries it,
the REST server.

The forms:

- ``str``, ``int``, ``float``, ``bool``, None: themselves.
- ``timedelta``: seconds. ``datetime``: ISO 8601.
- A ``StrEnum``: its value; any other enum: its name.
- A record (``api_surface.is_record``): an object by its published field
  names, its wire-marked properties among them.
- A path, out: ``{"name": <live-folder name> | null, "host_path": <path>}``,
  ``name`` set when the path is inside the live folder, both compared
  resolved, so a path stored in either form (``/var`` or ``/private/var``)
  is named. A path, in: a live-folder name, which the host resolves
  through ``ScopeSession.live_folder_path``.
- A tuple, list, set, frozenset or iterable: an array, a set sorted.
- A dict, mapping or ``pandas.Series``: an object, keys as strings (an
  enum key in its own form).
- A ``Future``: a job, from the host's job registry.
- Any other object with wire members: a handle, from the host's registry.
- A callable parameter (an ``on_progress``, a run's ``RunEvents``): not
  on the wire; the host passes its own.

No form: an image array, a table (``DataFrame``), a callable anywhere but
a parameter, ``Any`` or ``object``, a thread. A member that hands one out
is marked in-process.

Inbound, a parameter's value comes back the same way: seconds become a
``timedelta``, an enum's form its member, an object a record, a handle's
id its live object, a live-folder name an absolute path. ``wire_members``
describes each member a host routes to, and ``decode`` turns what a
client sent into what the member takes. A parameter type with no inbound
form -- a job, a path inside an array, a union a value could be read as
either side of -- is refused when the host is built, never guessed at a
call.
"""

from __future__ import annotations

import ast
import dataclasses
import datetime
import enum
import importlib
import inspect
import pathlib
import pkgutil
import typing
import warnings
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future

from modules.api_surface import API, fields_of, is_record, mark_of

# The forms a value takes on the wire.
SCALAR = 'scalar'
SECONDS = 'seconds'
ISO8601 = 'iso8601'
ENUM = 'enum'
RECORD = 'record'
PATH = 'path'
ARRAY = 'array'
OBJECT = 'object'
JOB = 'job'
HANDLE = 'handle'
SERVER_SUPPLIED = 'server_supplied'

# The forms of the names an annotation uses that are not project classes,
# read by their last part (``pathlib.Path`` is ``Path``).
NAMED_FORMS = {
    'str': SCALAR,
    'int': SCALAR,
    'float': SCALAR,
    'bool': SCALAR,
    'None': SCALAR,
    'timedelta': SECONDS,
    'datetime': ISO8601,
    'Path': PATH,
    'PurePath': PATH,
    'FilePath': PATH,
    'PathLike': PATH,
    'tuple': ARRAY,
    'list': ARRAY,
    'set': ARRAY,
    'frozenset': ARRAY,
    'Iterable': ARRAY,
    'Sequence': ARRAY,
    'dict': OBJECT,
    'Mapping': OBJECT,
    'Series': OBJECT,
    'Future': JOB,
}
# Names that wrap a type without changing its form.
_TRANSPARENT = {'ClassVar', 'Optional', 'Union'}
# A parameter of one of these types is filled by the host, never sent: a
# callback, or the bundle of a run's event handlers.
_HOST_FILLED = {'Callable', 'ProgressCallback', 'RunEvents'}


class NoWireFormError(TypeError):
    """A value whose type has no wire form reached ``encode``."""


class NotHandedOutError(NoWireFormError):
    """A parameter takes only a live object no wire member hands out, so no client can name one."""


def project_classes() -> dict[str, type]:
    """Every class defined under ``modules/``, by name, every module imported."""
    import modules

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        for info in pkgutil.walk_packages(modules.__path__, 'modules.'):
            importlib.import_module(info.name)
    found = {}
    for name, module in list(importlib.sys.modules.items()):
        if not name.startswith('modules'):
            continue
        for cname, obj in vars(module).items():
            if inspect.isclass(obj) and obj.__module__ == name:
                found[cname] = obj
    return found


def project_aliases() -> dict[str, str]:
    """Every ``type`` alias defined under ``modules/``, by name, as the text of its value."""
    project_classes()
    found = {}
    for name, module in list(importlib.sys.modules.items()):
        if not name.startswith('modules'):
            continue
        for alias_name, obj in vars(module).items():
            if isinstance(obj, typing.TypeAliasType):
                found[alias_name] = _annotation_text(obj.__value__)
    return found


def _marked(cls: type) -> dict[str, object]:
    """Each member *cls* or a project base marks, by name, nearest first."""
    members = {}
    for klass in reversed(cls.__mro__):
        if klass is object:
            continue
        for name, value in vars(klass).items():
            if mark_of(value) is not None:
                members[name] = value
    return members


def _fields(cls: type) -> tuple[str, ...]:
    names = []
    for klass in reversed(cls.__mro__):
        names += [n for n in fields_of(klass) if n not in names]
    return tuple(names)


def _wire_properties(cls: type) -> list[str]:
    return [n for n, v in _marked(cls).items() if isinstance(v, property) and mark_of(v) == API]


def _wire_methods(cls: type) -> list[str]:
    return [n for n, v in _marked(cls).items() if not isinstance(v, property) and mark_of(v) == API]


def class_form(cls: type) -> str | None:
    """The form a project class crosses in, or None when it has none."""
    if issubclass(cls, enum.Enum):
        return ENUM
    if is_record(cls):
        return RECORD
    if _wire_methods(cls) or _wire_properties(cls) or _fields(cls):
        return HANDLE
    return None


# --- The static half: every wire edge has a form -----------------------------


def _annotation_text(annotation: object) -> str | None:
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, str):
        return annotation
    return inspect.formatannotation(annotation)


def _names(text: str) -> list[str]:
    """The type names an annotation uses, each by its last part, forward references read."""
    found = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                found.extend(_names(node.value))
            elif node.value is None:
                found.append('None')
            return
        if isinstance(node, ast.Attribute):
            found.append(node.attr)
            return
        if isinstance(node, ast.Name):
            found.append(node.id)
            return
        if isinstance(node, ast.Subscript):
            base = node.value.attr if isinstance(node.value, ast.Attribute) else node.value.id
            found.append(base)
            if base in _HOST_FILLED:
                return
            visit(node.slice)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(text, mode='eval').body)
    return found


def _edges(cls: type, name: str, member: object) -> list[tuple[str, str, str | None]]:
    """``(edge, direction, annotation text)`` for one wire member."""
    if isinstance(member, property):
        function = member.fget
        return [('return', 'out', _annotation_text(function.__annotations__.get('return')))]
    function = inspect.unwrap(getattr(member, '__func__', member))
    annotations = function.__annotations__
    edges = [
        ('return', 'out', _annotation_text(annotations.get('return', inspect.Parameter.empty)))
    ]
    for parameter in inspect.signature(function).parameters.values():
        if parameter.name in ('self', 'cls'):
            continue
        edges.append(
            (
                f'parameter {parameter.name}',
                'in',
                _annotation_text(annotations.get(parameter.name, inspect.Parameter.empty)),
            )
        )
    return edges


def _field_text(cls: type, name: str) -> str | None:
    for klass in cls.__mro__:
        annotations = vars(klass).get('__annotations__', {})
        if name in annotations:
            return _annotation_text(annotations[name])
    return None


def wire_gaps(root: type, classes: dict[str, type], aliases: dict[str, str]) -> list[str]:
    """Every wire edge reachable from *root* whose type has no form, and every record method on the wire.

    Follows each wire-marked member and published field of each class it
    reaches, through the project classes their annotations name; a name
    that is neither a project class nor in ``NAMED_FORMS`` -- an image
    array, a table, a thread, ``Any`` -- is a gap, as is a callable in
    anything but a parameter, and a record with a wire-marked method (a
    record has no address to call it at).
    """
    gaps, seen, queue = [], set(), [root]
    while queue:
        cls = queue.pop()
        if cls in seen:
            continue
        seen.add(cls)
        form = class_form(cls)
        if form == RECORD:
            gaps += [f'{cls.__name__}.{m}: a method of a record' for m in _wire_methods(cls)]
        edges = []
        for name, member in _marked(cls).items():
            if mark_of(member) != API:
                continue
            if form == RECORD and not isinstance(member, property):
                continue
            edges += [(f'{cls.__name__}.{name} {e}', d, t) for e, d, t in _edges(cls, name, member)]
        edges += [(f'{cls.__name__}.{f} field', 'out', _field_text(cls, f)) for f in _fields(cls)]
        for where, direction, text in edges:
            if text is None:
                gaps.append(f'{where}: no annotation')
                continue
            used_names = _names(text)
            for used in used_names:
                if used in aliases and used not in NAMED_FORMS:
                    used_names += _names(aliases[used])
                    continue
                if used in _TRANSPARENT or used in NAMED_FORMS:
                    continue
                if used in _HOST_FILLED:
                    if direction != 'in':
                        gaps.append(f'{where}: {used} is not on the wire')
                    continue
                target = classes.get(used)
                if target is None or class_form(target) is None:
                    gaps.append(f'{where}: {used} has no wire form')
                    continue
                queue.append(target)
    reachable = handed_out(root, classes, aliases)
    for cls in seen:
        if class_form(cls) == RECORD:
            continue
        try:
            wire_members(cls, classes, aliases, handed_out=reachable)
        except (NoWireFormError, TypeError) as e:
            gaps.append(f'{cls.__name__}: {e}')
    return sorted(gaps)


def handed_out(root: type, classes: dict[str, type], aliases: dict[str, str]) -> frozenset[type]:
    """Every live object's class a wire client can be handed, walking from *root*.

    One a wire member returns, or a read or record holds, other than a
    sub-object the routes pass through (a read whose type is that one
    class), which a client reaches by its path and never holds.
    """
    found, seen, queue = set(), set(), [root]
    while queue:
        cls = queue.pop()
        if cls in seen:
            continue
        seen.add(cls)
        texts = [(_field_text(cls, f), True) for f in _fields(cls)]
        for name, member in _marked(cls).items():
            if mark_of(member) != API:
                continue
            for _edge, direction, text in _edges(cls, name, member):
                if direction == 'out':
                    texts.append((text, isinstance(member, property)))
        for text, read in texts:
            if text is None:
                continue
            segment = _segment(text, classes) if read else None
            used_names = _names(text)
            for used in used_names:
                if used in aliases and used not in NAMED_FORMS:
                    used_names += _names(aliases[used])
                    continue
                target = classes.get(used) if used not in NAMED_FORMS else None
                if target is None or class_form(target) not in (HANDLE, RECORD):
                    continue
                if class_form(target) == HANDLE and target is not segment:
                    found.add(target)
                queue.append(target)
    return frozenset(found)


def _segment(text: str, classes: dict[str, type]) -> type | None:
    """The live object's class a read of type *text* leads to as a sub-object, or None.

    A sub-object is a read whose type is one live object's class, or that
    class or None.
    """
    live = [n for n in (_base(a) for a in _alternatives(text)) if n != 'None']
    target = classes.get(live[0]) if len(live) == 1 and live[0] not in NAMED_FORMS else None
    return target if target is not None and class_form(target) == HANDLE else None


# --- The runtime half: encode a value -----------------------------------------


def _path(value: pathlib.PurePath, live_folder: pathlib.Path) -> dict:
    host = pathlib.Path(value).absolute()
    resolved, root = host.resolve(), pathlib.Path(live_folder).resolve()
    name = resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else None
    return {'name': name, 'host_path': str(host)}


def _key(key: object) -> str:
    if isinstance(key, enum.StrEnum):
        return key.value
    if isinstance(key, enum.Enum):
        return key.name
    return str(key)


def encode(
    value: object,
    *,
    live_folder: pathlib.Path,
    handle: Callable[[object], object],
    job: Callable[[Future], object],
) -> object:
    """*value* in its wire form, JSON-ready.

    ``handle`` and ``job`` are the host's registries: they give the wire
    form of a live object and of a call still running.

    Raises:
        NoWireFormError: *value*, or something in it, has no wire form.
    """

    def form(v: object) -> object:
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, enum.StrEnum):
            return v.value
        if isinstance(v, enum.Enum):
            return v.name
        if isinstance(v, datetime.timedelta):
            return v.total_seconds()
        if isinstance(v, (datetime.datetime, datetime.date)):
            return v.isoformat()
        if isinstance(v, pathlib.PurePath):
            return _path(v, live_folder)
        if isinstance(v, Future):
            return job(v)
        cls = type(v)
        if is_record(cls):
            names = _fields(cls) + tuple(_wire_properties(cls))
            return {n: form(getattr(v, n)) for n in names}
        if cls.__module__ == 'numpy' and getattr(v, 'shape', None) == ():
            # A numpy scalar where the annotation says int or float.
            return form(v.item())
        if type(v).__name__ == 'Series' and type(v).__module__.startswith('pandas'):
            return {_key(k): form(x) for k, x in v.items()}
        if isinstance(v, Mapping):
            return {_key(k): form(x) for k, x in v.items()}
        if isinstance(v, (set, frozenset)):
            return sorted(form(x) for x in v)
        if isinstance(v, (tuple, list)):
            return [form(x) for x in v]
        if cls.__module__.startswith('modules') and class_form(cls) == HANDLE:
            return handle(v)
        if (
            isinstance(v, Iterable)
            and not isinstance(v, (bytes, bytearray))
            and not hasattr(v, 'shape')
        ):
            return [form(x) for x in v]
        raise NoWireFormError(f'{cls.__module__}.{cls.__qualname__} has no wire form')

    return form(value)


# --- The inbound half: a value a wire client sends -----------------------------

# The forms whose value is turned into something else on the way in; every
# other inbound form is taken as the client sent it.
_CONVERTED = {SECONDS, PATH, ENUM, RECORD, HANDLE}
# The array types a JSON array becomes, by the name the annotation uses.
_ARRAY_TYPES = {'tuple': tuple, 'set': set, 'frozenset': frozenset}


@dataclasses.dataclass(frozen=True)
class Inbound:
    """One alternative of a parameter's type, as a wire client sends it.

    Attributes:
        form: The wire form.
        name: The type's name, by its last part (``str``, ``FilePath``,
            ``Remedy``).
        cls: The project class, for an enum, a record or a handle.
        items: An array's or object's value alternatives, scalars only; empty
            when the annotation does not say.
        fields: A record's published fields, each with its alternatives.
    """

    form: str
    name: str
    cls: type | None = None
    items: tuple[Inbound, ...] = ()
    fields: tuple[tuple[str, tuple[Inbound, ...]], ...] = ()


@dataclasses.dataclass(frozen=True)
class WireParameter:
    """A parameter a wire client sends by name.

    Attributes:
        name: The Python parameter name, the JSON key.
        alternatives: What the value may be.
        required: Whether the client must send it; one it leaves out takes
            the member's own default.
        default: The member's default, for a description; None when required.
    """

    name: str
    alternatives: tuple[Inbound, ...]
    required: bool
    default: object = None


@dataclasses.dataclass(frozen=True)
class WireMember:
    """A member a host routes a wire client to.

    Attributes:
        name: The Python name, the route's last segment.
        read: A property or published field, read rather than called.
        parameters: What a call takes from the client; the parameters the
            host fills (a callback, a run's handlers) are not among them.
        segment: The live object's class a read leads to, when the member is
            a sub-object the route continues through; None otherwise.
        doc: The member's docstring.
        hands_out: Whether its answer can carry a live object a client is
            handed; known only when ``wire_members`` is given ``handed_out``.
        returns_job: Whether its answer can carry a running call (a
            ``Future``), which a host hands out as a job.
        progress: The parameter a host fills with its own ``ProgressCallback``
            to hear how far the call has got; None when it takes none.
    """

    name: str
    read: bool
    parameters: tuple[WireParameter, ...] = ()
    segment: type | None = None
    doc: str = ''
    hands_out: bool = False
    returns_job: bool = False
    progress: str | None = None


def _alternatives(text: str) -> list[ast.AST]:
    """The top-level alternatives of an annotation's union, forward references read."""

    def split(node: ast.AST) -> list[ast.AST]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return split(node.left) + split(node.right)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return split(ast.parse(node.value, mode='eval').body)
        if isinstance(node, ast.Subscript) and _base(node) in ('Optional', 'Union'):
            inner = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            found = [n for e in inner for n in split(e)]
            return found + ([ast.Constant(None)] if _base(node) == 'Optional' else [])
        return [node]

    return split(ast.parse(text, mode='eval').body)


def _base(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and node.value is None:
        return 'None'
    if isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    raise NoWireFormError(f'{ast.unparse(node)} is not a type a wire carries')


def inbound(
    text: str,
    classes: dict[str, type],
    aliases: dict[str, str],
    *,
    handed_out: frozenset[type] | None = None,
) -> tuple[Inbound, ...]:
    """The alternatives a parameter annotated *text* takes from a wire client.

    Empty when the host fills the parameter (a callback, a run's handlers).
    Given ``handed_out``, the live objects a client can be handed
    (``handed_out``), a live object's class outside it is no alternative: a
    client has no id to send for one.

    Raises:
        NoWireFormError: the type has no inbound form -- a job, a thread, a
            path or record inside an array -- or is a union a sent value
            could be read as either side of (a name or a path), so which one
            the client meant is not knowable.
        NotHandedOutError: every alternative but None is a live object no
            wire member hands out.
    """
    found: list[Inbound] = []
    unobtainable = []
    for node in _alternatives(text):
        name = _base(node)
        if name in _HOST_FILLED:
            return ()
        if name in aliases and name not in NAMED_FORMS:
            found += inbound(aliases[name], classes, aliases, handed_out=handed_out)
            continue
        form = NAMED_FORMS.get(name)
        cls = classes.get(name) if form is None else None
        if cls is not None:
            form = class_form(cls)
        if form in (ARRAY, OBJECT):
            found.append(Inbound(form, name, items=_items(node, text, classes, aliases)))
        elif form == RECORD:
            fields = tuple(
                (
                    f,
                    inbound(
                        _field_text(cls, f) or 'object', classes, aliases, handed_out=handed_out
                    ),
                )
                for f in _fields(cls)
            )
            found.append(Inbound(form, name, cls, fields=fields))
        elif form == HANDLE and handed_out is not None and cls not in handed_out:
            unobtainable.append(name)
        elif form in (SCALAR, SECONDS, PATH, ENUM, HANDLE):
            found.append(Inbound(form, name, cls))
        else:
            raise NoWireFormError(f'{text}: {name} has no inbound form')
    if unobtainable and all(a.name == 'None' for a in found):
        raise NotHandedOutError(
            f'{text}: no wire member hands out the live object {unobtainable[0]}'
        )
    converted = [a for a in found if a.form in _CONVERTED]
    others = [a for a in found if a not in converted and a.name != 'None']
    if converted and (len(converted) > 1 or others):
        raise NoWireFormError(f'{text}: a sent value could be read as more than one of its types')
    return tuple(found)


def _items(
    node: ast.AST, text: str, classes: dict[str, type], aliases: dict[str, str]
) -> tuple[Inbound, ...]:
    """An array's or object's value alternatives, which are scalars or nothing said."""
    if not isinstance(node, ast.Subscript):
        return ()
    parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    # A mapping's value is its last part; a tuple's ``...`` says nothing.
    parts = [p for p in parts if not (isinstance(p, ast.Constant) and p.value is Ellipsis)]
    if _base(node) == 'tuple' and len(parts) > 1:
        # A fixed-shape tuple's parts each have their own type, which one item
        # type would misdescribe.
        raise NoWireFormError(f'{text}: a tuple of fixed shape is not sent')
    items = inbound(ast.unparse(parts[-1]), classes, aliases)
    if any(i.form != SCALAR for i in items):
        raise NoWireFormError(f'{text}: an array or object of anything but scalars is not sent')
    return items


def wire_members(
    cls: type,
    classes: dict[str, type],
    aliases: dict[str, str],
    *,
    handed_out: frozenset[type] | None = None,
) -> list[WireMember]:
    """Each member of *cls* a wire client reaches, sorted by name.

    Given ``handed_out``, a parameter with a default that takes only a live
    object no wire member hands out is not sent: its default stands.

    Raises:
        NoWireFormError: a parameter's type has no inbound form, or a
            parameter without a default takes only a live object no wire
            member hands out (``NotHandedOutError``).
        TypeError: a method takes ``*args`` or ``**kwargs``, which a client
            cannot name.
    """
    members = []
    for name, member in _marked(cls).items():
        if mark_of(member) != API:
            continue
        if isinstance(member, property):
            fget = member.fget
            text = _annotation_text(fget.__annotations__.get('return', inspect.Parameter.empty))
            members.append(_read(name, text, fget, classes, aliases, handed_out))
            continue
        function = inspect.unwrap(getattr(member, '__func__', member))
        parameters = []
        progress = None
        for p in inspect.signature(function).parameters.values():
            if p.name in ('self', 'cls'):
                continue
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                raise TypeError(
                    f'{cls.__name__}.{name} takes *{p.name}, which a client cannot name'
                )
            text = _annotation_text(function.__annotations__.get(p.name, p.empty))
            required = p.default is p.empty
            try:
                alternatives = inbound(text, classes, aliases, handed_out=handed_out)
            except NotHandedOutError as e:
                if required:
                    raise NotHandedOutError(f'{cls.__name__}.{name} parameter {p.name}: {e}') from e
                continue
            if not alternatives:
                if 'ProgressCallback' in _names(text):
                    progress = p.name
                continue
            parameters.append(
                WireParameter(p.name, alternatives, required, None if required else p.default)
            )
        returns = _annotation_text(function.__annotations__.get('return', inspect.Parameter.empty))
        members.append(
            WireMember(
                name,
                read=False,
                parameters=tuple(parameters),
                doc=function.__doc__ or '',
                hands_out=_hands_out(returns, handed_out, aliases),
                returns_job=_returns_job(returns, aliases),
                progress=progress,
            )
        )
    members += [
        _read(f, _field_text(cls, f), None, classes, aliases, handed_out) for f in _fields(cls)
    ]
    return sorted(members, key=lambda m: m.name)


def _read(
    name: str,
    text: str | None,
    function: object,
    classes: dict[str, type],
    aliases: dict[str, str],
    handed_out: frozenset[type] | None,
) -> WireMember:
    """A read member, with the live object's class it leads to when it is a sub-object."""
    return WireMember(
        name,
        read=True,
        segment=_segment(text, classes) if text else None,
        doc=getattr(function, '__doc__', '') or '',
        hands_out=_hands_out(text, handed_out, aliases),
        returns_job=_returns_job(text, aliases),
    )


def _returns_job(text: str | None, aliases: dict[str, str]) -> bool:
    """Whether a value of type *text* can carry a running call, which crosses as a job."""
    if not text:
        return False
    used_names = _names(text)
    for used in used_names:
        if used in aliases and used not in NAMED_FORMS:
            used_names += _names(aliases[used])
    return any(NAMED_FORMS.get(n) == JOB for n in used_names)


def _hands_out(
    text: str | None, handed_out: frozenset[type] | None, aliases: dict[str, str]
) -> bool:
    """Whether a value of type *text* can carry a live object in *handed_out*."""
    if not text or not handed_out:
        return False
    names = {c.__name__ for c in handed_out}
    used_names = _names(text)
    for used in used_names:
        if used in aliases and used not in NAMED_FORMS:
            used_names += _names(aliases[used])
    return bool(names & set(used_names))


def decode(
    value: object,
    alternatives: tuple[Inbound, ...],
    *,
    resolve_path: Callable[[str], pathlib.Path],
    handle: Callable[[str, type], object],
) -> object:
    """What a member takes for *value*, sent by a wire client for a parameter of *alternatives*.

    The value's JSON type is the host's to have checked against the
    alternatives (``inbound`` admits only unions a value picks one side
    of); this turns it into the Python value.

    ``resolve_path`` turns a live-folder name into an absolute path, refusing
    a name outside the live folder; ``handle`` turns a handle's id into its
    live object of the given class, refusing an unknown id.
    """
    converted = [a for a in alternatives if a.form in _CONVERTED]
    if value is None or not converted:
        array = next((a for a in alternatives if a.form == ARRAY), None)
        if isinstance(value, list) and array is not None and array.name in _ARRAY_TYPES:
            return _ARRAY_TYPES[array.name](value)
        return value
    (a,) = converted
    if a.form == PATH:
        return resolve_path(value)
    if a.form == SECONDS:
        return datetime.timedelta(seconds=value)
    if a.form == ENUM:
        return a.cls(value) if issubclass(a.cls, enum.StrEnum) else a.cls[value]
    if a.form == HANDLE:
        return handle(value, a.cls)
    return a.cls(
        **{
            f: decode(value[f], alts, resolve_path=resolve_path, handle=handle)
            for f, alts in a.fields
            if f in value
        }
    )
