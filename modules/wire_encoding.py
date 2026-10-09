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
"""

from __future__ import annotations

import ast
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
    return sorted(gaps)


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
