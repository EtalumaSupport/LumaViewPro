# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every route names the doors it enters, wherever the code shows the entry.

A hardware lane that asks the scope's activity claim refuses a call while a
run, a diagnostic or a home holds the scope, or while a control session is
held by someone else; a guarded write refuses the second. The reference
states those refusals once, so no member's ``Raises:`` lists them, and a
route's description adds them from the member's mark,
``@api(admitted_by=(...))``, each door's own ``Raises:`` read through
``rest.problems.DOORS``. Unmarked, a route that refuses a held session
would describe no such refusal. This guard holds the mark to the code over
every member a REST route serves:

1. **Named.** A member whose walk reaches a door names it: a call landing on
   ``SequentialIOExecutor.call`` or ``.submit`` of a lane that asks the
   claim, or on ``ActivityClaim.enter_guarded_write``.
2. **Not stale.** A member whose walk resolves every call it makes names no
   door the walk does not reach.

The walk follows the calls a function makes, and the functions it hands on
as arguments (a lane's task), through the project's code, the layers a
decorator wraps around a function included. A call's target is read from
what the simulated session's live objects hold, else from annotations, a
local's assignment, a closure's cell or an import inside the function. A
lane's receiver (``self._scope._camera_executor``) is evaluated on the live
owner, so the IO and CAMERA lanes, which ask the claim, are told from the
post-processing and file lanes, which do not.

Its limit: a member that enters a door only through a call the walk
cannot resolve is not caught by (1), and is not judged by (2).
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import importlib
import inspect
import os
import pathlib
import textwrap
import types
import typing
from collections.abc import Callable

import pytest

from modules import api_surface, wire_encoding
from modules.api_surface import Admission, admitted_by_of
from modules.post_processing_api import PostProcessingAPI
from modules.lumascope_api.camera import CameraAPI
from modules.scope_session import ScopeSession
from modules.sequential_io_executor import SequentialIOExecutor
from rest import problems
from rest import routes as rest_routes
from tests.ast_seams import Packages
from tests.settings_fixtures import complete_settings

_PROJECT = Packages({'modules', 'lib', 'drivers', 'rest'})
_LANE_DOORS = {SequentialIOExecutor.call, SequentialIOExecutor.submit}
_DOOR_OF = {function: door for door, function in problems.DOORS.items()}


# The packages whose files are the project's code, and the argument door's
# own file, read by where a function's code is: a wrapper takes its
# wrapped function's module name (``functools.wraps``), not its file.
_PACKAGES = tuple(str(pathlib.Path(api_surface.__file__).parents[1] / p) + os.sep for p in _PROJECT)
_ARGUMENT_DOOR_FILE = api_surface.__file__


def _own_code(function: object) -> bool:
    """Whether *function*'s code is a project file's, other than the argument door's."""
    filename = getattr(getattr(function, '__code__', None), 'co_filename', '')
    return filename.startswith(_PACKAGES) and filename != _ARGUMENT_DOOR_FILE


def _is_project(obj: object) -> bool:
    return (getattr(obj, '__module__', None) or '').partition('.')[0] in _PROJECT


@dataclasses.dataclass(frozen=True)
class _Of:
    """An instance of ``cls``, known by its type rather than as a live object."""

    cls: type


# An expression whose value is data: a literal, a display, a comprehension,
# arithmetic, a comparison.
_DATA = (
    ast.Constant,
    ast.JoinedStr,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
)

# A value the walk knows is not the project's code -- a library's, a
# builtin's, data -- on which a call enters no door.
_FOREIGN = _Of(object)


@dataclasses.dataclass
class _Reached:
    """What a function's walk reaches: each door with one path to it, and the calls it could not resolve."""

    doors: dict[Admission, str] = dataclasses.field(default_factory=dict)
    holes: set[str] = dataclasses.field(default_factory=set)

    def add(self, other: _Reached, via: str) -> None:
        for door, path in other.doors.items():
            self.doors.setdefault(door, f'{via} -> {path}')
        self.holes |= other.holes


class _Walk:
    """The door walk over the project's code, read against one live session."""

    def __init__(self, roots: list[object]) -> None:
        self._live: list[object] = []
        seen: set[int] = set()
        frontier = list(roots)
        while frontier:
            obj = frontier.pop()
            if id(obj) in seen or not _is_project(type(obj)):
                continue
            seen.add(id(obj))
            self._live.append(obj)
            frontier.extend(getattr(obj, '__dict__', {}).values())
        # Every project class by name: an annotation imported only for type
        # checking names a class its module's globals lack.
        self._classes = wire_encoding.project_classes()
        self._reached: dict[object, _Reached] = {}
        self._walking: set[object] = set()

    # -- a function's reach ------------------------------------------------

    def reach(self, member: object) -> _Reached:
        """What *member* reaches, from the outermost layer of its own code."""
        return self._reach(self._entry(api_surface._function_of(member)))

    def _entry(self, function: object) -> object:
        """The outermost layer of *function* that is the project's own, past the argument door.

        The argument door's wrapper (``api_surface``) refuses only a wrong
        argument type, every member's refusal; a library's wrapper
        (``contextlib``, ``functools``) runs the function it wraps.
        """
        while not _own_code(function) and hasattr(function, '__wrapped__'):
            function = function.__wrapped__
        return function

    def _reach(self, function: object) -> _Reached:
        if function in self._reached:
            return self._reached[function]
        if function in self._walking:
            return _Reached()
        self._walking.add(function)
        try:
            found = self._walk_body(function)
        finally:
            self._walking.discard(function)
        self._reached[function] = found
        return found

    def _walk_body(self, function: object) -> _Reached:
        found = _Reached()
        node = self._source(function)
        if node is None:
            return found
        here = function.__qualname__
        scope = _Scope(self, function, node)
        # The body alone: a decorator runs when the function is defined, not called.
        calls = [
            n for statement in node.body for n in ast.walk(statement) if isinstance(n, ast.Call)
        ]
        for call in calls:
            targets = scope.resolve(call.func)
            if not targets:
                found.holes.add(f'{here}: {ast.unparse(call.func)}')
            for target in targets:
                self._enter(target, call, scope, found)
            # A function handed on as an argument is called by whoever takes
            # it: a lane's task, a callback.
            for argument in [*call.args, *(k.value for k in call.keywords)]:
                if isinstance(argument, (ast.Name, ast.Attribute)):
                    for target in scope.resolve(argument):
                        if _is_function(target):
                            self._enter(target, None, scope, found)
        return found

    def _enter(self, target: object, call: ast.Call | None, scope: _Scope, found: _Reached) -> None:
        if not _is_function(target) or not _is_project(target):
            return
        function = self._entry(target)
        door = _DOOR_OF.get(_unwrapped(function))
        here = scope.function.__qualname__
        if door is None:
            found.add(self._reach(function), here)
        elif _unwrapped(function) not in _LANE_DOORS:
            found.doors.setdefault(door, f'{here} -> {door.value}')
        elif call is not None:
            asks = self._lane_asks_the_claim(call.func, scope)
            if asks is None:
                found.holes.add(f'{here}: the lane {ast.unparse(call.func)} is not read live')
            elif asks:
                found.doors.setdefault(door, f'{here} -> {door.value}')

    def _lane_asks_the_claim(self, func: ast.expr, scope: _Scope) -> bool | None:
        """Whether the lane a ``<receiver>.call``/``.submit`` lands on asks the claim; None when no live owner shows it.

        The receiver is a chain of attributes from ``self``, read on every
        live instance of the function's class.
        """
        names: list[str] = []
        receiver = func.value if isinstance(func, ast.Attribute) else None
        while isinstance(receiver, ast.Attribute):
            names.insert(0, receiver.attr)
            receiver = receiver.value
        owner = _owner_of(scope.function)
        if not (isinstance(receiver, ast.Name) and receiver.id == 'self' and owner is not None):
            return None
        lanes = []
        for obj in self._live:
            if isinstance(obj, owner):
                for name in names:
                    obj = getattr(obj, name, None)
                if isinstance(obj, SequentialIOExecutor):
                    lanes.append(obj)
        if not lanes:
            return None
        return any(lane._claim is not None for lane in lanes)

    @staticmethod
    def _source(function: object) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        if not isinstance(function, types.FunctionType) or function.__name__ == '<lambda>':
            return None
        try:
            # From the code object: given the function, ``getsource`` follows
            # ``__wrapped__`` and reads the wrapped function, not this layer.
            node = ast.parse(textwrap.dedent(inspect.getsource(function.__code__))).body[0]
        except (OSError, TypeError, SyntaxError):
            return None
        return node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None

    # -- what a name holds -------------------------------------------------

    def held(self, cls: type, name: str) -> set[object]:
        """What attribute *name* of a *cls* instance can hold: what the live ones hold, else its annotation."""
        found: set[object] = set()
        for obj in self._live:
            if isinstance(obj, cls) and name in getattr(obj, '__dict__', {}):
                value = obj.__dict__[name]
                found |= (
                    {_Of(type(value))} if _is_project(type(value)) else _kept(value) or {_FOREIGN}
                )
        if not found:
            found = self.annotated(self.hints(cls).get(name))
        return found

    def hints(self, obj: object) -> dict[str, object]:
        try:
            return typing.get_type_hints(obj, localns=self._classes)
        except Exception:
            # An annotation the module cannot evaluate: read as no annotation.
            return {}

    def annotated(self, annotation: object) -> set[object]:
        """What an annotation says a value is: an instance, a class, or foreign; empty for ``Any``.

        A callable parameter is foreign here: the function handed in is
        walked where its caller hands it on.
        """
        if annotation is None or annotation is typing.Any:
            return set()
        origin, args = typing.get_origin(annotation), typing.get_args(annotation)
        if origin in (typing.Union, types.UnionType):
            return set().union(*(self.annotated(a) for a in args))
        if origin is typing.Annotated:
            return self.annotated(args[0])
        if origin is type and args and isinstance(args[0], type):
            return {args[0]} if _is_project(args[0]) else {_FOREIGN}
        if origin is not None:
            # A generic: a builtin container, a Callable, an iterator.
            return {_Of(origin)} if _is_project(origin) else {_FOREIGN}
        if isinstance(annotation, type):
            return {_Of(annotation)} if _is_project(annotation) else {_FOREIGN}
        return set()


class _Scope:
    """One function's names, read for the walk."""

    def __init__(self, walk: _Walk, function: types.FunctionType, node: ast.AST) -> None:
        self.walk = walk
        self.function = function
        self.node = node
        self._owner = _owner_of(function)
        self._hints = walk.hints(function)
        self._binding: set[str] = set()

    def resolve(self, expr: ast.expr) -> set[object]:
        """What *expr* can be: an object, or an ``_Of`` instance; empty when the walk cannot say."""
        if isinstance(expr, ast.Name):
            return self._name(expr.id)
        if isinstance(expr, ast.Attribute):
            found: set[object] = set()
            for base in self.resolve(expr.value):
                found |= self._attribute(base, expr.attr)
            return found
        if isinstance(expr, ast.Call):
            if (
                isinstance(expr.func, ast.Name)
                and expr.func.id == 'getattr'
                and len(expr.args) >= 2
                and isinstance(expr.args[1], ast.Constant)
                and isinstance(expr.args[1].value, str)
            ):
                return self.resolve(ast.Attribute(expr.args[0], expr.args[1].value, ast.Load()))
            found = set()
            for target in self.resolve(expr.func):
                if isinstance(target, type):
                    found.add(_Of(target) if _is_project(target) else _FOREIGN)
                elif _is_function(target) and _is_project(target):
                    found |= self.walk.annotated(self.walk.hints(target).get('return'))
                elif not _is_project(target) or target is _FOREIGN:
                    found.add(_FOREIGN)
            return found
        if isinstance(expr, _DATA):
            return {_FOREIGN}
        if isinstance(expr, ast.Subscript):
            return {_FOREIGN} if _FOREIGN in self.resolve(expr.value) else set()
        if isinstance(expr, ast.Await):
            return self.resolve(expr.value)
        if isinstance(expr, ast.IfExp):
            return self.resolve(expr.body) | self.resolve(expr.orelse)
        if isinstance(expr, ast.BoolOp):
            return set().union(*(self.resolve(v) for v in expr.values))
        return set()

    def _name(self, name: str) -> set[object]:
        if name == 'self' and self._owner is not None:
            return {_Of(self._owner)}
        if name == 'cls' and self._owner is not None:
            return {self._owner}
        if name in self._hints:
            return self.walk.annotated(self._hints[name])
        bound = self._bound(name)
        if bound is not None:
            return bound
        code = self.function.__code__
        if name in code.co_freevars:
            try:
                return _known(self.function.__closure__[code.co_freevars.index(name)].cell_contents)
            except ValueError:
                return set()
        if name in self.function.__globals__:
            return _known(self.function.__globals__[name])
        if hasattr(builtins, name):
            return _known(getattr(builtins, name))
        return self._imported(name)

    def _bound(self, name: str) -> set[object] | None:
        """What a local *name* is assigned in the function; None when it is assigned nowhere."""
        values = [
            n.value
            for n in ast.walk(self.node)
            if isinstance(n, (ast.Assign, ast.NamedExpr))
            and any(
                isinstance(t, ast.Name) and t.id == name
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
            )
        ]
        values += [
            n.context_expr
            for n in ast.walk(self.node)
            if isinstance(n, ast.withitem)
            and isinstance(n.optional_vars, ast.Name)
            and n.optional_vars.id == name
        ]
        if not values or name in self._binding:
            return None if not values else set()
        self._binding.add(name)
        try:
            return set().union(*(self.resolve(v) for v in values))
        finally:
            self._binding.discard(name)

    def _imported(self, name: str) -> set[object]:
        """What an ``import`` inside the function binds to *name*."""
        for n in ast.walk(self.node):
            if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                for alias in n.names:
                    if (alias.asname or alias.name) == name:
                        if n.module.partition('.')[0] not in _PROJECT:
                            # A library, which may not be installed here (a camera SDK).
                            return {_FOREIGN}
                        module = importlib.import_module(n.module)
                        if not hasattr(module, alias.name):
                            # A submodule not imported yet: ``from lib import memory_profile``.
                            return _known(importlib.import_module(f'{n.module}.{alias.name}'))
                        return _known(getattr(module, alias.name))
            if isinstance(n, ast.Import):
                for alias in n.names:
                    if (alias.asname or alias.name.partition('.')[0]) == name:
                        if alias.name.partition('.')[0] not in _PROJECT:
                            return {_FOREIGN}
                        return {importlib.import_module(alias.name if alias.asname else name)}
        return set()

    def _attribute(self, base: object, name: str) -> set[object]:
        if base is _FOREIGN:
            return {_FOREIGN}
        if isinstance(base, _Of):
            member = next((vars(k)[name] for k in base.cls.__mro__ if name in vars(k)), None)
            if isinstance(member, property):
                return self.walk.annotated(self.walk.hints(member.fget).get('return')) | {
                    member.fget
                }
            if isinstance(member, (classmethod, staticmethod)):
                return {member.__func__}
            if _is_function(member):
                return {member}
            return self.walk.held(base.cls, name)
        if isinstance(base, (types.ModuleType, type)):
            value = getattr(base, name, None)
            if isinstance(value, property):
                return {value.fget}
            return set() if value is None else _known(value)
        if _is_function(base) or callable(base):
            # An attribute of a function: ``functools.partial``'s, a bound method's.
            return {_FOREIGN}
        return set()


def _kept(*values: object) -> set[object]:
    """Of *values*, what a call can land on or lead to: a function, a class, a module, an instance."""
    return {v for v in values if isinstance(v, (_Of, type, types.ModuleType)) or _is_function(v)}


def _known(value: object) -> set[object]:
    """What a name bound to *value* leads to: *value* when a call can land on it, else foreign data."""
    return _kept(value) or {_FOREIGN}


def _is_function(obj: object) -> bool:
    return isinstance(obj, types.FunctionType) or (callable(obj) and hasattr(obj, '__wrapped__'))


def _unwrapped(function: object) -> object:
    while hasattr(function, '__wrapped__'):
        function = function.__wrapped__
    return function


def _owner_of(function: object) -> type | None:
    """The class whose body defines *function*, read from its qualified name."""
    parts = getattr(function, '__qualname__', '').split('.')
    if len(parts) < 2 or parts[-2] == '<locals>':
        return None
    owner = getattr(inspect.getmodule(function), parts[-2], None)
    return owner if isinstance(owner, type) else None


@pytest.fixture(scope='module')
def walk(tmp_path_factory):
    live = tmp_path_factory.mktemp('live')
    session = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    try:
        yield _Walk([session, session.create_protocol_runner()])
    finally:
        session.shutdown()


def _route_members() -> dict[str, Callable]:
    """Every member a REST route serves, by ``Class.member``, as the server builds its routes."""
    handed_out = wire_encoding.handed_out(
        ScopeSession, wire_encoding.project_classes(), wire_encoding.project_aliases()
    )
    found = {}
    for root in (ScopeSession, *sorted(handed_out - {ScopeSession}, key=lambda c: c.__name__)):
        for route in rest_routes.routes(root, handed_out=handed_out):
            member = next(
                (
                    vars(k)[route.member.name]
                    for k in route.owner.__mro__
                    if route.member.name in vars(k)
                ),
                None,
            )
            if member is not None and api_surface.mark_of(member) is not None:
                found[f'{route.owner.__name__}.{route.member.name}'] = member
    return found


def test_the_walk_tells_a_lane_that_asks_the_claim_from_one_that_does_not(walk):
    """The instrument on known members: a camera write enters the CAMERA lane, a build only the post-processing lane."""
    assert Admission.LANE_CALL in walk.reach(vars(CameraAPI)['set_gain_db']).doors
    assert (
        walk.reach(vars(PostProcessingAPI)['build'])
        .doors.keys()
        .isdisjoint({Admission.LANE_CALL, Admission.LANE_SUBMIT})
    )
    assert Admission.GUARDED_WRITE in walk.reach(vars(ScopeSession)['update_settings']).doors


def test_every_route_names_the_doors_it_enters(walk):
    unnamed, stale = [], []
    for name, member in sorted(_route_members().items()):
        reached = walk.reach(member)
        marked = set(admitted_by_of(member))
        for door in sorted(reached.doors.keys() - marked):
            unnamed.append(f'{name}: enters {door.value} ({reached.doors[door]})')
        if not reached.holes:
            for door in sorted(marked - reached.doors.keys()):
                stale.append(f'{name}: names {door.value}, which it never enters')

    assert not unnamed, (
        'A route enters a door its mark does not name, so its description omits '
        "the door's refusals; add it to @api(admitted_by=...):\n  " + '\n  '.join(unnamed)
    )
    assert not stale, (
        'A route names a door its code never enters, so its description '
        'publishes refusals it cannot give:\n  ' + '\n  '.join(stale)
    )
