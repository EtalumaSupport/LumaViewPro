# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every read a client of the API makes of an API object's member that is neither API nor host wiring, counted.

The API is what carries ``@api`` or is published by ``@api_fields``; a
host's wiring is what its class declares with ``@host_wiring``
(``modules/api_surface.py``). Anything else on an API object stays
callable from Python, and 4.0 freezes none of it, so a client reaching it
reaches round the surface. This guard walks every client of the API in
LumaViewPro -- ``ui/`` and its ``.kv`` rules, ``lumaviewpro.py``,
``tools/``, ``scripts/``, ``docs/api_examples/`` -- and counts each such
read by its receiver: a name marked on one class and not on another
counts by the class it is read from.

The receiver is followed through annotations, read from the AST so a name
imported only under ``TYPE_CHECKING`` resolves (``typing.get_type_hints``
raises ``NameError`` there): ``app_context.ctx``'s, ``AppContext``'s
fields and its forwarders (a read of ``ctx.settings`` is a read of what
the forwarder reads, ``ScopeSession.settings``), each API object's own
fields, properties and return annotations, a parameter's annotation, a
local function's return annotation, and a name or a ``self`` attribute
assigned within the file. It cannot follow a ``getattr`` with a string,
a widget field (``AppContext``'s widgets are ``object``: their classes
live in ``ui/``, which ``modules/`` may not import), or anything typed
``object`` or not at all; what it cannot type it does not count.

The count is an equality ratchet per member: a new read fails, and a read
removed lowers its pin in the same commit. The engineering plugin's and
``Scripts/`` reads are Firmware's, outside this walk.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import importlib
import inspect
import pathlib
import re
import sys
import textwrap
import types
from collections import Counter

import pytest

from tests import ratchets as _ratchets
from tests.ast_seams import REPO_ROOT, Packages

from modules import app_context
from modules.api_surface import HOST_WIRING_ATTRIBUTE, fields_of, host_wiring_of, mark_of

# The clients walked: each a directory of them, or one module.
_CLIENTS = Packages({'ui', 'tools', 'scripts', 'docs/api_examples', 'lumaviewpro'})
# The packages an annotation or an import is resolved in; a client's own
# modules (ui, tools) are never imported here.
_PROJECT = Packages({'modules', 'drivers', 'rest'})
# The classes whose members are read through as forwarders: the GUI's
# context holds the session and forwards to it.
_FORWARDERS = (app_context.AppContext,)

# The reads at the pin, by receiver: each a member a client reaches round
# the surface for. The four single builds are the GUI's until its builds
# move onto ``build`` (NEXT_DOCS "A combined post-processing option").
_PINNED: dict[str, int] = {
    'CameraAPI.active': 1,
    'DeliveredRate.megabytes_per_s': 1,
    # tools/pylon_probe_sweep.py, which does not run (fw TODO
    # "tools/pylon_probe_sweep.py does not run"); its rebuild reads the API.
    'ImagingAPI._set_grab_strategy': 1,
    'ImagingAPI._set_max_num_buffer': 1,
    'ImagingAPI._set_max_transfer_size': 1,
    'ImagingAPI._set_num_max_queued_urbs': 1,
    'Labware.get_well_label': 1,
    'LiveWork.has': 2,
    'Lumascope.camera_connected': 2,
    'MotionAPI.get_current_position': 2,
    'MotionAPI.one_command': 1,
    'PostProcessingAPI.composite': 1,
    'PostProcessingAPI.stitch': 1,
    'PostProcessingAPI.video': 1,
    'PostProcessingAPI.zproject': 1,
    'Protocol.copy_for_execution': 2,
    'Protocol.step_row': 6,
    'RuntimeState.get_labware': 4,
}


@dataclasses.dataclass(frozen=True)
class Instance:
    """A value of ``cls``."""

    cls: type


@dataclasses.dataclass(frozen=True)
class Method:
    """A method or function; a call answers its return annotation."""

    function: object
    module: types.ModuleType


@dataclasses.dataclass(frozen=True)
class SelfOf:
    """``self`` inside a class defined in the walked file."""

    class_name: str


@dataclasses.dataclass(frozen=True)
class Reach:
    path: str
    line: int
    owner: str
    member: str
    owner_class: type = dataclasses.field(compare=False, repr=False, default=object)

    def __str__(self) -> str:
        return f'{self.path}:{self.line} {self.owner}.{self.member}'


# --- What a declaration names --------------------------------------------


@functools.cache
def _tree_of(path: str) -> ast.Module:
    return ast.parse(pathlib.Path(path).read_text())


def _parse_def(obj: object) -> ast.AST | None:
    try:
        source = textwrap.dedent(inspect.getsource(obj))
    except (OSError, TypeError):
        return None
    return ast.parse(source).body[0]


def _imports(statements: list[ast.stmt], package: str | None) -> dict[str, tuple[str, str | None]]:
    """Each import binding in ``statements`` and the blocks they hold: name -> (module, attribute)."""
    out: dict[str, tuple[str, str | None]] = {}
    for s in statements:
        if isinstance(s, ast.Import):
            for a in s.names:
                out[a.asname or a.name.split('.')[0]] = (
                    a.name if a.asname else a.name.split('.')[0],
                    None,
                )
        elif isinstance(s, ast.ImportFrom):
            base = s.module or ''
            if s.level:
                parts = (package or '').split('.')
                parts = parts[: len(parts) - s.level + 1]
                base = '.'.join(p for p in [*parts, base] if p)
            for a in s.names:
                out[a.asname or a.name] = (base, a.name)
        elif isinstance(s, (ast.If, ast.Try)):
            out.update(_imports(s.body, package))
            out.update(_imports(getattr(s, 'orelse', []), package))
    return out


def _bound(module: str, attribute: str | None) -> object | None:
    """What ``from module import attribute`` binds, for a project module; None otherwise."""
    if module.split('.')[0] not in _PROJECT:
        return None
    try:
        value = importlib.import_module(module)
    except ImportError:
        return None
    if attribute is None:
        return value
    if hasattr(value, attribute):
        return getattr(value, attribute)
    try:
        return importlib.import_module(f'{module}.{attribute}')
    except ImportError:
        return None


def _name_in(module: types.ModuleType, name: str) -> object | None:
    if name in vars(module):
        return vars(module)[name]
    path = getattr(module, '__file__', None)
    if not path:
        return None
    found = _imports(_tree_of(path).body, module.__package__).get(name)
    return _bound(*found) if found else None


def _value_of(module: types.ModuleType, node: ast.expr) -> object | None:
    if isinstance(node, ast.Name):
        return _name_in(module, node.id)
    if isinstance(node, ast.Attribute):
        base = _value_of(module, node.value)
        return getattr(base, node.attr, None) if base is not None else None
    return None


def _is_project_class(value: object) -> bool:
    return isinstance(value, type) and value.__module__.split('.')[0] in _PROJECT


def class_named(module: types.ModuleType, annotation: object) -> type | None:
    """The one project class ``annotation`` names, less ``None``; None when it names none."""
    if annotation is None:
        return None
    if isinstance(annotation, str):
        try:
            annotation = ast.parse(annotation, mode='eval').body
        except SyntaxError:
            return None
    if isinstance(annotation, ast.Constant):
        return class_named(module, annotation.value) if isinstance(annotation.value, str) else None
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        sides = [
            side
            for side in (annotation.left, annotation.right)
            if not (isinstance(side, ast.Constant) and side.value is None)
        ]
        return class_named(module, sides[0]) if len(sides) == 1 else None
    if isinstance(annotation, ast.Subscript):
        head = _value_of(module, annotation.value)
        if getattr(head, '__name__', None) == 'Optional':
            return class_named(module, annotation.slice)
        return head if _is_project_class(head) else None
    value = _value_of(module, annotation)
    return value if _is_project_class(value) else None


def _returns(function: object, module: types.ModuleType | None = None) -> type | None:
    function = inspect.unwrap(function)
    node = function if isinstance(function, ast.AST) else _parse_def(function)
    if module is None:
        module = sys.modules.get(getattr(function, '__module__', ''))
    if module is None or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return class_named(module, node.returns)


@functools.cache
def _data_members(cls: type) -> dict[str, type | None]:
    """The data attributes ``cls``'s own source declares, each with the class it is annotated as."""
    out: dict[str, type | None] = {}
    node = _parse_def(cls)
    module = sys.modules.get(cls.__module__)
    if not isinstance(node, ast.ClassDef) or module is None:
        return out
    for s in node.body:
        if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
            out[s.target.id] = class_named(module, s.annotation)
    for function in node.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        parameters = {a.arg: a.annotation for a in function.args.args + function.args.kwonlyargs}
        for s in ast.walk(function):
            if isinstance(s, ast.AnnAssign) and _is_self_attribute(s.target):
                out.setdefault(s.target.attr, class_named(module, s.annotation))
            elif isinstance(s, ast.Assign):
                for target in s.targets:
                    if not _is_self_attribute(target):
                        continue
                    annotated = isinstance(s.value, ast.Name) and parameters.get(s.value.id)
                    if out.get(target.attr) is None:
                        out[target.attr] = class_named(module, annotated) if annotated else None
    return out


def _is_self_attribute(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == 'self'
    )


def has_member(cls: type, name: str) -> bool:
    return any(name in vars(k) or name in _data_members(k) for k in cls.__mro__ if k is not object)


@functools.cache
def is_api_object(cls: type) -> bool:
    """Whether ``cls`` publishes anything: a field, or a marked member."""
    return any(fields_of(k) or any(mark_of(v) for v in vars(k).values()) for k in cls.__mro__)


def declared(cls: type, name: str) -> bool:
    """Whether ``name`` on ``cls`` is API (marked or published) or host wiring."""
    for k in cls.__mro__:
        member = vars(k).get(name)
        if (
            name in fields_of(k)
            or name in host_wiring_of(k)
            or (member is not None and mark_of(member))
        ):
            return True
    return False


def member_of(cls: type, name: str) -> object | None:
    """What ``value.name`` is for a value of ``cls``."""
    for k in cls.__mro__:
        member = vars(k).get(name)
        if member is None:
            continue
        module = sys.modules.get(k.__module__)
        if isinstance(member, property):
            returned = _returns(member.fget) if member.fget else None
            return Instance(returned) if returned else None
        if isinstance(member, (classmethod, staticmethod)):
            return Method(member.__func__, module)
        if callable(member) and not isinstance(member, type):
            return Method(member, module)
        return member if isinstance(member, type) else None
    for k in cls.__mro__:
        annotated = _data_members(k).get(name) if k is not object else None
        if annotated is not None:
            return Instance(annotated)
    return None


# --- The walk ------------------------------------------------------------


class Walk:
    """One client file's reads, with what each name and ``self`` attribute holds."""

    def __init__(self, label: str, module: types.ModuleType):
        self.label = label
        self.module = module
        self.reaches: list[Reach] = []
        self.self_members: dict[str, dict[str, object]] = {}
        self.global_names: set[str] = set()
        self.global_values: dict[str, object] = {}

    # Typing an expression records every read it makes.

    def value(self, node: ast.expr, env: dict) -> object | None:
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.Attribute):
            return self.attribute(self.value(node.value, env), node)
        if isinstance(node, ast.Call):
            called = self.value(node.func, env)
            for argument in [*node.args, *(k.value for k in node.keywords)]:
                self.expression(argument, env)
            if isinstance(called, type):
                return Instance(called)
            if isinstance(called, Method):
                returned = _returns(called.function, called.module)
                return Instance(returned) if returned else None
            return None
        self.expression(node, env)
        return None

    def record(self, line: int, cls: type, name: str) -> None:
        self.reaches.append(Reach(self.label, line, cls.__name__, name, cls))

    def attribute(self, base: object, node: ast.Attribute) -> object | None:
        name = node.attr
        if name.startswith('__'):
            return None
        if isinstance(base, SelfOf):
            return self.self_members.get(base.class_name, {}).get(name)
        if isinstance(base, types.ModuleType):
            member = getattr(base, name, None)
            if isinstance(member, (type, types.ModuleType)):
                return member
            if inspect.isfunction(member):
                return Method(member, base)
            path = getattr(base, '__file__', None)
            for s in _tree_of(path).body if path and path.endswith('.py') else ():
                if (
                    isinstance(s, ast.AnnAssign)
                    and isinstance(s.target, ast.Name)
                    and s.target.id == name
                ):
                    annotated = class_named(base, s.annotation)
                    return Instance(annotated) if annotated else None
            return None
        if isinstance(base, type):
            cls = base
        elif isinstance(base, Instance):
            cls = base.cls
        else:
            return None
        if cls in _FORWARDERS:
            for owner, member in forwarded(cls, name):
                self.record(node.lineno, owner, member)
        elif is_api_object(cls) and has_member(cls, name) and not declared(cls, name):
            self.record(node.lineno, cls, name)
        return member_of(cls, name)

    def expression(self, node: ast.expr, env: dict) -> None:
        if isinstance(node, (ast.Attribute, ast.Call)):
            self.value(node, env)
        elif isinstance(node, ast.Lambda):
            inner = dict(env)
            for a in node.args.args + node.args.kwonlyargs:
                inner.pop(a.arg, None)
            self.expression(node.body, inner)
        else:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self.expression(child, env)
                elif isinstance(child, ast.comprehension):
                    self.expression(child.iter, env)
                    for condition in child.ifs:
                        self.expression(condition, env)

    # Statements bind names, in order.

    def body(self, statements: list[ast.stmt], env: dict, class_name: str | None = None) -> None:
        for s in statements:
            self.statement(s, env, class_name)

    def statement(self, s: ast.stmt, env: dict, class_name: str | None) -> None:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in s.decorator_list:
                self.expression(decorator, env)
            inner = dict(env)
            parameters = s.args.posonlyargs + s.args.args + s.args.kwonlyargs
            for a in parameters:
                annotated = class_named(self.module, a.annotation)
                if annotated:
                    inner[a.arg] = Instance(annotated)
                else:
                    inner.pop(a.arg, None)
            if class_name and parameters:
                inner[parameters[0].arg] = SelfOf(class_name)
            env[s.name] = Method(s, self.module)
            self.body(s.body, inner)
        elif isinstance(s, ast.ClassDef):
            for base in s.bases:
                self.expression(base, env)
            self.body(s.body, dict(env), s.name)
        elif isinstance(s, (ast.Import, ast.ImportFrom)):
            for name, found in _imports([s], self.module.__package__).items():
                bound = _bound(*found)
                if bound is not None:
                    env[name] = bound
        elif isinstance(s, ast.Assign):
            held = self.value(s.value, env)
            for target in s.targets:
                self.bind(target, held, env)
        elif isinstance(s, ast.AnnAssign):
            held = self.value(s.value, env) if s.value is not None else None
            annotated = class_named(self.module, s.annotation)
            self.bind(s.target, Instance(annotated) if annotated else held, env)
        elif isinstance(s, ast.Global):
            self.global_names.update(s.names)
        else:
            for child in ast.iter_child_nodes(s):
                if isinstance(child, ast.stmt):
                    self.statement(child, env, class_name)
                elif isinstance(child, ast.expr):
                    self.expression(child, env)
                else:
                    for grandchild in ast.iter_child_nodes(child):
                        if isinstance(grandchild, ast.stmt):
                            self.statement(grandchild, env, class_name)
                        elif isinstance(grandchild, ast.expr):
                            self.expression(grandchild, env)

    def bind(self, target: ast.expr, held: object | None, env: dict) -> None:
        if isinstance(target, ast.Name):
            if held is not None and target.id in self.global_names:
                self.global_values[target.id] = held
            if held is None and target.id in self.global_values:
                held = self.global_values[target.id]
            if held is None:
                env.pop(target.id, None)
            else:
                env[target.id] = held
        elif isinstance(target, ast.Attribute) and isinstance(env.get(_root_name(target)), SelfOf):
            owner = env[_root_name(target)]
            if held is not None and isinstance(target.value, ast.Name):
                self.self_members.setdefault(owner.class_name, {})[target.attr] = held
        else:
            self.expression(target, env)


def _root_name(node: ast.expr) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


@functools.cache
def forwarded(cls: type, name: str) -> tuple[tuple[type, str], ...]:
    """What a forwarder's member reads, on API objects, that is neither API nor host wiring."""
    member = vars(cls).get(name)
    function = member.fget if isinstance(member, property) else member
    if not inspect.isfunction(function):
        return ()
    node = _parse_def(function)
    walk = Walk(cls.__name__, sys.modules[cls.__module__])
    walk.body(node.body, {'self': Instance(cls)})
    return tuple((r.owner_class, r.member) for r in walk.reaches)


def _client_module(label: str, tree: ast.Module) -> types.ModuleType:
    """A stand-in for a client module: its imports from the project bound, nothing of its own run."""
    parts = pathlib.PurePosixPath(label).with_suffix('').parts
    module = types.ModuleType('.'.join(parts))
    module.__package__ = '.'.join(parts[:-1]) or None
    for name, found in _imports(tree.body, module.__package__).items():
        bound = _bound(*found)
        if bound is not None:
            setattr(module, name, bound)
    return module


def walk_source(label: str, source: str) -> Walk:
    """Every counted read in ``source``, the file at ``label``."""
    tree = ast.parse(source)
    module = _client_module(label, tree)
    walk = Walk(label, module)
    env = {k: v for k, v in vars(module).items() if not k.startswith('__')}
    # Twice: the second walk knows what a global and a ``self`` attribute
    # were assigned anywhere in the file.
    for _ in range(2):
        walk.reaches = []
        walk.body(tree.body, {**env, **walk.global_values})
    return walk


# A kv rule's Python: a property's expression or a handler's statements.
_KV_PYTHON = re.compile(r'^\s*[a-z_][a-z0-9_]*\s*:\s*(?P<python>.*\bapp\..*)$')


def walk_kv(label: str, text: str, app: Walk) -> list[Reach]:
    """Every counted read in a ``.kv`` file's rules, ``app`` being the application the walk ``app`` typed."""
    walk = Walk(label, app.module)
    walk.self_members = app.self_members
    env = {'app': SelfOf('LumaViewProApp')}
    for number, line in enumerate(text.splitlines(), start=1):
        found = _KV_PYTHON.match(line)
        if found is None or line.lstrip().startswith('#'):
            continue
        try:
            tree = ast.parse(found['python'].strip())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            node.lineno = number
        walk.body(tree.body, dict(env))
    return walk.reaches


def _client_paths() -> list[pathlib.Path]:
    paths = []
    for client in sorted(_CLIENTS):
        root = REPO_ROOT / client
        if root.is_dir():
            paths += sorted(p for p in root.rglob('*.py') if '__pycache__' not in p.parts)
        else:
            paths.append(root.with_suffix('.py'))
    return paths


@functools.cache
def every_reach() -> tuple[Reach, ...]:
    reaches: list[Reach] = []
    app = None
    for path in _client_paths():
        label = path.relative_to(REPO_ROOT).as_posix()
        walk = walk_source(label, path.read_text())
        reaches += walk.reaches
        if label == 'lumaviewpro.py':
            app = walk
    for kv in sorted((REPO_ROOT / 'ui').rglob('*.kv')):
        reaches += walk_kv(kv.relative_to(REPO_ROOT).as_posix(), kv.read_text(), app)
    return tuple(reaches)


# --- The tests -----------------------------------------------------------


def test_no_client_reads_an_unmarked_member_past_its_pin():
    reaches = every_reach()
    counted = Counter(f'{r.owner}.{r.member}' for r in reaches)
    _ratchets.observe(
        'client reads of an unmarked member', sum(counted.values()), sum(_PINNED.values()), 'equal'
    )
    risen = {k: (v, _PINNED.get(k, 0)) for k, v in counted.items() if v > _PINNED.get(k, 0)}
    fallen = {k: (counted.get(k, 0), v) for k, v in _PINNED.items() if counted.get(k, 0) < v}
    sites = '\n'.join(f'  {r}' for r in reaches if f'{r.owner}.{r.member}' in risen)
    assert not risen, (
        'A client reads a member that is neither API nor host wiring (count, pin): '
        f'{risen}\n{sites}\nRead the API member that answers it, or add the member to the API; '
        'a public name a host wires is declared with @host_wiring on its class.'
    )
    assert not fallen, f'A read went: lower its pin in _PINNED (count, pin): {fallen}'


def test_a_host_wired_name_is_a_member_and_not_api(sim_session):
    """Each name a class declares as host wiring is a member of it, and none is also API."""
    from modules.plugins import PluginRegistry, UIRegistry
    from modules.post_processing_api import PostProcessingAPI
    from modules.scope_session import ScopeSession

    holders = {
        ScopeSession: sim_session,
        PostProcessingAPI: sim_session.post_processing,
        PluginRegistry: PluginRegistry(sim_session.scope),
        UIRegistry: PluginRegistry(sim_session.scope).ui,
    }
    wrong = []
    for cls, instance in holders.items():
        assert host_wiring_of(cls), f'{cls.__name__} declares no host wiring'
        for name in host_wiring_of(cls):
            if not hasattr(instance, name):
                wrong.append(f'{cls.__name__}.{name}: no such member')
            if name in fields_of(cls) or mark_of(vars(cls).get(name)):
                wrong.append(f'{cls.__name__}.{name}: also API')
    assert not wrong, wrong


def test_the_host_mark_states_its_sentence_and_refuses_an_api_member():
    """A host-wired method's docstring carries the mark's sentence after its
    summary, and a member marked API cannot also be declared host wiring."""
    from modules.api_surface import HOST_WIRING_SENTENCE, api, host_wiring

    @host_wiring('lane')
    class Wired:
        def lane(self):
            """The lane.

            What it is for.
            """

    assert inspect.getdoc(Wired.lane) == f'The lane.\n\n{HOST_WIRING_SENTENCE}\n\nWhat it is for.'
    with pytest.raises(TypeError, match='marked API'):

        @host_wiring('read')
        class Both:
            @api(reads=True)
            def read(self) -> int:
                """A read.

                Raises:
                    Nothing.
                """
                return 1


def _reads(source: str) -> list[str]:
    return [
        f'{r.owner}.{r.member}'
        for r in walk_source('ui/planted.py', textwrap.dedent(source)).reaches
    ]


def test_the_walk_counts_a_read_through_the_context_and_an_alias():
    assert _reads(
        """
        import modules.app_context as _app_ctx

        def through_the_context():
            _app_ctx.ctx.scope.motion.get_current_position('Z')

        def through_an_alias():
            ctx = _app_ctx.ctx
            motion = ctx.session.scope.motion
            motion.get_current_position('Z')
        """
    ) == ['MotionAPI.get_current_position', 'MotionAPI.get_current_position']


def test_the_walk_resolves_a_parameters_annotation_by_the_files_imports():
    assert _reads(
        """
        from modules.lumascope_api import Lumascope

        def f(scope: Lumascope):
            scope.camera.active
        """
    ) == ['CameraAPI.active']


def test_the_walk_counts_what_a_forwarder_reads(monkeypatch):
    """``ctx.settings`` is a read of ``ScopeSession.settings``: counted once the
    session no longer declares it host wiring, and not while it does."""
    from modules.scope_session import ScopeSession

    planted = """
        import modules.app_context as _app_ctx

        def f():
            _app_ctx.ctx.settings['live_folder']
        """
    assert _reads(planted) == []
    forwarded.cache_clear()
    monkeypatch.setattr(
        ScopeSession,
        HOST_WIRING_ATTRIBUTE,
        tuple(n for n in host_wiring_of(ScopeSession) if n != 'settings'),
    )
    try:
        assert _reads(planted) == ['ScopeSession.settings']
    finally:
        forwarded.cache_clear()


def test_the_walk_counts_nothing_marked_or_host_wired():
    assert (
        _reads(
            """
        import modules.app_context as _app_ctx

        def f():
            session = _app_ctx.ctx.session
            session.executor_bundle.diagnostics_executor
            session.scope.motion.get_axis_limits('Z')
            session.capture_still()
        """
        )
        == []
    )


def test_the_walk_follows_a_global_and_a_self_attribute():
    assert _reads(
        """
        from modules.scope_session import ScopeSession

        ctx = None

        def build():
            global ctx
            ctx = ScopeSession.create({})

        class App:
            def build(self):
                self.session = ctx

            def later(self):
                self.session.scope.camera.active
        """
    ) == ['CameraAPI.active']


def test_the_walk_reads_a_kv_rule_s_python():
    app = walk_source(
        'lumaviewpro.py',
        textwrap.dedent(
            """
            from modules.scope_session import ScopeSession

            class LumaViewProApp:
                def build(self):
                    self.session = ScopeSession.create({})
            """
        ),
    )
    reaches = walk_kv(
        'ui/planted.kv',
        '<Thing>:\n    disabled: app.session.scope.camera.active\n    # text: app.session.scope.camera.active\n',
        app,
    )
    assert [(r.line, f'{r.owner}.{r.member}') for r in reaches] == [(2, 'CameraAPI.active')]


def test_the_walk_resolves_every_project_import_of_every_client_file():
    """Every client file parses and each name it imports from the project is
    there to bind: a file the walk could not read, or a name it could not
    bind, would count nothing, and pass."""
    unresolved = []
    for path in _client_paths():
        tree = ast.parse(path.read_text())
        package = '.'.join(path.relative_to(REPO_ROOT).with_suffix('').parts[:-1]) or None
        for module, attribute in _imports(tree.body, package).values():
            if module.split('.')[0] not in _PROJECT:
                continue
            imported = importlib.import_module(module)
            if attribute is not None and not hasattr(imported, attribute):
                try:
                    importlib.import_module(f'{module}.{attribute}')
                except ImportError:
                    unresolved.append(f'{path.name}: {module}.{attribute}')
    assert not unresolved, f'imports the walk cannot resolve: {unresolved}'
