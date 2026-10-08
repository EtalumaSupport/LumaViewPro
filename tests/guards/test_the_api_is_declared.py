# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The API is the marked set, the reference indexes exactly it, and it is closed.

A member is API when it carries ``@api``; a class publishes the data
attributes its ``@api_fields`` names (``modules/api_surface.py``). Before the
marker, the surface was inferred from names, so whoever named a member decided
whether it was API, and nothing asked what a published member hands out. This
guard holds four things over every class defined under ``modules/``:

1. **Equal.** The marked set equals the ``## API reference`` index in
   ``docs/LumascopeSkills.md``, both ways, in-process labels included. The
   failure names each side's extras.
2. **Real.** Every published field is a field the class has: a dataclass or
   NamedTuple field, a class attribute, or assigned as ``self.<name>`` in the
   class body.
3. **Typed.** Every marked member and field says what it hands out and takes
   in: a return annotation, every parameter's, the field's. No ``Any`` or
   ``object`` appears in one and every ``Callable`` names its arguments, a
   callback's included. The one exception is ``object`` as a callback's
   return, the typing idiom for a return the API ignores.
4. **Closed.** Every project class an annotation names is an enum, an
   exception, or a class with marked members or fields, so nothing published
   hands out or asks for an object whose own members are not API.
5. **Paths declared.** A parameter of a member on the wire (marked, not
   in-process) that takes a file-system path is annotated exactly
   ``FilePath`` or ``FilePath | None``, so a wire client can tell a path
   from a string and pass it through ``ScopeSession.live_folder_path``.

Annotations are read as strings and resolved by class name over the loaded
classes, never through ``typing.get_type_hints``, which raises on the
``TYPE_CHECKING`` imports the API modules use. A static read: no session.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import importlib
import inspect
import pkgutil
import re
import warnings

import pytest

from tests.ast_seams import REPO_ROOT

DOC = REPO_ROOT / 'docs' / 'LumascopeSkills.md'
INDEX_HEADING = '## API reference'
IN_PROCESS_LABEL = '(in-process)'

# Members whose absence would mean the walk lost its way, one per kind of
# owner: a Session member, a sub-API member, a run handle's, a run outcome's
# field, and an outcome delivered to a listener.
NOT_VACUOUS = {
    'ScopeSession.add_step',
    'ImagingAPI.set_gain_db',
    'RunHandle.wait',
    'RunOutcome.status',
    'Notification.kind',
}

_BLIND = {'Any', 'object'}


@pytest.fixture(scope='module')
def universe() -> dict[str, type]:
    """Every class defined under modules/, by name, every module imported."""
    import modules

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        for info in pkgutil.walk_packages(modules.__path__, 'modules.'):
            importlib.import_module(info.name)
    classes: dict[str, set[type]] = {}
    for name, module in list(importlib.sys.modules.items()):
        if not name.startswith('modules'):
            continue
        for cname, obj in vars(module).items():
            if inspect.isclass(obj) and obj.__module__ == name:
                classes.setdefault(cname, set()).add(obj)
    ambiguous = sorted(n for n, found in classes.items() if len(found) > 1)
    assert not ambiguous, (
        f'two classes under modules/ share each of these names: {ambiguous}. An '
        'annotation names a class by name, so the closure check cannot tell them apart'
    )
    return {n: next(iter(found)) for n, found in classes.items()}


@pytest.fixture(scope='module')
def marked(universe: dict[str, type]) -> dict[str, str]:
    """``Class.member`` -> ``api`` or ``in_process``; fields -> ``api``."""
    from modules.api_surface import fields_of, mark_of

    found = {}
    for cname, cls in universe.items():
        for member, value in vars(cls).items():
            tag = mark_of(value)
            if tag is not None:
                found[f'{cname}.{member}'] = tag
        for field in fields_of(cls):
            found[f'{cname}.{field}'] = 'api'
    return found


def _index(text: str) -> dict[str, str]:
    """``Class.member`` -> ``api`` or ``in_process``, from the doc's index."""
    start = text.index(f'\n{INDEX_HEADING}\n')
    section = text[start + len(INDEX_HEADING) + 2 :]
    end = section.find('\n## ')
    section = section if end < 0 else section[:end]
    indexed, cls = {}, None
    for line in section.splitlines():
        heading = re.fullmatch(r'### `?([A-Za-z_][A-Za-z0-9_]*)`?', line.strip())
        if heading:
            cls = heading.group(1)
            continue
        item = re.fullmatch(r'- `([A-Za-z_][A-Za-z0-9_]*)`( \(in-process\))?', line.strip())
        if item:
            assert cls is not None, f'index item {line!r} sits under no class heading'
            indexed[f'{cls}.{item.group(1)}'] = 'in_process' if item.group(2) else 'api'
    return indexed


@pytest.fixture(scope='module')
def indexed() -> dict[str, str]:
    """pin-justified: the guard's subject is the reference's index text."""
    return _index(DOC.read_text(encoding='utf-8'))


def test_no_marked_name_is_private(marked: dict[str, str]) -> None:
    private = sorted(name for name in marked if name.split('.', 1)[1].startswith('_'))
    assert not private, f'members marked as API whose names say private: {private}'


def test_the_index_is_the_marked_set(marked: dict[str, str], indexed: dict[str, str]) -> None:
    unindexed = sorted(set(marked) - set(indexed))
    unmarked = sorted(set(indexed) - set(marked))
    mislabelled = sorted(
        f'{name}: marked {marked[name]}, indexed {indexed[name]}'
        for name in set(marked) & set(indexed)
        if marked[name] != indexed[name]
    )
    assert not (unindexed or unmarked or mislabelled), (
        f'{DOC.name} "{INDEX_HEADING}" and the @api / @api_fields marks disagree.\n'
        f'  marked, not indexed: {unindexed}\n'
        f'  indexed, not marked: {unmarked}\n'
        f'  labelled differently: {mislabelled}'
    )


def _self_assigned(cls: type) -> set[str]:
    """Names the class body assigns as ``self.<name>``, read from its source."""
    try:
        tree = ast.parse(inspect.getsource(cls).lstrip())
    except (OSError, TypeError):
        return set()
    return {
        target.attr
        for node in ast.walk(tree)
        for target in (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, (ast.AnnAssign, ast.AugAssign))
            else []
        )
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == 'self'
    }


def test_every_published_field_exists(universe: dict[str, type]) -> None:
    from modules.api_surface import fields_of

    missing = []
    for cname, cls in universe.items():
        names = fields_of(cls)
        if not names:
            continue
        known = set(getattr(cls, '_fields', ())) | _self_assigned(cls)
        if dataclasses.is_dataclass(cls):
            known |= {f.name for f in dataclasses.fields(cls)}
        known |= {
            name
            for name in dir(cls)
            if not callable(inspect.getattr_static(cls, name))
            and not isinstance(
                inspect.getattr_static(cls, name), (property, classmethod, staticmethod)
            )
        }
        missing += [f'{cname}.{name}' for name in names if name not in known]
    assert not missing, f'@api_fields names a field the class does not have: {missing}'


def _text(annotation: object) -> str | None:
    """An annotation as source text; None when there is none."""
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, str):
        return annotation
    return inspect.formatannotation(annotation)


def _field_annotation(cls: type, name: str) -> str | None:
    for klass in cls.__mro__:
        annotations = vars(klass).get('__annotations__', {})
        if name in annotations:
            return _text(annotations[name])
    return None


def _edges(cls: type, member: str) -> list[tuple[str, str | None]]:
    """(edge, annotation text) for a marked member: its return and parameters, or the field."""
    value = vars(cls).get(member)
    function = value.fget if isinstance(value, property) else getattr(value, '__func__', value)
    if value is None or not callable(function) or inspect.isclass(function):
        return [('field', _field_annotation(cls, member))]
    function = inspect.unwrap(function)
    annotations = function.__annotations__
    edges = [('return', _text(annotations.get('return', inspect.Parameter.empty)))]
    for parameter in inspect.signature(function).parameters.values():
        if parameter.name in ('self', 'cls'):
            continue
        edges.append(
            (
                f'parameter {parameter.name}',
                _text(annotations.get(parameter.name, inspect.Parameter.empty)),
            )
        )
    return edges


def _named(text: str, *, holds_callbacks: bool = True) -> tuple[set[str], set[str]]:
    """(blind names, every name) an annotation uses.

    ``Any`` and ``object`` say nothing, and neither does a ``Callable`` that
    does not name the arguments it is called with. ``object`` as a
    ``Callable[[...], R]``'s ``R`` is the idiom for a return the API ignores,
    so there it is not counted blind when the annotation is one that hands
    the API a callback -- a parameter's or a field's (``holds_callbacks``) --
    and ``object`` is the whole return; nested in it, in what a member
    returns, or as ``Any`` anywhere, it is.
    """
    blind, names = set(), set()

    def visit(node: ast.AST, in_callback_return: bool) -> None:
        if isinstance(node, ast.Subscript):
            base = (
                node.value.attr
                if isinstance(node.value, ast.Attribute)
                else getattr(node.value, 'id', '')
            )
            if (
                base == 'Callable'
                and isinstance(node.slice, ast.Tuple)
                and len(node.slice.elts) == 2
            ):
                names.add(base)
                arguments, returned = node.slice.elts
                if isinstance(arguments, ast.Constant):
                    blind.add('Callable[..., R]')
                visit(arguments, in_callback_return)
                # Only the callback's own return may be a bare ``object``;
                # ``list[object]`` or ``object | None`` there still says nothing.
                bare_object = isinstance(returned, ast.Name) and returned.id == 'object'
                visit(returned, holds_callbacks and bare_object)
                return
            visit(node.value, in_callback_return)
            visit(node.slice, in_callback_return)
            return
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            names.add(name)
            ignored_return = in_callback_return and name == 'object'
            if (name in _BLIND or name == 'Callable') and not ignored_return:
                blind.add(name)
            if isinstance(node, ast.Attribute):
                visit(node.value, in_callback_return)
            return
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            inner_blind, inner = _named(node.value, holds_callbacks=holds_callbacks)
            names.update(inner)
            blind.update(inner_blind - ({'object'} if in_callback_return else set()))
            return
        for child in ast.iter_child_nodes(node):
            visit(child, in_callback_return)

    visit(ast.parse(text, mode='eval'), False)
    return blind, names


def test_the_api_is_typed_and_closed(universe: dict[str, type], marked: dict[str, str]) -> None:
    owners = {name.split('.', 1)[0] for name in marked}
    untyped, blind, unpublished = [], [], {}
    for name in sorted(marked):
        cname, member = name.split('.', 1)
        for edge, text in _edges(universe[cname], member):
            if text is None:
                untyped.append(f'{name} {edge}')
                continue
            blind_names, names = _named(text, holds_callbacks=edge != 'return')
            if blind_names:
                blind.append(f'{name} {edge}: {text}')
            for used in names & universe.keys():
                cls = universe[used]
                if used in owners or issubclass(cls, (enum.Enum, BaseException)):
                    continue
                unpublished.setdefault(used, []).append(f'{name} {edge}')
    assert not (untyped or blind or unpublished), (
        'The API must say what every published member hands out and takes in, '
        'and publish every type it names.\n'
        f'  no annotation: {untyped}\n'
        f'  Any or object: {blind}\n'
        + ''.join(
            f'  {cls} is named by {where} but has no marked member\n'
            for cls, where in sorted(unpublished.items())
        )
    )


def test_the_walk_is_not_vacuous(marked: dict[str, str], indexed: dict[str, str]) -> None:
    assert marked.keys() >= NOT_VACUOUS, f'marks not found: {sorted(NOT_VACUOUS - marked.keys())}'
    assert indexed.keys() >= NOT_VACUOUS, (
        f'index entries not found: {sorted(NOT_VACUOUS - indexed.keys())}'
    )


# A parameter naming one of these takes a file-system path.
_PATH_NAMES = {'Path', 'PathLike', 'FilePath'}
_DECLARED_PATH = {'FilePath', 'FilePath | None'}
# A wire path parameter whose absence would mean the walk lost its way.
KNOWN_WIRE_PATH = 'ScopeSession.load_protocol parameter file_path'


def test_every_wire_path_is_declared(universe: dict[str, type], marked: dict[str, str]) -> None:
    seen, undeclared = set(), []
    for name in sorted(n for n, tag in marked.items() if tag == 'api'):
        cname, member = name.split('.', 1)
        for edge, text in _edges(universe[cname], member):
            if not edge.startswith('parameter ') or text is None:
                continue
            if not _named(text)[1] & _PATH_NAMES:
                continue
            seen.add(f'{name} {edge}')
            if text not in _DECLARED_PATH:
                undeclared.append(f'{name} {edge}: {text}')
    assert KNOWN_WIRE_PATH in seen, f'the walk did not reach {KNOWN_WIRE_PATH}'
    assert not undeclared, (
        'A path parameter on the wire must be annotated FilePath or FilePath | None '
        '(modules.api_surface), so a wire client knows which arguments are paths:\n  '
        + '\n  '.join(undeclared)
    )
