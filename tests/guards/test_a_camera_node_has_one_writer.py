# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera node is written one way, through its driver's ``write_node``, by one setter.

Each driver writes a named setting through one function, ``Camera.write_node``,
whose SDK call is the driver's ``_put``: a write lands or fails the same way
whichever setter makes it, and a timeout is answered one way. A node written
from two setters is two ways to change one setting, each with its own value
source, so a node is written by one function, the named exceptions below
aside.

This walk over the Pylon and IDS drivers refuses:

- an SDK write call (``SetValue``, ``SetCurrentEntry``, ``Execute``,
  ``SetIntValue``) anywhere but the driver's ``_put``;
- an assignment, or augmented assignment, to an attribute rooted at the
  camera handle (``self.active`` or a name bound to it) or to a ``Value``:
  pypylon's deprecated ``camera.Node = value`` and ``node.Value = value``
  writes;
- a ``write_node`` whose node name is not a literal, but
  ``_write_reverse``'s, whose names are the connect-time orientation's
  (``orientation_owed``);
- a node name written by a second function, but the exceptions named.
"""

from __future__ import annotations

import ast
from collections import defaultdict

from tests.ast_seams import MemberNames, Words, parse_module

_PYLON = 'src/_lumascope/drivers/pyloncamera.py'
_IDS = 'src/_lumascope/drivers/idscamera.py'
_DRIVERS = {_PYLON: 'PylonCamera', _IDS: 'IDSCamera'}

_SDK_WRITES = Words({'SetValue', 'SetCurrentEntry', 'Execute', 'SetIntValue'})
_ONE_SDK_CALL = 'write_node'
_NAMES_FROM_A_TABLE = MemberNames((_PYLON, _IDS), {'_write_reverse'})

# Nodes written from more than one function, each for its reason.
_WRITTEN_FROM_TWO = {
    # A lock taken for a stream and released after it, not a setting.
    (_IDS, 'TLParamsLocked'),
    # The window and the binning both zero the offsets, and agree on zero:
    # the binning's zeroing is what the next frame grid reads the sensor's
    # maximum on.
    (_IDS, 'OffsetX'),
    (_IDS, 'OffsetY'),
}


def _is_the_handle(node, aliases) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == 'active'
        and isinstance(node.value, ast.Name)
        and node.value.id == 'self'
    ) or (isinstance(node, ast.Name) and node.id in aliases)


def _handle_aliases(function) -> set[str]:
    """The names a function binds to the camera handle (``camera = self.active``)."""
    return {
        n.targets[0].id
        for n in ast.walk(function)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and _is_the_handle(n.value, set())
    }


def _node_write_target(target, aliases) -> bool:
    """An assignment target that writes a node: an attribute rooted at the
    camera handle (pypylon's ``camera.Node = value``), or a ``Value``."""
    if not isinstance(target, ast.Attribute):
        return False
    if target.attr == 'Value':
        return True
    # The handle itself (``self.active = ...``) is the driver's own state.
    root = target.value
    while isinstance(root, ast.Attribute):
        if _is_the_handle(root, aliases):
            return True
        root = root.value
    return _is_the_handle(root, aliases)


def _functions(tree: ast.Module, class_name: str):
    """Every function of the class, nested ones included, by qualified name."""
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)

    def walk(body, prefix):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f'{prefix}.{node.name}'
                yield name, node
                yield from walk(node.body, name)

    yield from walk(klass.body, class_name)


def _own_nodes(function):
    """The nodes of ``function`` that are not inside a function nested in it."""
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _findings(rel_path: str, tree: ast.Module) -> list[str]:
    class_name = _DRIVERS[rel_path]
    found = []
    writers = defaultdict(set)
    for qualname, function in _functions(tree, class_name):
        short = qualname.rsplit('.', 1)[-1]
        aliases = _handle_aliases(function)
        for node in _own_nodes(function):
            where = f'{rel_path}:{getattr(node, "lineno", "?")} {qualname}'
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if _node_write_target(target, aliases):
                        found.append(f'{where}: an attribute write: {ast.unparse(target)}')
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            attr = node.func.attr
            if attr in _SDK_WRITES and short != '_put':
                found.append(f'{where}: an SDK write outside _put: {ast.unparse(node)}')
            if attr == _ONE_SDK_CALL:
                name = node.args[0] if node.args else None
                if isinstance(name, ast.Constant) and isinstance(name.value, str):
                    writers[name.value].add(qualname)
                elif short not in _NAMES_FROM_A_TABLE:
                    found.append(
                        f'{where}: a write_node of a name not literal: {ast.unparse(node)}'
                    )
    for name, functions in sorted(writers.items()):
        if len(functions) > 1 and (rel_path, name) not in _WRITTEN_FROM_TWO:
            found.append(f'{rel_path}: {name} written by {sorted(functions)}')
    return found


def test_every_camera_node_is_written_through_write_node_by_one_function():
    found = [line for rel_path in _DRIVERS for line in _findings(rel_path, parse_module(rel_path))]
    assert not found, (
        'A camera node written outside write_node, or by a second function; write it '
        'through write_node in its one setter:\n' + '\n'.join(found)
    )


def test_every_node_written_from_two_is_still_written_from_two():
    """An exception no longer needed leaves the list."""
    stale = []
    for rel_path, name in sorted(_WRITTEN_FROM_TWO):
        writers = defaultdict(set)
        for qualname, function in _functions(parse_module(rel_path), _DRIVERS[rel_path]):
            for node in _own_nodes(function):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == _ONE_SDK_CALL
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ):
                    writers[node.args[0].value].add(qualname)
        if len(writers[name]) < 2:
            stale.append(f'{rel_path}: {name} written by {sorted(writers[name])}')
    assert not stale, '\n'.join(stale)


def test_the_walk_finds_each_way_a_node_is_written():
    planted = ast.parse(
        'class PylonCamera:\n'
        '    def _put(self, node, value):\n'
        '        node.SetValue(value)\n'
        '    def gain(self, value):\n'
        "        self.write_node('Gain', value)\n"
        '    def raw(self, value):\n'
        '        self.active.Gain.SetValue(value)\n'
        '    def by_assignment(self):\n'
        '        self.active.AutoFunctionROIUseBrightness = True\n'
        '    def by_alias(self):\n'
        '        camera = self.active\n'
        '        camera.ChunkEnable.Value = True\n'
        '    def computed(self, name):\n'
        '        self.write_node(name, 1)\n'
        '    def second_gain(self):\n'
        "        self.write_node('Gain', 0.0)\n"
        '    def own_state(self):\n'
        '        self._pixel_format_cache = None\n'
    )
    assert _findings(_PYLON, planted) == [
        f'{_PYLON}:7 PylonCamera.raw: an SDK write outside _put: self.active.Gain.SetValue(value)',
        f'{_PYLON}:9 PylonCamera.by_assignment: an attribute write: '
        'self.active.AutoFunctionROIUseBrightness',
        f'{_PYLON}:12 PylonCamera.by_alias: an attribute write: camera.ChunkEnable.Value',
        f'{_PYLON}:14 PylonCamera.computed: a write_node of a name not literal: '
        'self.write_node(name, 1)',
        f"{_PYLON}: Gain written by ['PylonCamera.gain', 'PylonCamera.second_gain']",
    ]
