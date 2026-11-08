# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A Pylon node is found one way: ``PylonCamera._find_node``.

pypylon's ``GetNode(name)`` raises ``genicam.LogicalErrorException`` for a
node the map does not hold; it never returns ``None``. A lookup that tests
the answer for ``None`` therefore takes an absent node for a failure: a
WARNING on every GigE connect for a USB-only node, a drain error on every
camera without Basler's error nodes, a traceback at ERROR from a bench knob
a USB3 camera does not offer. ``_find_node`` answers absence as ``None``
and lets every other error through. This walk over LVP's production tree
(``src/``, ``ui/``, ``tools/``, the root's modules) refuses a ``GetNode``
call with an argument anywhere else; the no-argument ``GetNode()`` that
turns a node handle into its node is not a lookup and passes.
"""

from __future__ import annotations

import ast

from tests.ast_seams import iter_package_modules, parse_module, tracked_files

_OWNER = ('src/_lumascope/drivers/pyloncamera.py', 'PylonCamera._find_node')


def _production_tree():
    yield from iter_package_modules(('src', 'ui', 'tools'))
    for rel_path in tracked_files(pattern='*.py'):
        yield rel_path, parse_module(rel_path)


def _lookups_outside_the_owner(rel_path: str, tree: ast.Module) -> list[str]:
    """Every ``GetNode(<name>)`` in ``tree`` outside ``_find_node``."""
    found = []

    def walk(node, qualname):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                walk(child, f'{qualname}.{child.name}' if qualname else child.name)
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == 'GetNode'
                and (child.args or child.keywords)
                and (rel_path, qualname) != _OWNER
            ):
                found.append(f'{rel_path}:{child.lineno}: {ast.unparse(child)}')
            walk(child, qualname)

    walk(tree, '')
    return found


def test_no_pylon_node_is_looked_up_but_through_find_node():
    found = [
        line
        for rel_path, tree in _production_tree()
        for line in _lookups_outside_the_owner(rel_path, tree)
    ]
    assert not found, (
        'GetNode(<name>) outside PylonCamera._find_node; an absent node raises there, '
        'so find it with _find_node (or read it with read_node):\n' + '\n'.join(found)
    )


def test_the_walk_finds_a_lookup_in_the_driver_and_in_a_tool():
    planted_driver = ast.parse(
        'class PylonCamera:\n'
        '    @staticmethod\n'
        '    def _find_node(nodemap, name):\n'
        '        return nodemap.GetNode(name)\n'
        '    def connect(self):\n'
        "        return self.nm.GetNode('X')\n"
    )
    planted_tool = ast.parse("def detect(camera):\n    return camera.GetTLNodeMap().GetNode('X')\n")
    handle = ast.parse('def name(nd):\n    return nd.GetNode().GetName()\n')

    assert _lookups_outside_the_owner(_OWNER[0], planted_driver) == [
        f"{_OWNER[0]}:6: self.nm.GetNode('X')"
    ]
    assert _lookups_outside_the_owner('tools/sweep.py', planted_tool) == [
        "tools/sweep.py:2: camera.GetTLNodeMap().GetNode('X')"
    ]
    assert _lookups_outside_the_owner('tools/sweep.py', handle) == []
