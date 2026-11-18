# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What counts as a number is ``lib/real_numbers``'s, and no production check restates it.

About eighty checks once wrote "a number" out by hand, and they disagreed:
one let NaN through and another refused it, one took ``True`` as 1, one
refused a numpy integer its neighbour took, one judged a count by type
where the store judged it by value, so a stored sum of ``2.0`` was taken
and then refused at the capture. Every numeric check now asks one of the
four predicates; an ``isinstance`` or ``type()`` test against a numeric
type anywhere else in production code fails here, whether or not it
excludes a bool.

Two shapes are not a test of a number and pass: a type tuple that holds
``str`` beside the numbers asks which scalar kind a value is, and the
functions named below each dispatch on an exact kind, with the reason.
"""

from __future__ import annotations

import ast

from tests.ast_seams import Words, parse_module, tracked_files, walk_defs

_HOME = 'src/_lumascope/lib/real_numbers.py'
# The numeric types, as builtins, ``numbers`` and numpy spell them.
_NUMERIC = Words(
    {'int', 'float', 'Real', 'Integral', 'Rational', 'Number', 'integer', 'floating', 'number'}
)

# Functions that ask which exact kind a value is, not whether it is a number.
_DISPATCHERS = {
    # The store holds only JSON's own types: a numpy scalar is its own kind.
    ('src/_lumascope/modules/settings_paths.py', '_kind'),
    # A non-finite float has no JSON form.
    ('src/_lumascope/modules/wire_encoding.py', '_walk.form'),
    # The protocol's cell parser: an empty cell is a float NaN, an integer
    # cell is read exactly, never through a float.
    ('src/_lumascope/modules/protocol.py', '_position_cell'),
    ('src/_lumascope/modules/protocol.py', '_int_cell'),
}


def _type_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Tuple):
        return set().union(*(_type_names(element) for element in node.elts))
    if isinstance(node, ast.BinOp):
        return _type_names(node.left) | _type_names(node.right)
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    return set()


def _a_number_test(node: ast.AST) -> bool:
    """Whether ``node`` is ``isinstance(x, T)`` or ``type(x) is/is not/in T`` for a numeric ``T``."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'isinstance'
        and len(node.args) == 2
    ):
        names = _type_names(node.args[1])
    elif (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Call)
        and isinstance(node.left.func, ast.Name)
        and node.left.func.id == 'type'
    ):
        names = set().union(*(_type_names(c) for c in node.comparators))
    else:
        return False
    return bool(names & _NUMERIC) and 'str' not in names


def _own_nodes(body):
    """The nodes of ``body``, not those of a def or class nested in it."""
    stack = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _scopes(tree: ast.Module):
    """Every function's own body, by qualified name, and the module's own as ``<module>``."""
    yield '<module>', tree.body
    for qualname, function in walk_defs(tree.body):
        yield qualname, function.body
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            yield f'{node.name} (class body)', node.body


def _hand_written_number_tests():
    paths = [*tracked_files(('src', 'ui', 'tools')), 'lumaviewpro.py']
    for path in paths:
        if path == _HOME:
            continue
        for qualname, body in _scopes(parse_module(path)):
            if (path, qualname) in _DISPATCHERS:
                continue
            for node in _own_nodes(body):
                if _a_number_test(node):
                    yield f'{path}:{node.lineno} {qualname}: {ast.unparse(node)}'


def test_every_numeric_check_asks_a_real_numbers_predicate():
    found = sorted(_hand_written_number_tests())
    assert found == [], 'ask lib/real_numbers instead:\n' + '\n'.join(found)


def test_the_guard_sees_a_hand_written_check():
    # The scan's known positive: each spelling it must catch, and the
    # scalar-kind tuple it must pass.
    tree = ast.parse(
        'def f(x):\n'
        '    a = isinstance(x, (int, float))\n'
        '    b = isinstance(x, bool) or not isinstance(x, numbers.Integral)\n'
        '    c = type(x) is not int\n'
        '    d = isinstance(x, (str, int, float))\n'
    )
    caught = [ast.unparse(n) for n in _own_nodes(tree.body[0].body) if _a_number_test(n)]
    assert sorted(caught) == sorted(
        [
            'isinstance(x, (int, float))',
            'isinstance(x, numbers.Integral)',
            'type(x) is not int',
        ]
    )
