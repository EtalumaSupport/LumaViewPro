# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The session gives no second name to a member of its scope.

``ScopeSession`` carried unmarked properties whose whole body handed back a
scope member -- ``camera``, ``wellplate_loader``, ``objective_helper`` -- so
an in-process caller reached one object by two names, and the GUI and the
run reached the same camera and catalogues by different ones. A caller
holding a session reaches the scope's members through ``session.scope``; a
declared (``@api``) session door over a scope member is the architecture,
since L2 callers go through the session, and is not read here.

The walk reads one shape: an unmarked method or property whose body is a
single ``return`` of a chain rooted at ``self.scope``. A forward spelled
another way (``getattr``, two statements, a class-level ``property(...)``,
``__getattr__``, a held copy) is not claimed.
"""

from __future__ import annotations

import ast

from tests.ast_seams import MemberNames, parse_module

_SESSION_MODULE = 'modules/scope_session.py'
_SESSION_CLASS = 'ScopeSession'

# The scope's two lanes, which the host's context, the executor bundle and
# the run engine also hold under their own names: decided together in their
# own queue row (NEXT_PRIMARY, "The scope's lanes are held under second
# names"), not one holder at a time here.
_ALLOWED = MemberNames(_SESSION_MODULE, {'io_executor', 'camera_executor'})


def _is_marked(fn: ast.FunctionDef) -> bool:
    for d in fn.decorator_list:
        if isinstance(d, ast.Name) and d.id == 'api':
            return True
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'api':
            return True
    return False


def _is_self_scope(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == 'scope'
        and isinstance(node.value, ast.Name)
        and node.value.id == 'self'
    )


def _forwards_the_scope(value: ast.AST | None) -> bool:
    """Whether ``value`` is ``self.scope.<member>``, any further reads or calls on it."""
    node = value
    while isinstance(node, (ast.Attribute, ast.Call)):
        inner = node.value if isinstance(node, ast.Attribute) else node.func
        if _is_self_scope(inner):
            return True
        node = inner
    return False


def _statements(fn: ast.FunctionDef) -> list[ast.stmt]:
    body = fn.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return body


def forwarders(cls: ast.ClassDef) -> list[str]:
    """The unmarked members of ``cls`` whose whole body returns a ``self.scope`` chain."""
    found = []
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef) or _is_marked(fn):
            continue
        body = _statements(fn)
        if (
            len(body) == 1
            and isinstance(body[0], ast.Return)
            and _forwards_the_scope(body[0].value)
        ):
            found.append(fn.name)
    return found


def _session_class() -> ast.ClassDef:
    (cls,) = [
        node
        for node in parse_module(_SESSION_MODULE).body
        if isinstance(node, ast.ClassDef) and node.name == _SESSION_CLASS
    ]
    return cls


def test_the_session_forwards_no_scope_member_it_does_not_declare():
    extra = sorted(set(forwarders(_session_class())) - _ALLOWED)
    assert extra == [], (
        f'{_SESSION_CLASS} hands back a scope member under its own unmarked name: {extra}. '
        "A caller holding a session reads the scope's member through session.scope; "
        'a member L2 callers need is declared with @api.'
    )


def test_each_allowed_forward_is_still_one():
    # An allowance that no longer names a forward exempts nothing, and is
    # the cue that its row has decided the lanes.
    stale = sorted(_ALLOWED - set(forwarders(_session_class())))
    assert stale == [], f'allowed forwards that are no longer forwards; remove them: {stale}'


def test_the_walk_finds_the_shapes_it_claims():
    planted = ast.parse(
        'class S:\n'
        '    @property\n'
        '    def camera(self):\n'
        '        """The scope\'s camera."""\n'
        '        return self.scope.camera\n'
        '    def lane(self):\n'
        '        return self.scope.io_lane()\n'
        '    @api\n'
        '    @property\n'
        '    def source_path(self):\n'
        '        return self.scope.source_path\n'
        '    @api(in_process=True)\n'
        '    def clock(self):\n'
        '        return self.scope.host_steady_ns()\n'
        '    def own(self):\n'
        '        return self._own\n'
        '    def bare(self):\n'
        '        return self.scope\n'
        '    def two(self):\n'
        '        x = 1\n'
        '        return self.scope.camera\n'
    ).body[0]
    assert forwarders(planted) == ['camera', 'lane']
