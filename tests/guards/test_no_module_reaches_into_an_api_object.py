# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""No module above the Lumascope API reaches a private member of one of its objects.

A layer calls only the one below it, through what that layer offers. The
scope's sub-APIs (``scope.camera``, ``scope.imaging``, ``scope.illumination``
and the rest) hold their mechanisms in private members: the camera lane's
admission, a capture's wait arithmetic, a lane-inline body. A module that
reaches one of them depends on the mechanism rather than the member that
owns it, so the owner can no longer change how it does the work, and the
reach's caller rebuilds part of the owner's decision (a manual still once
summed the capture's wait, the camera's write bound and its own stall
itself). Where a module needs what a private member does, the owner gains
a public member for it.

The walk reads every module of ``src/_lumascope`` outside the API's own
package for an attribute read ``<scope>.<sub-API>._<name>``, the scope
reached as ``scope`` or ``_scope`` on any receiver. A name read through a
local alias of a sub-API is not followed.
"""

import ast

from tests.ast_seams import REPO_ROOT, MemberNames, Words, tracked_files

# The scope's sub-API attributes (``Lumascope.__init__``).
_SUB_APIS = MemberNames(
    'src/_lumascope/modules/lumascope_api/_lumascope.py',
    {'camera', 'imaging', 'illumination', 'motion', 'diagnostics', 'protocols'},
)
# The names a module holds the scope by.
_SCOPE_NAMES = Words({'scope', '_scope'})
_API_PACKAGE = 'src/_lumascope/modules/lumascope_api/'


def _names_the_scope(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in _SCOPE_NAMES
    return isinstance(node, ast.Attribute) and node.attr in _SCOPE_NAMES


def _private_reaches(path: str) -> list[str]:
    tree = ast.parse((REPO_ROOT / path).read_text(encoding='utf-8'))
    return [
        f'{path}:{node.lineno}: {ast.unparse(node)}'
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr.startswith('_')
        and not node.attr.startswith('__')
        and isinstance(node.value, ast.Attribute)
        and node.value.attr in _SUB_APIS
        and _names_the_scope(node.value.value)
    ]


def test_no_module_reads_a_private_member_of_a_scope_sub_api():
    modules = [p for p in tracked_files(['src/_lumascope']) if not p.startswith(_API_PACKAGE)]
    assert len(modules) > 100, f'the walk found {len(modules)} modules; re-check its scope'
    reaches = [reach for path in modules for reach in _private_reaches(path)]
    assert reaches == [], (
        'reaches into an API object; give its owner a public member:\n' + '\n'.join(reaches)
    )
