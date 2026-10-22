# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Nothing above the Session writes the live settings dict; it calls the writer.

``ScopeSession.update_settings(path, value)`` is the one write: it refuses a
path that is not a setting, a value of the wrong kind or range, and a
setting that has its own Session member. A write into the dict itself takes
none of those checks and no lock, and is the door a REST caller does not
have. This walk over the GUI, the startup module, the built-in plugins and
the tools finds every subscript store, augmented assignment, delete and
mutating call whose target is the live settings -- reached as ``settings``,
``<x>.settings``, or a local bound from one of those in the same function.

The pin holds the writes still waiting for their Session member. Lower an
entry in the commit that gives its setting a member; a rise is a new
direct write and is moved to ``update_settings`` instead.
"""

from __future__ import annotations

import ast

from tests.ast_seams import (
    Words,
    iter_package_modules,
    parse_module,
    settings_store_aliases,
    settings_store_root,
    walk_defs,
)

_MUTATORS = Words(
    {'update', 'pop', 'setdefault', 'append', 'clear', 'extend', 'insert', 'remove', 'popitem'}
)

# Every write above the Session goes through its writer; a site added
# here is a direct write waiting for its Session member, lowered to 0 in
# the commit that gives it one.
_DIRECT_WRITE_PIN: dict[tuple[str, str], int] = {}


def _direct_writes(fn: ast.AST) -> set[int]:
    aliases = settings_store_aliases(fn)
    lines: set[int] = set()
    for node in ast.walk(fn):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = node.targets
        for target in targets:
            if isinstance(target, ast.Subscript) and settings_store_root(target.value, aliases):
                lines.add(node.lineno)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _MUTATORS
            and settings_store_root(node.func.value, aliases)
        ):
            lines.add(node.lineno)
    return lines


def _guarded_modules():
    yield from iter_package_modules(('ui', 'modules/plugins', 'tools'))
    yield 'lumaviewpro.py', parse_module('lumaviewpro.py')


def _direct_write_counts() -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for rel_path, tree in _guarded_modules():
        defs = list(walk_defs(tree.body))
        claimed: set[int] = set()
        # Innermost def first, so a closure's write is its own, not its parent's.
        for qualname, fn in sorted(defs, key=lambda d: -d[1].lineno):
            lines = _direct_writes(fn) - claimed
            claimed |= lines
            if lines:
                counts[(rel_path, qualname)] = len(lines)
    return counts


def test_the_walk_sees_a_direct_write():
    """The instrument reports a write it must see, before its silence counts."""
    tree = ast.parse(
        'def f(ctx):\n'
        "    layer = ctx.settings['BF']\n"
        "    layer['gain_db'] = 3\n"
        "    ctx.settings.setdefault('video', {})\n"
    )
    assert _direct_writes(tree.body[0]) == {3, 4}


def test_nothing_above_the_session_writes_the_live_settings():
    counts = _direct_write_counts()
    over = {site: n for site, n in counts.items() if n > _DIRECT_WRITE_PIN.get(site, 0)}
    stale = {
        site: (pinned, counts.get(site, 0))
        for site, pinned in _DIRECT_WRITE_PIN.items()
        if counts.get(site, 0) < pinned
    }
    assert not over, (
        f'direct writes into the live settings; call update_settings(path, value) instead: {over}'
    )
    assert not stale, f'lower the pin to the tree (pinned, found): {stale}'


def test_inside_the_session_only_its_writer_writes_the_live_settings():
    """Every Session write of the live settings goes through ``_store_setting``.

    One writer is the one place a write can be marked for saving; a Session
    member that subscripts ``self.settings`` itself is a write that place
    never sees.
    """
    tree = parse_module('modules/scope_session.py')
    found = {
        qualname: sorted(lines)
        for qualname, fn in walk_defs(tree.body)
        if not qualname.endswith('._store_setting') and (lines := _direct_writes(fn))
    }
    assert not found, f'write through _store_setting instead (member: lines): {found}'
