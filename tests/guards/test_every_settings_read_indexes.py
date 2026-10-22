# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A reader of the live settings indexes the key and names no second value.

``settings_init.prepare_settings`` is the one preparation of the live
configuration, and ``ScopeSession.create`` prepares a dict handed to it the
same way, so every dict the application runs on carries every key the
shipped template carries, at every depth. A read of ``settings.get(key,
default)`` or ``settings.get(key) or default`` on that store can never see
the key missing: the default is dead code that nonetheless names a second
value for the key, a second store for the setting (#789 was a reader
returning its own default for a key the file did not have). This walk over
every production module finds each store read, reached as ``settings``,
``<x>.settings``, a chain rooted there, or a local bound from one in the
same function, that carries a second value in either form.

A read of the raw file before it is prepared, or of a dict that is not
the store but is reached by the same names, keeps its default:
``_NOT_THE_STORE`` names each by function, with its count and reason. A
key the template does not ship is not read from the store at all; it is a
migration read of the raw file, said so where it is read.
"""

from __future__ import annotations

import ast

from tests.ast_seams import (
    production_modules,
    settings_store_aliases,
    settings_store_root,
    walk_defs,
)

# Each a dict that is not the prepared store, with its reason; a rise is a
# new such read to justify in the same commit.
_NOT_THE_STORE = {
    # The raw file's migration of two retired keys, run before the merge.
    ('modules/image_mode.py', 'migrate_settings_dict'): 2,
    # The report reads the installed file to find the live folder when the
    # application is not running; there is no prepared store to ask.
    ('modules/tech_support_report.py', '_get_capture_dir'): 1,
    # The auto-gain arm's own settings dict, whose bounds are optional.
    ('modules/lumascope_api/camera.py', 'CameraAPI._lock_auto_gain_impl'): 2,
}


def _second_value_reads(fn: ast.AST) -> set[int]:
    """Lines in ``fn`` where a store read names a second value."""
    aliases = settings_store_aliases(fn)

    def is_store_get(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'get'
            and settings_store_root(node.func.value, aliases)
        )

    def names_a_second_value(node: ast.AST) -> bool:
        # ``.get(key, default)``, or ``.get(key) or default``.
        if is_store_get(node):
            return len(node.args) + len(node.keywords) >= 2
        return (
            isinstance(node, ast.BoolOp)
            and isinstance(node.op, ast.Or)
            and is_store_get(node.values[0])
            and len(node.values[0].args) == 1
        )

    return {node.lineno for node in ast.walk(fn) if names_a_second_value(node)}


def _second_value_counts() -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for rel_path, tree in production_modules():
        defs = list(walk_defs(tree.body))
        claimed: set[int] = set()
        # Innermost def first, so a closure's read is its own, not its parent's.
        for qualname, fn in sorted(defs, key=lambda d: -d[1].lineno):
            lines = _second_value_reads(fn) - claimed
            claimed |= lines
            if lines:
                counts[(rel_path, qualname)] = len(lines)
    return counts


def test_the_walk_sees_both_forms_of_a_second_value():
    """The instrument reports each form it must see, before its silence counts."""
    tree = ast.parse(
        'def f(ctx):\n'
        "    video = ctx.settings.get('video', {})\n"
        "    a = video.get('max_duration_s', 60)\n"
        "    b = settings.get('turret_objectives') or {}\n"
        "    c = ctx.settings['frame'].get('width', 1900)\n"
        '    d = ctx.settings.get(key, default=1)\n'
        "    e = ctx.settings['frame']['width']\n"
        "    f = ctx.settings.get('optional_block')\n"
        "    g = other.get('x', 1)\n"
    )
    assert _second_value_reads(tree.body[0]) == {2, 3, 4, 5, 6}


def test_every_settings_read_indexes():
    counts = _second_value_counts()
    over = {site: n for site, n in counts.items() if n > _NOT_THE_STORE.get(site, 0)}
    stale = {
        site: (pinned, counts.get(site, 0))
        for site, pinned in _NOT_THE_STORE.items()
        if counts.get(site, 0) < pinned
    }
    assert not over, (
        'store reads naming a second value for a key the template ships; index the key '
        f'(a key the template does not ship is a raw-file migration read): {over}'
    )
    assert not stale, f'lower the pin to the tree (pinned, found): {stale}'
