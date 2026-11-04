# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""No production code lists channels by name: the catalogue is the one list.

Every set of channels the program uses -- the transmitted, fluorescence and
luminescence sets, the channels with an LED, which a composite takes -- is
computed from the release catalogue's records. A literal list of channel
keys written in a module is a second list, which a channel the catalogue
adds is missing from: its drawer, settings, composite membership or sweep
silently leave it out. This walk over every production module refuses a
tuple, list, set or mapping naming three or more catalogue keys or old
channel names (``OldChannelNames``): the old names are the release's one
table too, and a list of them is a second one.

Two tables read other programs' files and stay: Quick Enhance's legacy
one-plane colour order (an old RGB TIFF's plane is red, green or blue) and
its filename tokens for foreign images.
"""

from __future__ import annotations

import ast

from modules.layer_record import stored_channel_names
from tests.ast_seams import iter_package_modules, production_modules, walk_defs

_MINIMUM = 3

_FOREIGN_FILE_TABLES = {
    ('modules/quick_enhance.py', 'QuickEnhancer._legacy_false_color_channel'),
    ('modules/quick_enhance.py', 'QuickEnhancer._source_channel'),
}


def _all_production_modules():
    yield from production_modules()
    yield from iter_package_modules(('drivers', 'tools', 'rest'))


def _named_keys(node: ast.AST, keys: frozenset[str]) -> set[str]:
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        elements = node.elts
    elif isinstance(node, ast.Dict):
        elements = [k for k in node.keys if k is not None] + node.values
    else:
        return set()
    return {
        e.value
        for e in elements
        if isinstance(e, ast.Constant) and isinstance(e.value, str) and e.value in keys
    }


def _lists(tree: ast.Module, keys: frozenset[str]):
    """Yield (qualname or '<module>', lineno) for every literal naming enough keys."""
    seen = set()
    for qualname, node in walk_defs(tree.body):
        for inner in ast.walk(node):
            if id(inner) not in seen and len(_named_keys(inner, keys)) >= _MINIMUM:
                seen.add(id(inner))
                yield qualname, inner.lineno
    for inner in ast.walk(tree):
        if id(inner) not in seen and len(_named_keys(inner, keys)) >= _MINIMUM:
            yield '<module>', inner.lineno


def _found():
    keys = frozenset(stored_channel_names())
    for rel_path, tree in _all_production_modules():
        for qualname, lineno in _lists(tree, keys):
            yield rel_path, qualname, lineno


def test_no_production_module_lists_channels_by_name():
    found = [
        f'{rel_path}:{lineno} {qualname}'
        for rel_path, qualname, lineno in _found()
        if (rel_path, qualname) not in _FOREIGN_FILE_TABLES
    ]
    assert found == [], (
        'a list of channel keys leaves out every channel the catalogue adds; '
        f'compute the set from the catalogue records (common_utils): {found}'
    )


def test_the_guard_sees_the_tables_it_exempts():
    # Each exemption must name a real table, or a rename would leave the
    # guard exempting nothing while the check above still passed.
    seen = {(rel_path, qualname) for rel_path, qualname, _lineno in _found()}
    assert seen >= _FOREIGN_FILE_TABLES
