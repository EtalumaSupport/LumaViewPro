# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""src/_lumascope/lib/ and src/_lumascope/drivers/ import nothing from the layers above them.

The layering is that each layer calls only the one below it: src/_lumascope/lib/ is the
floor, the drivers sit on it, and src/_lumascope/modules/, ui/, src/_lumascope/rest/ and the files at the
repository root sit above both. A lower layer that imports upward cannot
run without the layer above it, and decides something that layer's host
knows.

What is "above" is read from the tree, not listed: every top-level name the
repository tracks -- a package or a root module -- is first-party, and a
lower layer may import only its own floor of those. What the repository does
not track is not ours and is not judged, so no list of third-party packages
is kept here to drift. (The MicroPython stand-ins under src/_lumascope/drivers/sim_wire/mp/
import their siblings by bare name, `import channel`; that passes only
because no root module is named `channel`.)

The walk sees every import at any depth and the literal dynamic ones, in
every tracked file of the package, subpackages and `__init__` included.
The guard this replaced read lines, skipped every indented import as
"deferred", missed `from _lumascope.modules import x`, and listed only the files
directly under each folder, so four upward imports and the logger's reach
into src/_lumascope/modules/ stood under a green suite.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT, imported_names, tracked_files

# Each lower layer, and the first-party names it may import. The API's own
# layers are named inside its package, ``_lumascope.<folder>``.
_MAY_IMPORT = {
    '_lumascope.lib': frozenset({'_lumascope.lib'}),
    '_lumascope.drivers': frozenset({'_lumascope.lib', '_lumascope.drivers'}),
}
_API = '_lumascope'


def _first_party_name(path: str) -> str | None:
    """The name a tracked path is imported under: ``_lumascope.<folder>`` inside the API,
    else its first folder or its own stem; None for a path no import names."""
    parts = path.split('/')
    if parts[0] == 'src':
        parts = parts[1:]
        if parts[0] == _API and len(parts) > 1:
            return f'{_API}.{parts[1].removesuffix(".py")}'
    if len(parts) > 1:
        return parts[0]
    return parts[0][: -len('.py')] if parts[0].endswith('.py') else None


def _first_party_names() -> frozenset[str]:
    """Every name the repository's tracked tree can be imported by."""
    return frozenset(
        name
        for path in tracked_files(pattern='**/*')
        if not path.startswith('.') and (name := _first_party_name(path))
    )


def _head(name: str) -> str:
    """The first-party name an imported module is read under."""
    parts = name.split('.')
    return '.'.join(parts[:2]) if parts[0] == _API else parts[0]


def _layer_files(layer: str) -> list[str]:
    return tracked_files(('src/' + layer.replace('.', '/'),))


def _upward_imports() -> list[str]:
    first_party = _first_party_names()
    found = []
    for layer, allowed in _MAY_IMPORT.items():
        forbidden = first_party - allowed
        for rel_path in _layer_files(layer):
            source = (REPO_ROOT / rel_path).read_text(encoding='utf-8')
            for lineno, name in imported_names(ast.parse(source, filename=rel_path)):
                if _head(name) in forbidden:
                    found.append(f'{rel_path}:{lineno} imports {name}')
    return found


def test_lib_and_the_drivers_import_nothing_above_them():
    found = _upward_imports()
    assert not found, (
        'A lower layer imports from a layer above it. Hand the lower layer what '
        'it needs (an argument, a callback, a name in src/_lumascope/lib/) from the host that '
        'knows it:\n  ' + '\n  '.join(found)
    )


def test_the_walk_reads_every_tracked_file_of_each_layer():
    """The gap that hid the last violations: files the sweep never opened.

    Subpackages and package `__init__` files are where a narrower listing
    stops, so both are pinned here by name.
    """
    assert any(
        path.startswith('src/_lumascope/drivers/sim_wire/')
        for path in _layer_files('_lumascope.drivers')
    )
    assert 'src/_lumascope/lib/__init__.py' in _layer_files('_lumascope.lib')


def test_the_upper_layers_are_first_party():
    """The forbidden set is computed; pin that it holds what it must."""
    assert {
        '_lumascope.modules',
        '_lumascope.rest',
        '_lumascope.lvp_logger',
        'ui',
        'tests',
        'tools',
        'lumaviewpro',
    } <= _first_party_names()
