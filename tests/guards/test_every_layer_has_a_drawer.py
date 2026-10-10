# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every layer the catalogue names has a drawer in the image-settings panel.

The drawers are built, not derived from the catalogue: five come from the
lookup maps in ``ImageSettings`` and the rest from kv ids. A catalogue layer
with no drawer would have no controls on screen, and the GUI used to skip
its title and its histogram with a DEBUG line. The catalogue and the drawers
are both in the repo, so the gap is refused here, at commit, instead.
"""

from __future__ import annotations

import ast
import json
import re

from tests.ast_seams import REPO_ROOT, find_def


def _catalogue_layers() -> set[str]:
    scopes = json.loads((REPO_ROOT / 'data' / 'scopes.json').read_text())
    models = scopes.get('Models', scopes)
    return {
        layer['key_name']
        for row in models.values()
        if isinstance(row, dict)
        for layer in row.get('Layers', [])
    }


def _mapped(method: str) -> set[str]:
    node = find_def('ui/image_settings.py', method, class_name='ImageSettings')
    assert node is not None, method
    maps = [n for n in ast.walk(node) if isinstance(n, ast.Dict)]
    assert len(maps) == 1, f'{method} is expected to hold one lookup map'
    return {k.value for k in maps[0].keys}


def _kv_ids() -> set[str]:
    kv = (REPO_ROOT / 'ui' / 'lumaviewpro.kv').read_text()
    return set(re.findall(r'^\s*id:\s*(\w+)\s*$', kv, flags=re.MULTILINE))


def _without_a_drawer(layers: set[str]) -> dict[str, list[str]]:
    """Each layer the panel cannot look up, with the lookup that misses it."""
    ids = _kv_ids()
    accordion = _mapped('accordion_item_lookup')
    control = _mapped('layer_lookup')
    missing: dict[str, list[str]] = {}
    for layer in sorted(layers):
        if layer not in accordion and f'{layer}_accordion' not in ids:
            missing.setdefault(layer, []).append('accordion_item_lookup')
        if layer not in control and layer not in ids:
            missing.setdefault(layer, []).append('layer_lookup')
    return missing


def test_every_catalogue_layer_has_a_drawer():
    assert _without_a_drawer(_catalogue_layers()) == {}


def test_a_layer_with_no_drawer_is_named():
    # The guard's known positive: a layer nothing builds is reported by both
    # lookups, so an empty answer above is the catalogue's, not the reader's.
    assert _without_a_drawer({'BF', 'UV'}) == {'UV': ['accordion_item_lookup', 'layer_lookup']}
