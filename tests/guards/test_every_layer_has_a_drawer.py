# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every layer the catalogue names has a drawer, because no drawer code names a layer.

The image-settings panel builds one ``ChannelDrawer`` per catalogue key and
draws each from its layer's record, so a key added to the catalogue gets a
drawer with no code edit. The drawers were five hand-written classes, two
inline kv items and two lookup maps, and a catalogue layer with none of
them had no controls on screen. A key named in the drawer code is the start
of that again, so it is refused here, at commit. The live view's shader is
held to the same: it shows a colour, never a key's.
"""

from __future__ import annotations

import ast
import json
import re

from tests.ast_seams import REPO_ROOT, parse_module

# The Python modules that build and fill the drawers, and colour the view.
_DRAWER_MODULES = ('ui/image_settings.py', 'ui/shader.py')


def _catalogue_keys() -> set[str]:
    scopes = json.loads((REPO_ROOT / 'data' / 'scopes.json').read_text())
    return {record['key'] for record in scopes['LayerOrder']}


def _keys_named_in_python(keys: set[str], tree: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value in keys
    }


def _keys_named_in_kv(keys: set[str], kv: str) -> set[str]:
    """Keys a kv rule sets as a layer or builds an id from."""
    named = set(re.findall(r"^\s*layer:\s*'(\w+)'", kv, flags=re.MULTILINE))
    named |= set(re.findall(r'^\s*id:\s*(\w+?)_accordion\s*$', kv, flags=re.MULTILINE))
    return named & keys


def _named() -> dict[str, set[str]]:
    keys = _catalogue_keys()
    found = {path: _keys_named_in_python(keys, parse_module(path)) for path in _DRAWER_MODULES}
    found['ui/lumaviewpro.kv'] = _keys_named_in_kv(
        keys, (REPO_ROOT / 'ui' / 'lumaviewpro.kv').read_text()
    )
    return {path: names for path, names in found.items() if names}


def test_no_drawer_code_names_a_catalogue_layer():
    assert _named() == {}


def test_a_layer_named_in_drawer_code_is_found():
    # The guard's known positives, one per form it reads, so an empty answer
    # above is the code's, not the reader's.
    keys = {'BF', 'Lumi'}
    assert _keys_named_in_python(keys, ast.parse("x = {'Lumi': 1}")) == {'Lumi'}
    kv = "<A>:\n\tLayerControl:\n\t\tlayer: 'BF'\n<B>:\n\tid: Lumi_accordion\n"
    assert _keys_named_in_kv(keys, kv) == {'BF', 'Lumi'}
