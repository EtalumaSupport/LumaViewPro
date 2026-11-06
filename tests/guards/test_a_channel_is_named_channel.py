# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A channel is called a channel: no new identifier says ``layer``.

A channel (``BF``, ``PC``, ``Ex405``, ..., ``Composite``) is named
``channel`` wherever code names it; ``layer`` is the old word for the same
fact, and every member written since the word was ruled chose its own.
The sweep renames the old sites a commit at a time, so this is a
ratchet first: each guarded file holds at most the ``layer`` identifiers
its pin states, and a commit that renames some lowers its file's pin to
what it leaves. When every pin is zero the table goes and this is a ban.

Counted, in the guarded folders (``drivers/`` keeps its own vocabulary):
every def, class, parameter, name, attribute, keyword argument and
imported name or module whose words include ``layer`` or ``layers``,
split on underscores and on case (``LayerControl``, ``_shown_layer``,
``_LAYER_DISPLAY_ORDER``, ``ui.layer_control``). Not counted: string
contents, which no walk can judge (the protocol file's ``Layer`` column
stays as saved), and ``color``, whose channel sites are per-site
judgment, read at each commit's review.

A file over its pin fails with its sites named: rename the new one to
``channel``. A file under its pin fails too, so the pin falls in the
commit that renamed the sites.
"""

from __future__ import annotations

import ast
import re
from collections import Counter

from tests import ratchets as _ratchets
from tests.ast_seams import Packages, Words, iter_package_modules, parse_module

GUARDED = Packages({'src/_lumascope/modules', 'src/_lumascope/rest', 'src/_lumascope/lib', 'ui'})
GUARDED_MODULES = ('lumaviewpro.py',)

THE_OLD_WORD = Words({'layer', 'layers'})

# Each guarded file holding a `layer` identifier, and how many it holds.
_PIN: dict[str, int] = {
    'lumaviewpro.py': 7,
    'src/_lumascope/modules/scope_display_thread.py': 1,
    'ui/composite_capture.py': 4,
    'ui/histogram.py': 12,
    'ui/image_settings.py': 154,
    'ui/layer_control.py': 79,
    'ui/listener_bridge.py': 12,
    'ui/main_display.py': 4,
    'ui/microscope_settings.py': 30,
    'ui/motion_settings.py': 9,
    'ui/protocol_settings.py': 15,
    'ui/scope_display.py': 17,
    'ui/step_navigation.py': 11,
    'ui/ui_helpers.py': 31,
    'ui/vertical_control.py': 9,
    'ui/zstack.py': 4,
}

_WORD = re.compile(r'[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+')


def _says_layer(identifier: str) -> bool:
    return any(w.lower() in THE_OLD_WORD for w in _WORD.findall(identifier))


def _identifiers(tree: ast.AST):
    """Yield ``(lineno, identifier)`` for every name the module writes or reads."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.lineno, node.name
        elif isinstance(node, ast.arg):
            yield node.lineno, node.arg
        elif isinstance(node, ast.Name):
            yield node.lineno, node.id
        elif isinstance(node, ast.Attribute):
            yield node.lineno, node.attr
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.lineno, node.arg
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.lineno, node.module
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                yield node.lineno, alias.name
                if alias.asname:
                    yield node.lineno, alias.asname


def _sites(tree: ast.AST) -> list[tuple[int, str]]:
    return sorted((line, name) for line, name in _identifiers(tree) if _says_layer(name))


def _guarded_modules():
    yield from iter_package_modules(sorted(GUARDED))
    for path in GUARDED_MODULES:
        yield path, parse_module(path)


def test_no_new_identifier_says_layer():
    sites = {path: _sites(tree) for path, tree in _guarded_modules()}
    counts = Counter({path: len(found) for path, found in sites.items() if found})
    _ratchets.observe(
        'Names: identifiers saying `layer` for a channel',
        sum(counts.values()),
        sum(_PIN.values()),
        'equal',
    )
    over = {path: sites[path] for path in counts if counts[path] > _PIN.get(path, 0)}
    assert not over, (
        'a channel is called `channel`; these files name more `layer` sites '
        'than their pins: '
        + '; '.join(
            f'{path} ({counts[path]} > pin {_PIN.get(path, 0)}): '
            + ', '.join(f'{line} {name}' for line, name in found)
            for path, found in over.items()
        )
    )
    under = {
        path: (counts.get(path, 0), pin) for path, pin in _PIN.items() if counts.get(path, 0) < pin
    }
    assert not under, 'lower these pins to what the rename left (count, pin): ' + repr(under)


def test_the_census_sees_each_shape_of_name():
    source = (
        'from _lumascope.modules.layer_record import release_layers as rl\n'
        'import _lumascope.modules.layer_record\n'
        'class LayerControl: pass\n'
        'def get_layers(layer, *, a_layer=None): pass\n'
        'self._shown_layer = 1\n'
        'x = obj.layer_obj\n'
        'f(layer_configs=1)\n'
        '_LAYER_DISPLAY_ORDER = ()\n'
        'player = relayer = 1\n'
        "s = 'layer'\n"
        'channel = led_channel = 1\n'
    )
    assert Counter(name for _, name in _sites(ast.parse(source))) == Counter(
        [
            '_lumascope.modules.layer_record',
            'release_layers',
            '_lumascope.modules.layer_record',
            'LayerControl',
            'get_layers',
            'layer',
            'a_layer',
            '_shown_layer',
            'layer_obj',
            'layer_configs',
            '_LAYER_DISPLAY_ORDER',
        ]
    )
