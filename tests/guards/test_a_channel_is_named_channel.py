# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A channel is called a channel: no identifier says ``layer``.

A channel (``BF``, ``PC``, ``Ex405``, ..., ``Composite``) is named
``channel`` wherever code names it; ``layer`` is the old word for the same
fact, and every member written before the word was enforced chose its own.

Refused, in the guarded folders (``drivers/`` keeps its own vocabulary):
every def, class, parameter, name, attribute, keyword argument and
imported name or module whose words include ``layer`` or ``layers``,
split on underscores and on case (``LayerControl``, ``_shown_layer``,
``_LAYER_DISPLAY_ORDER``, ``ui.layer_control``). Not read: string
contents, which no walk can judge (the protocol file's ``Layer`` column
stays as saved), and ``color``, whose channel sites are per-site
judgment, read at each commit's review.

A site fails with its file and line named: rename it to ``channel``.
"""

from __future__ import annotations

import ast
import re
from collections import Counter

from tests.ast_seams import Packages, Words, iter_package_modules, parse_module

GUARDED = Packages({'src/_lumascope/modules', 'src/_lumascope/rest', 'src/_lumascope/lib', 'ui'})
GUARDED_MODULES = ('lumaviewpro.py',)

THE_OLD_WORD = Words({'layer', 'layers'})

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


def test_no_identifier_says_layer():
    sites = {path: found for path, tree in _guarded_modules() if (found := _sites(tree))}
    assert not sites, 'a channel is called `channel`; rename these `layer` sites: ' + '; '.join(
        f'{path}: ' + ', '.join(f'{line} {name}' for line, name in found)
        for path, found in sites.items()
    )


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
