# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""No driver names an LED address.

Which layer an LED address lights is the unit's layer identity's to say
(``IlluminationAPI.ch2color`` / ``color2ch``), and which addresses are lit
is the illumination API's store, keyed by address. A colour table on a
driver was a second answer to the first question, and the one the store
was keyed by: an address the table could not name was written to the
board and recorded nowhere, and the per-colour ``led_ma`` dict was a
second store of the second. A driver takes and reports addresses only.
"""

import ast

from tests.ast_seams import RetiredNames, iter_package_modules

# The names a colour table, its lookups and the driver's state cache went by.
_NAMING = RetiredNames(
    {'ch2color', 'color2ch', 'available_colors', '_COLOR_TO_CH', '_CH_TO_COLOR', 'led_ma'}
)


def _named(node: ast.AST) -> str | None:
    """The name *node* defines or assigns, when it is one a driver may not hold."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        name = node.name
    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        name = node.id
    elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
        name = node.attr
    else:
        return None
    return name if name in _NAMING else None


def test_no_driver_defines_a_colour_table_or_an_led_state_cache():
    offenders = [
        f'{rel_path}:{node.lineno} {name}'
        for rel_path, tree in iter_package_modules(('drivers',))
        for node in ast.walk(tree)
        if (name := _named(node)) is not None
    ]
    assert offenders == [], (
        'a driver takes and reports LED addresses; the layer an address lights is '
        "the identity's (IlluminationAPI.ch2color), and what is lit is the API's store:\n"
        + '\n'.join(offenders)
    )
