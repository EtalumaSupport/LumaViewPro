# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Structural guard: the retired L2 spellings stay retired in production.

The L2 surface converged on one word per fact:

    LED drive current   mA        -> illumination_ma
    turret move         tmove     -> move_turret
    turret homed query  has_thomed -> has_turret_homed

Drivers deliberately keep the old vocabulary -- `mA` mirrors the board
command format and `thome` / `has_thomed` are the motor board's own
names -- so `drivers/` is NOT swept. The translation happens at the
facade, whose call into the driver is positional.

This is an AST sweep, not a grep, and the distinction is the whole
point. `mA` is also a legitimate UNIT: it is the value of the
`IlluminationUnit` tag written into every saved TIFF, it appears in the
regex that parses `I_SENS` firmware output, and it labels GUI sliders.
A text search cannot tell those from an identifier; the parser can, so
prose and string literals are invisible here by construction.

Precedent for the shape: the Wave-7 attribute rename claimed 259 sites
and still shipped live misses, each of which produced a user-visible
failure on real hardware. A completed rename needs a guard that fails
if the old name comes back, not a one-time sweep.
"""

import ast

import pytest

from tests.ast_seams import iter_package_modules, parse_module

PROD_PACKAGES = ('modules', 'ui')

RETIRED = {
    'mA': 'illumination_ma',
    'tmove': 'move_turret',
    '_tmove_impl': '_move_turret_impl',
    'has_thomed': 'has_turret_homed',
    '_thome_impl': '_home_turret_impl',
}


def _retired_identifier_hits(tree: ast.AST) -> list[tuple[int, str]]:
    """Every use of a retired name AS AN IDENTIFIER, with its line."""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.arg in RETIRED:
            hits.append((node.lineno, node.arg))
        elif isinstance(node, ast.keyword) and node.arg in RETIRED:
            hits.append((node.value.lineno, node.arg))
        elif isinstance(node, ast.Name) and node.id in RETIRED:
            hits.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and node.attr in RETIRED:
            hits.append((node.lineno, node.attr))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in RETIRED:
            hits.append((node.lineno, node.name))
    return hits


@pytest.mark.parametrize(
    'rel_path, tree',
    list(iter_package_modules(PROD_PACKAGES)),
    ids=lambda v: v if isinstance(v, str) else '',
)
def test_no_retired_l2_spelling_in_production(rel_path, tree):
    hits = _retired_identifier_hits(tree)
    assert not hits, '\n'.join(
        f'{rel_path}:{line} uses retired `{name}` -- the L2 surface spells this '
        f'`{RETIRED[name]}`. If this is driver vocabulary it belongs in drivers/, '
        f'not here.'
        for line, name in hits
    )


def test_the_unit_string_is_untouched():
    """`mA` as a UNIT must survive -- it is written into every saved TIFF.

    This pins the distinction the guard above rests on: banning the
    identifier must never cost the unit. If a future sweep renames this,
    every image this application writes carries a wrong unit.

    Parsed, not read as text: a source-text read would add this file to
    the read-text ratchet for a claim the AST answers exactly.
    """
    tree = parse_module('modules/image_utils.py')
    # The tag is written either as a dict-literal pair or as a subscript
    # assignment (it became conditional when Illumination joined the
    # optional-fields contract); the unit fact is the same in both forms.
    units = [
        value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values, strict=True)
        if isinstance(key, ast.Constant)
        and key.value == 'IlluminationUnit'
        and isinstance(value, ast.Constant)
    ]
    units += [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Subscript)
        and isinstance(node.targets[0].slice, ast.Constant)
        and node.targets[0].slice.value == 'IlluminationUnit'
        and isinstance(node.value, ast.Constant)
    ]
    assert units == ['mA'], (
        f'the TIFF IlluminationUnit tag must still read mA, got {units!r} -- '
        'a saved image records the unit, not the parameter name'
    )
