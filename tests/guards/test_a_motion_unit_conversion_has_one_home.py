# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A motor board's position converts between user units and microsteps in
one place, ``motion_units``.

Each board once carried its own copy of the conversion, and the copies
drifted: one rounded a tie to even where the others rounded it up, two
converted Y by X's row of the config. A function under the drivers that
reads a board's microsteps per mm or its turret slot table is converting,
so only ``motion_units`` reads them, the exceptions below aside, each
named by function.
"""

from __future__ import annotations

import ast

from tests.ast_seams import MemberNames, iter_package_modules, walk_defs

_DRIVERS = 'src/_lumascope/drivers'
_HOME = 'src/_lumascope/drivers/motion_units.py'
_TABLE_READS = MemberNames(
    ('src/_lumascope/drivers/motorconfig.py', 'src/_lumascope/drivers/tmcm6110_config.py'),
    {'usteps_per_mm', 'turret_position_usteps'},
)

# Functions that read a step table and convert no position.
_EXEMPT = {
    # The connect's log line names the Z resolution it loaded.
    ('src/_lumascope/drivers/motorboard.py', 'MotorBoard._load_board_config'),
    # The simulated TMCM-6110 places its own limit switches in its own
    # microstep frame; nothing LVP reads back is converted there.
    ('src/_lumascope/drivers/simulated_tmcm6110.py', 'SimulatedTmcm6110.__init__'),
}


def _own_calls(function):
    """The calls in ``function``'s own body, not in a def nested inside it."""
    stack = list(function.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def _table_readers():
    for rel_path, tree in iter_package_modules((_DRIVERS,)):
        for qualname, function in walk_defs(tree.body):
            if any(
                isinstance(call.func, ast.Attribute) and call.func.attr in _TABLE_READS
                for call in _own_calls(function)
            ):
                yield rel_path, qualname


def test_only_motion_units_converts_a_motor_position():
    readers = set(_table_readers())
    assert {path for path, _ in readers} >= {_HOME}, (
        'the walk found no read in motion_units itself: it is reading nothing'
    )
    stray = sorted(r for r in readers if r[0] != _HOME and r not in _EXEMPT)
    assert not stray, (
        'a function converts a motor position outside motion_units; call '
        f'its position_to_steps / steps_to_position instead: {stray}'
    )


def test_every_exemption_still_reads_a_step_table():
    stale = sorted(_EXEMPT - set(_table_readers()))
    assert not stale, f'exempt functions that no longer read a step table: {stale}'
