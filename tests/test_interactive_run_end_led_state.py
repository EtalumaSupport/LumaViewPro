# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Run-end LED policy is decided by whether the run leaves the user's field.

Every sequenced-capture starter hands ``SequencedCaptureRunner.prepare`` a
``leds_state_at_end`` literal, and the two correct answers depend on one fact
about the run:

* A run that stays at the position the user is already watching (the standalone
  autofocus button, a manual z-stack) is an interruption of live view, not an
  acquisition. It ends by putting the live view back the way the user set it,
  illumination included.
* A run that traverses the plate (autofocus-all-steps, a protocol scan) ends
  dark, so the sample is never left lit between positions or after the run.

The bug this pins: a starter for the first kind copying the second kind's
policy, so a manual operation at one position silently extinguishes the LED the
user was imaging with. Nothing downstream can catch it -- the LED authority
correctly does what the literal tells it -- so the guard has to be on the
literal at the starter.

The runtime half of the invariant (a 'return_to_original' run really does
re-light the pre-run channel, without a blink) is pinned behaviourally by
test_led_lifecycle_sequence.py; this file pins which starters ask for it.
"""

from __future__ import annotations

import ast

import pytest

from tests.ast_seams import parse_module

# starter function -> (module, expected policy, why that policy)
STARTERS = {
    'run_autofocus_from_ui': (
        'ui/vertical_control.py',
        'return_to_original',
        'the standalone autofocus button runs at the current position and '
        'returns the user to the live view they were focusing',
    ),
    'run_zstack_acquire_from_ui': (
        'ui/zstack.py',
        'return_to_original',
        'a manual z-stack runs at the current position and returns to it',
    ),
    'run_autofocus_scan_from_ui': (
        'ui/protocol_settings.py',
        'off',
        'autofocus-all-steps traverses every protocol position; holding the '
        'excitation LED across those moves would photobleach the sample',
    ),
    'run_sequenced_capture': (
        'ui/protocol_settings.py',
        'off',
        'a protocol scan traverses the plate and may end unattended',
    ),
}


def _prepare_policies(module_path: str, func_name: str) -> list[ast.expr]:
    """Every leds_state_at_end argument passed to prepare() inside func_name.

    Walks nested defs too: the starters build their plan inside a local
    prepare_and_start() closure handed to the refusal boundary.
    """
    tree = parse_module(module_path)
    target = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name
        ),
        None,
    )
    assert target is not None, f'{module_path} must define {func_name}'
    return [
        kw.value
        for node in ast.walk(target)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'prepare'
        for kw in node.keywords
        if kw.arg == 'leds_state_at_end'
    ]


@pytest.mark.parametrize('func_name', sorted(STARTERS))
def test_starter_run_end_led_policy_matches_whether_it_leaves_the_field(func_name):
    module_path, expected, why = STARTERS[func_name]
    policies = _prepare_policies(module_path, func_name)

    assert policies, (
        f'{module_path}::{func_name} must pass leds_state_at_end to prepare() '
        'explicitly -- the run-end LED state is never left to a default'
    )
    for value in policies:
        assert isinstance(value, ast.Constant), (
            f'{module_path}::{func_name} must pass a literal leds_state_at_end '
            f'so this invariant is checkable; got {ast.unparse(value)}'
        )
        assert value.value == expected, (
            f'{module_path}::{func_name} passes leds_state_at_end='
            f'{value.value!r}, expected {expected!r}: {why}'
        )
