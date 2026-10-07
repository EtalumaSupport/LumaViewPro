# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The run loop and the step runner end a scan and end a run one way each.

Six fatal endings each read the run's state, then wrote ERROR outside the
state lock, and three of them swallowed the ValueError that a state
another thread wrote in between turns that write into; two scan endings
did the same with SCANNING -> RUNNING, one with the swallow. The runner's
end_run_fatally and end_scan read and write under the state lock, so the
transition the table refuses is never asked for and nothing is caught.
"""

from __future__ import annotations

import ast
from unittest.mock import MagicMock

import pytest

from modules.protocol_state_machine import ProtocolState
from tests.ast_seams import parse_module
from tests.protocol_drives import bare_capture_runner

_RUN_PATHS = ('modules/protocol_run_loop.py', 'modules/protocol_step_runner.py')


@pytest.mark.parametrize('rel_path', _RUN_PATHS)
def test_the_run_paths_write_no_scan_or_error_state_themselves(rel_path):
    writes = [
        node.lineno
        for node in ast.walk(parse_module(rel_path))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == '_set_state'
        and node.args
        and isinstance(node.args[0], ast.Attribute)
        and node.args[0].attr in ('ERROR', 'RUNNING')
    ]
    assert writes == [], (
        f'{rel_path} writes ERROR or RUNNING outside the state lock at lines {writes}: '
        'end a run with end_run_fatally, a scan with end_scan'
    )


@pytest.mark.parametrize('rel_path', _RUN_PATHS)
def test_the_run_paths_catch_no_refused_transition(rel_path):
    caught = [
        node.lineno
        for node in ast.walk(parse_module(rel_path))
        if isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == 'ValueError'
    ]
    assert caught == [], f'{rel_path} swallows a ValueError at lines {caught}'


def _runner_in(*states):
    runner = bare_capture_runner()
    for state in states:
        runner._set_state(state)
    runner.abort_run_fatal = MagicMock()
    return runner


@pytest.mark.parametrize(
    'states', [(ProtocolState.RUNNING,), (ProtocolState.RUNNING, ProtocolState.SCANNING)]
)
def test_a_fatal_ending_puts_a_live_run_in_error_and_aborts_it(states):
    runner = _runner_in(*states)
    runner.end_run_fatally('motion_timeout', 'Protocol Error', 'timed out')
    assert runner._state is ProtocolState.ERROR
    runner.abort_run_fatal.assert_called_once_with('motion_timeout', 'Protocol Error', 'timed out')


@pytest.mark.parametrize(
    'states',
    [
        (ProtocolState.RUNNING, ProtocolState.COMPLETING),
        (ProtocolState.RUNNING, ProtocolState.ERROR),
    ],
)
def test_a_fatal_ending_of_a_run_already_ending_keeps_its_state_and_still_aborts(states):
    runner = _runner_in(*states)
    runner.end_run_fatally('motion_timeout', 'Protocol Error', 'timed out')
    assert runner._state is states[-1]
    runner.abort_run_fatal.assert_called_once()


def test_a_scan_ends_back_to_running_and_an_ended_run_keeps_its_state():
    runner = _runner_in(ProtocolState.RUNNING, ProtocolState.SCANNING)
    runner.end_scan()
    assert runner._state is ProtocolState.RUNNING

    runner = _runner_in(ProtocolState.RUNNING, ProtocolState.ERROR)
    runner.end_scan()
    assert runner._state is ProtocolState.ERROR
