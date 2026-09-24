# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The bench record of an LS850T motor board, and its replay through the API.

The record is the board characterization tool's: four runs of the same API
calls on a bench unit on field firmware, with each reply and how long each
call took. One run began at power-up (`FRESH`); the others began where an
earlier session left the stage. A test replays the fresh run's calls on a
simulated board and holds the results to the record.
"""

import functools
import json
import pathlib
import re
import time

import pytest

FIXTURE = pathlib.Path(__file__).parent / 'data' / 'sim_wire_conformance_ls850t_field.json'
BENCH = json.loads(FIXTURE.read_text())
RUNS = BENCH['runs']
FRESH = next(run for run in RUNS if run['fresh_board'])['records']

# The characterization tool's own waits: the multi-line drain, the pause
# between starting a move and stopping it, and the settle after STOP.
_MULTILINE_TIMEOUT_S = 5
_MULTILINE_END_MARKERS = ['T:']
_STOP_AFTER_S = 0.3
_STOP_SETTLE_TIMEOUT_S = 30.0

_TARGET = re.compile(r'move_absolute\((\w), ([-\d.]+)\)')
_AXIS_ARG = re.compile(r'\((\w+)\)')


def replayed(record) -> bool:
    """One repetition of each call is replayed."""
    return record.get('rep', 0) == 0


def _call(scope, record):
    """The API call the characterization tool made for this record."""
    diag, motion = scope.diagnostics, scope.motion
    kind, command = record['kind'], record['command']
    if kind in ('query', 'error_probe'):
        return lambda: diag.send_diagnostic_command('motor', command)
    if kind == 'query_multiline':
        return lambda: diag.send_diagnostic_command_multiline(
            'motor', command, timeout_s=_MULTILINE_TIMEOUT_S, end_markers=_MULTILINE_END_MARKERS
        )
    if kind == 'home':
        return functools.partial(motion.home, _AXIS_ARG.search(command).group(1))
    if kind in ('position', 'move', 'stop_move_start'):
        axis, target = _TARGET.fullmatch(command).groups()
        return functools.partial(
            motion.move_absolute,
            axis,
            float(target),
            wait_until_complete=kind != 'stop_move_start',
            overshoot_enabled=False,
        )
    if kind == 'turret':
        return functools.partial(motion.move_turret, record['slot'])
    if kind == 'stop':
        return motion.stop_motion
    if kind == 'stop_settle':
        return functools.partial(
            motion.wait_until_finished_moving, timeout_s=_STOP_SETTLE_TIMEOUT_S
        )
    if kind == 'stop_position':
        return functools.partial(motion.get_current_position, record['axis'])
    raise AssertionError(f'no replay for a {kind!r} record')


def replay(session) -> dict:
    """Replay the fresh run's calls on the session's scope, then shut the
    session down; index -> (reply, ms)."""
    results = {}
    try:
        for index, record in enumerate(FRESH):
            if not replayed(record):
                continue
            call = _call(session.scope, record)
            started = time.perf_counter()
            reply = call()
            results[index] = (reply, (time.perf_counter() - started) * 1000.0)
            if record['kind'] == 'stop_move_start':
                time.sleep(_STOP_AFTER_S)
    finally:
        session.shutdown()
    return results


def reply_group(record) -> str:
    kind = record['kind']
    if kind in ('query', 'query_multiline', 'error_probe', 'home'):
        return f'{kind} {record["command"]}'
    if kind == 'turret':
        return 'turret'
    return f'{kind} {record["axis"]}'


def groups(group_of) -> dict:
    """Group name -> the indices of the replayed records in it; a record
    whose group is None is in none."""
    grouped = {}
    for index, record in enumerate(FRESH):
        name = group_of(record)
        if replayed(record) and name is not None:
            grouped.setdefault(name, []).append(index)
    return grouped


def marked(grouped: dict, gaps: dict, flaky: dict | None = None) -> list:
    """The group names as test parameters: a known difference is a strict
    xfail naming it, and one whose outcome varies from run to run is a
    non-strict one."""
    flaky = flaky or {}
    params = []
    for name in grouped:
        if name in gaps:
            params.append(
                pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=gaps[name]))
            )
        elif name in flaky:
            params.append(
                pytest.param(name, marks=pytest.mark.xfail(strict=False, reason=flaky[name]))
            )
        else:
            params.append(name)
    return params
