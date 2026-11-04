# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The bench record of an LS850T's motor and LED boards, and its replay
through the API.

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
# Its LED queries read a fixed number of lines, each waiting up to the
# timeout: the INFO banner's, and LEDREAD's, which field firmware lacks.
_LED_QUERY_TIMEOUT_S = 2.0
_LED_QUERY_LINES = {'INFO': 6}
_LED_READ_LINES = 3

# A move's record names its axis and target. A record taken before the plugin
# asked for a move by what it moves says move_absolute(X, 123.000); one taken
# after says move_xy(x_um=123.000) or move_z(123.000). A recorded session is
# what that bench did, so both are read.
_TARGETS = (
    re.compile(r'move_absolute\((?P<axis>\w), (?P<target>[-\d.]+)\)'),
    re.compile(r'move_xy\((?P<axis>[xy])_um=(?P<target>[-\d.]+)\)'),
    re.compile(r'move_(?P<axis>z)\((?P<target>[-\d.]+)\)'),
)
_AXIS_ARG = re.compile(r'\((\w+)\)')
_LED_ON = re.compile(r'led_on\((\d+), ([\d.]+)\)')


def replayed(record) -> bool:
    """One repetition of each call is replayed."""
    return record.get('rep', 0) == 0


def _homed(motion, axis) -> bool:
    """The recorded reply of a home: True when it homed.

    The record was taken when ``home`` answered a bool. It now returns
    when it homed and raises when it did not, so its True is a return.
    """
    motion.home(axis)
    return True


def _target(command) -> tuple[str, float]:
    """The axis and target a move's record names, in either record's words."""
    for pattern in _TARGETS:
        found = pattern.fullmatch(command)
        if found:
            return found['axis'].upper(), float(found['target'])
    raise AssertionError(f'no move in the record {command!r}')


def _moved(motion, axis, target) -> None:
    """The recorded move, waited, through the member that moves ``axis``."""
    if axis == 'Z':
        motion.move_z(target, overshoot_enabled=False)
    else:
        motion.move_xy(**{f'{axis.lower()}_um': target})


def _started(motion, axis, target) -> None:
    """The recorded reply of a move started and not waited on: None.

    The record was taken when an unwaited move returned None. The start now
    returns the started move, which the record never held.
    """
    if axis == 'Z':
        motion.start_move_z(target, overshoot_enabled=False)
    else:
        motion.start_move_xy(**{f'{axis.lower()}_um': target})


def _settled(motion) -> bool:
    """The recorded reply of the stop-settle wait: True when every axis stopped.

    The record was taken when ``wait_until_finished_moving`` answered a
    bool. It now returns when every axis stopped well and raises when one
    did not, so its True is a return.
    """
    motion.wait_until_finished_moving(timeout_s=_STOP_SETTLE_TIMEOUT_S)
    return True


def _call(scope, record):
    """The API call the characterization tool made for this record."""
    if record['board'] == 'led':
        return _led_call(scope, record)
    diag, motion = scope.diagnostics, scope.motion
    kind, command = record['kind'], record['command']
    if kind in ('query', 'error_probe'):
        return lambda: diag.send_diagnostic_command('motor', command)
    if kind == 'query_multiline':
        return lambda: diag.send_diagnostic_command_multiline(
            'motor', command, timeout_s=_MULTILINE_TIMEOUT_S, end_markers=_MULTILINE_END_MARKERS
        )
    if kind == 'home':
        return functools.partial(_homed, motion, _AXIS_ARG.search(command).group(1))
    if kind == 'stop_move_start':
        return functools.partial(_started, motion, *_target(command))
    if kind in ('position', 'move'):
        return functools.partial(_moved, motion, *_target(command))
    if kind == 'turret':
        return functools.partial(motion.move_turret, record['slot'])
    if kind == 'stop':
        return motion.stop_motion
    if kind == 'stop_settle':
        return functools.partial(_settled, motion)
    if kind == 'stop_position':
        return functools.partial(motion.get_current_position, record['axis'])
    raise AssertionError(f'no replay for a {kind!r} record')


def led_on_args(record) -> tuple[int, float]:
    """The board address and current an `led_on` record lit."""
    address, ma = _LED_ON.fullmatch(record['command']).groups()
    return int(address), float(ma)


def _led_call(scope, record):
    diag, illumination = scope.diagnostics, scope.illumination
    kind, command = record['kind'], record['command']
    if kind == 'query':
        return functools.partial(
            diag.send_diagnostic_command,
            'led',
            command,
            response_numlines=_LED_QUERY_LINES.get(command, _LED_READ_LINES),
            timeout_s=_LED_QUERY_TIMEOUT_S,
        )
    if kind == 'error_probe':
        return functools.partial(
            diag.send_diagnostic_command, 'led', command, timeout_s=_LED_QUERY_TIMEOUT_S
        )
    if kind == 'led_on':
        return functools.partial(illumination.led_on_by_address, *led_on_args(record))
    if kind == 'led_off':
        return functools.partial(illumination.led_off_by_address, record['channel'])
    if kind == 'leds_off':
        return illumination.leds_off
    raise AssertionError(f'no replay for an LED {kind!r} record')


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
    if record['board'] == 'led':
        if kind in ('query', 'error_probe'):
            return f'LED {kind} {record["command"]}'
        return f'LED {kind}'
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


# The bench recorded led_on's return when it returned None; it now returns the
# current commanded, so the recording, not the simulator, is out of date. A
# re-recorded bench run passes it, and the strict xfail then fails until the
# entry naming it is removed.
LED_ON_RECORDED_BEFORE_ITS_RETURN = (
    'the bench recording predates led_on returning the commanded current; re-record at the '
    'next bench'
)


def marked(grouped: dict, gaps: dict, flaky: dict | None = None, slow=()) -> list:
    """The group names as test parameters: a known difference is a strict
    xfail naming it, one whose outcome varies from run to run is a
    non-strict one, and a group in ``slow`` is marked so (it needs the
    realistic replay, a minute of real waits)."""
    flaky = flaky or {}
    params = []
    for name in grouped:
        marks = []
        if name in gaps:
            marks.append(pytest.mark.xfail(strict=True, reason=gaps[name]))
        elif name in flaky:
            marks.append(pytest.mark.xfail(strict=False, reason=flaky[name]))
        if name in slow:
            marks.append(pytest.mark.slow)
        params.append(pytest.param(name, marks=marks) if marks else name)
    return params
