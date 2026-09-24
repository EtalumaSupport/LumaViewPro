# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The simulated motor board against a real one, through the API.

The fixture is the board characterization tool's record of an LS850T bench
unit on field firmware: four runs of the same API calls, with each reply and
how long each call took. The simulated board is booted as that unit (its own
config, read from its CONFIG reply), the calls are replayed through the API
in realistic timing, and each result is held to the bench:

- a reply must equal the reply of the one run that began at power-up, as
  the simulated board does; the other runs began where an earlier session
  left the stage, so their replies before the first home are not this
  board's. A position must fall within the bench's positions, give or take
  one microstep.
- a duration must fall within every bench sample of the same call from the
  same state, pooled across runs and repetitions. Four samples of one record
  are too few: correct query times fall outside them half the time.

One repetition of each call is replayed; the pools hold all of them.

Where the model does not yet behave like the board, the check is a strict
xfail naming the difference, so it turns red when the model is fixed and the
marker goes with the fix.
"""

import ast
import functools
import json
import pathlib
import re
import sys
import time

import pytest

import drivers.sim_wire.backend as sim_backend

from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

if not (sys.platform == 'darwin' or sys.platform.startswith('linux')):
    pytest.skip(
        'the firmware-backed simulator runs on macOS and Linux only', allow_module_level=True
    )

FIXTURE = pathlib.Path(__file__).parent / 'data' / 'sim_wire_conformance_ls850t_field.json'
_BENCH = json.loads(FIXTURE.read_text())
_RUNS = _BENCH['runs']
_FRESH = next(run for run in _RUNS if run['fresh_board'])['records']

# The characterization tool's own waits: the multi-line drain, the pause
# between starting a move and stopping it, and the settle after STOP.
_MULTILINE_TIMEOUT_S = 5
_MULTILINE_END_MARKERS = ['T:']
_STOP_AFTER_S = 0.3
_STOP_SETTLE_TIMEOUT_S = 30.0


def _unit_config() -> dict:
    config = ast.literal_eval(next(r['reply'] for r in _FRESH if r['command'] == 'CONFIG'))
    # The unit's XY register table is not in the simulator's image; the
    # field table stands in until the unit's own is added, and XY timing is
    # held as a known difference until then.
    config['IniFiles']['XY'] = 'xymotorconfig.ini'
    return config


_UNIT_CONFIG = _unit_config()


def _replayed(record) -> bool:
    return record.get('rep', 0) == 0


def _state_keys(records) -> list:
    """For each record, the key of the calls that are the same call from the
    same state. A query or a move of a given size is; a turret move is the
    same from the same slot; anything else is only itself."""
    keys = []
    slot = 1  # every turret move follows a turret home
    for index, r in enumerate(records):
        kind = r['kind']
        if kind in ('query', 'query_multiline', 'error_probe'):
            keys.append((kind, r['command']))
        elif kind == 'move':
            keys.append((kind, r['axis'], round(r['distance_um'], 3), r['direction']))
        elif kind == 'turret':
            keys.append((kind, slot, r['slot']))
            slot = r['slot']
        else:
            keys.append((kind, index))
    return keys


_KEYS = _state_keys(_FRESH)


def _pools(field) -> dict:
    pools = {}
    for run in _RUNS:
        for key, record in zip(_KEYS, run['records'], strict=True):
            pools.setdefault(key, []).append(record[field])
    return pools


_DURATIONS = _pools('api_ms')
_POSITIONS = _pools('reply')

_TARGET = re.compile(r'move_absolute\((\w), ([-\d.]+)\)')
_AXIS_ARG = re.compile(r'\((\w+)\)')


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


@pytest.fixture(scope='module')
def replay():
    """Replay the bench's calls on the simulated unit; index -> (reply, ms)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            sim_backend,
            'MotorBoardSpec',
            functools.partial(
                sim_backend.MotorBoardSpec,
                dialect='field',
                timing='realistic',
                unit_config=_UNIT_CONFIG,
            ),
        )
        session = ScopeSession.create(
            complete_settings(simulator_tier='firmware', microscope=_BENCH['unit']['model']),
            simulate=True,
            warn_pre_release=False,
        )
    results = {}
    try:
        for index, record in enumerate(_FRESH):
            if not _replayed(record):
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


def _reply_group(record) -> str:
    kind = record['kind']
    if kind in ('query', 'query_multiline', 'error_probe', 'home'):
        return f'{kind} {record["command"]}'
    if kind == 'turret':
        return 'turret'
    return f'{kind} {record["axis"]}'


def _duration_group(record) -> str | None:
    """Durations are grouped by what sets them. A position read is answered
    from the API's cache and never reaches the board, so it has none."""
    kind = record['kind']
    if kind in ('query', 'error_probe'):
        return 'one-line exchanges'
    if kind == 'query_multiline':
        return 'multi-line drains'
    if kind == 'home':
        return 'homes'
    if kind in ('position', 'move'):
        return f'moves {record["axis"]}'
    if kind == 'turret':
        return 'turret'
    if kind == 'stop_position':
        return None
    return 'STOP'


def _groups(group_of) -> dict:
    groups = {}
    for index, record in enumerate(_FRESH):
        name = group_of(record)
        if _replayed(record) and name is not None:
            groups.setdefault(name, []).append(index)
    return groups


_REPLY_GROUPS = _groups(_reply_group)
_DURATION_GROUPS = _groups(_duration_group)

# What the model does not yet do as the board does. Each is fixed in the
# model, and its entry removed, on its own.
_STALLGUARD = 'StallGuard status (bit 13) is not modelled'
_REPLY_GAPS = {
    'query STATUS_RX': _STALLGUARD,
    'query STATUS_RY': _STALLGUARD,
    'query STATUS_RZ': _STALLGUARD,
    'query STATUS_RT': f"{_STALLGUARD}, nor the turret's two switch bits at power-up",
    'query CONFIG': 'the runtime computes in double precision where the board computes in '
    'single (47.92 prints as 47.92000000000001), and the key order follows the '
    "unit's own motorconfig.json file, which the simulator does not have",
}
# A gap whose outcome varies from run to run cannot be strict.
_STOP_REST = (
    'after STOP mid-move the position sometimes reads up to 94 um short of the target; '
    'on the board it always read within a microstep of it'
)
_FLAKY_REPLY_GAPS = {'stop_position X': _STOP_REST, 'stop_position Y': _STOP_REST}
_DURATION_GAPS = {
    'one-line exchanges': 'the board takes longer to answer than the simulator: a few ms on a '
    'short reply, up to 85 ms on a long one (CONFIG, FULLINFO)',
    'multi-line drains': 'the drain ends a few ms sooner than on the board',
    'homes': 'a home takes about half the time it takes on the board',
    'moves X': "short X moves lack the board's floor of about 160 ms, and the unit's own XY "
    'register table is not in the simulator',
    'moves Y': "the unit's own XY register table is not in the simulator; moves finish a few "
    'percent sooner than on the board',
    'moves Z': 'moves finish 1-5% sooner than on the board',
    'turret': 'a slot change finishes about 5% sooner than on the board',
    'STOP': 'the settle after STOP on X ends about 110 ms sooner than on the board',
}


def _marked(groups: dict, gaps: dict, flaky: dict | None = None) -> list:
    flaky = flaky or {}
    params = []
    for name in groups:
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


def _microstep_um(axis: str) -> float:
    return 1000.0 / _UNIT_CONFIG['Axis Microsteps per mm / Objective'][axis]


@pytest.mark.slow
@pytest.mark.parametrize('group', _marked(_REPLY_GROUPS, _REPLY_GAPS, _FLAKY_REPLY_GAPS))
def test_replies_match_the_bench(replay, group):
    wrong = []
    for index in _REPLY_GROUPS[group]:
        record = _FRESH[index]
        reply, _ms = replay[index]
        if record['kind'] == 'stop_position':
            slack = _microstep_um(record['axis'])
            bench = _POSITIONS[_KEYS[index]]
            if not min(bench) - slack <= reply <= max(bench) + slack:
                wrong.append((record['command'], reply, (min(bench), max(bench))))
        elif reply != record['reply']:
            wrong.append((record['command'], reply, record['reply']))
    assert not wrong, '\n'.join(f'{c}: simulator {s!r}, bench {b!r}' for c, s, b in wrong)


@pytest.mark.slow
@pytest.mark.parametrize('group', _marked(_DURATION_GROUPS, _DURATION_GAPS))
def test_durations_fall_within_the_bench(replay, group):
    outside = []
    for index in _DURATION_GROUPS[group]:
        _reply, ms = replay[index]
        bench = _DURATIONS[_KEYS[index]]
        if not min(bench) <= ms <= max(bench):
            outside.append((_FRESH[index]['command'], round(ms), (min(bench), max(bench))))
    assert not outside, '\n'.join(
        f'{c}: simulator {s} ms, bench {lo:.0f}-{hi:.0f} ms' for c, s, (lo, hi) in outside
    )


def test_a_unit_config_is_booted_exactly_as_it_is():
    spec = sim_backend.MotorBoardSpec('LS850T', frozenset('XYZT'), unit_config=_UNIT_CONFIG)
    assert spec.motorconfig() == _UNIT_CONFIG


def test_a_unit_config_for_another_scope_is_refused():
    with pytest.raises(
        ValueError, match="unit config is a LS850T with axes \\['T', 'X', 'Y', 'Z'\\]"
    ):
        sim_backend.MotorBoardSpec('LS820', frozenset('Z'), unit_config=_UNIT_CONFIG)
