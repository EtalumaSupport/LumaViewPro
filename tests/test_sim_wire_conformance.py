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
import subprocess
import sys

import pytest

import drivers.sim_wire.backend as sim_backend

from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.sim_wire_bench import BENCH, FRESH, RUNS, groups, marked, replay, reply_group

if not (sys.platform == 'darwin' or sys.platform.startswith('linux')):
    pytest.skip(
        'the firmware-backed simulator runs on macOS and Linux only', allow_module_level=True
    )

# The replay is a module fixture of about a minute; under xdist each worker
# that draws one of these tests would run it again. One group sends them all
# to one worker (`--dist loadgroup`, in the pytest addopts).
pytestmark = pytest.mark.xdist_group('sim_wire_conformance')


def _unit_config() -> dict:
    config = ast.literal_eval(next(r['reply'] for r in FRESH if r['command'] == 'CONFIG'))
    # The unit's XY register table is not in the simulator's image; the
    # field table stands in until the unit's own is added, and XY timing is
    # held as a known difference until then.
    config['IniFiles']['XY'] = 'xymotorconfig.ini'
    return config


_UNIT_CONFIG = _unit_config()


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


_KEYS = _state_keys(FRESH)


def _pools(field) -> dict:
    pools = {}
    for run in RUNS:
        for key, record in zip(_KEYS, run['records'], strict=True):
            pools.setdefault(key, []).append(record[field])
    return pools


_DURATIONS = _pools('api_ms')
_POSITIONS = _pools('reply')


@pytest.fixture(scope='module')
def replayed_on_the_firmware():
    """The bench's calls replayed on the simulated unit; index -> (reply, ms)."""
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
            complete_settings(simulator_tier='firmware', microscope=BENCH['unit']['model']),
            simulate=True,
            warn_pre_release=False,
        )
    return replay(session)


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


_REPLY_GROUPS = groups(reply_group)
_DURATION_GROUPS = groups(_duration_group)

# What the model does not yet do as the board does. Each is fixed in the
# model, and its entry removed, on its own.
_STALLGUARD = 'StallGuard status (bit 13) is not modelled'
_REPLY_GAPS = {
    'query STATUS_RX': _STALLGUARD,
    'query STATUS_RY': _STALLGUARD,
    'query STATUS_RZ': _STALLGUARD,
    'query STATUS_RT': f"{_STALLGUARD}, nor the turret's two switch bits at power-up",
    'query CONFIG': "the key order follows the unit's own motorconfig.json file, which the "
    'simulator does not have',
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


def _microstep_um(axis: str) -> float:
    return 1000.0 / _UNIT_CONFIG['Axis Microsteps per mm / Objective'][axis]


@pytest.mark.slow
@pytest.mark.parametrize('group', marked(_REPLY_GROUPS, _REPLY_GAPS, _FLAKY_REPLY_GAPS))
def test_replies_match_the_bench(replayed_on_the_firmware, group):
    wrong = []
    for index in _REPLY_GROUPS[group]:
        record = FRESH[index]
        reply, _ms = replayed_on_the_firmware[index]
        if record['kind'] == 'stop_position':
            slack = _microstep_um(record['axis'])
            bench = _POSITIONS[_KEYS[index]]
            if not min(bench) - slack <= reply <= max(bench) + slack:
                wrong.append((record['command'], reply, (min(bench), max(bench))))
        elif reply != record['reply']:
            wrong.append((record['command'], reply, record['reply']))
    assert not wrong, '\n'.join(f'{c}: simulator {s!r}, bench {b!r}' for c, s, b in wrong)


@pytest.mark.slow
@pytest.mark.parametrize('group', marked(_DURATION_GROUPS, _DURATION_GAPS))
def test_durations_fall_within_the_bench(replayed_on_the_firmware, group):
    outside = []
    for index in _DURATION_GROUPS[group]:
        _reply, ms = replayed_on_the_firmware[index]
        bench = _DURATIONS[_KEYS[index]]
        if not min(bench) <= ms <= max(bench):
            outside.append((FRESH[index]['command'], round(ms), (min(bench), max(bench))))
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


@pytest.mark.parametrize('dialect', sim_backend.DIALECTS)
def test_the_runtime_computes_in_single_precision_as_the_board_does(dialect):
    # 2**24 + 1 is the smallest integer a single-precision float cannot hold.
    out = subprocess.run(
        [str(sim_backend.runtime_path(dialect)), '-c', 'print(16777217.0 == 16777216.0, 47.92)'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out == ['True', '47.92']
