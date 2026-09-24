# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The fast simulated motor board answers as a real one does.

Most tests run on `SimulatedMotorBoard`, the fast tier: a board written by
hand rather than the real firmware. Its replies are held to the bench
record the firmware tier is held to (`tests/sim_wire_bench.py`), content
only and not timing: the fresh run's calls are replayed through the API
and each reply must equal the board's. A reply the board never gave fails
a test instead of drifting unseen.

A stop position is a position, not a reply, and is held by the firmware
tier's conformance test alone.

Where the fast tier answers otherwise today, the check is a strict xfail
naming the difference, so it turns red when the difference is removed and
the marker goes with it.
"""

import pytest

from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings
from tests.sim_wire_bench import BENCH, FRESH, groups, marked, replay, reply_group


def _content_group(record) -> str | None:
    return None if record['kind'] == 'stop_position' else reply_group(record)


_GROUPS = groups(_content_group)

_IDENTITY = "the fast tier names itself SIMULATED rather than giving the board's identity"
_ERROR_WORDING = "an unknown command is answered 'ERROR: unknown command', not the board's wording"
_BAD_PARAMETER = "a malformed parameter is answered with Python's error, not 'Invalid parameter'"
_NEWER_FIRMWARE = (
    'the field firmware has no such command; the fast tier answers it as newer firmware '
    'does, and the tech-support report tests rely on the answer'
)
_STATUS = (
    "the status before the first home lacks the board's bits 10 and 13 (and the turret's "
    'two switch bits); the driver reads bits 0 and 9, which agree'
)
_GAPS = {
    'query INFO': _IDENTITY,
    'query FULLINFO': _IDENTITY,
    'query CONFIG': 'CONFIG is not answered',
    'query VOLTAGE': _NEWER_FIRMWARE,
    'query FANSPEED': _NEWER_FIRMWARE,
    'query_multiline MOTORDETECT': _NEWER_FIRMWARE,
    'query_multiline CURRENT': _NEWER_FIRMWARE,
    'query_multiline DRVSTAT': _ERROR_WORDING,
    'error_probe NOSUCHCMD': _ERROR_WORDING,
    'error_probe ' + 'Q' * 100: _ERROR_WORDING,
    'error_probe STATUS_RQ': _BAD_PARAMETER,
    'error_probe ACTUAL_R': _BAD_PARAMETER,
    'query STATUS_RX': _STATUS,
    'query STATUS_RY': _STATUS,
    'query STATUS_RZ': _STATUS,
    'query STATUS_RT': _STATUS,
}


@pytest.fixture(scope='module')
def replayed_on_the_fast_tier():
    session = ScopeSession.create(
        complete_settings(simulator_tier='fast', microscope=BENCH['unit']['model']),
        simulate=True,
        warn_pre_release=False,
    )
    return replay(session)


def test_every_known_difference_names_a_replayed_call():
    assert set(_GAPS) <= set(_GROUPS)


@pytest.mark.parametrize('group', marked(_GROUPS, _GAPS))
def test_replies_match_the_bench(replayed_on_the_fast_tier, group):
    wrong = []
    for index in _GROUPS[group]:
        record = FRESH[index]
        reply, _ms = replayed_on_the_fast_tier[index]
        if reply != record['reply']:
            wrong.append((record['command'], reply, record['reply']))
    assert not wrong, '\n'.join(f'{c}: fast tier {s!r}, bench {b!r}' for c, s, b in wrong)
