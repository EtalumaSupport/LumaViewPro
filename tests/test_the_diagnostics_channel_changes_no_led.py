# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The diagnostics channel changes no LED: only the illumination API does.

``send_diagnostic_command('led', 'LED0_10')`` lit the board while the API
went on believing every LED dark, past the LED record, the lease and frame
validity. Every LED-board command that changes what an LED emits is now
refused before it is sent, matched as both firmwares dispatch it, as the
motor target refuses a command that moves a motor; the reads still pass.
"""

import pytest

from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

# Each firmware's form of every command that changes what an LED emits.
_CHANGERS = (
    'LED0_10',
    'led3_50',
    ' LEDS_OFF ',
    'LED1_OFF',
    'LEDS_ENT',
    'LED2_ENF',
    'LED9_10',
    'RAW0_2000',
    'SELFTEST',
    'selftestslow',
    'CALIBRATE',
    'CALSAVE',
    'CALCLEAR',
    'FWUPDATE',
    'FIGHT',
    '{TON,1,H,50}00',
    '{TOF}00',
    'XLED1_50',
)
_READS = ('INFO', 'LEDREAD0', 'LEDREADS', 'LEDREAD', 'LEDREAD9', 'I2CSCAN', 'ADCREAD', 'NOSUCHCMD')


@pytest.fixture(scope='module')
def session():
    session = ScopeSession.create(
        complete_settings(microscope='LS850'), simulate=True, warn_pre_release=False
    )
    yield session
    session.shutdown()


@pytest.fixture
def sent(session):
    """Every command written to the LED board during the test."""
    board = session.scope._led_driver
    written = []
    single, multi = board.exchange_command, board.exchange_multiline
    board.exchange_command = lambda command, *a, **k: (
        written.append(command),
        single(command, *a, **k),
    )[1]
    board.exchange_multiline = lambda command, *a, **k: (
        written.append(command),
        multi(command, *a, **k),
    )[1]
    yield written
    del board.exchange_command, board.exchange_multiline


def test_every_command_that_changes_an_led_is_refused_unsent(session, sent):
    diag = session.scope.diagnostics
    carried = []
    for command in _CHANGERS:
        for send in (diag.send_diagnostic_command, diag.send_diagnostic_command_multiline):
            try:
                send('led', command)
            except ValueError as refused:
                if 'illumination API' not in str(refused):
                    carried.append((send.__name__, command, str(refused)))
            else:
                carried.append((send.__name__, command, 'sent'))
    assert not carried, carried
    assert sent == []


def test_the_led_reads_are_still_carried(session, sent):
    diag = session.scope.diagnostics
    replies = {command: diag.send_diagnostic_command('led', command) for command in _READS}
    assert all(isinstance(reply, str) for reply in replies.values()), replies
    assert sent == list(_READS)
