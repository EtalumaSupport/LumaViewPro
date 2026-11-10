# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The diagnostic channel answers with the board's reply, or raises.

``send_diagnostic_command`` and ``send_diagnostic_command_multiline``
answered a failure in the reply's own channel -- ``'None'``, ``'No
response'``, ``'Board not connected'``, ``'Error: <msg>'`` -- and a caller
that did not tell them from a reply wrote them as a register value, a
position or a version. Now a board that is not there is refused as every
command for it is, a board that does not answer raises the driver's error,
and a firmware ``ERROR:`` line is the board's reply.
"""

import pytest

from _lumascope.drivers.exceptions import BoardNoReplyError
from _lumascope.modules.exceptions import HardwareCommandRefusedError, MissingPart
from _lumascope.modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings


@pytest.fixture
def scope_of(tmp_path):
    sessions = []

    def make(model):
        session = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path / 'live'), microscope=model),
            simulate=True,
        )
        sessions.append(session)
        return session.scope

    yield make
    for session in sessions:
        session.shutdown()


def test_a_firmware_error_line_is_the_boards_reply(scope_of):
    diagnostics = scope_of('LS850').diagnostics
    assert diagnostics.send_diagnostic_command('motor', 'NOSUCH') == (
        'ERROR: unknown command NOSUCH'
    )


def test_a_board_that_does_not_answer_raises(scope_of):
    scope = scope_of('LS850')
    scope.simulation.drop_replies('motor', command='FULLINFO')
    with pytest.raises(BoardNoReplyError):
        scope.diagnostics.send_diagnostic_command('motor', 'FULLINFO')


@pytest.mark.parametrize(
    'send',
    [
        lambda d: d.send_diagnostic_command('motor', 'INFO'),
        lambda d: d.send_diagnostic_command_multiline('motor', 'INFO'),
        lambda d: d.read_motor_drv_status('X'),
        lambda d: d.read_motor_fan_rpm(),
    ],
    ids=['single', 'multiline', 'drv_status', 'fan_rpm'],
)
def test_a_manual_scope_is_refused_a_motor_read_for_its_missing_motors(scope_of, send):
    with pytest.raises(HardwareCommandRefusedError) as refused:
        send(scope_of('LS620').diagnostics)
    assert (refused.value.reason, refused.value.missing) == ('axis_absent', MissingPart.MOTORS)


def test_a_board_without_text_commands_is_refused_text(scope_of):
    with pytest.raises(HardwareCommandRefusedError) as refused:
        scope_of('LS720').diagnostics.send_diagnostic_command('motor', 'INFO')
    assert (refused.value.reason, refused.value.missing) == (
        'not_on_this_scope',
        MissingPart.TEXT_COMMANDS,
    )


def test_an_unknown_target_is_refused_before_anything_is_sent(scope_of):
    with pytest.raises(ValueError, match="unknown target 'camera'"):
        scope_of('LS850').diagnostics.send_diagnostic_command('camera', 'INFO')
