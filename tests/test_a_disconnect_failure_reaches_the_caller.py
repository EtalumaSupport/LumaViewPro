# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A part that does not shut down cleanly reaches whoever shut the scope down.

Before, `disconnect` caught each part's failure, logged it, posted its own
popup and returned a bool no caller read. The GUI mutes notifications before
it shuts down, so the popup was never seen, and `shutdown` returned nothing,
so an SDK or REST caller was never told. Now `disconnect` runs every step,
then raises one `ScopeDisconnectError` naming each part that failed; the
paths already failing for another reason report it and keep their own fault.
"""

from unittest.mock import MagicMock

import pytest

from modules.exceptions import MotorStopFailedError, ScopeDisconnectError
from modules.lumascope_api import Lumascope
from modules.notification_center import _TYPED_FAULTS, _UNTYPED_FAULT_BODY
from modules.scope_session import ScopeSession
from tests.scope_fakes import build_scope
from tests.settings_fixtures import complete_settings


@pytest.fixture
def sim_scope():
    """A simulated scope, not streaming; its own disconnect is repeatable."""
    scope = build_scope(simulate=True, warn_pre_release=False)
    yield scope
    scope.disconnect()


def test_the_fault_names_the_part_in_its_own_words_and_is_chained(sim_scope):
    driver_error = RuntimeError('port gone')
    sim_scope._led_driver.disconnect = MagicMock(side_effect=driver_error)

    with pytest.raises(ScopeDisconnectError) as excinfo:
        sim_scope.disconnect()

    fault = excinfo.value
    assert fault.__cause__ is driver_error
    assert fault.causes == {'LED board': driver_error}
    assert 'LED board did not shut down cleanly' in str(fault)
    assert isinstance(fault, _TYPED_FAULTS), 'the person reads its words, not the generic body'
    assert str(fault) != _UNTYPED_FAULT_BODY


def test_two_parts_failing_are_one_fault_naming_both_in_teardown_order(sim_scope):
    led_error, camera_error = RuntimeError('led'), RuntimeError('camera')
    sim_scope._led_driver.disconnect = MagicMock(side_effect=led_error)
    sim_scope._camera_driver.disconnect = MagicMock(side_effect=camera_error)

    with pytest.raises(ScopeDisconnectError) as excinfo:
        sim_scope.disconnect()

    assert excinfo.value.parts == ('LED board', 'camera')
    assert excinfo.value.__cause__ is led_error


def test_a_camera_that_answers_false_is_not_a_failed_part(sim_scope):
    # False also means "already gone": a camera unplugged earlier must not
    # read as a teardown failure when the app closes.
    sim_scope._camera_driver.disconnect = MagicMock(return_value=False)

    sim_scope.disconnect()

    assert sim_scope._camera_driver is None


@pytest.mark.parametrize(
    'stop_error', [MotorStopFailedError(), RuntimeError('bus gone')], ids=['stop', 'other']
)
def test_a_failed_stop_does_not_skip_the_board_teardown(sim_scope, monkeypatch, stop_error):
    from drivers.null_ledboard import NullLEDBoard
    from drivers.null_motorboard import NullMotionBoard

    monkeypatch.setattr(sim_scope.motion, 'stop_motion', MagicMock(side_effect=stop_error))

    with pytest.raises(ScopeDisconnectError) as excinfo:
        sim_scope.disconnect()

    assert excinfo.value.parts == ('motor stop',)
    assert excinfo.value.__cause__ is stop_error
    assert isinstance(sim_scope._led_driver, NullLEDBoard)
    assert isinstance(sim_scope._motion_driver, NullMotionBoard)
    assert sim_scope._camera_driver is None


def test_session_shutdown_raises_it_and_a_second_call_completes(tmp_path):
    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    scope = session.scope
    scope._motion_driver.disconnect = MagicMock(side_effect=RuntimeError('port gone'))

    with pytest.raises(ScopeDisconnectError) as excinfo:
        session.shutdown()

    assert excinfo.value.parts == ('motor board',)
    assert session._shut_down is False
    session.shutdown()
    assert session._shut_down is True


def _disconnect_then_fail(monkeypatch):
    real = Lumascope.disconnect

    def disconnect(self):
        real(self)
        raise ScopeDisconnectError({'camera': None})

    monkeypatch.setattr(Lumascope, 'disconnect', disconnect)


def _record_reports(monkeypatch):
    from modules.notification_center import notifications

    reported = []
    monkeypatch.setattr(
        notifications, 'report_outcome', lambda exception, **kw: reported.append(exception)
    )
    return reported


def test_a_failed_construction_keeps_its_fault_and_reports_the_disconnect(monkeypatch):
    _disconnect_then_fail(monkeypatch)
    reported = _record_reports(monkeypatch)
    construction_error = RuntimeError('construction')
    monkeypatch.setattr(ScopeSession, '__init__', MagicMock(side_effect=construction_error))

    with pytest.raises(RuntimeError) as excinfo:
        ScopeSession.create(complete_settings(), simulate=True)

    assert excinfo.value is construction_error
    assert [type(e) for e in reported] == [ScopeDisconnectError]


def test_a_failed_bring_up_keeps_its_fault_and_reports_the_disconnect(monkeypatch, tmp_path):
    _disconnect_then_fail(monkeypatch)
    reported = _record_reports(monkeypatch)
    configure_error = RuntimeError('configure')
    monkeypatch.setattr(ScopeSession, 'configure_scope', MagicMock(side_effect=configure_error))

    with pytest.raises(RuntimeError) as excinfo:
        ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)

    assert excinfo.value is configure_error
    assert [type(e) for e in reported] == [ScopeDisconnectError]
