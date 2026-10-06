"""A camera temperature read that fails raises; an empty answer means no sensor.

The read answered ``{}`` for a camera with no temperature sensor, for an
inactive camera, and for a read that failed, at three layers, so no caller
could tell a camera that has no sensor from one whose read broke. A test that
records the temperature beside a measurement then wrote "no sensor" for a
failure. Now ``capabilities.camera_reports_temperature``, probed from the
camera's temperature node at connect, says whether a sensor exists; ``{}`` is
the answer only where it does not; a failed or impossible read raises
``HardwareError``, without tearing the camera down. Each caller reports the
failure as a temperature failure: the diagnostic snapshot records it in its
field, the support report writes it, the periodic log logs it at its tick.
"""

from unittest.mock import MagicMock, patch

import pytest
from pypylon import genicam

from drivers.exceptions import HardwareError
from drivers.fx2driver import FX2Camera
from tests.camera_fakes import bare_ids_camera, bare_pylon_camera
from tests.scope_fakes import spec_scope
from tests.test_ids_diagnostics import _Nodemap, _temp_nodemap, _TempNode, _TempSelector


def _boom(*_args, **_kwargs):
    raise genicam.RuntimeException('transient node read failure')


# --- Pylon ------------------------------------------------------------------


def test_a_pylon_read_that_fails_raises_and_keeps_the_camera():
    cam = bare_pylon_camera()
    cam.active.GetNodeMap.side_effect = _boom

    with pytest.raises(HardwareError, match='temperature'):
        cam.get_all_temperatures()
    cam._mark_disconnected.assert_not_called()


def test_a_pylon_camera_with_no_temperature_node_has_no_sensor():
    cam = bare_pylon_camera()
    cam.active.GetNodeMap.return_value.GetNode.side_effect = genicam.LogicalErrorException(
        'node not in map'
    )

    assert cam.supports_temperature() is False
    assert cam.get_all_temperatures() == {}


def test_an_inactive_pylon_camera_raises():
    cam = bare_pylon_camera()
    cam.active = None

    with pytest.raises(HardwareError):
        cam.get_all_temperatures()


# --- IDS --------------------------------------------------------------------


class _FailingTemp(_TempNode):
    def Value(self):
        if self._selector.current == 'FpgaCore':
            raise RuntimeError('simulated node read failure')
        return super().Value()


def test_an_ids_sensor_whose_read_fails_raises_and_is_not_dropped():
    cam = bare_ids_camera()
    temps = {'Sensor': 42.5, 'FpgaCore': 55.0}
    selector = _TempSelector(temps, current='Sensor')
    cam.remote_nodemap = _Nodemap(
        special={
            'DeviceTemperatureSelector': selector,
            'DeviceTemperature': _FailingTemp(selector, temps),
        }
    )

    with pytest.raises(HardwareError, match='temperature'):
        cam.get_all_temperatures()
    assert selector.current == 'Sensor'


def test_an_ids_camera_with_no_temperature_node_has_no_sensor():
    cam = bare_ids_camera()
    cam.remote_nodemap = _Nodemap(missing=('DeviceTemperature',))

    assert cam.supports_temperature() is False
    assert cam.get_all_temperatures() == {}


def test_an_ids_camera_with_a_temperature_node_has_a_sensor():
    cam = bare_ids_camera()
    cam.remote_nodemap = _temp_nodemap()

    assert cam.supports_temperature() is True


def test_an_inactive_ids_camera_raises():
    cam = bare_ids_camera()
    cam.active = False

    with pytest.raises(HardwareError):
        cam.get_all_temperatures()


def test_the_ids_snapshot_raises_a_failed_temperature_read():
    cam = bare_ids_camera()
    cam.remote_nodemap = _Nodemap()
    cam._read_stream_stats = lambda: {}

    def _fail():
        raise HardwareError('temperature read failed')

    cam.get_all_temperatures = _fail
    with pytest.raises(HardwareError, match='temperature read failed'):
        cam.read_diagnostic_snapshot(duration_s=0)


# --- The other drivers --------------------------------------------------------


def test_the_classic_camera_has_no_sensor():
    cam = FX2Camera.__new__(FX2Camera)
    assert cam.supports_temperature() is False
    assert cam.get_all_temperatures() == {}


# --- The API and its callers ------------------------------------------------


def test_the_simulated_camera_reports_temperature(sim_scope):
    assert sim_scope.capabilities.camera_reports_temperature is True
    assert sim_scope.diagnostics.get_camera_temperatures_degc()


def test_a_failed_read_reaches_the_api_caller(sim_scope):
    with (
        patch.object(
            sim_scope._camera_driver,
            'get_all_temperatures',
            side_effect=HardwareError('temperature read failed'),
        ),
        pytest.raises(HardwareError, match='temperature read failed'),
    ):
        sim_scope.diagnostics.get_camera_temperatures_degc()


def test_no_active_camera_answers_none_at_the_api(sim_scope):
    """None is not {}: no camera to ask is not a camera without a sensor."""
    sim_scope._camera_driver.active = False

    assert sim_scope.diagnostics.get_camera_temperatures_degc() is None


def test_the_diagnostic_snapshot_records_the_failure_in_its_field(sim_scope):
    with patch.object(
        sim_scope._camera_driver,
        'get_all_temperatures',
        side_effect=HardwareError('temperature read failed'),
    ):
        info = sim_scope.diagnostics.get_camera_diagnostic_info()

    assert info['connected'] is True
    assert info['temperatures'] == 'Error: temperature read failed'


def test_a_failed_read_on_a_tick_reaches_the_scheduler(sim_scope):
    """The session's scheduler reports a raising tick and keeps the schedule."""
    with (
        patch.object(
            sim_scope._camera_driver,
            'get_all_temperatures',
            side_effect=HardwareError('temperature read failed'),
        ),
        pytest.raises(HardwareError, match='temperature read failed'),
    ):
        sim_scope.imaging._log_camera_temps()


def test_a_failed_startup_sample_is_reported_and_the_schedule_starts(sim_scope, centre_posts):
    schedule = MagicMock(return_value='handle')
    with patch.object(
        sim_scope._camera_driver,
        'get_all_temperatures',
        side_effect=HardwareError('temperature read failed'),
    ):
        sim_scope.imaging.start_camera_temp_logging(schedule, MagicMock(), interval_s=60.0)

    reported = [n for n in centre_posts if 'temperature read failed' in n.message]
    assert len(reported) == 1
    assert reported[0].category == 'Camera'
    schedule.assert_called_once()


def test_the_support_report_writes_a_temperature_failure_as_one(tmp_path):
    from modules.tech_support_report import TechSupportReport

    report = TechSupportReport.__new__(TechSupportReport)
    report.scope = spec_scope()
    report.scope.diagnostics.get_camera_diagnostic_info.return_value = {
        'connected': True,
        'model': 'daA3840-45um',
        'temperatures': 'Error: temperature read failed',
    }

    report._step_camera_diagnostics(tmp_path)

    written = (tmp_path / 'camera_info' / 'camera_info.txt').read_text()
    assert 'temperature read failed' in written
    assert not (tmp_path / 'camera_info' / 'no_camera.txt').exists()
