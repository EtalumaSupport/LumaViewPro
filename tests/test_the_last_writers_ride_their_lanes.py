# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The public members that transmitted beside the lanes now go through them.

Acceleration, the LED restore, the raw diagnostic channel, the fan duty, LED
engineering mode, streaming start and stop, the grab benchmark and the Pylon
probe each wrote to a board or the camera on the caller's thread, so a REST
or script call landed in the middle of a run or a diagnostic with nothing to
refuse it. Each is now one task on its device's lane: refused to anyone not
acting under the holder's taking, and run on the lane's worker for the
holder. The camera temperature read goes on the camera lane too -- it sets a
selector that must not interleave with another camera write -- but a hold
does not refuse it, so the temperature log keeps running through a run.
"""

from unittest.mock import patch

import pytest

from modules import sequential_io_executor
from modules.activity_claim import acting
from modules.exceptions import HardwareCommandRefusedError


@pytest.fixture
def sim_session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    s = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), microscope='LS850T'),
        simulate=True,
    )
    try:
        yield s
    finally:
        s.shutdown()


def _lane_now():
    return getattr(sequential_io_executor._lane_worker, 'executor', None)


# (the member, the call, the driver slot, the driver method it transmits through, its lane)
_GATED = [
    (
        'set_acceleration_limit',
        lambda sc: sc.motion.set_acceleration_limit(val_pct=50),
        '_motion_driver',
        'set_acceleration_limits',
        'io',
    ),
    (
        'restore_led_state',
        lambda sc: sc.illumination.restore_led_state(
            {'tag': 't', 'states': {'Blue': {'enabled': True, 'illumination_ma': 10}}}
        ),
        '_led_driver',
        'led_on',
        'io',
    ),
    (
        'send_diagnostic_command',
        lambda sc: sc.diagnostics.send_diagnostic_command('led', 'INFO'),
        '_led_driver',
        'exchange_command',
        'io',
    ),
    (
        'send_diagnostic_command_multiline',
        lambda sc: sc.diagnostics.send_diagnostic_command_multiline('led', 'INFO', timeout_s=1),
        '_led_driver',
        'exchange_multiline',
        'io',
    ),
    (
        'set_motor_fan_duty',
        lambda sc: sc.diagnostics.set_motor_fan_duty(0),
        '_motion_driver',
        'set_fan_duty',
        'io',
    ),
    (
        'enter_led_engineering_mode',
        lambda sc: sc.diagnostics.enter_led_engineering_mode(timeout_s=1),
        '_led_driver',
        'enter_engineering_mode',
        'io',
    ),
    (
        'exit_led_engineering_mode',
        lambda sc: sc.diagnostics.exit_led_engineering_mode(),
        '_led_driver',
        'exit_engineering_mode',
        'io',
    ),
    (
        'stop_streaming',
        lambda sc: sc.imaging.stop_streaming(),
        '_camera_driver',
        'stop_grabbing',
        'camera',
    ),
    (
        'start_streaming',
        lambda sc: sc.imaging.start_streaming(),
        '_camera_driver',
        'open_and_start',
        'camera',
    ),
    (
        'run_grab_lifecycle_benchmark',
        lambda sc: sc.diagnostics.run_grab_lifecycle_benchmark(num_cycles=1),
        '_camera_driver',
        'stop_grabbing',
        'camera',
    ),
]


def _spy(sc, slot, method):
    """Wrap one driver method so each call records the lane it ran on."""
    driver = getattr(sc, slot)
    real = getattr(driver, method)
    lanes = []

    def _record(*a, **k):
        lanes.append(_lane_now())
        return real(*a, **k)

    return patch.object(driver, method, side_effect=_record), lanes


@pytest.mark.parametrize(
    ('member', 'call', 'slot', 'method', 'lane'), _GATED, ids=[g[0] for g in _GATED]
)
class TestAGatedMember:
    def test_a_non_holder_is_refused_and_nothing_is_sent(
        self, sim_session, member, call, slot, method, lane
    ):
        sc = sim_session.scope
        spy, lanes = _spy(sc, slot, method)
        held = sim_session.activity_claim.try_claim('diagnostic')
        try:
            with spy, pytest.raises(HardwareCommandRefusedError) as refused:
                call(sc)
        finally:
            held.release()
        assert refused.value.holder == 'diagnostic'
        assert lanes == [], f'{member} reached the driver while a diagnostic held the scope'

    def test_the_holder_runs_it_on_the_lane_worker(
        self, sim_session, member, call, slot, method, lane
    ):
        sc = sim_session.scope
        spy, lanes = _spy(sc, slot, method)
        expected = sim_session.io_executor if lane == 'io' else sim_session.camera_executor
        with sim_session.diagnostic_claim() as held, acting(held), spy:
            call(sc)
        assert lanes, f'{member} under the holder never reached the driver'
        assert all(ln is expected for ln in lanes), (
            f'{member} ran on {lanes!r}, not on its lane worker'
        )


class TestThePylonProbe:
    def test_a_non_holder_is_refused(self, sim_session):
        held = sim_session.activity_claim.try_claim('diagnostic')
        try:
            with pytest.raises(HardwareCommandRefusedError):
                sim_session.scope.diagnostics.run_pylon_diagnostic_probe(duration_s=0.1)
        finally:
            held.release()


class TestTheTemperatureRead:
    @pytest.mark.parametrize('kind', ['diagnostic', 'protocol'])
    def test_a_hold_does_not_refuse_it_and_it_runs_on_the_camera_lane(self, sim_session, kind):
        sc = sim_session.scope
        spy, lanes = _spy(sc, '_camera_driver', 'get_all_temperatures')
        held = sim_session.activity_claim.try_claim(kind)
        try:
            with spy:
                temps = sc.diagnostics.get_camera_temperatures_degc()
        finally:
            held.release()
        assert temps, 'the temperature read was refused or returned nothing during a hold'
        assert lanes == [sim_session.camera_executor]

    def test_it_passes_a_runs_protocol_fence(self, sim_session):
        sc = sim_session.scope
        held = sim_session.activity_claim.try_claim('protocol')
        sim_session.camera_executor.protocol_start(held)
        try:
            assert sc.diagnostics.get_camera_temperatures_degc()
        finally:
            sim_session.camera_executor.protocol_end()
            held.release()


class TestAcceleration:
    def test_a_scope_with_no_motor_board_sends_nothing(self, sim_session, monkeypatch):
        sc = sim_session.scope
        monkeypatch.setattr(type(sc), 'motor_connected', property(lambda self: False))
        spy, lanes = _spy(sc, '_motion_driver', 'set_acceleration_limits')
        with spy:
            sc.motion.set_acceleration_limit(val_pct=50)
        assert lanes == []


class TestConfigureScope:
    @pytest.mark.parametrize('kind', ['protocol', 'diagnostic', 'recording'])
    def test_it_is_refused_while_the_scope_is_held(self, sim_session, kind):
        held = sim_session.activity_claim.try_claim(kind)
        try:
            with pytest.raises(HardwareCommandRefusedError) as refused:
                sim_session.configure_scope()
        finally:
            held.release()
        assert refused.value.holder == kind


class TestBringUp:
    def test_the_camera_is_streaming_after_create(self, sim_session):
        assert sim_session.scope.imaging.is_streaming()


class TestTheSupportReportDuringARun:
    def test_it_sends_no_board_command_and_still_records_identity_and_temperature(
        self, sim_session, tmp_path
    ):
        from modules.tech_support_report import TechSupportReport

        sc = sim_session.scope
        report = TechSupportReport(session=sim_session)
        led_cmd, led_sent = _spy(sc, '_led_driver', 'exchange_command')
        led_multi, led_multi_sent = _spy(sc, '_led_driver', 'exchange_multiline')
        run = sim_session.activity_claim.try_claim('protocol')
        try:
            with led_cmd, led_multi:
                sn = report._run_scope_steps(tmp_path, lambda pct, msg: None)
        finally:
            run.release()

        assert led_sent == [] and led_multi_sent == [], 'a board command went out during a run'
        assert sn == sc.diagnostics.get_motor_info()['serial_number']
        motor_info = (tmp_path / 'firmware_info' / 'motor_info.txt').read_text()
        assert sn in motor_info and 'SKIPPED' in motor_info
        for rel in (
            'firmware_configs/config_backup.txt',
            'hardware_checks/tmc5072_registers.txt',
            'hardware_checks/serial_latency.txt',
        ):
            text = (tmp_path / rel).read_text()
            assert 'SKIPPED' in text and 'protocol' in text, (rel, text)
        camera = (tmp_path / 'camera_info' / 'camera_info.txt').read_text()
        assert 'emperature' in camera, camera


class TestBringUpWithLanesNotYetStarted:
    def test_create_does_not_dispatch_the_stream_start(self, tmp_path):
        # A caller may hand create() lanes it has not started yet; a stream
        # start dispatched at bring-up would wait out its whole bound on them.
        import time

        from modules.scope_session import ScopeSession
        from modules.sequential_io_executor import SequentialIOExecutor
        from tests.settings_fixtures import complete_settings

        io = SequentialIOExecutor(name='NOT_STARTED_IO')
        camera = SequentialIOExecutor(name='NOT_STARTED_CAMERA')
        t0 = time.monotonic()
        s = ScopeSession.create(
            complete_settings(live_folder=str(tmp_path), microscope='LS850T'),
            simulate=True,
            io_executor=io,
            camera_executor=camera,
        )
        try:
            assert time.monotonic() - t0 < 10.0
            assert s.scope.imaging.is_streaming()
        finally:
            s.shutdown()
