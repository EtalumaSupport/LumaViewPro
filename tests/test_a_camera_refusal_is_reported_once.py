# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A gain or exposure the camera takes is recorded as what it took; one it
refuses is reported once, by whoever ends the refusal's flight.

The setters answered with the driver's raw result -- ``True`` for gain,
microseconds for exposure -- and recorded the request as the camera's value,
so a driver that clamped, snapped or quantized left the cache, the listeners
and the display naming a value the sensor was not at. And a refusal was
reported three times: the impl logged it and posted a popup, the public setter
raised it, and the run's catcher logged it again.

Now the setters return the value in effect in the caller's unit, the impls
post nothing, the public setters raise, and the in-run catchers -- the image
writer, the autofocus sweep and the step's auto-gain arm -- report through the
one reporter and carry on at the value the camera holds.
"""

import pytest

from modules.exceptions import CameraSettingRejected
from modules.notification_center import Severity


def _rejected(setting='gain_db', requested=7.0):
    return CameraSettingRejected(setting, requested, message='x')


def _posted(centre_posts):
    told = {Severity.ERROR, Severity.WARNING, Severity.INFO}
    return [n for n in centre_posts if n.severity in told]


@pytest.fixture
def reported(monkeypatch):
    calls = []
    monkeypatch.setattr(
        'modules.notification_center.notifications.report_outcome',
        lambda exc, **kw: calls.append((exc, kw)),
    )
    return calls


class TestTheSetterAnswersWithTheValueInEffect:
    def test_a_gain_the_driver_snapped_is_recorded_as_snapped(self, sim_camera, monkeypatch):
        camera, cam = sim_camera
        heard = []
        camera.add_gain_listener(heard.append)
        # The simulated camera applies the gain asked for; a body that snaps
        # it is this member answering another value.
        monkeypatch.setattr(cam, 'gain', lambda v: 12.4)

        assert camera.set_gain_db(12.5) == pytest.approx(12.4)
        assert camera.gain_db == pytest.approx(12.4)
        assert camera.frame_validity.target('gain') == pytest.approx(12.4)
        assert pytest.approx(12.4) in heard

    def test_an_exposure_answers_in_milliseconds(self, sim_camera):
        camera, _cam = sim_camera
        assert camera.set_exposure_ms(33.0) == pytest.approx(33.0)

    def test_an_exposure_the_driver_clamped_is_recorded_as_clamped(self, sim_camera, monkeypatch):
        """A request the body raises to its own floor answers with the floor,
        and the cache says so -- not the request. The request is inside the
        declared range, so it reaches the body."""
        camera, cam = sim_camera
        # The simulated camera applies the exposure asked for; a body that
        # raises it to its floor is this member answering the floor.
        monkeypatch.setattr(cam, 'exposure_t', lambda v: 50.0)

        assert camera.set_exposure_ms(0.02) == pytest.approx(0.05)
        assert camera.exposure_ms == pytest.approx(0.05)


class TestTheApiPostsNothing:
    def test_a_refused_gain_raises_and_posts_nothing(self, sim_scope, centre_posts):
        camera = sim_scope.camera
        sim_scope.simulation.refuse_camera_write('gain', 'false')

        with pytest.raises(CameraSettingRejected):
            camera.set_gain_db(7.0)

        assert _posted(centre_posts) == []

    def test_a_refused_exposure_raises_and_posts_nothing(self, sim_scope, centre_posts):
        camera = sim_scope.camera
        sim_scope.simulation.refuse_camera_write('exposure', 'false')

        with pytest.raises(CameraSettingRejected):
            camera.set_exposure_ms(25.0)

        assert _posted(centre_posts) == []


class TestTheLayerApply:
    def test_a_refused_gain_still_writes_the_exposure_then_raises(self, sim_scope, centre_posts):
        camera = sim_scope.camera
        sim_scope.simulation.refuse_camera_write('gain', 'false')

        with pytest.raises(CameraSettingRejected) as excinfo:
            camera.apply(gain_db=5.0, exposure_ms=40.0, channel='BF')

        assert excinfo.value.setting == 'gain_db'
        assert camera.exposure_ms == pytest.approx(40.0)
        assert _posted(centre_posts) == []


class TestTheRunReportsAndCarriesOn:
    def test_the_autofocus_sweep_reports_a_refused_target_once(self, sim_scope, reported):
        from modules.autofocus_runner import AutofocusRunner
        from modules.lumascope_api.camera import AutoGainLock

        held_gain = sim_scope.camera.gain_db
        sim_scope.simulation.refuse_camera_write('gain', 'false')
        runner = AutofocusRunner(sim_scope)
        runner._camera_gain = 7.0
        runner._camera_exposure = 20.0

        runner._apply_sweep_camera_targets(AutoGainLock(state=None))

        assert [(exc.setting, exc.requested) for exc, _kw in reported] == [('gain_db', 7.0)]
        assert reported[0][1]['solicited'] is False
        assert sim_scope.camera.gain_db == pytest.approx(held_gain), 'the camera keeps its gain'
        assert sim_scope.camera.exposure_ms == pytest.approx(20.0), 'the sweep went on to exposure'

    def test_the_step_auto_gain_arm_reports_a_refusal_and_goes_on(self, reported):
        from tests.protocol_drives import protocol_step, scan_ready_runner

        runner = scan_ready_runner(protocol_step(Auto_Gain=True))
        refusal = _rejected()
        runner._scope.camera.apply.side_effect = refusal

        runner._step_executor.scan_iterate()

        assert [exc for exc, _kw in reported] == [refusal]
        assert reported[0][1]['solicited'] is False
        assert runner._auto_gain_armed_step == 0, 'the arm is recorded, so the step goes on'


AG_SETTINGS = {'target_brightness': 0.3, 'min_gain_db': 0.0, 'max_gain_db': 20.0}


class TestTheApiReportsTheWritesNoCallerHears:
    """The lock's write-back, the ceiling clamp, the live-view re-arm and the
    restore carry on with the camera as it is, so no caller hears a refusal;
    each reports it once, unsolicited, where its flight ends."""

    def test_the_locks_refused_write_back_is_reported(self, sim_scope, reported):
        camera = sim_scope.camera
        camera.set_auto_gain(True, AG_SETTINGS)
        sim_scope.simulation.refuse_camera_write('gain', 'false')

        lock = camera.lock_auto_gain()

        assert [exc.setting for exc, _kw in reported] == ['gain_db']
        assert reported[0][1]['solicited'] is False
        assert lock.state is not None, 'the camera holds the achieved values; the lock stands'

    def test_a_refused_ceiling_clamp_is_reported_and_the_arm_goes_on(self, sim_scope, reported):
        camera = sim_scope.camera
        camera.set_exposure_ms(100.0)
        sim_scope.simulation.refuse_camera_write('exposure', 'false')

        camera.set_auto_gain(True, {**AG_SETTINGS, 'max_exposure_ms': 10.0})

        assert [(exc.setting, exc.requested) for exc, _kw in reported] == [('exposure_ms', 10.0)]
        assert camera._auto_gain_arm is not None

    def test_a_refused_live_view_re_arm_is_reported(self, sim_scope, reported):
        camera = sim_scope.camera
        camera.set_auto_gain(True, AG_SETTINGS, resume_after_capture=True)
        lock = camera.lock_auto_gain()
        sim_scope.simulation.refuse_camera_write('auto_gain', 'false')

        camera._resume_auto_gain_impl(lock)

        assert [(exc.setting, exc.requested) for exc, _kw in reported] == [('auto_gain', True)]

    def test_a_refused_restore_reports_each_write(self, sim_scope, reported):
        camera = sim_scope.camera
        snapshot = camera.save_camera_state('test')
        sim_scope.simulation.refuse_camera_write('gain', 'false')
        sim_scope.simulation.refuse_camera_write('exposure', 'false')

        camera.restore_camera_state(snapshot)

        assert sorted(exc.setting for exc, _kw in reported) == ['exposure_ms', 'gain_db']

    def test_a_refused_restore_re_arm_is_reported(self, sim_scope, reported):
        camera = sim_scope.camera
        camera.set_auto_gain(True, AG_SETTINGS)
        snapshot = camera.save_camera_state('test')
        camera.set_auto_gain(False, AG_SETTINGS)
        sim_scope.simulation.refuse_camera_write('auto_gain', 'false')

        camera.restore_camera_state(snapshot)

        assert [(exc.setting, exc.requested) for exc, _kw in reported] == [('auto_gain', True)]


def test_a_driver_answering_a_bare_true_leaves_the_request_recorded(sim_scope):
    """``bool`` is an ``int``: a driver that answers ``True`` for applied is
    not saying it applied 1 dB."""
    camera = sim_scope.camera
    sim_scope.simulation.refuse_camera_write('gain', 'true')

    assert camera.set_gain_db(12.5) == pytest.approx(12.5)
    assert camera.gain_db == pytest.approx(12.5)
