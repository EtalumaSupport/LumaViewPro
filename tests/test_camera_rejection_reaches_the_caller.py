# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera setting write that did not apply reaches its caller as one raise.

Both setters once detected a rejection, notified the user and returned
nothing, so the failure existed only as a popup; later the setting bodies
answered ``False`` while a raising driver raised, two channels for one
failure, so a caller that handled one met the other unhandled.

Now every setting write applied or raises ``CameraSettingRejected``, from
``_camera_write``, the one path every write takes: ``'camera_rejected'``
when the camera refused it (an explicit ``False``) or its driver raised,
``'camera_no_answer'`` when the camera answered neither the write nor the
write sent again at once. A refusal is not resent. The API's own callers
that carry on with the camera as it is -- the auto-gain lock, the exposure
clamp, the restore, the re-arm -- catch the one type and report it; the layer
apply runs every write and raises one rejection at the end.

A driver with no confirmation signal answers ``None``, which is not a refusal
and does not raise. A camera that is not there, before or during the write,
is refused naming it (``tests/test_a_camera_command_with_no_camera_is_refused.py``).
"""

import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.exceptions import CameraSettingRejected, HardwareCommandRefusedError, MissingPart
from modules.lumascope_api import Lumascope
from modules.notification_center import Severity
from tests.scope_fakes import (
    bind_settings_like_a_session,
    give_camera_and_imaging,
    give_camera_capabilities,
    give_stub_lanes,
)


@pytest.fixture
def sim_camera():
    cam = SimulatedCamera()
    cam.active = True
    cam.open_and_start()
    scope = Lumascope.__new__(Lumascope)
    bind_settings_like_a_session(scope)
    scope._camera_driver = cam
    give_stub_lanes(scope)
    give_camera_capabilities(scope, cam)
    camera = give_camera_and_imaging(scope)
    return camera, cam


_AUTO_GAIN = {'target_brightness': 0.5, 'min_gain_db': 0.0, 'max_gain_db': 20.0}


def _errors(centre_posts):
    return [n for n in centre_posts if n.severity == Severity.ERROR]


class TestAConfirmedRejectionReachesTheCaller:
    def test_a_refused_gain_raises_by_name(self, sim_camera, monkeypatch):
        camera, cam = sim_camera
        monkeypatch.setattr(cam, 'gain', lambda v: False)

        with pytest.raises(CameraSettingRejected) as excinfo:
            camera.set_gain_db(7.0)

        assert excinfo.value.setting == 'gain_db'
        assert excinfo.value.requested == 7.0
        # What the camera holds, not what happens next: true for a live view
        # and for a protocol step, which then does not capture.
        assert str(excinfo.value) == (
            'The camera rejected the gain change to 7.0 dB. It is still at its previous '
            'gain. Check that the value is within the camera limits.'
        )

    def test_a_refused_exposure_raises_by_name(self, sim_camera, monkeypatch):
        camera, cam = sim_camera
        monkeypatch.setattr(cam, 'exposure_t', lambda v: False)

        with pytest.raises(CameraSettingRejected) as excinfo:
            camera.set_exposure_ms(25.0)

        assert excinfo.value.setting == 'exposure_ms'
        assert excinfo.value.requested == 25.0
        assert str(excinfo.value) == (
            'The camera rejected the exposure change to 25 ms. It is still at its previous '
            'exposure. Check that the value is within the camera limits.'
        )

    def test_the_api_shows_nothing_itself(self, sim_camera, centre_posts, monkeypatch):
        """The raise carries the words; whoever ends its flight shows them, so
        the API posting too would show the one refusal twice."""
        camera, cam = sim_camera
        monkeypatch.setattr(cam, 'gain', lambda v: False)

        with pytest.raises(CameraSettingRejected):
            camera.set_gain_db(7.0)

        assert not _errors(centre_posts), 'the refusal is shown by its reporter, not by the API'

    def test_a_refused_one_shot_auto_gain_raises_by_name(self, sim_scope):
        sim_scope.simulation.refuse_camera_write('auto_gain_once', 'false')

        with pytest.raises(CameraSettingRejected) as excinfo:
            sim_scope.camera.auto_gain_once(True, 0.5, 0.0, 20.0)

        assert excinfo.value.setting == 'auto_gain'


class TestOnlyAConfirmedRejectionRaises:
    def test_a_driver_without_a_confirmation_signal_does_not_raise(self, sim_camera, monkeypatch):
        """``None`` means "this driver cannot confirm", never "refused"."""
        camera, cam = sim_camera
        monkeypatch.setattr(cam, 'gain', lambda v: None)

        camera.set_gain_db(7.0)

    def test_a_successful_write_does_not_raise(self, sim_camera):
        camera, _cam = sim_camera

        camera.set_gain_db(7.0)
        camera.set_exposure_ms(25.0)

    def test_no_camera_is_refused_naming_it_not_rejected(self, sim_camera):
        """No camera is not a rejection: it is refused, naming the camera."""
        camera, cam = sim_camera
        cam.active = False

        for write in (lambda: camera.set_gain_db(7.0), lambda: camera.set_exposure_ms(25.0)):
            with pytest.raises(HardwareCommandRefusedError) as exc:
                write()
            assert exc.value.missing == MissingPart.CAMERA


class TestARefusalIsNotRecordedAsTruth:
    """The API's judgement rule, pinned independently of any one driver.

    Driver-agnostic and it passes before the Pylon fix as well as after: what it
    guards is ``_camera_write``'s rule that only an explicit ``False`` is a
    refusal, and the state consequence that rule carries. A future change that
    loosened the rule -- or that recorded the request before consulting the
    result -- would leave the cache describing a gain the camera is not at and a
    chunk target no frame can ever carry. The driver half of the same contract
    is ``tests/test_pylon_gain_reports_its_rejection.py``.
    """

    def test_a_refused_gain_moves_neither_the_cache_nor_the_target(self, sim_camera, monkeypatch):
        camera, cam = sim_camera
        camera.set_gain_db(3.0)
        monkeypatch.setattr(cam, 'gain', lambda v: False)

        with pytest.raises(CameraSettingRejected):
            camera.set_gain_db(9.0)

        assert camera.gain_db == 3.0
        assert camera.frame_validity.target('gain') == 3.0

    def test_an_applied_gain_moves_both(self, sim_camera):
        camera, _cam = sim_camera
        camera.set_gain_db(3.0)

        camera.set_gain_db(9.0)

        assert camera.gain_db == 9.0
        assert camera.frame_validity.target('gain') == 9.0

    def test_a_driver_that_cannot_confirm_is_believed(self, sim_camera, monkeypatch):
        """``None`` is not a refusal, so the request IS recorded -- that is the
        deliberate rule, and it is why a driver that CAN confirm must."""
        camera, cam = sim_camera
        monkeypatch.setattr(cam, 'gain', lambda v: None)

        camera.set_gain_db(9.0)

        assert camera.gain_db == 9.0


class TestOneChannel:
    """A setting body raises a write that did not apply, unreported: the one
    channel its callers catch, whichever way the driver failed."""

    def test_the_value_bodies_raise_a_refusal_unreported(
        self, sim_camera, centre_posts, monkeypatch
    ):
        camera, cam = sim_camera
        monkeypatch.setattr(cam, 'gain', lambda v: False)
        monkeypatch.setattr(cam, 'exposure_t', lambda v: False)

        for body, setting in (
            (lambda: camera._set_gain_db_impl(7.0), 'gain_db'),
            (lambda: camera._set_exposure_ms_impl(25.0), 'exposure_ms'),
        ):
            with pytest.raises(CameraSettingRejected) as rejected:
                body()
            assert (rejected.value.setting, rejected.value.reason) == (setting, 'camera_rejected')
        assert not _errors(centre_posts), 'the body raises; its caller reports it'

    def test_an_auto_gain_lock_survives_a_refused_write(self, sim_camera, monkeypatch):
        """The lock consumes the arm; a raise would strand it, disarmed."""
        camera, cam = sim_camera
        monkeypatch.setattr(cam, 'gain', lambda v: False)
        monkeypatch.setattr(cam, 'exposure_t', lambda v: False)

        lock = camera._lock_auto_gain_impl()

        assert lock is not None


class TestARaisingWriteIsARejection:
    """A driver that raises from a node write is refused like one that
    answers ``False``: every public setter raises ``CameraSettingRejected``
    with the driver's exception as its cause, never the driver's own, and the
    write still marks readiness stale. ``_camera_write`` is the one place it
    is done, so no setter can let a driver's exception out of the API."""

    def test_every_setter_raises_a_chained_rejection(self, sim_scope):
        camera = sim_scope.camera
        setters = {
            'gain': lambda: camera.set_gain_db(9.0),
            'exposure': lambda: camera.set_exposure_ms(20.0),
            'auto_exposure': lambda: camera.set_auto_exposure(True),
            'frame_size': lambda: camera.set_frame_size(1024, 1024),
            'binning': lambda: camera.set_binning(2),
            'pixel_format': lambda: camera.set_image_mode('12bit_scientific'),
            'black_level': lambda: camera.set_black_level(10.0),
            'auto_gain': lambda: camera.set_auto_gain(True, _AUTO_GAIN),
            'auto_gain_once': lambda: camera.auto_gain_once(True, 0.5, 0.0, 20.0),
            'auto_gain_target_brightness': lambda: camera.update_auto_gain_target_brightness(0.6),
        }
        escaped = {}
        for node, setter in setters.items():
            sim_scope.simulation.refuse_camera_write(node, 'raise')
            try:
                setter()
            except CameraSettingRejected as rejected:
                if not isinstance(rejected.__cause__, RuntimeError):
                    escaped[node] = f'unchained: cause {rejected.__cause__!r}'
            except Exception as other:  # the defect under test: anything else escaped
                escaped[node] = repr(other)
            else:
                escaped[node] = 'no raise'
            sim_scope.simulation.clear()
        assert not escaped, f'a raising write did not reach the caller as a rejection: {escaped}'

    def test_a_raising_gain_write_still_marks_readiness_stale(self, sim_scope):
        validity = sim_scope.imaging.frame_validity
        before = validity.invalidation_counts.get('gain', 0)
        sim_scope.simulation.refuse_camera_write('gain', 'raise')

        with pytest.raises(CameraSettingRejected):
            sim_scope.camera.set_gain_db(9.0)

        assert validity.invalidation_counts.get('gain', 0) == before + 1


class TestAWriteTheCameraDidNotAnswer:
    """Sent once more, at once; not answered again, it is told as not answered."""

    def test_one_unanswered_write_is_sent_again_and_applies(self, sim_scope):
        sim_scope.simulation.refuse_camera_write('gain', 'no_answer')

        assert sim_scope.camera.set_gain_db(9.0) == 9.0
        assert sim_scope.imaging.frame_validity.target('gain') == 9.0

    def test_two_unanswered_raise_no_answer_and_clear_the_target(self, sim_scope, centre_posts):
        camera = sim_scope.camera
        camera.set_gain_db(3.0)
        sim_scope.simulation.refuse_camera_write('gain', 'no_answer', count=2)

        with pytest.raises(CameraSettingRejected) as rejected:
            camera.set_gain_db(9.0)

        assert (rejected.value.reason, rejected.value.title, rejected.value.setting) == (
            'camera_no_answer',
            'Camera Did Not Answer',
            'gain_db',
        )
        assert str(rejected.value).startswith('The camera did not answer a gain change')
        assert type(rejected.value.__cause__).__name__ == 'CameraNoAnswerError'
        # Whether the camera took 9 dB is not known: no gain target is kept, so
        # the next captures settle by frame count rather than match 3 dB.
        assert camera.frame_validity.target('gain') is None
        assert not _errors(centre_posts), 'the raise is the report'

    def test_a_refused_write_is_not_sent_again(self, sim_scope):
        # One refusal injected: sent again, the write would meet the node and
        # apply, and nothing would be raised.
        sim_scope.simulation.refuse_camera_write('gain', 'false')

        with pytest.raises(CameraSettingRejected) as rejected:
            sim_scope.camera.set_gain_db(9.0)

        assert rejected.value.reason == 'camera_rejected'


class TestTheCallersThatGoOn:
    def test_a_capture_whose_lock_write_back_raises_returns_its_frame(
        self, sim_scope, centre_posts
    ):
        """The lock pins the values auto-gain reached, which the camera
        already holds: a write-back that fails is reported, and the capture
        it belongs to goes on."""
        sim_scope.camera.set_auto_gain(True, _AUTO_GAIN)
        sim_scope.simulation.refuse_camera_write('exposure', 'raise')

        frame = sim_scope.imaging.capture_and_wait(timeout_s=5.0)

        assert frame is not None
        assert [(n.title, n.category) for n in _errors(centre_posts)] == [
            ('Camera Setting Not Applied', 'Camera')
        ]

    def test_a_layer_apply_whose_gain_write_raises_still_writes_the_exposure(self, sim_scope):
        camera = sim_scope.camera
        sim_scope.simulation.refuse_camera_write('gain', 'raise')

        with pytest.raises(CameraSettingRejected) as rejected:
            camera.apply_layer_camera_settings(4.0, 30.0, layer='BF')

        assert (rejected.value.setting, rejected.value.reason) == ('gain_db', 'camera_rejected')
        assert camera.exposure_ms == 30.0, 'the exposure was left at a previous layer'

    def test_a_layer_apply_with_any_part_unanswered_is_not_answered(self, sim_scope):
        camera = sim_scope.camera
        sim_scope.simulation.refuse_camera_write('gain', 'no_answer', count=2)
        sim_scope.simulation.refuse_camera_write('exposure', 'false')

        with pytest.raises(CameraSettingRejected) as rejected:
            camera.apply_layer_camera_settings(4.0, 30.0, layer='BF')

        assert (rejected.value.setting, rejected.value.reason, rejected.value.title) == (
            'gain_db,exposure_ms',
            'camera_no_answer',
            'Camera Did Not Answer',
        )


def test_every_setter_raises_a_refusal_as_rejected(sim_scope):
    """An explicit False from any setting write is the one raise, not an
    answer a caller could forget to check."""
    camera = sim_scope.camera
    setters = {
        'exposure': lambda: camera.set_exposure_ms(20.0),
        'auto_exposure': lambda: camera.set_auto_exposure(True),
        'binning': lambda: camera.set_binning(2),
        'pixel_format': lambda: camera.set_image_mode('12bit_scientific'),
        'black_level': lambda: camera.set_black_level(10.0),
        'auto_gain': lambda: camera.set_auto_gain(True, _AUTO_GAIN),
        'auto_gain_target_brightness': lambda: camera.update_auto_gain_target_brightness(0.6),
    }
    answered = {}
    for node, setter in setters.items():
        sim_scope.simulation.refuse_camera_write(node, 'false')
        try:
            setter()
        except CameraSettingRejected as rejected:
            if rejected.reason != 'camera_rejected':
                answered[node] = rejected.reason
        else:
            answered[node] = 'no raise'
        sim_scope.simulation.clear()
    assert not answered, f'a refused write was not raised as camera_rejected: {answered}'


class TestTheRestTheApiCarriesOnFrom:
    def test_a_capture_whose_disarm_and_re_arm_are_refused_returns_its_frame(
        self, sim_scope, centre_posts
    ):
        sim_scope.camera.set_auto_gain(True, _AUTO_GAIN)
        # The lock's disarm and the re-arm after the frame are both auto-gain
        # writes: each is refused once.
        sim_scope.simulation.refuse_camera_write('auto_gain', 'raise', count=2)

        frame = sim_scope.imaging.capture_and_wait(timeout_s=5.0)

        assert frame is not None
        assert [n.title for n in _errors(centre_posts)] == ['Camera Setting Not Applied'] * 2

    def test_an_arm_whose_ceiling_clamp_is_refused_still_arms(self, sim_scope, centre_posts):
        camera = sim_scope.camera
        camera.set_exposure_ms(100.0)
        sim_scope.simulation.refuse_camera_write('exposure', 'raise')

        camera.set_auto_gain(True, {**_AUTO_GAIN, 'max_exposure_ms': 50.0})

        assert camera.lock_auto_gain().state is not None, 'the arm did not go ahead'
        assert [n.title for n in _errors(centre_posts)] == ['Camera Setting Not Applied']

    def test_a_restore_whose_gain_is_refused_still_restores_the_exposure(
        self, sim_scope, centre_posts
    ):
        camera = sim_scope.camera
        camera.set_gain_db(2.0)
        camera.set_exposure_ms(20.0)
        snapshot = camera.save_camera_state('test')
        camera.set_gain_db(6.0)
        camera.set_exposure_ms(40.0)
        sim_scope.simulation.refuse_camera_write('gain', 'raise')

        camera.restore_camera_state(snapshot)

        assert camera.exposure_ms == 20.0
        assert [n.title for n in _errors(centre_posts)] == ['Camera Setting Not Applied']
