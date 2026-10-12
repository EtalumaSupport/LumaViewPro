# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A confirmed camera rejection of gain or exposure reaches a non-GUI caller.

Both setters detected the rejection, notified the user and returned nothing, so
the failure existed only as a popup. An SDK, headless or REST caller received
``None`` for a refusal exactly as it did for a success, and the published
contract in ``docs/LumascopeSkills.md`` described these setters as returning
True/False -- false in both directions.

The raise lives on the PUBLIC setters, not on the ``_impl`` bodies. The impls
are the composition primitive several other API methods build on -- the
auto-gain lock, the exposure ceiling, the layer apply -- and are what the
protocol step and the autofocus sweep bind directly. Raising from them would
send a refused gain into the protocol scan loop, which classifies any exception
raised with the boards still connected as transient and abandons the rest of
the scan. The public setters are the L2 surface and have no in-run callers, so
the contract is repaired where it is consumed.

Only an explicit ``False`` is a confirmed rejection. A driver with no
confirmation signal answers ``None``, which is not a refusal and does not
raise. A camera that is not there, before or during the write, is refused
naming it (``tests/test_a_camera_command_with_no_camera_is_refused.py``).
"""

import threading

import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.exceptions import CameraSettingRejected, HardwareCommandRefusedError, MissingPart
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI
from modules.notification_center import Severity
from tests.scope_fakes import give_stub_lanes


@pytest.fixture
def sim_imaging():
    cam = SimulatedCamera()
    cam.active = True
    cam.open_and_start()
    scope = Lumascope.__new__(Lumascope)
    scope._camera_driver = cam
    give_stub_lanes(scope)
    scope._cam_lock = threading.RLock()
    scope._state_lock = threading.RLock()
    imaging = ImagingAPI(scope, cam)
    scope.imaging = imaging
    return imaging, cam


_AUTO_GAIN = {'target_brightness': 0.5, 'min_gain_db': 0.0, 'max_gain_db': 20.0}


def _errors(centre_posts):
    return [n for n in centre_posts if n.severity == Severity.ERROR]


class TestAConfirmedRejectionReachesTheCaller:
    def test_a_refused_gain_raises_by_name(self, sim_imaging, monkeypatch):
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'gain', lambda v: False)

        with pytest.raises(CameraSettingRejected) as excinfo:
            imaging.set_gain_db(7.0)

        assert excinfo.value.setting == 'gain_db'
        assert excinfo.value.requested == 7.0
        assert str(excinfo.value).startswith('The camera rejected the gain change to 7.0 dB.')

    def test_a_refused_exposure_raises_by_name(self, sim_imaging, monkeypatch):
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'exposure_t', lambda v: False)

        with pytest.raises(CameraSettingRejected) as excinfo:
            imaging.set_exposure_ms(25.0)

        assert excinfo.value.setting == 'exposure_ms'
        assert excinfo.value.requested == 25.0
        assert str(excinfo.value).startswith('The camera rejected the exposure change to 25 ms.')

    def test_the_api_shows_nothing_itself(self, sim_imaging, centre_posts, monkeypatch):
        """The raise carries the words; whoever ends its flight shows them, so
        the API posting too would show the one refusal twice."""
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'gain', lambda v: False)

        with pytest.raises(CameraSettingRejected):
            imaging.set_gain_db(7.0)

        assert not _errors(centre_posts), 'the refusal is shown by its reporter, not by the API'

    def test_a_refused_one_shot_auto_gain_raises_by_name(self, sim_scope):
        sim_scope.simulation.refuse_camera_write('auto_gain_once', 'false')

        with pytest.raises(CameraSettingRejected) as excinfo:
            sim_scope.imaging.auto_gain_once(True, 0.5, 0.0, 20.0)

        assert excinfo.value.setting == 'auto_gain'


class TestOnlyAConfirmedRejectionRaises:
    def test_a_driver_without_a_confirmation_signal_does_not_raise(self, sim_imaging, monkeypatch):
        """``None`` means "this driver cannot confirm", never "refused"."""
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'gain', lambda v: None)

        imaging.set_gain_db(7.0)

    def test_a_successful_write_does_not_raise(self, sim_imaging):
        imaging, _cam = sim_imaging

        imaging.set_gain_db(7.0)
        imaging.set_exposure_ms(25.0)

    def test_no_camera_is_refused_naming_it_not_rejected(self, sim_imaging):
        """No camera is not a rejection: it is refused, naming the camera."""
        imaging, cam = sim_imaging
        cam.active = False

        for write in (lambda: imaging.set_gain_db(7.0), lambda: imaging.set_exposure_ms(25.0)):
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

    def test_a_refused_gain_moves_neither_the_cache_nor_the_target(self, sim_imaging, monkeypatch):
        imaging, cam = sim_imaging
        imaging.set_gain_db(3.0)
        monkeypatch.setattr(cam, 'gain', lambda v: False)

        with pytest.raises(CameraSettingRejected):
            imaging.set_gain_db(9.0)

        assert imaging.gain_db_cached == 3.0
        assert imaging.frame_validity.target('gain') == 3.0

    def test_an_applied_gain_moves_both(self, sim_imaging):
        imaging, _cam = sim_imaging
        imaging.set_gain_db(3.0)

        imaging.set_gain_db(9.0)

        assert imaging.gain_db_cached == 9.0
        assert imaging.frame_validity.target('gain') == 9.0

    def test_a_driver_that_cannot_confirm_is_believed(self, sim_imaging, monkeypatch):
        """``None`` is not a refusal, so the request IS recorded -- that is the
        deliberate rule, and it is why a driver that CAN confirm must."""
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'gain', lambda v: None)

        imaging.set_gain_db(9.0)

        assert imaging.gain_db_cached == 9.0


class TestTheImplsStayNonRaising:
    """The in-run composition primitive must not throw into the scan loop.

    An exception here reaches protocol_run_loop's classifier, which sees the
    boards still connected, calls the failure transient, and abandons the rest
    of the scan -- so one refused gain on an early step would discard every
    step after it. The refusal is reported and the caller continues at the
    value the camera actually holds.
    """

    def test_the_gain_impl_reports_without_raising(self, sim_imaging, centre_posts, monkeypatch):
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'gain', lambda v: False)

        assert imaging._set_gain_db_impl(7.0) is False
        assert not _errors(centre_posts), 'the impl answers the refusal; its caller reports it'

    def test_the_exposure_impl_reports_without_raising(
        self, sim_imaging, centre_posts, monkeypatch
    ):
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'exposure_t', lambda v: False)

        assert imaging._set_exposure_ms_impl(25.0) is False
        assert not _errors(centre_posts), 'the impl answers the refusal; its caller reports it'

    def test_an_auto_gain_lock_survives_a_refused_write(self, sim_imaging, monkeypatch):
        """The lock consumes the arm; a raise would strand it, disarmed."""
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'gain', lambda v: False)
        monkeypatch.setattr(cam, 'exposure_t', lambda v: False)

        lock = imaging._lock_auto_gain_impl()

        assert lock is not None


class TestARaisingWriteIsARejection:
    """A driver that raises from a node write is refused like one that
    answers ``False``: every public setter raises ``CameraSettingRejected``
    with the driver's exception as its cause, never the driver's own, and the
    write still marks readiness stale. ``_camera_write`` is the one place it
    is done, so no setter can let a driver's exception out of the API."""

    def test_every_setter_raises_a_chained_rejection(self, sim_scope):
        imaging = sim_scope.imaging
        setters = {
            'gain': lambda: imaging.set_gain_db(9.0),
            'exposure': lambda: imaging.set_exposure_ms(20.0),
            'auto_exposure': lambda: imaging.set_auto_exposure_time(True),
            'frame_size': lambda: imaging.set_frame_size(1024, 1024),
            'binning': lambda: imaging.set_binning_size(2),
            'pixel_format': lambda: imaging.set_pixel_format('Mono12'),
            'black_level': lambda: imaging.set_black_level(10.0),
            'auto_gain': lambda: imaging.set_auto_gain(True, _AUTO_GAIN),
            'auto_gain_once': lambda: imaging.auto_gain_once(True, 0.5, 0.0, 20.0),
            'auto_gain_target_brightness': lambda: imaging.update_auto_gain_target_brightness(0.6),
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
            sim_scope.imaging.set_gain_db(9.0)

        assert validity.invalidation_counts.get('gain', 0) == before + 1
