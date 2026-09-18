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
confirmation signal answers ``None``, and no camera at all is a quiet no-op;
neither is a refusal and neither raises.
"""

import threading

import pytest

from drivers.simulated_camera import SimulatedCamera
from modules.exceptions import CameraSettingRejected
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI


@pytest.fixture
def sim_imaging():
    cam = SimulatedCamera()
    cam.active = True
    cam.open_and_start()
    scope = Lumascope.__new__(Lumascope)
    scope._camera_driver = cam
    scope._camera_executor = None
    scope._cam_lock = threading.RLock()
    scope._state_lock = threading.RLock()
    imaging = ImagingAPI(scope, cam)
    scope.imaging = imaging
    return imaging, cam


@pytest.fixture
def notified(monkeypatch):
    """Capture the popup: it must survive the change, not be replaced by it."""
    captured = []
    monkeypatch.setattr(
        'modules.lumascope_api.imaging.notifications.error',
        lambda *a, **kw: captured.append(a),
    )
    return captured


class TestAConfirmedRejectionReachesTheCaller:
    def test_a_refused_gain_raises_by_name(self, sim_imaging, notified, monkeypatch):
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'gain', lambda v: False)

        with pytest.raises(CameraSettingRejected) as excinfo:
            imaging.set_gain_db(7.0)

        assert excinfo.value.setting == 'gain_db'
        assert excinfo.value.requested == 7.0

    def test_a_refused_exposure_raises_by_name(self, sim_imaging, notified, monkeypatch):
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'exposure_t', lambda v: False)

        with pytest.raises(CameraSettingRejected) as excinfo:
            imaging.set_exposure_ms(25.0)

        assert excinfo.value.setting == 'exposure_ms'
        assert excinfo.value.requested == 25.0

    def test_the_user_is_still_told(self, sim_imaging, notified, monkeypatch):
        """The raise is added to the notification, it does not replace it."""
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'gain', lambda v: False)

        with pytest.raises(CameraSettingRejected):
            imaging.set_gain_db(7.0)

        assert notified, 'a rejection must still reach the user at the GUI'


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

    def test_no_camera_is_a_quiet_no_op(self, sim_imaging):
        imaging, cam = sim_imaging
        cam.active = False

        imaging.set_gain_db(7.0)
        imaging.set_exposure_ms(25.0)


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

    def test_a_refused_gain_moves_neither_the_cache_nor_the_target(
        self, sim_imaging, notified, monkeypatch
    ):
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

    def test_the_gain_impl_reports_without_raising(self, sim_imaging, notified, monkeypatch):
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'gain', lambda v: False)

        assert imaging._set_gain_db_impl(7.0) is False
        assert notified, 'the impl still surfaces the refusal'

    def test_the_exposure_impl_reports_without_raising(self, sim_imaging, notified, monkeypatch):
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'exposure_t', lambda v: False)

        assert imaging._set_exposure_ms_impl(25.0) is False
        assert notified, 'the impl still surfaces the refusal'

    def test_an_auto_gain_lock_survives_a_refused_write(self, sim_imaging, notified, monkeypatch):
        """The lock consumes the arm; a raise would strand it, disarmed."""
        imaging, cam = sim_imaging
        monkeypatch.setattr(cam, 'gain', lambda v: False)
        monkeypatch.setattr(cam, 'exposure_t', lambda v: False)

        lock = imaging._lock_auto_gain_impl()

        assert lock is not None
