# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera without hardware auto-gain is never asked to change it.

On a bench IDS U3-34LxXLS-M every BF LED toggle showed "Camera Setting Not
Applied: The camera did not take the auto-gain change", and every run start
showed it again for the target brightness. The IDS driver's auto-mode
members write nothing; they answered whether the camera has the node, which
this body does not, so even turning auto-gain OFF read as a refusal once
refusals were raised.

The API now reads the camera's capability before any auto-mode write.
Turning a mode off, or applying a layer whose stored preference is on, asks
nothing of such a camera and refuses nothing; asking it to turn a mode on,
or to set that mode's target, is refused and says the camera has no such
mode.
"""

import threading

import pytest

import drivers.camera_profiles as camera_profiles
from drivers.simulated_camera import SimulatedCamera
from modules.exceptions import CameraSettingUnsupportedError
from modules.lumascope_api import Lumascope
from modules.lumascope_api.imaging import ImagingAPI
from tests.scope_fakes import give_stub_lanes
from tests.test_composite_run_e2e import headless_settings, open_composite_session

AG_SETTINGS = {'target_brightness': 0.3, 'min_gain_db': 0.0, 'max_gain_db': 20.0}
_AUTO_MODE_MEMBERS = (
    'auto_gain',
    'auto_gain_once',
    'update_auto_gain_target_brightness',
    'update_auto_gain_min_max',
    'auto_exposure_t',
)


@pytest.fixture
def ag_less_profile(monkeypatch):
    """The simulator's profile, declaring no hardware auto-gain or auto-exposure.

    Patched in the registry, so every camera connected under it -- and the
    scope capabilities built from that camera -- agree, as they do for an
    IDS body.
    """
    monkeypatch.setattr(camera_profiles._simulated, 'has_auto_gain', False)
    monkeypatch.setattr(camera_profiles._simulated, 'has_auto_exposure', False)


@pytest.fixture
def asked(monkeypatch):
    """Every driver auto-mode call, recorded; each answers as the IDS stub did."""
    calls = []
    for name in _AUTO_MODE_MEMBERS:

        def _record(self, *args, _name=name, **kwargs):
            calls.append(_name)
            return False

        monkeypatch.setattr(SimulatedCamera, name, _record)
    return calls


@pytest.fixture
def reported(monkeypatch):
    calls = []
    monkeypatch.setattr(
        'modules.notification_center.notifications.report_outcome',
        lambda exc, **kw: calls.append((exc, kw)),
    )
    return calls


@pytest.fixture
def ag_less_imaging(ag_less_profile, asked):
    cam = SimulatedCamera()
    cam.open_and_start()
    assert cam.profile.has_auto_gain is False
    scope = Lumascope.__new__(Lumascope)
    scope._camera_driver = cam
    give_stub_lanes(scope)
    scope._cam_lock = threading.RLock()
    scope._state_lock = threading.RLock()
    imaging = ImagingAPI(scope, cam)
    scope.imaging = imaging
    yield imaging
    cam.disconnect()


class TestTheLayerApply:
    def test_auto_gain_off_asks_nothing_and_refuses_nothing(self, ag_less_imaging, asked):
        applied = ag_less_imaging.apply_layer_camera_settings(
            gain_db=5.0,
            exposure_ms=10.0,
            auto_gain=False,
            auto_gain_settings=AG_SETTINGS,
            layer='BF',
        )

        assert applied is not None
        assert asked == []

    def test_a_stored_auto_gain_on_is_applied_as_manual(self, ag_less_imaging, asked):
        applied = ag_less_imaging.apply_layer_camera_settings(
            gain_db=5.0,
            exposure_ms=10.0,
            auto_gain=True,
            auto_gain_settings=AG_SETTINGS,
            layer='BF',
        )

        assert applied is not None
        assert asked == []
        assert ag_less_imaging._auto_gain_arm is None


class TestTheAnswerTheGuiDisplays:
    def test_a_stored_preference_is_off_on_a_camera_without_auto_gain(self, ag_less_imaging):
        held = ag_less_imaging.applied_auto_gain_for(True)
        assert (held.stored, held.applied, held.capped) == (True, False, True)
        off = ag_less_imaging.applied_auto_gain_for(False)
        assert (off.stored, off.applied, off.capped) == (False, False, False)

    def test_a_stored_preference_stands_on_a_camera_with_auto_gain(self):
        cam = SimulatedCamera()
        cam.open_and_start()
        try:
            scope = Lumascope.__new__(Lumascope)
            scope._camera_driver = cam
            imaging = ImagingAPI(scope, cam)
            on = imaging.applied_auto_gain_for(True)
            assert (on.stored, on.applied, on.capped) == (True, True, False)
            assert imaging.applied_auto_gain_for(False).applied is False
        finally:
            cam.disconnect()


class TestThePublicSetters:
    def test_turning_auto_gain_off_is_not_a_refusal(self, ag_less_imaging, asked, monkeypatch):
        writes = []
        monkeypatch.setattr(ag_less_imaging, '_camera_write', lambda *a, **kw: writes.append(kw))

        ag_less_imaging.set_auto_gain(False, AG_SETTINGS)

        assert asked == []
        # Off is such a camera's state, not a write: no validity invalidation,
        # no target clear, no cache resync.
        assert writes == []

    def test_turning_auto_exposure_off_is_not_a_refusal(self, ag_less_imaging, asked):
        ag_less_imaging.set_auto_exposure_time(False)

        assert asked == []

    @pytest.mark.parametrize(
        ('member', 'call', 'setting'),
        [
            ('set_auto_gain', lambda im: im.set_auto_gain(True, AG_SETTINGS), 'auto_gain'),
            (
                'set_auto_exposure_time',
                lambda im: im.set_auto_exposure_time(True),
                'auto_exposure',
            ),
            (
                'update_auto_gain_target_brightness',
                lambda im: im.update_auto_gain_target_brightness(0.4),
                'auto_gain_target_brightness',
            ),
            (
                'auto_gain_once',
                lambda im: im.auto_gain_once(
                    state=True, target_brightness=0.3, min_gain_db=0.0, max_gain_db=20.0
                ),
                'auto_gain',
            ),
        ],
    )
    def test_asking_for_a_mode_the_camera_lacks_is_refused(
        self, ag_less_imaging, asked, member, call, setting
    ):
        with pytest.raises(CameraSettingUnsupportedError) as excinfo:
            call(ag_less_imaging)

        assert excinfo.value.setting == setting, member
        assert asked == [], f'{member} asked the camera for a mode it does not have'


class TestARun:
    def test_a_run_on_a_camera_without_auto_gain_reports_no_auto_gain_refusal(
        self, tmp_path, ag_less_profile, asked, reported
    ):
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            assert session.scope.capabilities.camera_supports_auto_gain is False

            outcome = runner.start_composite(sequence_name='ag_less', parent_dir=str(tmp_path))
            result = outcome.wait(timeout_s=30.0)

        assert result is not None and result.status == 'completed', result
        assert asked == []
        assert reported == [], [str(exc) for exc, _ in reported]
