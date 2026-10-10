"""The frame-validity chunk target must describe the exposure the hardware
actually applied, not the one that was requested.

The bench defect this locks: a stored 0.01 ms exposure was requested as 10 us,
the driver clamped it up to the node minimum (~14 us) and reported nothing, and
the API stamped the REQUEST as the chunk-match target. Every frame then arrived
carrying the applied value, missed the 2.0 us tolerance by 4 us, and was
rejected -- ten captures in a row, with no value the camera would ever produce
that could close the gate.

The clamp is correct and stays. What must hold is that the target follows the
applied value through it. The tests set the exposure through the public
setter on a simulated scope and read the real owner, ``FrameValidity``,
through ``target`` and ``chunk_match``. The simulated camera applies every
in-range request exactly, so each test scripts the driver's answer on the
scope's own camera.
"""

import pytest

from modules.exceptions import CameraSettingRejected
from tests.scope_fakes import bind_settings_like_a_session, build_scope

# Inside the simulated camera's declared exposure range, so the request
# reaches the driver; 20 us.
REQUEST_MS = 0.02
REQUEST_US = 20.0


@pytest.fixture
def sim_imaging():
    """A simulated scope's imaging and its camera."""
    scope = build_scope(simulate=True)
    bind_settings_like_a_session(scope)
    return scope.imaging, scope.imaging._driver


def test_clamped_write_targets_the_applied_value_not_the_request(sim_imaging, monkeypatch):
    """The bench case: the driver raises the request to its floor and answers
    with the microseconds it applied; a frame carrying them passes the gate."""
    imaging, cam = sim_imaging
    # a stand-in by design: the simulated camera has no exposure floor to clamp to; the gap stage 7 fills
    monkeypatch.setattr(cam, 'exposure_t', lambda exposure_ms: 50.0)

    imaging.set_exposure_ms(REQUEST_MS)

    assert imaging.frame_validity.target('exposure') == 50.0, (
        'the chunk target must be the applied microseconds; recording the '
        'request is what rejected every frame at the bench'
    )
    assert imaging.frame_validity.chunk_match('exposure', 50.0), (
        'a frame carrying the applied exposure must pass the gate'
    )


def test_refused_write_records_no_target(sim_imaging, monkeypatch):
    """A driver that refused did not move the hardware: the prior target stands."""
    imaging, cam = sim_imaging
    imaging.set_exposure_ms(2.5)
    # a stand-in by design: the simulated camera refuses only above its maximum, which the API refuses first; the fault stage 7 injects
    monkeypatch.setattr(cam, 'exposure_t', lambda exposure_ms: False)

    with pytest.raises(CameraSettingRejected) as caught:
        imaging.set_exposure_ms(REQUEST_MS)

    assert caught.value.setting == 'exposure_ms'
    assert imaging.frame_validity.target('exposure') == 2500.0, (
        'a refused write must not claim a target the camera never took'
    )


def test_driver_reporting_no_value_falls_back_to_the_request(sim_imaging, monkeypatch):
    """None means applied-but-unknown; the request, in microseconds, is the
    best available target."""
    imaging, cam = sim_imaging
    # a stand-in by design: a driver with no confirmation signal; the simulator always reports what it applied
    monkeypatch.setattr(cam, 'exposure_t', lambda exposure_ms: None)

    imaging.set_exposure_ms(REQUEST_MS)

    assert imaging.frame_validity.target('exposure') == REQUEST_US


def test_a_bare_true_is_not_mistaken_for_microseconds(sim_imaging, monkeypatch):
    """bool is an int subclass; a driver reporting True must not stamp 1.0 us."""
    imaging, cam = sim_imaging
    # a stand-in by design: a driver that confirms with a bare True; the simulator reports microseconds
    monkeypatch.setattr(cam, 'exposure_t', lambda exposure_ms: True)

    imaging.set_exposure_ms(REQUEST_MS)

    assert imaging.frame_validity.target('exposure') == REQUEST_US, (
        'True is applied-but-unknown, not a 1 microsecond exposure'
    )
