"""Regression tests for the set_exposure_time WARNING threshold.

Bug shape: ``set_exposure_time`` emitted a WARNING at threshold
``t < 0.1`` ms saying "image will be nearly black. Value should be in
milliseconds." But Pylon physical ExposureTime minimum is 10-35 us
across Basler USB3 sensors, and bright-field captures legitimately use
0.03 ms (30 us). The warning fired on every BF capture and every
protocol BF step (multiple times per scan in beta11 field logs),
generating user-visible log noise for fully valid values.

Logs must be accurate: the "nearly black" wording was wrong -- Pylon
silently clamps below the sensor's physical minimum; the image is
at-minimum, not zero.

Fix shape: lower threshold to 0.005 ms (5 us), below any Basler
sensor physical minimum. The warning now fires only for genuinely
impossible values (unit-confusion bugs). Wording corrected to name
the actual behavior (clamping) and prompt the user to verify units.

These tests drive set_exposure_time on a simulator-backed ImagingAPI
with a recording logger (the shared lvp_logger is conftest-mocked, so
caplog cannot observe it) and lock the no-warn/warn boundary plus the
wording, so a threshold bump back toward 0.1 ms or a wording revert
trips immediately.
"""

from __future__ import annotations

from types import SimpleNamespace


def _warnings_for(exposure_ms: float, monkeypatch) -> list:
    """Call set_exposure_time on a sim-backed ImagingAPI; return the
    warning messages the imaging module logged."""
    from drivers.simulated_camera import SimulatedCamera
    from modules.lumascope_api import Lumascope
    from modules.lumascope_api.imaging import ImagingAPI

    cam = SimulatedCamera()
    cam.connect()
    scope = Lumascope.__new__(Lumascope)
    scope._camera_driver = cam
    scope._camera_executor = None
    imaging = ImagingAPI(scope, cam)

    records = []
    recorder = SimpleNamespace(
        warning=lambda msg, *a, **kw: records.append(msg),
        info=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        exception=lambda *a, **kw: None,
    )
    monkeypatch.setattr('modules.lumascope_api.imaging.logger', recorder)
    imaging.set_exposure_time(exposure_ms)
    return records


class TestSetExposureTimeWarningThreshold:
    """Lock the lower-than-100us threshold so a regression to 0.1 ms
    (which fires on every legitimate bright-BF capture) trips."""

    def test_legitimate_bright_bf_value_does_not_warn(self, monkeypatch):
        # 0.03 ms (30 us) is real bright-BF, bench-validated in beta11
        # field logs; the old 0.1 ms threshold warned on it every capture.
        assert _warnings_for(0.03, monkeypatch) == []

    def test_value_just_above_contract_does_not_warn(self, monkeypatch):
        # The warning gate must sit at or below 0.01 ms (10 us); a value
        # just above that bound must pass silently.
        assert _warnings_for(0.011, monkeypatch) == []

    def test_impossible_value_warns_describing_clamp(self, monkeypatch):
        # Below 5 us is impossible on any shipped Basler sensor --
        # a unit-confusion bug. The warning must name the clamping
        # behavior, not claim the image will be "nearly black" (Pylon
        # clamps to the sensor minimum; the image is at-minimum, not zero).
        records = _warnings_for(0.004, monkeypatch)
        assert len(records) == 1, f'a below-physical-minimum exposure must warn once; got {records}'
        assert 'clamp' in records[0].lower(), (
            f'warning must describe the clamping behavior; got {records[0]}'
        )
        assert 'nearly black' not in records[0], (
            'warning must not claim the image will be nearly black'
        )
