# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A camera driver's ``gain()`` answers with the dB it applied.

The caller records the cache, the listener value and the frame-validity
target from this answer. A driver that clamps, snaps or quantizes the request
and then answers a bare ``True`` leaves all three naming a gain the sensor is
not at -- the same defect ``exposure_t`` already closes by returning the
microseconds in effect. So ``gain()`` has ``exposure_t``'s shape: the dB in
effect when applied, ``False`` when refused, ``None`` when not attempted.

The simulated camera honours the range its profile declares, so a refusal is
reachable in the simulator rather than only on a bench.
"""

from __future__ import annotations

import math

import pytest

from drivers import fx2driver
from drivers.simulated_camera import SimulatedCamera
from tests.camera_fakes import bare_ids_camera, bare_pylon_camera

# The FX2 fixture lives beside the FX2 driver tests.
from tests.test_fx2_driver import fake_fx2_conn  # noqa: F401
from tests.test_ids_driver import _RecordingNode, _RecordingNodemap


class TestPylon:
    @pytest.fixture
    def cam(self):
        cam = bare_pylon_camera()
        gain = cam.active.Gain
        gain.GetValue.return_value = 0.0
        gain.SetValue.side_effect = lambda v: setattr(gain.GetValue, 'return_value', v)
        return cam

    def test_a_written_gain_answers_its_db(self, cam):
        assert cam.gain(12.5) == pytest.approx(12.5)

    def test_a_gain_the_node_snaps_answers_the_snapped_db(self, cam):
        """Bodies with a Gain increment snap an off-increment request; the
        answer is what the node reads back, not the request."""
        gain = cam.active.Gain
        gain.SetValue.side_effect = lambda v: setattr(gain.GetValue, 'return_value', 12.4)
        assert cam.gain(12.5) == pytest.approx(12.4)

    def test_a_short_circuited_gain_answers_the_db_in_effect(self, cam):
        cam.active.Gain.GetValue.return_value = 12.5
        assert cam.gain(12.5) == pytest.approx(12.5)
        cam.active.Gain.SetValue.assert_not_called()


class TestIDS:
    def _cam(self, maximum):
        cam = bare_ids_camera()
        cam.remote_nodemap = _RecordingNodemap(
            {
                'Gain': _RecordingNode(value=1.0, minimum=1.0, maximum=maximum),
                'GainSelector': _RecordingNode(),
            }
        )
        cam._gain_selector = 'AnalogAll'
        return cam

    def test_an_in_range_gain_answers_its_db(self):
        assert self._cam(maximum=31.62).gain(20.0) == pytest.approx(20.0)

    def test_a_gain_the_node_clamps_answers_the_clamped_db(self):
        """The node's maximum factor is 10x (20 dB); a 30 dB request is
        written as 10x, so the answer is 20 dB, not the 30 asked for."""
        assert self._cam(maximum=10.0).gain(30.0) == pytest.approx(20.0)


@pytest.mark.skipif(
    not fx2driver._FX2_AVAILABLE,
    reason='FX2 prerequisites (pyusb / libusb1) not installed',
)
class TestFX2:
    @pytest.fixture
    def cam(self, fake_fx2_conn):
        return fx2driver.FX2Camera()

    def test_a_gain_answers_the_register_quantized_db(self, cam):
        _, expected = fx2driver._register_to_gain_db(fx2driver._gain_db_to_register(13.0))
        assert cam.gain(13.0) == pytest.approx(expected)

    def test_a_gain_above_the_sensor_ceiling_answers_the_clamped_db(self, cam):
        _, expected = fx2driver._register_to_gain_db(fx2driver._gain_db_to_register(42.1))
        assert cam.gain(60.0) == pytest.approx(expected)

    def test_an_inactive_camera_attempts_no_gain(self, cam):
        cam._active = None
        cam._fx2.sensor_reg_write.reset_mock()
        assert cam.gain(6.0) is None
        cam._fx2.sensor_reg_write.assert_not_called()

    def test_an_inactive_camera_refuses_an_exposure(self, cam):
        """``exposure_t``'s ``None`` means "applied, value unknown", so an
        inactive camera answers refused, as the pylon and simulated drivers do."""
        cam._active = None
        cam._fx2.sensor_reg_write.reset_mock()
        assert cam.exposure_t(20.0) is False
        cam._fx2.sensor_reg_write.assert_not_called()


class TestSimulated:
    @pytest.fixture
    def cam(self):
        cam = SimulatedCamera()
        cam.connect()
        return cam

    def test_an_in_range_gain_answers_its_db(self, cam):
        assert cam.gain(5.0) == pytest.approx(5.0)
        assert cam.get_gain() == pytest.approx(5.0)

    def test_a_gain_above_the_declared_maximum_is_refused(self, cam):
        before = cam.get_gain()
        assert cam.max_gain < 25.0
        assert cam.gain(25.0) is False
        assert cam.get_gain() == before

    def test_a_gain_below_the_declared_minimum_is_refused(self, cam):
        before = cam.get_gain()
        assert cam.gain(-5.0) is False
        assert cam.get_gain() == before


class TestMinimumGain:
    def test_the_minimum_gain_is_the_profile_declared_one(self):
        cam = SimulatedCamera()
        cam.connect()
        assert cam.min_gain == pytest.approx(cam.profile.gain.total_min_db)

    def test_an_undeclared_minimum_gain_is_unknown(self):
        cam = SimulatedCamera()
        cam.connect()
        cam.profile.gain.total_min_db = None
        assert cam.min_gain is None


def test_the_ids_answer_is_the_written_factor_in_db():
    """The dB answer is derived from the factor actually written, so it
    carries the node's own reconciliation, not the request."""
    cam = TestIDS()._cam(maximum=31.622776)
    answer = cam.gain(30.0)
    written = cam.remote_nodemap.nodes['Gain'].value
    assert answer == pytest.approx(20.0 * math.log10(written))


class TestSimulatedAutoGain:
    """A real body's auto loop drives the gain node and cannot leave its
    range, so the simulator's convergence stays inside the profile too."""

    def test_bounds_past_the_ceiling_converge_inside_it(self):
        cam = SimulatedCamera()
        cam.active = True
        cam.auto_gain(True, min_gain_db=0.0, max_gain_db=48.0)
        assert cam.get_gain() <= cam.max_gain

    def test_the_converged_gain_is_one_the_camera_accepts_back(self):
        cam = SimulatedCamera()
        cam.active = True
        cam.auto_gain_once(True, min_gain_db=0.0, max_gain_db=48.0)
        assert cam.gain(cam.get_gain()) is not False
