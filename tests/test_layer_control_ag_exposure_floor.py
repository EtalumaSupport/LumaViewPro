# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for the AG-feedback exposure floor stored on leaving auto-gain.

When auto-gain runs on a bright transmitted sample (BF/PC/DF), the Pylon
camera can drive ExposureTime to its physical minimum (~30 us = 0.030 ms on
common sensors). Leaving auto-gain stores the exposure the lock measured;
stored raw, 0.030 ms re-fired set_exposure_ms's unit-confusion warning on
every later apply (recurring spam in a beta tester's logs), while the
fluorescence and luminescence layers already took a 1.0 ms floor.

The floor is the API's decision: ``stored_exposure_after_lock`` applies the
per-class floor in ``config_helpers.DEFAULT_AG_AE_MIN_EXPOSURE_MS`` to the
STORED value and the lock result carries it, so a GUI and a REST caller
store the same thing. Live AG output to the camera is untouched.

Test approach
-------------
- Structural lock (source-level): the Session member applies no floor of
  its own; it stores what the lock carries.
- Behavioral: on a simulated session, arm the shown layer's auto-gain at a
  sub-floor exposure (the simulated loop converges the gain and leaves the
  exposure where it was) and turn it off through
  ScopeSession.set_channel_auto_gain; assert the store holds the floored
  value, not the raw one.
"""

from __future__ import annotations

import inspect

import pytest

import modules.config_helpers as config_helpers
from modules.lumascope_api.camera import stored_exposure_after_lock
from modules.scope_session import ScopeSession
from tests.stock_channels import LED0


class TestExposureFloorSourceStructure:
    """Source-level lock on the AG-feedback floor logic in
    ScopeSession.set_channel_auto_gain."""

    def test_the_per_class_floors_live_beside_the_ag_ae_ceiling(self):
        """The floors live beside the AG/AE ceiling in config_helpers: 0.1 ms
        for the transmitted class, 1.0 ms for fluorescence and luminescence."""
        assert config_helpers.DEFAULT_AG_AE_MIN_EXPOSURE_MS['transmitted'] == 0.1
        assert config_helpers.DEFAULT_AG_AE_MIN_EXPOSURE_MS['fluorescence'] == 1.0
        assert config_helpers.DEFAULT_AG_AE_MIN_EXPOSURE_MS['luminescence'] == 1.0

    def test_floor_is_decided_by_the_api_not_the_callback(self):
        """Leaving auto-gain stores the value the lock result carries and
        applies no floor of its own; the floor is the API's decision so a
        REST caller storing the same result gets the same value. A
        hand-written per-class branch in the GUI is how the BF AG ->
        0.03 ms -> warning-spam path came back once before."""
        body = inspect.getsource(ScopeSession.set_channel_auto_gain)
        assert 'stored_exposure_ms' in body
        assert 'get_ag_ae_min_exposure_ms' not in body
        assert 'FLUORESCENCE_MIN_EXPOSURE_MS' not in body
        assert 'TRANSMITTED_MIN_EXPOSURE_MS' not in body
        assert stored_exposure_after_lock(0.03, 0.1) == 0.1
        assert stored_exposure_after_lock(5.0, 0.1) == 5.0
        assert stored_exposure_after_lock(0.4, 1.0) == 1.0


# ---------------------------------------------------------------------------
# Behavioral test: a simulated session leaves auto-gain
# ---------------------------------------------------------------------------


@pytest.fixture
def leave(sim_session):
    """Leave *layer*'s auto-gain on ``sim_session`` after the loop ran at *raw_exp_ms*.

    The layer is shown at that exposure, its auto-gain armed and then
    turned off; returns the layer's stored exposure.
    """

    def leave(layer: str, raw_exp_ms: float) -> float:
        sim_session.update_settings(f'{layer}.exposure_ms', raw_exp_ms)
        sim_session.show_channel(layer)
        sim_session.set_channel_auto_gain(layer, True)
        sim_session.set_channel_auto_gain(layer, False)
        return sim_session.get_setting(f'{layer}.exposure_ms')

    return leave


class TestExposureFloorBehavior:
    """Behavioral verification that AG-feedback writes are floored before
    landing in settings[layer]['exposure_ms']."""

    @pytest.mark.parametrize(
        'raw_exp_ms,expected_floor',
        [
            (0.030, 0.1),  # Pylon ExposureTime.Min for ace 2 etc.
            (0.09, 0.1),  # just below threshold, at the stored resolution (0.01 ms)
            (0.1, 0.1),  # at threshold (still allowed)
            (5.0, 5.0),  # above threshold -> passes through
        ],
    )
    def test_bf_ag_feedback_floored_to_transmitted_min(self, leave, raw_exp_ms, expected_floor):
        """For BF (transmitted), AG-feedback exp values < 0.1 ms must be
        floored to 0.1 before being written to settings. Without this,
        the next apply_settings fires the set_exposure_ms(<0.1ms)
        WARNING on every layer switch."""
        stored = leave('BF', raw_exp_ms)

        assert stored == expected_floor, (
            f'BF AG-feedback raw_exp={raw_exp_ms}ms should floor to '
            f'{expected_floor}ms, got {stored}ms. See class docstring '
            f'for the WARNING-spam bug this floor prevents.'
        )

    @pytest.mark.parametrize(
        'raw_exp_ms,expected_floor',
        [
            (0.030, 1.0),  # camera minimum -> 1ms fluorescence floor
            (0.9, 1.0),  # below fluo floor, at the stored resolution (0.01 ms)
            (1.0, 1.0),  # at fluo floor
            (15.0, 15.0),  # above floor -> passes through
        ],
    )
    def test_blue_ag_feedback_floored_to_fluorescence_min(self, leave, raw_exp_ms, expected_floor):
        """For LED 0's channel (fluorescence), AG-feedback exp values < 1 ms must be
        floored to 1.0, the fluorescence class floor."""
        stored = leave(LED0, raw_exp_ms)

        assert stored == expected_floor, (
            f'{LED0} AG-feedback raw_exp={raw_exp_ms}ms should floor to '
            f'{expected_floor}ms (the fluorescence class floor), '
            f'got {stored}ms.'
        )

    def test_pc_uses_transmitted_floor(self, leave):
        """PC is in the transmitted class (not in get_image_layers).
        Must use the 0.1 floor, not the 1.0 floor."""
        assert leave('PC', 0.050) == 0.1

    def test_df_uses_transmitted_floor(self, leave):
        """DF is in the transmitted class (not in get_image_layers).
        Must use the 0.1 floor, not the 1.0 floor."""
        assert leave('DF', 0.050) == 0.1

    @pytest.mark.parametrize('sim_session', [{'microscope': 'Lumi'}], indirect=True)
    def test_lumi_uses_fluorescence_floor(self, leave):
        """Lumi (luminescence) is in get_image_layers; must use the 1.0
        floor like fluorescence, not the 0.1 transmitted floor."""
        assert leave('Lumi', 0.5) == 1.0
