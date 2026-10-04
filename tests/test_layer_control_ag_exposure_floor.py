# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression tests for the AG-feedback exposure floor stored on leaving auto-gain.

Bug
---
When auto-gain runs on a bright transmitted sample (BF/PC/DF), the Pylon
camera can drive ExposureTime to its physical minimum (~30 us = 0.030 ms
on common sensors). update_auto_gain_cb then reads that value via
get_exposure_ms() and writes it to settings[layer]['exposure_ms'] without an
appropriate floor:

- Fluorescence + luminescence already get a 1.0 ms floor via
  FLUORESCENCE_MIN_EXPOSURE_MS in the original conditional.
- Transmitted (BF/PC/DF) had NO conditional floor; the slider's .kv
  default min is 0.01 ms, so np.clip(0.030, 0.01, max) returned 0.030.
- The 0.030 ms write then fires set_exposure_ms's <0.1 ms
  "Value should be in milliseconds" WARNING on every subsequent
  apply_settings (visible in the beta tester's beta9 logs as recurring spam).

Fix
---
Add TRANSMITTED_MIN_EXPOSURE_MS = 0.1 (matching set_exposure_ms's
internal warning gate) and apply it via an else branch on the existing
get_image_layers conditional. Live AG output to the camera is untouched
(the floor applies only to the settings write-back).

Test approach
-------------
- Structural lock (source-level): the store applies no floor of its own.
- Behavioral: turn auto-gain off through ScopeSession.set_layer_auto_gain
  with sub-floor exp values; assert settings store the floored value, not
  the raw value. (The write-back moved from the GUI's toggle callback to
  that Session member.)
"""

from __future__ import annotations

import inspect
import threading
from types import SimpleNamespace

import pytest

import modules.config_helpers as config_helpers
from modules.lumascope_api.imaging import (
    AutoGainConvergence,
    AutoGainLock,
    stored_exposure_after_lock,
)
from modules.scope_session import ScopeSession


class TestExposureFloorSourceStructure:
    """Source-level lock on the AG-feedback floor logic in
    ScopeSession.set_layer_auto_gain."""

    def test_transmitted_floor_is_the_warning_gate(self):
        """The transmitted class floor lives beside the AG/AE ceiling in
        config_helpers and equals set_exposure_ms's <0.1 ms warning gate;
        changing it changes which AG-feedback values fire the warning."""
        assert config_helpers.DEFAULT_AG_AE_MIN_EXPOSURE_MS['transmitted'] == 0.1
        assert config_helpers.DEFAULT_AG_AE_MIN_EXPOSURE_MS['fluorescence'] == 1.0
        assert config_helpers.DEFAULT_AG_AE_MIN_EXPOSURE_MS['luminescence'] == 1.0

    def test_floor_is_decided_by_the_api_not_the_callback(self):
        """Leaving auto-gain stores the value the lock result carries and
        applies no floor of its own; the floor is the API's decision so a
        REST caller storing the same result gets the same value. A
        hand-written per-class branch in the GUI is how the BF AG ->
        0.03 ms -> warning-spam path came back once before."""
        body = inspect.getsource(ScopeSession.set_layer_auto_gain)
        assert 'stored_exposure_ms' in body
        assert 'get_ag_ae_min_exposure_ms' not in body
        assert 'FLUORESCENCE_MIN_EXPOSURE_MS' not in body
        assert 'TRANSMITTED_MIN_EXPOSURE_MS' not in body
        assert stored_exposure_after_lock(0.03, 0.1) == 0.1
        assert stored_exposure_after_lock(5.0, 0.1) == 5.0
        assert stored_exposure_after_lock(0.4, 1.0) == 1.0


# ---------------------------------------------------------------------------
# Behavioral test: AST-extract + exec pattern
# ---------------------------------------------------------------------------


def _leave(settings: dict, layer: str, lock: AutoGainLock) -> None:
    """Turn the layer's auto-gain off through the Session member, over a
    scope whose lock returns ``lock``."""
    session = SimpleNamespace(
        scope=SimpleNamespace(imaging=SimpleNamespace(lock_auto_gain=lambda: lock)),
        settings=settings,
        settings_lock=threading.Lock(),
    )
    ScopeSession.set_layer_auto_gain(session, layer, False)


def _lock(layer: str, gain_db: float, exposure_ms: float) -> AutoGainLock:
    """The lock result a toggle-off hands the callback, built the way the
    API builds it: the floor is applied to the STORED value by the API's
    own rule, so these tests exercise production's decision, not a copy."""
    floor = config_helpers.get_ag_ae_min_exposure_ms(layer)
    return AutoGainLock(
        AutoGainConvergence.CONVERGED,
        exposure_ms,
        gain_db,
        floor,
        stored_exposure_ms=stored_exposure_after_lock(exposure_ms, floor),
    )


class TestExposureFloorBehavior:
    """Behavioral verification that AG-feedback writes are floored before
    landing in settings[layer]['exposure_ms']."""

    @pytest.mark.parametrize(
        'raw_exp_ms,expected_floor',
        [
            (0.030, 0.1),  # Pylon ExposureTime.Min for ace 2 etc.
            (0.05, 0.1),  # below threshold
            (0.099, 0.1),  # just below threshold
            (0.1, 0.1),  # at threshold (still allowed)
            (5.0, 5.0),  # above threshold -> passes through
        ],
    )
    def test_bf_ag_feedback_floored_to_transmitted_min(self, raw_exp_ms, expected_floor):
        """For BF (transmitted), AG-feedback exp values < 0.1 ms must be
        floored to 0.1 before being written to settings. Without this,
        the next apply_settings fires the set_exposure_ms(<0.1ms)
        WARNING on every layer switch."""
        settings = {'BF': {'exposure_ms': 999.0, 'gain_db': 0.0, 'auto_gain': True}}

        _leave(settings, 'BF', _lock('BF', 0.0, raw_exp_ms))

        stored = settings['BF']['exposure_ms']
        assert stored == expected_floor, (
            f'BF AG-feedback raw_exp={raw_exp_ms}ms should floor to '
            f'{expected_floor}ms, got {stored}ms. See class docstring '
            f'for the WARNING-spam bug this floor prevents.'
        )

    @pytest.mark.parametrize(
        'raw_exp_ms,expected_floor',
        [
            (0.030, 1.0),  # camera minimum -> 1ms fluorescence floor
            (0.5, 1.0),  # below fluo floor
            (0.999, 1.0),  # just below fluo floor
            (1.0, 1.0),  # at fluo floor
            (15.0, 15.0),  # above floor -> passes through
        ],
    )
    def test_blue_ag_feedback_floored_to_fluorescence_min(self, raw_exp_ms, expected_floor):
        """For Blue (fluorescence), AG-feedback exp values < 1 ms must be
        floored to 1.0 (FLUORESCENCE_MIN_EXPOSURE_MS). Pre-existing
        behavior; this test locks it against accidental regression
        during the transmitted-floor refactor."""
        settings = {'Blue': {'exposure_ms': 999.0, 'gain_db': 0.0, 'auto_gain': True}}

        _leave(settings, 'Blue', _lock('Blue', 0.0, raw_exp_ms))

        stored = settings['Blue']['exposure_ms']
        assert stored == expected_floor, (
            f'Blue AG-feedback raw_exp={raw_exp_ms}ms should floor to '
            f'{expected_floor}ms (FLUORESCENCE_MIN_EXPOSURE_MS), '
            f'got {stored}ms.'
        )

    def test_pc_uses_transmitted_floor(self):
        """PC is in the transmitted class (not in get_image_layers).
        Must use the 0.1 floor, not the 1.0 floor."""
        settings = {'PC': {'exposure_ms': 999.0, 'gain_db': 0.0, 'auto_gain': True}}
        _leave(settings, 'PC', _lock('PC', 0.0, 0.050))
        assert settings['PC']['exposure_ms'] == 0.1

    def test_df_uses_transmitted_floor(self):
        """DF is in the transmitted class (not in get_image_layers).
        Must use the 0.1 floor, not the 1.0 floor."""
        settings = {'DF': {'exposure_ms': 999.0, 'gain_db': 0.0, 'auto_gain': True}}
        _leave(settings, 'DF', _lock('DF', 0.0, 0.050))
        assert settings['DF']['exposure_ms'] == 0.1

    def test_lumi_uses_fluorescence_floor(self):
        """Lumi (luminescence) is in get_image_layers; must use the 1.0
        floor like fluorescence, not the 0.1 transmitted floor."""
        settings = {'Lumi': {'exposure_ms': 999.0, 'gain_db': 0.0, 'auto_gain': True}}
        _leave(settings, 'Lumi', _lock('Lumi', 0.0, 0.5))
        assert settings['Lumi']['exposure_ms'] == 1.0

    def test_unknown_exposure_keeps_previous_settings(self):
        """A non-physical exposure reading (nothing was ever successfully
        read from the camera) must not overwrite the layer's stored
        exposure -- the previous value is the best truth available."""
        settings = {'BF': {'exposure_ms': 42.0, 'gain_db': 7.0, 'auto_gain': True}}
        _leave(settings, 'BF', _lock('BF', 3.0, 0.0))
        assert settings['BF']['exposure_ms'] == 42.0
        # The valid gain reading in the same lock still lands.
        assert settings['BF']['gain_db'] == 3.0

    def test_unknown_gain_keeps_previous_settings(self):
        """A non-physical gain reading must not overwrite the layer's
        stored gain, while a valid exposure in the same lock still
        floors and lands."""
        settings = {'BF': {'exposure_ms': 42.0, 'gain_db': 7.0, 'auto_gain': True}}
        _leave(settings, 'BF', _lock('BF', -1.0, 5.0))
        assert settings['BF']['gain_db'] == 7.0
        assert settings['BF']['exposure_ms'] == 5.0
