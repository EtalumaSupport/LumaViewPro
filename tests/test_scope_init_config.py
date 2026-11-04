# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for ScopeInitConfig.

The config carries what the selected scope's model expects: an LS620 has no
motor board (`Focus=false, XYStage=false, Turret=false` in scopes.json), so
its missing board is not reported as a failure. What bring-up reports from
those expectations is pinned in `tests/test_bring_up_is_a_record.py`.
"""

# Heavy deps are mocked by tests/conftest.py at module-import time.

import pytest

from modules.layer_record import load_scope_models
from modules.scope_init_config import ScopeInitConfig
from modules.lumascope_api._constants import ACCELERATION_PCT_MAX
from tests.scope_fakes import build_scope, identity_with_layers
from tests.stock_channels import LED0, LED1, LED2


# ---------- ScopeInitConfig.from_settings ----------

_BASE_SETTINGS = {
    'binning': {'size': '1x1'},
    'camera': {'high_conversion_gain': False, 'line_noise_reduction': False},
    'frame': {'width': 1900, 'height': 1900},
    'image_mode': '8bit',
    'objective_id': '4x',
    'turret_objectives': None,
    'motion': {'acceleration_max_pct': 100},
    'stage_offset': {'x': 0, 'y': 0},
    'scale_bar': {'enabled': False},
    'microscope': 'LS820',
}

# The shipped catalogue's records for the three models the table walks.
_MODELS = load_scope_models()
_LS620_CONFIG = _MODELS['LS620']
_LS820_CONFIG = _MODELS['LS820']
_LS850T_CONFIG = _MODELS['LS850T']

# The truth-table fixtures used to carry scopes.json category booleans;
# expects_led derives from the scope's resolved layer identity, so the
# fixtures carry the same models' layers instead.
_LS620_IDENTITY = identity_with_layers(['BF', LED0, LED1, LED2])
_LS820_IDENTITY = identity_with_layers(['BF', 'PC', 'DF', LED0, LED1, LED2])
_LS850T_IDENTITY = _LS820_IDENTITY
_NO_LED_IDENTITY = identity_with_layers(['Lumi'])


class TestFromSettings:
    def test_default_no_scope_config_preserves_pre_filter_behavior(self):
        config = ScopeInitConfig.from_settings(_BASE_SETTINGS)
        assert config.expects_motion is True
        assert config.expects_led is True

    def test_capture_depth_resolved_from_image_mode(self):
        # The stored mode is carried as itself.
        config = ScopeInitConfig.from_settings(_BASE_SETTINGS)
        assert config.image_mode == '8bit'
        # A 12-bit image mode is carried as itself, so initialize() applies a
        # 12-bit native pixel format up front, or says it cannot.
        twelve = {**_BASE_SETTINGS, 'image_mode': '12bit_scientific'}
        config = ScopeInitConfig.from_settings(twelve)
        assert config.image_mode == '12bit_scientific'

    def test_ls620_no_motor_expected(self):
        config = ScopeInitConfig.from_settings(
            _BASE_SETTINGS,
            scope_config=_LS620_CONFIG,
            channel_identity=_LS620_IDENTITY,
        )
        assert config.expects_motion is False
        assert config.expects_led is True

    def test_ls820_motor_expected_via_focus(self):
        config = ScopeInitConfig.from_settings(
            _BASE_SETTINGS,
            scope_config=_LS820_CONFIG,
            channel_identity=_LS820_IDENTITY,
        )
        assert config.expects_motion is True
        assert config.expects_led is True

    def test_ls850t_motor_expected_via_xystage_and_turret(self):
        config = ScopeInitConfig.from_settings(
            _BASE_SETTINGS,
            scope_config=_LS850T_CONFIG,
            channel_identity=_LS850T_IDENTITY,
        )
        assert config.expects_motion is True
        assert config.expects_led is True

    def test_no_led_driving_layer_means_no_led_expected(self):
        config = ScopeInitConfig.from_settings(
            _BASE_SETTINGS,
            scope_config=_LS820_CONFIG,
            channel_identity=_NO_LED_IDENTITY,
        )
        assert config.expects_led is False


class TestAccelerationBound:
    """A settings dict's acceleration limit is carried as stored.

    The limit is held to its range before this reads it: the file's at
    the load, a handed dict's at the session's door, by the writer's one
    rule (``tests/test_a_settings_dict_handed_to_create_is_prepared_as_the_file_is.py``).
    """

    def test_an_in_range_value_is_carried(self):
        settings = {**_BASE_SETTINGS, 'motion': {'acceleration_max_pct': 50}}
        config = ScopeInitConfig.from_settings(settings)
        assert config.acceleration_pct == 50

    def test_the_api_refuses_on_the_same_bound(self):
        """from_settings and the motion API read the one check."""
        scope = build_scope(simulate=True)
        with pytest.raises(ValueError):
            scope.motion.set_acceleration_limit(limit_pct=ACCELERATION_PCT_MAX + 1)
        scope.motion.set_acceleration_limit(limit_pct=ACCELERATION_PCT_MAX)
