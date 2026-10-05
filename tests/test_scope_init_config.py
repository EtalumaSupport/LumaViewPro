# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Tests for ScopeInitConfig.

The config carries what the selected scope's model expects: an LS620 has no
motor board (`Focus=false, XYStage=false, Turret=false` in scopes.json), so
its missing board is not reported as a failure. What bring-up reports from
those expectations is pinned in `tests/test_bring_up_is_a_record.py`.
"""

# Heavy deps are mocked by tests/conftest.py at module-import time.

import pytest

from modules.scope_init_config import ScopeInitConfig
from modules.lumascope_api._constants import ACCELERATION_PCT_MAX, ACCELERATION_PCT_MIN
from tests.scope_fakes import build_scope


# ---------- ScopeInitConfig.from_settings ----------

_BASE_SETTINGS = {
    'binning': {'size': '1x1'},
    'frame': {'width': 1900, 'height': 1900},
    'objective_id': '4x',
    'turret_objectives': None,
    'motion': {'acceleration_max_pct': 100},
    'stage_offset': {'x': 0, 'y': 0},
    'scale_bar': {'enabled': False},
    'microscope': 'LS820',
}

_LS620_CONFIG = {
    'Focus': False,
    'XYStage': False,
    'Turret': False,
    'Layers': {
        'Lumi': False,
        'Fluorescence': True,
        'Darkfield': False,
        'Brightfield': True,
        'PhaseContrast': True,
    },
}
_LS820_CONFIG = {
    'Focus': True,
    'XYStage': False,
    'Turret': False,
    'Layers': {
        'Lumi': False,
        'Fluorescence': True,
        'Darkfield': True,
        'Brightfield': True,
        'PhaseContrast': True,
    },
}
_LS850T_CONFIG = {
    'Focus': True,
    'XYStage': True,
    'Turret': True,
    'Layers': {
        'Lumi': False,
        'Fluorescence': True,
        'Darkfield': True,
        'Brightfield': True,
        'PhaseContrast': True,
    },
}


def identity_from_rows(rows):
    """Build a resolved-identity snapshot from (key_name, led_channel) pairs.

    The truth-table fixtures used to carry scopes.json category booleans;
    expects_led now derives from the scope's resolved layer identity, so
    the fixtures carry the same models as identity rows instead. The
    assertion values are unchanged.
    """
    from modules.layer_record import LayerIdentity, LayerRecord

    records = tuple(
        LayerRecord(
            id=i,
            key_name=key,
            display_name=key,
            led_channel=(channel,) if channel is not None else (),
            excitation_nm=None,
        )
        for i, (key, channel) in enumerate(rows)
    )
    return LayerIdentity(layers=records, filterset='', source='scopes', model=None)


_LS620_IDENTITY = identity_from_rows([('BF', 3), ('Blue', 0), ('Green', 1), ('Red', 2)])
_LS820_IDENTITY = identity_from_rows(
    [('BF', 3), ('PC', 4), ('DF', 5), ('Blue', 0), ('Green', 1), ('Red', 2)]
)
_LS850T_IDENTITY = _LS820_IDENTITY
_NO_LED_IDENTITY = identity_from_rows([('Lumi', None)])


class TestFromSettings:
    def test_default_no_scope_config_preserves_pre_filter_behavior(self):
        config = ScopeInitConfig.from_settings(_BASE_SETTINGS, labware=None, turreted=False)
        assert config.expects_motion is True
        assert config.expects_led is True

    def test_capture_depth_resolved_from_image_mode(self):
        # No image_mode key -> the 8-bit default mode.
        config = ScopeInitConfig.from_settings(_BASE_SETTINGS, labware=None, turreted=False)
        assert config.image_mode == '8bit'
        # A 12-bit image mode is carried as itself, so initialize() applies a
        # 12-bit native pixel format up front, or says it cannot.
        twelve = {**_BASE_SETTINGS, 'image_mode': '12bit_scientific'}
        config = ScopeInitConfig.from_settings(twelve, labware=None, turreted=False)
        assert config.image_mode == '12bit_scientific'

    def test_ls620_no_motor_expected(self):
        config = ScopeInitConfig.from_settings(
            _BASE_SETTINGS,
            labware=None,
            turreted=False,
            scope_config=_LS620_CONFIG,
            layer_identity=_LS620_IDENTITY,
        )
        assert config.expects_motion is False
        assert config.expects_led is True

    def test_ls820_motor_expected_via_focus(self):
        config = ScopeInitConfig.from_settings(
            _BASE_SETTINGS,
            labware=None,
            turreted=False,
            scope_config=_LS820_CONFIG,
            layer_identity=_LS820_IDENTITY,
        )
        assert config.expects_motion is True
        assert config.expects_led is True

    def test_ls850t_motor_expected_via_xystage_and_turret(self):
        config = ScopeInitConfig.from_settings(
            _BASE_SETTINGS,
            labware=None,
            turreted=False,
            scope_config=_LS850T_CONFIG,
            layer_identity=_LS850T_IDENTITY,
        )
        assert config.expects_motion is True
        assert config.expects_led is True

    def test_no_led_driving_layer_means_no_led_expected(self):
        scope_config = {'Focus': True, 'XYStage': False, 'Turret': False}
        config = ScopeInitConfig.from_settings(
            _BASE_SETTINGS,
            labware=None,
            turreted=False,
            scope_config=scope_config,
            layer_identity=_NO_LED_IDENTITY,
        )
        assert config.expects_led is False


class TestAccelerationBound:
    """The stored acceleration percentage is bounded where settings become
    hardware commands, not where a GUI happens to draw a slider.

    A settings dict arrives here from a file a user can hand-edit and from a
    caller that hands one straight to a session, and neither path passes a
    slider. Before it was bounded at this read, an out-of-range stored value
    reached the motor driver and only a swallowed exception kept bring-up
    alive.
    """

    @pytest.mark.parametrize(
        'stored, expected',
        [
            (ACCELERATION_PCT_MAX + 400, ACCELERATION_PCT_MAX),
            (0, ACCELERATION_PCT_MIN),
            (-3, ACCELERATION_PCT_MIN),
            (50, 50),
            # A hand-edited file can carry the number as a string. Coercing
            # before clamping is what keeps this from raising TypeError out of
            # bring-up once the driver's rejection is no longer swallowed.
            ('50', 50),
            ('', ACCELERATION_PCT_MAX),
            (None, ACCELERATION_PCT_MAX),
        ],
    )
    def test_stored_value_is_bounded_to_what_the_driver_accepts(self, stored, expected):
        settings = {**_BASE_SETTINGS, 'motion': {'acceleration_max_pct': stored}}
        config = ScopeInitConfig.from_settings(settings, labware=None, turreted=False)
        assert config.acceleration_pct == expected

    def test_an_absent_motion_section_still_yields_a_legal_value(self):
        settings = {key: value for key, value in _BASE_SETTINGS.items() if key != 'motion'}
        config = ScopeInitConfig.from_settings(settings, labware=None, turreted=False)
        assert ACCELERATION_PCT_MIN <= config.acceleration_pct <= ACCELERATION_PCT_MAX

    def test_the_bound_is_the_apis_own_rather_than_a_second_copy(self):
        """The clamp here and the motion API's refusal read the same constants.

        Asserted as a pairing rather than against the numbers: a future edit
        that hand-copies 1 and 100 into either side would still satisfy a
        literal assertion, and a silently drifting duplicate of this pair is
        how the value got out of range to begin with.
        """
        settings = {**_BASE_SETTINGS, 'motion': {'acceleration_max_pct': 10**6}}
        config = ScopeInitConfig.from_settings(settings, labware=None, turreted=False)
        assert config.acceleration_pct == ACCELERATION_PCT_MAX

        scope = build_scope(simulate=True)
        with pytest.raises(ValueError):
            scope.motion.set_acceleration_limit(val_pct=ACCELERATION_PCT_MAX + 1)
