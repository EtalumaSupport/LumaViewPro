"""A camera states the most gain it applies before a digital stage.

Above the analog maximum a camera multiplies its digitised values: the noise
rises with the signal, and a measurement of the sensor stops being one. The
capability is the profile's documented analog maximum; a camera whose profile
states no digital stage applies analog gain over its whole range, so its live
maximum is its analog maximum (the IDS U3-34L0XCP-M, whose profile leaves the
figure to the SDK). A camera with neither states none.
"""

import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

from drivers.camera_profiles import GainInfo, lookup_profile
from modules.layer_record import UNRESOLVED
from modules.scope_capabilities import ScopeCapabilities


def _capabilities(profile):
    camera = SimpleNamespace(
        model_name='STUB',
        device_serial=None,
        timestamp_tick_frequency_hz=None,
        profile=profile,
        get_max_frame_size=lambda: None,
    )
    return ScopeCapabilities.from_drivers(
        motion=MagicMock(motorconfig=None),
        led=MagicMock(),
        camera=camera,
        layer_identity=UNRESOLVED,
        scope_models={},
    )


def test_the_simulated_camera_states_the_profiles_analog_maximum(sim_scope):
    assert sim_scope.capabilities.camera_analog_gain_max_db == 24.0


def test_a_camera_with_no_digital_stage_states_its_live_maximum():
    profile = copy.deepcopy(lookup_profile('U3-34L0XCP-M'))
    assert profile.gain.analog_max_db is None and profile.gain.has_digital is False
    profile.gain.total_max_db = 29.99

    assert _capabilities(profile).camera_analog_gain_max_db == 29.99


def test_a_digital_stage_with_no_documented_split_states_none():
    profile = copy.deepcopy(lookup_profile('U3-34L0XCP-M'))
    profile.gain = replace(profile.gain, has_digital=True, total_max_db=48.0)

    assert _capabilities(profile).camera_analog_gain_max_db is None


def test_a_camera_whose_maximum_was_not_read_states_none():
    profile = copy.deepcopy(lookup_profile('U3-34L0XCP-M'))
    profile.gain = GainInfo(has_digital=False)

    assert _capabilities(profile).camera_analog_gain_max_db is None


def test_no_camera_states_none():
    caps = ScopeCapabilities.from_drivers(
        motion=MagicMock(motorconfig=None),
        led=MagicMock(),
        camera=None,
        layer_identity=UNRESOLVED,
        scope_models={},
    )
    assert caps.camera_analog_gain_max_db is None
