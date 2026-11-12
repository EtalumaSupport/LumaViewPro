"""A camera states the most gain it applies before a digital stage.

Above the analog maximum a camera multiplies its digitised values: the noise
rises with the signal, and a measurement of the sensor stops being one. The
capability is the profile's documented analog maximum; a camera whose profile
states no digital stage applies analog gain over its whole range, so its live
maximum is its analog maximum (the IDS U3-34L0XCP-M, whose profile leaves the
figure to the SDK). A camera with neither states none. The figure is part of
the camera's range (``scope.camera.setting_limits``), read again after each
readout change; a read that fails raises
(``tests/test_the_camera_range_follows_each_write.py``).
"""

from dataclasses import replace

from _lumascope.modules.image_mode import ImageMode
from tests.scope_fakes import capabilities_of


def _analog_maximum(sim_scope, monkeypatch, *, documented, has_digital, live_max_db):
    """The analog maximum the camera API holds for a camera whose profile
    documents ``documented`` and ``has_digital`` (the capabilities' static
    facts) and whose gain maximum reads ``live_max_db``."""
    import dataclasses

    # a stand-in by design: another camera's documented gain split, which the
    # simulator's one profile does not have.
    monkeypatch.setattr(
        sim_scope,
        'capabilities',
        dataclasses.replace(
            sim_scope.capabilities,
            camera_analog_gain_max_db=documented,
            camera_gain_has_digital=has_digital,
        ),
    )
    cam = sim_scope._camera_driver
    monkeypatch.setattr(cam.profile, 'gain', replace(cam.profile.gain, total_max_db=live_max_db))
    # A readout change reads the range again.
    sim_scope.camera.set_image_mode(ImageMode.SCIENTIFIC)
    return sim_scope.camera.setting_limits.analog_gain_max_db


def test_the_simulated_camera_states_the_profiles_analog_maximum(sim_scope):
    assert sim_scope.capabilities.camera_analog_gain_max_db == 24.0
    assert sim_scope.camera.setting_limits.analog_gain_max_db == 24.0


def test_a_camera_with_no_digital_stage_states_its_live_maximum(sim_scope, monkeypatch):
    assert (
        _analog_maximum(
            sim_scope, monkeypatch, documented=None, has_digital=False, live_max_db=29.99
        )
        == 29.99
    )


def test_a_digital_stage_with_no_documented_split_states_none(sim_scope, monkeypatch):
    assert (
        _analog_maximum(sim_scope, monkeypatch, documented=None, has_digital=True, live_max_db=48.0)
        is None
    )


def test_no_camera_states_none():
    assert capabilities_of(None).camera_limits is None
