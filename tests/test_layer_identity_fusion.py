# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The identity/drivability fusion and the two-resolver split.

Identity (the layer record) answers what a layer IS; the driver answers
what the board can DRIVE; state bookkeeping follows the DRIVER's table.
These tests pin the seams that keep those from bleeding into each other:
an ON request for a layer identity lacks fails loudly by name, an OFF
stays a silent no-op, a no-LED layer resolves to no channel without
being an error, and -- the load-bearing one -- a channel that
physically lit stays in the state store when the CURRENT identity can
no longer name it, so a mid-session identity change can never strand a
lit LED outside the extinguish machinery.
"""

import pytest

from modules.exceptions import ConfigError, HardwareCommandRefusedError, MissingPart
from tests.scope_fakes import build_scope


@pytest.fixture
def scope():
    s = build_scope(simulate=True, register_atexit=False, sim_model='LS850T')
    yield s
    s.disconnect()


class TestFusion:
    def test_identity_layer_on_drivable_channel_lights(self, scope):
        scope.illumination.led_on(channel='Green', illumination_ma=50)
        assert scope.illumination.get_led_state('Green')['enabled'] is True
        scope.illumination.led_off(channel='Green')

    def test_unknown_layer_on_raises_by_name(self, scope):
        with pytest.raises(ConfigError, match='Skylight'):
            scope.illumination.led_on(channel='Skylight', illumination_ma=50)

    def test_unknown_layer_off_is_a_silent_noop(self, scope):
        scope.illumination.led_off(channel='Skylight')

    def test_identity_narrowed_layer_on_raises_by_name(self, scope):
        scope.refresh_layer_identity(override_model='LS850-0')
        with pytest.raises(HardwareCommandRefusedError, match='Green') as refused:
            scope.illumination.led_on(channel='Green', illumination_ma=50)
        assert refused.value.missing == MissingPart.led('Green')

    def test_identity_narrowed_layer_off_still_noops(self, scope):
        scope.refresh_layer_identity(override_model='LS850-0')
        scope.illumination.led_off(channel='Green')

    def test_no_led_layer_resolves_to_no_channel_without_error(self, scope):
        scope.refresh_layer_identity(override_model='Lumi')
        assert scope.layer_identity.find('Lumi') is not None
        assert scope.illumination.color2ch('Lumi') is None
        scope.illumination.led_off(channel='Lumi')


class TestTwoResolverSplit:
    def test_identity_answers_by_record(self, scope):
        assert scope.illumination.color2ch('Green') == 1
        assert scope.illumination.ch2color(1) == 'Green'
        scope.refresh_layer_identity(override_model='LS850-0')
        assert scope.illumination.color2ch('Green') is None
        assert scope.illumination.ch2color(1) is None

    def test_state_answers_by_driver_whatever_identity_says(self, scope):
        scope.refresh_layer_identity(override_model='LS850-0')
        assert scope.illumination.state_color2ch('Green') == 1
        assert scope.illumination.state_ch2color(1) == 'Green'

    def test_lit_channel_stays_recorded_under_narrowed_identity(self, scope):
        """A channel lit before the identity narrowed stays in the state
        store, and the extinguish that rides that record still reaches it."""
        scope.illumination.led_on(channel='Green', illumination_ma=50)
        scope.refresh_layer_identity(override_model='LS850-0')
        assert scope.illumination.get_led_states()['Green']['enabled'] is True
        scope.illumination.led_off(channel=1)
        assert scope.illumination.get_led_state('Green')['enabled'] is False

    def test_a_light_the_narrowed_identity_lacks_is_refused(self, scope):
        """Numbered, or replayed from a snapshot taken before the identity
        narrowed, a light of an LED the model now lacks is refused and
        lights nothing."""
        scope.illumination.led_on(channel='Green', illumination_ma=50)
        snapshot = scope.illumination.save_led_state('fusion-test')
        scope.illumination.led_off(channel='Green')
        scope.refresh_layer_identity(override_model='LS850-0')
        for light in (
            lambda: scope.illumination.led_on(channel=1, illumination_ma=50),
            lambda: scope.illumination.restore_led_state(snapshot),
        ):
            with pytest.raises(HardwareCommandRefusedError) as refused:
                light()
            assert refused.value.missing == MissingPart.led(1)
        assert scope.illumination.get_led_state('Green')['enabled'] is False


class TestOverrideScope:
    def test_override_is_session_scoped_not_sticky(self, scope):
        scope.refresh_layer_identity(override_model='LS850-0')
        assert scope.layer_identity.find('Green') is None
        scope.refresh_layer_identity()
        assert scope.layer_identity.find('Green') is not None
