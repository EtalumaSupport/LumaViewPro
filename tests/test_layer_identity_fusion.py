# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The identity/drivability fusion and the two-resolver split.

Identity (the layer record) answers what a layer IS; the driver answers
which addresses the board can DRIVE; the state store is keyed by address
and names each through the identity at the read. These tests pin the
seams that keep those from bleeding into each other: an ON request for a
layer identity lacks fails loudly by name, an OFF stays a silent no-op,
a no-LED layer resolves to no channel without being an error, and -- the
load-bearing one -- an address that physically lit stays in the state
store and is reported by its address when the CURRENT identity can no
longer name it, so a mid-session identity change can never strand a lit
LED outside the extinguish machinery.
"""

import pytest

from modules.exceptions import ArgumentRefusedError, HardwareCommandRefusedError, MissingPart
from modules.layer_record import UNRESOLVED
from modules.lumascope_api.illumination import LedReading, LedState
from tests.scope_fakes import build_scope, led_reading


@pytest.fixture
def scope():
    s = build_scope(simulate=True, register_atexit=False, sim_model='LS850T')
    yield s
    s.disconnect()


class TestFusion:
    def test_identity_layer_on_drivable_channel_lights(self, scope):
        scope.illumination.led_on(channel='Green', illumination_ma=50)
        assert scope.illumination.get_led_state('Green').state is LedState.LIT
        scope.illumination.led_off(channel='Green')

    @pytest.mark.parametrize('member', ['led_on', 'led_off'])
    def test_a_name_that_is_no_layer_is_refused_by_name(self, scope, member):
        call = {
            'led_on': lambda: scope.illumination.led_on(channel='Skylight', illumination_ma=50),
            'led_off': lambda: scope.illumination.led_off(channel='Skylight'),
        }[member]
        with pytest.raises(ArgumentRefusedError) as refused:
            call()
        assert (refused.value.reason, refused.value.value) == ('channel_unknown', 'Skylight')

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
        assert scope.channel_identity.find('Lumi') is not None
        assert scope.illumination.color2ch('Lumi') is None
        scope.illumination.led_off(channel='Lumi')


class TestTwoResolverSplit:
    def test_identity_answers_by_record(self, scope):
        assert scope.illumination.color2ch('Green') == 1
        assert scope.illumination.ch2color(1) == 'Green'
        scope.refresh_layer_identity(override_model='LS850-0')
        assert scope.illumination.color2ch('Green') is None
        assert scope.illumination.ch2color(1) is None

    def test_lit_channel_stays_recorded_under_narrowed_identity(self, scope):
        """An address lit before the identity narrowed stays in the state
        store, reported by its address with no name, and the extinguish
        that rides that record still reaches it."""
        scope.illumination.led_on(channel='Green', illumination_ma=50)
        scope.refresh_layer_identity(override_model='LS850-0')
        assert LedReading(1, None, LedState.LIT, 50.0) in scope.illumination.get_led_states()
        scope.illumination.led_off_by_address(1)
        assert led_reading(scope, 1).state is LedState.DARK

    def test_an_address_no_identity_names_is_reported_lit_by_address(self, scope):
        """With no layer identity every address is the board's: one lit,
        6 among them, is held and reported by its address, never dropped.
        The listeners are told the same reading."""
        scope.channel_identity = UNRESOLVED  # a unit with no resolvable layers
        heard = []
        scope.illumination.add_led_listener(lambda *reading: heard.append(LedReading(*reading)))
        scope.illumination.led_on_by_address(6, 50)
        lit = LedReading(6, None, LedState.LIT, 50.0)
        assert lit in scope.illumination.get_led_states()
        assert heard == [lit]

    def test_a_light_the_narrowed_identity_lacks_is_refused(self, scope):
        """Numbered, or replayed from a snapshot taken before the identity
        narrowed, a light of an LED the model now lacks is refused and
        lights nothing."""
        scope.illumination.led_on(channel='Green', illumination_ma=50)
        snapshot = scope.illumination.save_led_state('fusion-test')
        scope.illumination.led_off(channel='Green')
        scope.refresh_layer_identity(override_model='LS850-0')
        for light in (
            lambda: scope.illumination.led_on_by_address(1, 50),
            lambda: scope.illumination.restore_led_state(snapshot),
        ):
            with pytest.raises(HardwareCommandRefusedError) as refused:
                light()
            assert refused.value.missing == MissingPart.led_address(1)
        assert led_reading(scope, 1).state is LedState.DARK


class TestOverrideScope:
    def test_override_is_session_scoped_not_sticky(self, scope):
        scope.refresh_layer_identity(override_model='LS850-0')
        assert scope.channel_identity.find('Green') is None
        scope.refresh_layer_identity()
        assert scope.channel_identity.find('Green') is not None
