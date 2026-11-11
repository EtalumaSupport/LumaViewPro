# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The identity/drivability fusion and the two-resolver split.

Identity (the layer record) answers what a layer IS; the driver answers
which addresses the board can DRIVE; the state store is keyed by address
and names each through the identity at the read. These tests pin the
seams that keep those from bleeding into each other: an ON request for a
layer identity lacks fails loudly by name, an OFF stays a silent no-op,
a no-LED layer resolves to no channel without being an error, and an
address no identity names is held and reported by its address. A scope's
identity is resolved once, when it is built, so each model is a scope
built as that model.
"""

import pytest

from _lumascope.modules.exceptions import (
    ArgumentRefusedError,
    HardwareCommandRefusedError,
    MissingPart,
)
from _lumascope.modules.channel_record import UNRESOLVED
from _lumascope.modules.lumascope_api.illumination import LedReading, LedState
from tests.scope_fakes import build_scope, channel_reading, led_reading
from tests.stock_channels import LED1


@pytest.fixture
def scope():
    s = build_scope(simulate=True, register_atexit=False, sim_model='LS850T')
    yield s
    s.disconnect()


@pytest.fixture
def ls850_0():
    """A simulated LS850-0: brightfield only, so no channel drives LED 1."""
    s = build_scope(simulate=True, register_atexit=False, sim_model='LS850-0')
    yield s
    s.disconnect()


class TestFusion:
    def test_identity_layer_on_drivable_channel_lights(self, scope):
        scope.illumination.led_on(channel=LED1, illumination_ma=50)
        assert channel_reading(scope.illumination, LED1).state is LedState.LIT
        scope.illumination.led_off(channel=LED1)

    @pytest.mark.parametrize('member', ['led_on', 'led_off'])
    def test_a_name_that_is_no_layer_is_refused_by_name(self, scope, member):
        call = {
            'led_on': lambda: scope.illumination.led_on(channel='Skylight', illumination_ma=50),
            'led_off': lambda: scope.illumination.led_off(channel='Skylight'),
        }[member]
        with pytest.raises(ArgumentRefusedError) as refused:
            call()
        assert (refused.value.reason, refused.value.value) == ('channel_unknown', 'Skylight')

    def test_a_layer_the_model_lacks_is_refused_on_by_name(self, ls850_0):
        with pytest.raises(HardwareCommandRefusedError, match=LED1) as refused:
            ls850_0.illumination.led_on(channel=LED1, illumination_ma=50)
        assert refused.value.missing == MissingPart.led(LED1)

    def test_a_layer_the_model_lacks_is_turned_off_as_a_no_op(self, ls850_0):
        ls850_0.illumination.led_off(channel=LED1)
        assert led_reading(ls850_0, 1).state is LedState.DARK

    def test_no_led_layer_resolves_to_no_channel_without_error(self):
        lumi = build_scope(simulate=True, register_atexit=False, sim_model='Lumi')
        try:
            assert lumi.channel_identity.find('Lumi') is not None
            assert lumi.illumination.led_address_of('Lumi') is None
            lumi.illumination.led_off(channel='Lumi')
        finally:
            lumi.disconnect()


class TestTwoResolverSplit:
    def test_identity_answers_by_record(self, scope, ls850_0):
        assert scope.illumination.led_address_of(LED1) == 1
        assert scope.channel_identity.key_at(1) == LED1
        assert ls850_0.illumination.led_address_of(LED1) is None
        assert ls850_0.channel_identity.key_at(1) is None

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

    def test_a_snapshot_lighting_an_led_the_model_lacks_is_refused(self, scope, ls850_0):
        """Replayed from a snapshot taken on a scope that has the LED, a
        light of an LED this model lacks is refused and lights nothing."""
        scope.illumination.led_on(channel=LED1, illumination_ma=50)
        snapshot = scope.illumination.get_led_states()
        scope.illumination.led_off(channel=LED1)

        with pytest.raises(HardwareCommandRefusedError) as refused:
            ls850_0.illumination.restore_led_state(snapshot)

        assert refused.value.missing == MissingPart.led_address(1)
        assert led_reading(ls850_0, 1).state is LedState.DARK
