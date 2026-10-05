# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Save Focus and Apply Focus to Steps are Session members, refused at the API.

The layer panel's handlers owned both decisions: which Z to read and
whether it could be trusted, the layer focus write, which step took the Z
and which were left alone. A script or REST caller could do neither, and a
scope with no Z axis saved Z 0 as a focus (the refusal it was owed was the
GUI hiding the button). The Session now does both, the protocols API
refuses what cannot be saved, and the GUI passes the selected step.
"""

import logging

import pytest

from modules.exceptions import (
    AxisStateUnknownError,
    ConfigError,
    ProtocolError,
    ProtocolRunRefusedError,
)
from modules.lumascope_api import AxisState
from modules.scope_session import SavedFocus, ScopeSession
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

LAYER_FOCUS = 7000.0


def _session(tmp_path, microscope):
    settings = complete_settings(live_folder=str(tmp_path), microscope=microscope)
    for layer in ('BF', 'Blue'):
        settings[layer]['acquire'] = 'image'
        settings[layer]['focus'] = LAYER_FOCUS
    return ScopeSession.create(settings, simulate=True)


def _shutdown(session):
    try:
        session.shutdown()
    except Exception:
        logging.getLogger(__name__).debug('teardown noise is not the measurement', exc_info=True)


@pytest.fixture
def session(tmp_path):
    built = _session(tmp_path, 'LS850')
    home_sim_scope(built.scope)
    yield built
    _shutdown(built)


def _protocol_bf_bf_blue(session):
    """Three steps, all at the layer focus: BF, BF, Blue."""
    protocol = session.create_empty_protocol()
    for x_um in (10000, 20000):
        session.scope.motion.move_absolute('X', x_um)
        session.set_layer_acquire('Blue', None)
        session.add_step(protocol, after_step=protocol.num_steps() - 1)
    session.set_layer_acquire('Blue', 'image')
    session.set_layer_acquire('BF', None)
    session.add_step(protocol, after_step=protocol.num_steps() - 1)
    session.set_layer_acquire('BF', 'image')
    assert [protocol.step(i)['Color'] for i in range(3)] == ['BF', 'BF', 'Blue']
    assert [protocol.step(i)['Z'] for i in range(3)] == [LAYER_FOCUS] * 3
    return protocol


def _go_to_z(session, z):
    session.scope.motion.move_absolute('Z', z)


def _zs(protocol):
    return [protocol.step(i)['Z'] for i in range(protocol.num_steps())]


class TestSaveFocus:
    def test_the_selected_step_of_the_layer_takes_the_z_and_no_other(self, session):
        protocol = _protocol_bf_bf_blue(session)
        _go_to_z(session, 5000)

        saved = session.save_focus(protocol, 'BF', step_idx=1)

        assert saved == SavedFocus(z=5000.0, step_idx=1)
        assert session.saved_focus('BF') == 5000.0
        assert _zs(protocol) == [LAYER_FOCUS, 5000.0, LAYER_FOCUS]

    def test_three_saves_on_three_steps_stay_three_values(self, session):
        # #734: saving step after step collapsed every step to the last save.
        protocol = _protocol_bf_bf_blue(session)
        _go_to_z(session, 5000)
        session.save_focus(protocol, 'BF', step_idx=0)
        _go_to_z(session, 6000)
        session.save_focus(protocol, 'BF', step_idx=1)

        assert _zs(protocol) == [5000.0, 6000.0, LAYER_FOCUS]

    def test_a_step_of_another_channel_is_left_alone(self, session):
        protocol = _protocol_bf_bf_blue(session)
        _go_to_z(session, 5000)

        saved = session.save_focus(protocol, 'BF', step_idx=2)

        assert saved == SavedFocus(z=5000.0, step_idx=None)
        assert session.saved_focus('BF') == 5000.0
        assert _zs(protocol) == [LAYER_FOCUS] * 3

    def test_no_step_saves_the_layer_focus_only(self, session):
        protocol = _protocol_bf_bf_blue(session)
        _go_to_z(session, 5000)

        saved = session.save_focus(protocol, 'BF')

        assert saved == SavedFocus(z=5000.0, step_idx=None)
        assert session.saved_focus('BF') == 5000.0
        assert _zs(protocol) == [LAYER_FOCUS] * 3

    def test_a_step_the_protocol_lacks_is_refused_and_nothing_is_written(self, session):
        protocol = _protocol_bf_bf_blue(session)
        _go_to_z(session, 5000)

        with pytest.raises(ProtocolError):
            session.save_focus(protocol, 'BF', step_idx=3)

        assert session.saved_focus('BF') == LAYER_FOCUS
        assert _zs(protocol) == [LAYER_FOCUS] * 3

    def test_a_z_that_does_not_know_its_position_saves_nothing(self, session):
        protocol = _protocol_bf_bf_blue(session)
        session.scope.motion._axis_state['Z'] = AxisState.UNKNOWN

        with pytest.raises(AxisStateUnknownError):
            session.save_focus(protocol, 'BF', step_idx=0)

        assert session.saved_focus('BF') == LAYER_FOCUS
        assert _zs(protocol) == [LAYER_FOCUS] * 3


class TestApplyFocusToLayerSteps:
    def test_every_step_of_the_layer_takes_the_z(self, session):
        protocol = _protocol_bf_bf_blue(session)
        _go_to_z(session, 5000)

        updated = session.apply_focus_to_layer_steps(protocol, 'BF')

        assert updated == 2
        assert session.saved_focus('BF') == 5000.0
        assert _zs(protocol) == [5000.0, 5000.0, LAYER_FOCUS]

    def test_a_z_that_does_not_know_its_position_applies_nothing(self, session):
        protocol = _protocol_bf_bf_blue(session)
        session.scope.motion._axis_state['Z'] = AxisState.UNKNOWN

        with pytest.raises(AxisStateUnknownError):
            session.apply_focus_to_layer_steps(protocol, 'BF')

        assert session.saved_focus('BF') == LAYER_FOCUS
        assert _zs(protocol) == [LAYER_FOCUS] * 3


class TestALayerThisScopeLacks:
    """A focus is saved for a layer of this scope; any other name is refused first."""

    @pytest.mark.parametrize('layer', ['Purple', 'Lumi'])
    def test_save_focus_is_refused_and_nothing_is_written(self, session, layer):
        protocol = _protocol_bf_bf_blue(session)
        before = {k: v.get('focus') for k, v in session.settings.items() if isinstance(v, dict)}

        with pytest.raises(ConfigError, match=f'no {layer} layer'):
            session.save_focus(protocol, layer, step_idx=0)

        after = {k: v.get('focus') for k, v in session.settings.items() if isinstance(v, dict)}
        assert after == before
        assert _zs(protocol) == [LAYER_FOCUS] * 3

    @pytest.mark.parametrize('layer', ['Purple', 'Lumi'])
    def test_apply_focus_is_refused_and_nothing_is_written(self, session, layer):
        protocol = _protocol_bf_bf_blue(session)

        with pytest.raises(ConfigError, match=f'no {layer} layer'):
            session.apply_focus_to_layer_steps(protocol, layer)

        assert _zs(protocol) == [LAYER_FOCUS] * 3


class TestAScopeWithNoZ:
    """LS620: no motor controller, so no Z to save a focus from."""

    @pytest.fixture
    def no_z(self, tmp_path):
        built = _session(tmp_path, 'LS620')
        assert not built.scope.capabilities.has_focus
        yield built
        _shutdown(built)

    def test_save_focus_is_refused_and_the_focus_is_kept(self, no_z):
        protocol = no_z.create_empty_protocol()

        with pytest.raises(ProtocolRunRefusedError) as refused:
            no_z.save_focus(protocol, 'BF')

        assert refused.value.reason == 'positions_unreachable'
        assert no_z.saved_focus('BF') == LAYER_FOCUS

    def test_apply_focus_is_refused_and_the_focus_is_kept(self, no_z):
        protocol = no_z.create_empty_protocol()

        with pytest.raises(ProtocolRunRefusedError) as refused:
            no_z.apply_focus_to_layer_steps(protocol, 'BF')

        assert refused.value.reason == 'positions_unreachable'
        assert no_z.saved_focus('BF') == LAYER_FOCUS
