# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A headless caller captures a z-stack by naming a layer.

The sibling of run_autofocus, on the same lane selector: one position,
one layer, the stack's shape read from settings. It differs in the four
things that make a stack a stack rather than a focus measurement -- it
expands into slices, it saves them, it does not autofocus, and it puts
the stage back where the stack was centred.

These pin the SURFACE: what reaches the prepare boundary, and what the
selector's config builds when the real protocol builder gets hold of it.
"""

import pathlib
from unittest.mock import MagicMock

import pytest

from modules.exceptions import ConfigError
from modules.protocol_state_machine import SequencedCaptureRunMode
from tests.test_composite_run_config import _settings

_POSITION = {'x': 1.0, 'y': 2.0, 'z': 3.0}


def _zstack_settings(**overrides):
    settings = _settings(acquiring=('BF',))
    # The store keeps the reference under 'position', holding the
    # spinner's display LABEL rather than the config token.
    settings['zstack'] = {
        'range': 20.0,
        'step_size': 5.0,
        'position': 'Current Position at Center',
    }
    settings.update(overrides)
    return settings


def _runner():
    from modules.protocol_runner import ProtocolRunner

    session = MagicMock()
    session.settings = _zstack_settings()
    session.get_current_plate_position.return_value = dict(_POSITION)
    session.objective_helper.get_objective_info.return_value = {'magnification': 10}
    return ProtocolRunner(session)


def _prepared(runner):
    return runner._executor.prepare.call_args.kwargs


def _built_config(runner):
    create = runner.session.scope.protocols.create_protocol
    return create.call_args.kwargs['input_config']


class TestTheRunIsAZStackOfItsOwn:
    def test_it_prepares_as_a_zstack(self):
        runner = _runner()
        runner.run_zstack(layer='BF')
        assert _prepared(runner)['run_mode'] is SequencedCaptureRunMode.SINGLE_ZSTACK

    def test_it_runs_exactly_one_scan(self):
        runner = _runner()
        runner.run_zstack(layer='BF')
        assert _prepared(runner)['max_scans'] == 1

    def test_the_trigger_source_names_the_api_caller(self):
        runner = _runner()
        runner.run_zstack(layer='BF')
        assert _prepared(runner)['run_trigger_source'] == 'api_zstack'

    def test_the_slices_are_saved(self):
        # The images are the product; a stack that saved nothing would
        # have run the stage for no result.
        runner = _runner()
        runner.run_zstack(layer='BF')
        assert _prepared(runner)['enable_image_saving'] is True

    def test_it_writes_its_run_artifacts(self):
        # Unlike an autofocus: this run produced data a user will come
        # back to, so it leaves the record of what it did.
        runner = _runner()
        runner.run_zstack(layer='BF')
        assert _prepared(runner)['disable_saving_artifacts'] is False

    def test_it_asks_for_no_autofocus_characterization_data(self):
        runner = _runner()
        runner.run_zstack(layer='BF')
        assert _prepared(runner)['save_autofocus_data'] is False

    def test_the_slices_land_under_the_manual_zstack_folder(self):
        runner = _runner()
        runner.run_zstack(layer='BF')
        parent = _prepared(runner)['parent_dir']
        assert parent.parts[-2:] == ('Manual', 'Z-Stacks')

    def test_a_stated_parent_dir_wins(self, tmp_path):
        runner = _runner()
        runner.run_zstack(layer='BF', parent_dir=tmp_path)
        assert _prepared(runner)['parent_dir'] == tmp_path

    def test_it_hands_the_illumination_back_as_it_found_it(self):
        runner = _runner()
        runner.run_zstack(layer='BF')
        assert _prepared(runner)['leds_state_at_end'] == 'return_to_original'


class TestTheStageGoesBackWhereTheStackWasCentred:
    def test_by_default_the_run_returns_to_the_starting_position(self):
        """A stack ends at whichever end of the range it finished on."""
        runner = _runner()
        runner.run_zstack(layer='BF')
        assert _prepared(runner)['return_to_position'] == _POSITION

    def test_a_caller_can_decline_the_return(self):
        runner = _runner()
        runner.run_zstack(layer='BF', return_to_start=False)
        assert _prepared(runner)['return_to_position'] is None

    def test_the_return_position_is_the_one_the_stack_was_built_around(self):
        # Read once and used for both, so the stack cannot be centred on
        # one position and returned to another.
        runner = _runner()
        runner.run_zstack(layer='BF')
        built = _built_config(runner)['positions'][0]
        returned = _prepared(runner)['return_to_position']
        assert (built['x'], built['y']) == (returned['x'], returned['y'])


class TestTheStackIsNotFlattenedByAnAutofocus:
    def test_autofocus_is_off_for_the_step(self):
        """Refocusing at each slice re-centres the range being swept."""
        runner = _runner()
        runner.run_zstack(layer='BF')
        assert _built_config(runner)['layer_configs']['BF']['autofocus'] is False

    def test_autofocus_stays_off_even_when_the_layer_has_it_stored_on(self):
        runner = _runner()
        runner.session.settings['BF']['autofocus'] = True
        runner.run_zstack(layer='BF')
        assert _built_config(runner)['layer_configs']['BF']['autofocus'] is False

    def test_the_method_offers_no_way_to_turn_it_on(self):
        import inspect

        from modules.protocol_runner import ProtocolRunner

        params = inspect.signature(ProtocolRunner.run_zstack).parameters
        assert 'autofocus' not in params


class TestTheStackIsConfiguredFromSettings:
    def test_zstacking_is_on(self):
        runner = _runner()
        runner.run_zstack(layer='BF')
        assert _built_config(runner)['use_zstacking'] is True

    def test_the_stack_parameters_come_from_settings(self):
        runner = _runner()
        runner.run_zstack(layer='BF')
        params = _built_config(runner)['zstack_params']
        assert params['range'] == 20.0
        assert params['step_size'] == 5.0

    def test_stimulation_configs_are_carried_as_stored(self):
        """GUI parity: every layer's config rides along, enabled or not.

        Stated rather than inherited silently -- if the API should carry
        only the enabled ones, this test is what changes with it.
        """
        import modules.config_helpers as config_helpers

        runner = _runner()
        runner.run_zstack(layer='BF')
        expected = config_helpers.get_stim_configs(runner.session.settings)
        assert _built_config(runner)['stim_config'] == expected

    def test_an_unknown_layer_is_refused_by_name(self):
        runner = _runner()
        with pytest.raises(ConfigError) as excinfo:
            runner.run_zstack(layer='Purple')
        assert 'Purple' in str(excinfo.value)

    def test_an_unknown_layer_refuses_before_the_engine_is_touched(self):
        runner = _runner()
        with pytest.raises(ConfigError):
            runner.run_zstack(layer='Purple')
        runner._executor.prepare.assert_not_called()
        runner._executor.start.assert_not_called()


class TestTheConfigActuallyBuildsASliceEach:
    """Fed to the REAL protocol builder.

    The rest of this file mocks the scope, so a config the builder turned
    into one step -- or rejected -- would leave every assertion above
    green while the stack was not a stack.
    """

    def _protocol(self, zstack=None):
        import modules.config_helpers as config_helpers
        from modules.lumascope_api import Lumascope
        from modules.protocol import Protocol

        settings = _zstack_settings()
        if zstack is not None:
            settings['zstack'] = zstack
        objective_helper = MagicMock()
        objective_helper.get_objective_info.return_value = {'magnification': 10}
        wellplate_loader = MagicMock()
        wellplate_loader.get_plate_list.return_value = ['96 well microplate']

        config = config_helpers.get_standalone_capture_config_from_settings(
            settings,
            objective_helper,
            wellplate_loader,
            layer='BF',
            position=dict(_POSITION),
            position_name='ZStack',
            autofocus=False,
            use_zstacking=True,
            stim_config={},
        )
        scope = Lumascope(simulate=True)
        try:
            return Protocol.from_config(
                input_config=config,
                tiling_configs_file_loc=(
                    pathlib.Path(__file__).parent.parent / 'data' / 'tiling.json'
                ),
                capabilities=scope.capabilities,
            )
        finally:
            scope.disconnect()

    def test_a_configured_stack_builds_more_than_one_slice(self):
        protocol = self._protocol()
        assert protocol.num_steps() > 1

    def test_every_slice_sits_at_its_own_z(self):
        protocol = self._protocol()
        z_values = protocol.steps()['Z'].tolist()
        assert len(set(z_values)) == len(z_values), 'slices must not share a Z'

    def test_no_slice_autofocuses(self):
        protocol = self._protocol()
        assert not any(protocol.steps()['Auto_Focus'].tolist())
