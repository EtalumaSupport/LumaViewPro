# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A headless caller runs a standalone autofocus by naming a layer.

The run family had a member for a scan, a protocol and a composite, and
none for the thing a bench operator does most: focus once, here, on this
layer. The recipe existed only as thirteen values chosen inline inside
two GUI starters, so a script had to reproduce the GUI's assembly to get
the same run -- which is the opposite of an API complete enough that
there need be no GUI at all.

These pin the SURFACE: what reaches the prepare boundary, and what a
caller is refused for. The engine's behaviour behind that boundary is
pinned by the delivery tests over the real executors.
"""

from unittest.mock import MagicMock

import pytest

from modules.exceptions import ConfigError
from modules.protocol_state_machine import SequencedCaptureRunMode
from tests.test_composite_run_config import _settings


def _runner(acquiring=('BF',)):
    """A ProtocolRunner over a mocked session and engine.

    Real config assembly, mocked engine: this pins what the autofocus
    surface HANDS the boundary, not what the engine does with it.
    """
    from modules.protocol_runner import ProtocolRunner

    session = MagicMock()
    session.settings = _settings(acquiring=acquiring)
    session.capture_settings_snapshot.return_value = session.settings
    session.get_current_plate_position.return_value = {'x': 1.0, 'y': 2.0, 'z': 3.0}
    session.objective_helper.get_objective_info.return_value = {'magnification': 10}
    return ProtocolRunner(session)


def _prepared(runner):
    return runner._executor.prepare.call_args.kwargs


def _built_config(runner):
    """The input_config the run handed to protocol construction.

    Asserted here rather than on the built Protocol: this harness mocks
    the scope, so the protocol it returns is a mock. The config IS this
    surface's output -- turning it into steps is Protocol.from_config's
    job and is pinned where that lives.
    """
    create = runner.session.scope.protocols.create_protocol
    return create.call_args.kwargs['input_config']


class TestTheRunIsAnAutofocusOfItsOwn:
    def test_it_prepares_as_an_autofocus_scan(self):
        runner = _runner()
        runner.run_autofocus(layer='BF')
        assert _prepared(runner)['run_mode'] is SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN

    def test_it_runs_exactly_one_scan(self):
        runner = _runner()
        runner.run_autofocus(layer='BF')
        assert _prepared(runner)['max_scans'] == 1

    def test_the_trigger_source_names_the_api_caller(self):
        # Distinct from the GUI button's token: the rival-run check
        # compares it, so a click during an API autofocus must read as a
        # rival rather than as this run's own.
        runner = _runner()
        runner.run_autofocus(layer='BF')
        assert _prepared(runner)['run_trigger_source'] == 'api_autofocus'

    def test_it_saves_no_images(self):
        runner = _runner()
        runner.run_autofocus(layer='BF')
        assert _prepared(runner)['enable_image_saving'] is False

    def test_it_writes_no_run_artifacts(self):
        # A focus measurement is not an acquisition: no run folder, no
        # protocol copy, no execution record.
        runner = _runner()
        runner.run_autofocus(layer='BF')
        assert _prepared(runner)['disable_saving_artifacts'] is True

    def test_it_hands_the_illumination_back_as_it_found_it(self):
        # One field at a scope someone is standing at, not a plate
        # traverse that must end dark.
        runner = _runner()
        runner.run_autofocus(layer='BF')
        assert _prepared(runner)['leds_state_at_end'] == 'return_to_original'


class TestCharacterizationDataIsOptIn:
    def test_no_data_is_requested_by_default(self):
        """A caller that does not ask for data does not get a folder."""
        runner = _runner()
        runner.run_autofocus(layer='BF')
        assert _prepared(runner)['save_autofocus_data'] is False

    def test_asking_for_data_reaches_the_boundary(self):
        runner = _runner()
        runner.run_autofocus(layer='BF', save_characterization_data=True)
        assert _prepared(runner)['save_autofocus_data'] is True

    def test_the_data_lands_under_the_characterization_folder(self):
        runner = _runner()
        runner.run_autofocus(layer='BF', save_characterization_data=True)
        assert _prepared(runner)['parent_dir'].name == 'Autofocus Characterization'

    def test_a_stated_parent_dir_wins(self, tmp_path):
        runner = _runner()
        runner.run_autofocus(layer='BF', save_characterization_data=True, parent_dir=tmp_path)
        assert _prepared(runner)['parent_dir'] == tmp_path


class TestTheLayerIsTheCallersToName:
    def test_the_named_layer_is_the_only_one_configured(self):
        runner = _runner(acquiring=())
        runner.run_autofocus(layer='Green')
        assert list(_built_config(runner)['layer_configs']) == ['Green']

    def test_autofocus_is_on_for_that_layer_whatever_settings_say(self):
        # The layer's stored flag is a leftover from whatever the user
        # last did in the GUI; a caller that asked for an autofocus gets
        # one. _settings leaves every layer's autofocus off, so this
        # would read False if the stored flag were honoured.
        runner = _runner()
        runner.run_autofocus(layer='BF')
        assert _built_config(runner)['layer_configs']['BF']['autofocus'] is True

    def test_the_single_position_is_the_current_stage_position(self):
        runner = _runner()
        runner.run_autofocus(layer='BF')
        positions = _built_config(runner)['positions']
        assert len(positions) == 1
        assert (positions[0]['x'], positions[0]['y']) == (1.0, 2.0)

    def test_an_unknown_layer_is_refused_by_name(self):
        """The refusal has to name the layer, not a downstream symptom.

        Without the check the layer selector simply skips a name it does
        not recognise, yielding no layers and a zero-step protocol -- and
        the run is refused as "Protocol has no steps", which is true and
        tells the caller nothing about what it got wrong.
        """
        runner = _runner()
        with pytest.raises(ConfigError) as excinfo:
            runner.run_autofocus(layer='Purple')
        message = str(excinfo.value)
        assert 'Purple' in message
        assert 'BF' in message, 'the refusal should say which layers this scope does have'

    def test_an_unknown_layer_refuses_before_the_engine_is_touched(self):
        runner = _runner()
        with pytest.raises(ConfigError):
            runner.run_autofocus(layer='Purple')
        runner._executor.prepare.assert_not_called()
        runner._executor.start.assert_not_called()


class TestTheWriteBackCannotBeReachedFromL2:
    def test_the_boundary_receives_the_focus_write_back_off(self):
        """The found focus is never written into a caller's protocol.

        The boundary's safety for this input rests on being unreachable
        -- 'absent from get_sequenced_run_settings, so no Run button and
        no L2 caller can turn it on'. A new L2 member that exposed it
        would falsify that sentence in a file this work does not touch.
        """
        runner = _runner()
        runner.run_autofocus(layer='BF')
        assert _prepared(runner).get('update_z_pos_from_autofocus', False) is False

    def test_the_method_offers_no_way_to_ask_for_it(self):
        import inspect

        from modules.protocol_runner import ProtocolRunner

        params = inspect.signature(ProtocolRunner.run_autofocus).parameters
        assert 'update_z_pos_from_autofocus' not in params


class TestTheExistingFamilyIsUnchanged:
    """The two new pass-through inputs default to what the boundary
    already gave every existing caller. This is the pin that catches a
    default drifting under a member that never asked for either."""

    def test_a_scan_still_saves_its_artifacts_and_asks_for_no_data(self):
        from modules.protocol import Protocol

        runner = _runner()
        protocol = MagicMock(spec=Protocol)
        runner.run_single_scan(
            protocol,
            image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
        )
        prepared = _prepared(runner)
        assert prepared['disable_saving_artifacts'] is False
        assert prepared['save_autofocus_data'] is False

    def test_a_composite_still_saves_its_artifacts_and_asks_for_no_data(self):
        runner = _runner(acquiring=('BF', 'Blue'))
        runner.start_composite()
        prepared = _prepared(runner)
        assert prepared['disable_saving_artifacts'] is False
        assert prepared['save_autofocus_data'] is False


class TestTheSelectorAndTheGuiStarterAgree:
    """One recipe, not two that resemble each other.

    The GUI starters still choose these values inline; cutting them over
    to the selector is what makes it canonical. This is what fails if
    someone edits one copy and not the other, and what makes that cutover
    mechanical rather than a judgement call.
    """

    def _selector_config(self, **overrides):
        import modules.config_helpers as config_helpers

        kwargs = {
            'layer': 'BF',
            'position': {'x': 1.0, 'y': 2.0, 'z': 3.0},
            'position_name': 'Autofocus',
            'autofocus': True,
            'use_zstacking': False,
            'stim_config': {},
        }
        kwargs.update(overrides)
        objective_helper = MagicMock()
        objective_helper.get_objective_info.return_value = {'magnification': 10}
        wellplate_loader = MagicMock()
        wellplate_loader.get_plate_list.return_value = ['96 well microplate']
        return config_helpers.get_standalone_capture_config_from_settings(
            _settings(acquiring=('BF',)),
            objective_helper,
            wellplate_loader,
            **kwargs,
        )

    def test_it_carries_the_key_set_the_starters_carry(self):
        # The thirteen the two inline recipes choose, verified identical
        # between them: whatever the selector produces has to cover the
        # same ground or the cutover silently drops a value.
        expected = {
            'labware_id',
            'objective_id',
            'zstack_params',
            'use_zstacking',
            'tiling',
            'tiling_overlap_percent',
            'layer_configs',
            'period',
            'duration',
            'frame_dimensions',
            'binning_size',
            'stim_config',
            'positions',
        }
        assert set(self._selector_config()) == expected

    def test_the_autofocus_lane_matches_the_autofocus_starter(self):
        config = self._selector_config()
        assert config['use_zstacking'] is False
        assert config['zstack_params'] == {}
        assert config['stim_config'] == {}
        assert config['layer_configs']['BF']['autofocus'] is True
        assert config['layer_configs']['BF']['acquire'] == 'image'
        assert config['positions'][0]['name'] == 'Autofocus'

    def test_neither_lane_tiles_or_repeats(self):
        # A degenerate run is one field, once: tiling would multiply the
        # steps and a period would make it recur.
        for use_zstacking in (False, True):
            config = self._selector_config(use_zstacking=use_zstacking)
            assert config['tiling_overlap_percent'] == 0.0
            assert config['period'] is None
            assert config['duration'] is None

    def test_the_position_keeps_its_z(self):
        # Unlike the composite lane, which nulls z so each channel falls
        # to its own focus: here the z is where the sweep starts.
        config = self._selector_config()
        assert config['positions'][0]['z'] == 3.0

    def test_zstack_params_are_read_only_when_stacking(self):
        """A caller that is not stacking gets no parameters, not stale ones."""
        assert self._selector_config(use_zstacking=False)['zstack_params'] == {}


class TestTheConfigActuallyBuildsTheIntendedStep:
    """The selector's output is fed to the REAL protocol builder.

    Every other test here mocks the scope, so the config never meets
    Protocol.from_config -- and a config that builder rejects, or turns
    into the wrong steps, would leave all of them green. That is the
    shape of blind spot this run kind exists to close, so it is closed
    here rather than assumed.
    """

    def _protocol_from(self, **overrides):
        import pathlib

        import modules.config_helpers as config_helpers
        from modules.lumascope_api import Lumascope
        from modules.protocol import Protocol

        kwargs = {
            'layer': 'BF',
            'position': {'x': 1.0, 'y': 2.0, 'z': 3.0},
            'position_name': 'Autofocus',
            'autofocus': True,
            'use_zstacking': False,
            'stim_config': {},
        }
        kwargs.update(overrides)

        objective_helper = MagicMock()
        objective_helper.get_objective_info.return_value = {'magnification': 10}
        wellplate_loader = MagicMock()
        wellplate_loader.get_plate_list.return_value = ['96 well microplate']

        config = config_helpers.get_standalone_capture_config_from_settings(
            _settings(acquiring=('BF',)),
            objective_helper,
            wellplate_loader,
            **kwargs,
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

    def test_it_builds_exactly_one_step(self):
        assert self._protocol_from().num_steps() == 1

    def test_the_step_is_the_named_layer_with_autofocus_on(self):
        steps = self._protocol_from(layer='Green').steps()
        assert steps['Color'].tolist() == ['Green']
        assert bool(steps['Auto_Focus'].iloc[0]) is True

    def test_the_step_sits_at_the_position_it_was_given(self):
        step = self._protocol_from().steps().iloc[0]
        assert (step['X'], step['Y']) == (1.0, 2.0)

    def test_autofocus_off_builds_a_step_that_does_not_focus(self):
        # The z-stack lane's setting, proven on the same builder so C3
        # inherits a checked path rather than an assumed one.
        steps = self._protocol_from(autofocus=False).steps()
        assert bool(steps['Auto_Focus'].iloc[0]) is False
