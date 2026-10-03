# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Step data read from a protocol cannot change it.

A protocol changes only through its writers, which judge and record each
change. ``steps()`` handed out the stored frame, so a caller's write landed
in the protocol unjudged; ``steps().copy()`` and ``copy_for_execution()``
copied the frame but shared the dicts inside its cells, so a write into a
step's Stim_Config or Video Config landed in the caller's protocol as
well, including from a run's own copy.

``steps()`` and the run's copy hand out a copy of the frame. The dicts in
its cells are read-only wherever they are, so a write into one raises
rather than landing anywhere; copying the frame stays as cheap as copying
a table, which a deep copy of every cell was not. A deep copy of a cell is
a plain dict, for the caller that means to edit its own.
"""

import copy
import pickle

import pandas as pd
import pytest

from modules.protocol import Protocol
from tests.test_protocol_roundtrip import TILING_CONFIGS, _build_protocol, _make_step

DICT_COLUMNS = ['Stim_Config', 'Video Config']


@pytest.fixture
def protocol() -> Protocol:
    return _build_protocol(
        [
            _make_step(name='A1_BF', color='BF', acquire='image'),
            _make_step(name='A1_Blue', color='Blue', acquire='video'),
        ]
    )


def _snapshot(protocol: Protocol) -> pd.DataFrame:
    return protocol._config['steps'].copy(deep=True).map(repr)


def _writes(cell: dict):
    """Every way a caller could change a dict cell in place."""
    yield lambda: cell.__setitem__('written', True)
    yield lambda: cell.update(written=True)
    yield lambda: cell.setdefault('written', True)
    yield lambda: cell.pop(next(iter(cell)))
    yield lambda: cell.clear()


class TestReadersHandOutCopies:
    def test_a_write_to_the_frame_steps_returns_leaves_the_protocol(self, protocol):
        before = _snapshot(protocol)

        protocol.steps().loc[0, 'X'] = 999.0

        pd.testing.assert_frame_equal(_snapshot(protocol), before)

    @pytest.mark.parametrize('column', DICT_COLUMNS)
    def test_a_write_into_a_dict_cell_steps_returns_is_refused(self, protocol, column):
        before = _snapshot(protocol)

        for write in _writes(protocol.steps().iloc[0][column]):
            with pytest.raises(TypeError):
                write()

        pd.testing.assert_frame_equal(_snapshot(protocol), before)

    def test_a_write_into_a_nested_stim_dict_is_refused(self, protocol):
        before = _snapshot(protocol)
        stim = protocol.step(0)['Stim_Config']
        layer = next(iter(stim))

        with pytest.raises(TypeError):
            stim[layer]['enabled'] = True

        pd.testing.assert_frame_equal(_snapshot(protocol), before)

    @pytest.mark.parametrize('column', DICT_COLUMNS)
    def test_a_write_into_a_dict_one_step_carries_is_refused(self, protocol, column):
        before = _snapshot(protocol)

        with pytest.raises(TypeError):
            protocol.step(0)[column]['written'] = True

        pd.testing.assert_frame_equal(_snapshot(protocol), before)

    def test_a_write_into_the_layer_settings_leaves_the_protocol(self, tmp_path, protocol):
        path = tmp_path / 'with_block.tsv'
        protocol.to_file(
            path,
            layer_settings={
                'BF': {
                    'Layer': 'BF',
                    'Acquire': 'image',
                    'Illumination': 5.0,
                    'Gain': 0.0,
                    'Auto_Gain': False,
                    'Exposure': 2.0,
                    'False_Color': False,
                    'Sum': 1,
                    'Stim_Enabled': '',
                }
            },
        )
        loaded = Protocol.from_file(path, tiling_configs_file_loc=TILING_CONFIGS)

        loaded.layer_settings()['BF']['Illumination'] = 999.0

        assert loaded.layer_settings()['BF']['Illumination'] == 5.0


class TestEveryStoredCellIsReadOnly:
    @pytest.mark.parametrize('column', DICT_COLUMNS)
    def test_a_loaded_protocols_cells_are_read_only(self, tmp_path, protocol, column):
        path = tmp_path / 'saved.tsv'
        protocol.to_file(path)
        loaded = Protocol.from_file(path, tiling_configs_file_loc=TILING_CONFIGS)

        with pytest.raises(TypeError):
            loaded.step(0)[column]['written'] = True

    @pytest.mark.parametrize('column', DICT_COLUMNS)
    def test_a_modified_steps_cells_are_read_only(self, protocol, column):
        step = protocol.step(0)
        protocol.modify_step(
            step_idx=0,
            layer='BF',
            layer_config={
                'autofocus': False,
                'false_color': False,
                'illumination_ma': 5.0,
                'gain_db': 0.0,
                'auto_gain': False,
                'exposure_ms': 2.0,
                'sum': 1,
                'acquire': 'image',
                'video_config': {'fps': 5, 'duration': 5},
            },
            plate_position={'x': step['X'], 'y': step['Y'], 'z': step['Z']},
            objective_id=step['Objective'],
            stim_configs={'Blue': {'enabled': False}},
        )

        with pytest.raises(TypeError):
            protocol.step(0)[column]['written'] = True


class TestACopyOfACellIsTheCallersOwn:
    @pytest.mark.parametrize('column', DICT_COLUMNS)
    def test_a_deep_copy_is_a_plain_dict_the_caller_can_edit(self, protocol, column):
        cell = protocol.step(0)[column]

        own = copy.deepcopy(cell)
        own['written'] = True

        assert type(own) is dict
        assert 'written' not in protocol.step(0)[column]
        assert own == {**cell, 'written': True}

    def test_a_deep_copy_of_a_stim_config_is_plain_all_the_way_down(self, protocol):
        own = copy.deepcopy(protocol.step(0)['Stim_Config'])

        assert all(type(v) is dict for v in own.values() if isinstance(v, dict))

    @pytest.mark.parametrize('column', DICT_COLUMNS)
    def test_a_cell_survives_pickle_unchanged_and_read_only(self, protocol, column):
        cell = protocol.step(0)[column]

        back = pickle.loads(pickle.dumps(cell))

        assert back == cell
        with pytest.raises(TypeError):
            back['written'] = True


class TestTheRunsCopyIsItsOwn:
    @pytest.mark.parametrize('column', DICT_COLUMNS)
    def test_a_write_into_the_run_copys_dict_cell_is_refused(self, protocol, column):
        before = _snapshot(protocol)
        run_copy = protocol.copy_for_execution()

        with pytest.raises(TypeError):
            run_copy._config['steps'].iloc[0][column]['written'] = True

        pd.testing.assert_frame_equal(_snapshot(protocol), before)

    def test_the_run_copys_layer_settings_are_its_own(self, tmp_path, protocol):
        protocol._config['layer_settings'] = {'BF': {'Layer': 'BF', 'Acquire': 'image'}}
        run_copy = protocol.copy_for_execution()

        run_copy._config['layer_settings']['BF']['Acquire'] = 'video'

        assert protocol._config['layer_settings']['BF']['Acquire'] == 'image'
