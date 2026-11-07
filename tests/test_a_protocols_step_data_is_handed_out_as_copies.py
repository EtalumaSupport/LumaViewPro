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
import dataclasses
import pickle

import pandas as pd
import pytest

from _lumascope.modules.acquire_mode import AcquireMode
from _lumascope.modules.protocol import ChannelSettings, Protocol, ReadOnlyDict
from tests.test_protocol_roundtrip import _build_protocol, _make_step
from _lumascope.modules.coord_transformations import PlateLocation
from tests.stock_channels import LED0
from tests.protocol_fixtures import protocol_from_file
from tests.test_adding_a_step_is_an_api_capability import session  # noqa: F401 -- pytest fixture

DICT_COLUMNS = ['Stim_Config', 'Video Config']


@pytest.fixture
def protocol() -> Protocol:
    return _build_protocol(
        [
            _make_step(name='A1_BF', color='BF', acquire='image'),
            _make_step(name=f'A1_{LED0}', color=LED0, acquire='video'),
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
        stim = protocol.step_row(0)['Stim_Config']
        layer = next(iter(stim))

        with pytest.raises(TypeError):
            stim[layer]['enabled'] = True

        pd.testing.assert_frame_equal(_snapshot(protocol), before)

    @pytest.mark.parametrize('column', DICT_COLUMNS)
    def test_a_write_into_a_dict_one_step_carries_is_refused(self, protocol, column):
        before = _snapshot(protocol)

        with pytest.raises(TypeError):
            protocol.step_row(0)[column]['written'] = True

        pd.testing.assert_frame_equal(_snapshot(protocol), before)

    def test_a_write_into_a_channels_settings_is_refused(self, tmp_path, protocol):
        path = tmp_path / 'with_block.tsv'
        protocol.to_file(
            path,
            channel_settings={
                'BF': ChannelSettings(
                    acquire=AcquireMode.IMAGE,
                    illumination_ma=5.0,
                    gain_db=0.0,
                    auto_gain=False,
                    exposure_ms=2.0,
                    false_color=False,
                    frames_summed=1,
                    stim_enabled=None,
                )
            },
        )
        loaded = protocol_from_file(
            path,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            loaded.channel_settings()['BF'].illumination_ma = 999.0
        loaded.channel_settings()['BF'] = None

        assert loaded.channel_settings()['BF'].illumination_ma == 5.0


class TestEveryStoredCellIsReadOnly:
    @pytest.mark.parametrize('column', DICT_COLUMNS)
    def test_a_loaded_protocols_cells_are_read_only(self, tmp_path, protocol, column):
        path = tmp_path / 'saved.tsv'
        protocol.to_file(path)
        loaded = protocol_from_file(
            path,
        )

        with pytest.raises(TypeError):
            loaded.step_row(0)[column]['written'] = True

    @pytest.mark.parametrize('column', DICT_COLUMNS)
    def test_a_modified_steps_cells_are_read_only(self, protocol, column):
        step = protocol.step_row(0)
        protocol.modify_step(
            turret_slot=-1,
            step_index=0,
            channel='BF',
            channel_config={
                'autofocus': False,
                'false_color': False,
                'illumination_ma': 5.0,
                'gain_db': 0.0,
                'auto_gain': False,
                'exposure_ms': 2.0,
                'sum': 1,
                'acquire': 'image',
                'video_config': {'duration': 5},
            },
            plate_position=PlateLocation(x_mm=step['X'], y_mm=step['Y'], z_um=step['Z']),
            objective_id=step['Objective'],
            stim_configs={LED0: {'enabled': False}},
        )

        with pytest.raises(TypeError):
            protocol.step_row(0)[column]['written'] = True


class TestACopyOfACellIsTheCallersOwn:
    @pytest.mark.parametrize('column', DICT_COLUMNS)
    def test_a_deep_copy_is_a_plain_dict_the_caller_can_edit(self, protocol, column):
        cell = protocol.step_row(0)[column]

        own = copy.deepcopy(cell)
        own['written'] = True

        assert type(own) is dict
        assert 'written' not in protocol.step_row(0)[column]
        assert own == {**cell, 'written': True}

    def test_a_deep_copy_of_a_stim_config_is_plain_all_the_way_down(self, protocol):
        own = copy.deepcopy(protocol.step_row(0)['Stim_Config'])

        assert all(type(v) is dict for v in own.values() if isinstance(v, dict))

    @pytest.mark.parametrize('column', DICT_COLUMNS)
    def test_a_cell_survives_pickle_unchanged_and_read_only(self, protocol, column):
        cell = protocol.step_row(0)[column]

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


class TestACellIsFrozenAllTheWayDown:
    """A list in a config cell is a tuple: a cell cannot change through anything it holds."""

    @pytest.fixture
    def with_a_list(self) -> Protocol:
        return _build_protocol(
            [_make_step(name='A1_BF', color='BF', video_config={'duration': 5, 'notes': [1, 2]})]
        )

    def test_a_list_in_a_cell_cannot_be_appended_to(self, with_a_list):
        notes = with_a_list.step_row(0)['Video Config']['notes']

        assert notes == (1, 2)
        with pytest.raises(AttributeError):
            notes.append(99)

    def test_a_list_in_a_cell_is_saved_and_loaded_as_the_list_it_was(self, tmp_path, with_a_list):
        path = tmp_path / 'list.tsv'
        with_a_list.to_file(path)

        assert '""notes"": [1, 2]' in path.read_text()
        loaded = protocol_from_file(path)
        assert loaded.step_row(0)['Video Config'] == {'duration': 5, 'notes': (1, 2)}

    def test_a_deep_copy_of_a_cell_gives_its_lists_back(self, with_a_list):
        own = copy.deepcopy(with_a_list.step_row(0)['Video Config'])

        own['notes'].append(3)

        assert own == {'duration': 5, 'notes': [1, 2, 3]}
        assert with_a_list.step_row(0)['Video Config']['notes'] == (1, 2)

    def test_a_cell_built_from_a_mutable_dict_shares_nothing_with_it(self):
        given = {'stim': {'enabled': False}, 'notes': [1]}
        cell = ReadOnlyDict(given)
        given['stim']['enabled'] = True
        given['notes'].append(2)

        assert cell == {'stim': {'enabled': False}, 'notes': (1,)}
        with pytest.raises(TypeError):
            cell['stim']['enabled'] = True


class TestAWritersConfigIsCopiedOnce:
    """A writer's plain dict is copied where it is read; a stored cell is shared, never copied again."""

    def test_a_write_to_the_callers_config_after_a_modify_leaves_the_protocol(self, protocol):
        step = protocol.step_row(0)
        video_config = {'duration': 5}
        stim_configs = {LED0: {'enabled': False}}
        protocol.modify_step(
            turret_slot=-1,
            step_index=0,
            channel='BF',
            channel_config={
                'autofocus': False,
                'false_color': False,
                'illumination_ma': 5.0,
                'gain_db': 0.0,
                'auto_gain': False,
                'exposure_ms': 2.0,
                'sum': 1,
                'acquire': 'image',
                'video_config': video_config,
            },
            plate_position=PlateLocation(x_mm=step['X'], y_mm=step['Y'], z_um=step['Z']),
            objective_id=step['Objective'],
            stim_configs=stim_configs,
        )

        video_config['duration'] = 99
        stim_configs[LED0]['enabled'] = True

        assert protocol.step_row(0)['Video Config'] == {'duration': 5}
        assert protocol.step_row(0)['Stim_Config'] == {LED0: {'enabled': False}}

    def test_the_tiles_of_one_step_share_its_config_cells(self, session):
        protocol = session.create_empty_protocol()
        session.add_step(
            protocol, channels=('BF',), at=PlateLocation(x_mm=40.0, y_mm=40.0, z_um=5000.0)
        )

        session.apply_tiling(protocol, '2x2')

        assert protocol.num_steps() == 4
        for column in DICT_COLUMNS:
            cells = [protocol.step_row(i)[column] for i in range(4)]
            assert all(cell is cells[0] for cell in cells), column
