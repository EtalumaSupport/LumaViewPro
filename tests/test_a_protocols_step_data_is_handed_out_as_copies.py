# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Step data read from a protocol is a copy, all the way down.

A protocol changes only through its writers, which judge and record each
change. ``steps()`` handed out the stored frame, so a caller's write landed
in the protocol unjudged; ``steps().copy()`` and ``copy_for_execution()``
copied the frame but shared the dicts inside its cells, so a write into a
step's Stim_Config or Video Config landed in the caller's protocol as
well, including from a run's own copy.
"""

import pandas as pd
import pytest

from modules.protocol import Protocol
from tests.test_protocol_roundtrip import TILING_CONFIGS, _build_protocol, _make_step


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


class TestReadersHandOutCopies:
    def test_a_write_to_the_frame_steps_returns_leaves_the_protocol(self, protocol):
        before = _snapshot(protocol)

        protocol.steps().loc[0, 'X'] = 999.0

        pd.testing.assert_frame_equal(_snapshot(protocol), before)

    @pytest.mark.parametrize('column', ['Stim_Config', 'Video Config'])
    def test_a_write_into_a_dict_cell_steps_returns_leaves_the_protocol(self, protocol, column):
        before = _snapshot(protocol)

        protocol.steps().iloc[0][column]['written'] = True

        pd.testing.assert_frame_equal(_snapshot(protocol), before)

    @pytest.mark.parametrize('column', ['Stim_Config', 'Video Config'])
    def test_a_write_into_a_dict_one_step_carries_leaves_the_protocol(self, protocol, column):
        before = _snapshot(protocol)

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


class TestTheRunsCopyIsItsOwn:
    @pytest.mark.parametrize('column', ['Stim_Config', 'Video Config'])
    def test_a_write_into_the_run_copys_dict_cell_leaves_the_protocol(self, protocol, column):
        before = _snapshot(protocol)
        run_copy = protocol.copy_for_execution()

        run_copy._config['steps'].iloc[0][column]['written'] = True

        pd.testing.assert_frame_equal(_snapshot(protocol), before)

    def test_the_run_copys_layer_settings_are_its_own(self, tmp_path, protocol):
        protocol._config['layer_settings'] = {'BF': {'Layer': 'BF', 'Acquire': 'image'}}
        run_copy = protocol.copy_for_execution()

        run_copy._config['layer_settings']['BF']['Acquire'] = 'video'

        assert protocol._config['layer_settings']['BF']['Acquire'] == 'image'
