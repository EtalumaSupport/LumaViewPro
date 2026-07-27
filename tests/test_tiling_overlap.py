# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

import pathlib
import datetime

import pytest

from modules.protocol import Protocol
from modules.tiling_config import TilingConfig


TILING_CONFIGS = pathlib.Path(__file__).parent.parent / 'data' / 'tiling.json'


def _tile_spacing(tiles):
    x_spacing = abs(tiles['A2']['x'] - tiles['A1']['x'])
    y_spacing = abs(tiles['B1']['y'] - tiles['A1']['y'])
    return x_spacing, y_spacing


def _make_protocol_from_config(overlap_percent):
    input_config = {
        'positions': [
            {
                'x': 10.0,
                'y': 20.0,
                'z': 5000.0,
                'name': 'A1',
            }
        ],
        'labware_id': 'custom',
        'objective_id': '10x Oly',
        'zstack_params': {
            'range': 0,
            'step_size': 0,
            'z_reference': 'center',
        },
        'use_zstacking': False,
        'tiling': '2x2',
        'tiling_overlap_percent': overlap_percent,
        'layer_configs': {
            'BF': {
                'acquire': 'image',
                'focus': 5000.0,
                'autofocus': False,
                'false_color': False,
                'illumination_ma': 50.0,
                'sum': 1,
                'gain_db': 1.0,
                'auto_gain': False,
                'exposure_ms': 10.0,
                'video_config': {},
            }
        },
        'period': datetime.timedelta(minutes=1),
        'duration': datetime.timedelta(hours=1),
        'frame_dimensions': {'width': 1920, 'height': 1080},
        'binning_size': 1,
        'stim_config': {},
    }
    return Protocol.from_config(
        input_config=input_config,
        tiling_configs_file_loc=TILING_CONFIGS,
    )


def _protocol_tile_spacing(protocol):
    steps = protocol.steps().set_index('Tile')
    x_spacing = abs(steps.loc['A2', 'X'] - steps.loc['A1', 'X'])
    y_spacing = abs(steps.loc['B1', 'Y'] - steps.loc['A1', 'Y'])
    return x_spacing, y_spacing


def test_overlap_percent_to_fill_factor():
    assert TilingConfig.fill_factor_from_overlap_percent(0) == pytest.approx(1.0)
    assert TilingConfig.fill_factor_from_overlap_percent(10) == pytest.approx(0.9)
    assert TilingConfig.fill_factor_from_overlap_percent(15) == pytest.approx(0.85)
    assert TilingConfig.fill_factor_from_overlap_percent(20) == pytest.approx(0.8)


@pytest.mark.parametrize('overlap_percent', [-1, 51, 'abc', None])
def test_invalid_overlap_percent_rejected(overlap_percent):
    with pytest.raises(ValueError):
        TilingConfig.fill_factor_from_overlap_percent(overlap_percent)


def test_ten_percent_overlap_reduces_tile_spacing_by_ten_percent(scale_ctx):
    tiling_config = TilingConfig(tiling_configs_file_loc=TILING_CONFIGS)

    common_kwargs = {
        'config_label': '2x2',
        'focal_length': 50.0,
        'frame_size': {'width': 1920, 'height': 1080},
        'binning_size': 1,
    }

    tiles_no_overlap = tiling_config.get_tile_centers(
        **common_kwargs,
        fill_factor=TilingConfig.fill_factor_from_overlap_percent(0),
    )
    tiles_ten_percent_overlap = tiling_config.get_tile_centers(
        **common_kwargs,
        fill_factor=TilingConfig.fill_factor_from_overlap_percent(10),
    )

    x_spacing_no_overlap, y_spacing_no_overlap = _tile_spacing(tiles_no_overlap)
    x_spacing_overlap, y_spacing_overlap = _tile_spacing(tiles_ten_percent_overlap)

    assert x_spacing_overlap == pytest.approx(x_spacing_no_overlap * 0.9, abs=0.02)
    assert y_spacing_overlap == pytest.approx(y_spacing_no_overlap * 0.9, abs=0.02)


def test_protocol_from_config_overlap_keeps_tile_count_and_reduces_spacing(scale_ctx):
    protocol_no_overlap = _make_protocol_from_config(overlap_percent=0)
    protocol_ten_percent_overlap = _make_protocol_from_config(overlap_percent=10)

    assert len(protocol_no_overlap.steps()) == 4
    assert len(protocol_ten_percent_overlap.steps()) == 4

    x_spacing_no_overlap, y_spacing_no_overlap = _protocol_tile_spacing(protocol_no_overlap)
    x_spacing_overlap, y_spacing_overlap = _protocol_tile_spacing(protocol_ten_percent_overlap)

    assert x_spacing_overlap == pytest.approx(x_spacing_no_overlap * 0.9, abs=0.0001)
    assert y_spacing_overlap == pytest.approx(y_spacing_no_overlap * 0.9, abs=0.0001)


def test_overlap_never_adds_tiles(scale_ctx):
    """The tiling label is a tile-count contract: the requested grid is
    what runs, at every allowed overlap. Overlap shrinks the step (and
    the total covered area) instead of growing the grid -- an earlier
    coverage-preserving design silently turned a 2x2 into a 3x3."""
    tiling_config = TilingConfig(tiling_configs_file_loc=TILING_CONFIGS)

    common_kwargs = {
        'config_label': '2x2',
        'focal_length': 50.0,
        'frame_size': {'width': 1920, 'height': 1080},
        'binning_size': 1,
    }

    for overlap_percent in (0, 10, 15, 20, 50):
        tiles = tiling_config.get_tile_centers(
            **common_kwargs,
            fill_factor=TilingConfig.fill_factor_from_overlap_percent(overlap_percent),
        )
        assert len(tiles) == 4, (
            f'2x2 produced {len(tiles)} tiles at {overlap_percent}% overlap; '
            f'the requested count is the contract'
        )


class TestTileCentersWithoutScale:
    """A scale-less scope (no pixel size -> no field of view) must still
    lay out the trivial 1x1 grid: its offsets are identically zero
    whatever the step, so the field of view is not an input. The default
    protocol built at startup is 1x1, so requiring a field of view there
    crashed the whole app on a scope that honestly reports no scale.
    Grids with real spacing must still refuse."""

    def _scaleless(self, monkeypatch):
        import modules.app_context as app_context

        monkeypatch.setattr(app_context, 'ctx', None)
        return TilingConfig(tiling_configs_file_loc=TILING_CONFIGS)

    def test_1x1_layout_needs_no_field_of_view(self, monkeypatch):
        tiling_config = self._scaleless(monkeypatch)
        tiles = tiling_config.get_tile_centers(
            config_label='1x1',
            focal_length=9.0,
            frame_size={'width': 1900, 'height': 1900},
            fill_factor=1.0,
            binning_size=1,
        )
        assert tiles == {'': {'x': 0.0, 'y': 0.0}}

    def test_real_grid_still_refuses_without_field_of_view(self, monkeypatch):
        from modules.exceptions import ConfigError

        tiling_config = self._scaleless(monkeypatch)
        with pytest.raises(ConfigError):
            tiling_config.get_tile_centers(
                config_label='2x2',
                focal_length=9.0,
                frame_size={'width': 1900, 'height': 1900},
                fill_factor=1.0,
                binning_size=1,
            )
