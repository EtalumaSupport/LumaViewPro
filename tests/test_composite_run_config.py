# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Composite capture assembles a degenerate sequenced run, headlessly.

A composite is a single-position, one-step-per-channel run through the
sequenced-capture engine, so every host -- GUI, SDK, REST -- reaches it
through the same config assembly and the same typed refusals. The old
path built its frames inside a GUI widget worker, which no headless
caller could reach and whose failures surfaced only as popups.

Covered here:
1. The assembled input_config's shape: one step per acquiring channel,
   a single named position whose z defers to each layer's own focus.
2. One-transmitted-wins -- a plate carries at most one transmitted
   channel into the merge.
3. Fewer than two acquiring channels refuses loudly at assembly.
4. The run's sequenced output format: the user's preference when the
   merge can read it back, else coerced with the override logged.
"""

from unittest.mock import MagicMock

import pytest

from modules.exceptions import ProtocolRunRefusedError
from modules.image_mode import (
    OUTPUT_FORMAT_HYPERSTACK,
    OUTPUT_FORMAT_JPG,
    OUTPUT_FORMAT_OME_TIFF,
    OUTPUT_FORMAT_TIFF,
)

import modules.config_helpers as config_helpers

# The plate position a composite starts from; every channel step shares
# it, so only the name and the deferred z carry meaning here.
_POSITION = {'x': 1234.5, 'y': 678.9, 'z': 4321.0}

_ALL_LAYERS = ('BF', 'PC', 'DF', 'Blue', 'Green', 'Red', 'Lumi')


def _layer(acquire='none', focus=0.0):
    return {
        'acquire': acquire,
        'video_config': {},
        'stim_config': {'enabled': False},
        'autofocus': False,
        'false_color': True,
        'illumination_ma': 50.0,
        'gain_db': 2.0,
        'auto_gain': False,
        'exposure_ms': 10.0,
        'sum': 1,
        'focus': focus,
    }


def _settings(acquiring=(), sequenced_format=OUTPUT_FORMAT_TIFF, focus_by_layer=None):
    """A settings snapshot with *acquiring* layers set to capture images."""
    focus_by_layer = focus_by_layer or {}
    settings = {
        layer: _layer(
            acquire='image' if layer in acquiring else 'none',
            focus=focus_by_layer.get(layer, 0.0),
        )
        for layer in _ALL_LAYERS
    }
    settings.update(
        {
            'objective_id': '10x Oly',
            'binning_size': 1,
            'frame': {'width': 800, 'height': 600},
            'image_output_format': {'live': OUTPUT_FORMAT_TIFF, 'sequenced': sequenced_format},
            'live_folder': '.',
            'protocol': {
                'labware': '96 well microplate',
                'autogain': {
                    'target_brightness': 0.3,
                    'max_duration_seconds': 1.0,
                    'min_gain_db': 0.0,
                    'max_gain_db': 20.0,
                },
            },
        }
    )
    return settings


def _objective_helper():
    helper = MagicMock()
    helper.get_objective_info.return_value = {'magnification': 10}
    return helper


def _assemble(settings):
    return config_helpers.get_composite_capture_config_from_settings(
        settings,
        _objective_helper(),
        position=dict(_POSITION),
    )


def _capture_notifications(monkeypatch):
    """Route both severities of the notification singleton to one list."""
    import modules.notification_center as notification_center

    captured = []
    for severity in ('error', 'warning'):
        monkeypatch.setattr(
            notification_center.notifications,
            severity,
            lambda *args, _s=severity, **kwargs: captured.append((_s, args)),
        )
    return captured


# ---------------------------------------------------------------------------
# 1. The assembled config's shape
# ---------------------------------------------------------------------------


class TestCompositeConfigShape:
    def test_one_step_per_acquiring_channel(self):
        config = _assemble(_settings(acquiring=('BF', 'Blue', 'Green')))
        assert set(config['layer_configs']) == {'BF', 'Blue', 'Green'}

    def test_non_acquiring_channels_are_excluded(self):
        config = _assemble(_settings(acquiring=('BF', 'Blue')))
        assert 'Red' not in config['layer_configs']
        assert 'Lumi' not in config['layer_configs']

    def test_single_position_named_composite(self):
        config = _assemble(_settings(acquiring=('BF', 'Blue')))
        assert len(config['positions']) == 1
        assert config['positions'][0]['name'] == 'Composite'

    def test_position_z_defers_to_each_layers_own_focus(self):
        # A numeric z pins every step to one plane and silently discards
        # each channel's focus; None is the fallthrough that lets
        # Protocol.from_config resolve z per layer.
        config = _assemble(
            _settings(
                acquiring=('BF', 'Blue'),
                focus_by_layer={'BF': 1000.0, 'Blue': 2000.0},
            )
        )
        assert config['positions'][0]['z'] is None

    def test_position_keeps_the_stages_xy(self):
        config = _assemble(_settings(acquiring=('BF', 'Blue')))
        assert config['positions'][0]['x'] == _POSITION['x']
        assert config['positions'][0]['y'] == _POSITION['y']

    def test_each_channel_carries_its_own_focus(self):
        config = _assemble(
            _settings(
                acquiring=('BF', 'Blue'),
                focus_by_layer={'BF': 1000.0, 'Blue': 2000.0},
            )
        )
        assert config['layer_configs']['BF']['focus'] == 1000.0
        assert config['layer_configs']['Blue']['focus'] == 2000.0

    def test_no_zstacking_and_no_tiling(self):
        # A composite is one frame per channel at one point: z-stacking or
        # a tiling mosaic would multiply the steps and leave the merge
        # grouping several frames per channel.
        config = _assemble(_settings(acquiring=('BF', 'Blue')))
        assert config['use_zstacking'] is False
        assert config['zstack_params'] == {}
        assert config['tiling'] == '1x1'
        assert config['tiling_overlap_percent'] == 0.0

    def test_is_a_single_shot_run(self):
        config = _assemble(_settings(acquiring=('BF', 'Blue')))
        assert config['period'] is None
        assert config['duration'] is None

    def test_a_composite_never_stimulates_the_sample(self):
        # Protocol.from_config stamps this one value onto every step, so a
        # populated dict here would fire stim hardware at the sample during
        # what is only a multi-channel image capture.
        settings = _settings(acquiring=('BF', 'Blue'))
        settings['Blue']['stim_config'] = {'enabled': True, 'duration_ms': 500}
        assert _assemble(settings)['stim_config'] == {}

    def test_routes_through_the_canonical_builder_key_set(self):
        from tests.test_sequenced_capture_config_builder import _CANONICAL_KEYS

        config = _assemble(_settings(acquiring=('BF', 'Blue')))
        assert set(config) >= _CANONICAL_KEYS


# ---------------------------------------------------------------------------
# 2. One transmitted channel wins
# ---------------------------------------------------------------------------


class TestOneTransmittedWins:
    def test_only_the_first_acquiring_transmitted_channel_is_kept(self):
        config = _assemble(_settings(acquiring=('BF', 'PC', 'DF', 'Blue')))
        transmitted = {'BF', 'PC', 'DF'} & set(config['layer_configs'])
        assert transmitted == {'BF'}

    def test_a_later_transmitted_channel_wins_when_it_is_the_only_one(self):
        config = _assemble(_settings(acquiring=('DF', 'Blue')))
        assert 'DF' in config['layer_configs']

    def test_every_fluorescence_channel_survives(self):
        config = _assemble(_settings(acquiring=('BF', 'Blue', 'Green', 'Red')))
        assert {'Blue', 'Green', 'Red'} <= set(config['layer_configs'])


# ---------------------------------------------------------------------------
# 3. Fewer than two channels refuses at assembly
# ---------------------------------------------------------------------------


class TestTwoChannelFloor:
    @pytest.mark.parametrize(
        'acquiring',
        [(), ('BF',), ('Blue',), ('BF', 'PC')],
        ids=['none', 'one_transmitted', 'one_fluorescence', 'two_transmitted_collapse_to_one'],
    )
    def test_fewer_than_two_channels_is_refused(self, acquiring, monkeypatch):
        # The merge skips groups of one, so a one-channel composite cannot
        # be produced at all -- refusing here is the difference between a
        # loud "pick another channel" and a run that quietly makes no file.
        _capture_notifications(monkeypatch)
        with pytest.raises(ProtocolRunRefusedError) as excinfo:
            _assemble(_settings(acquiring=acquiring))
        assert excinfo.value.reason == 'composite_needs_two_channels'

    def test_the_refusal_notifies_exactly_once(self, monkeypatch):
        captured = _capture_notifications(monkeypatch)
        with pytest.raises(ProtocolRunRefusedError):
            _assemble(_settings(acquiring=('BF',)))
        assert len(captured) == 1, f'expected one notification, got {captured}'

    def test_two_channels_is_enough(self, monkeypatch):
        _capture_notifications(monkeypatch)
        config = _assemble(_settings(acquiring=('BF', 'Blue')))
        assert len(config['layer_configs']) == 2


# ---------------------------------------------------------------------------
# 4. The run's sequenced output format
# ---------------------------------------------------------------------------


class TestCompositeOutputFormat:
    @pytest.mark.parametrize(
        'user_format',
        [OUTPUT_FORMAT_TIFF, OUTPUT_FORMAT_OME_TIFF],
        ids=['tiff', 'ome_tiff'],
    )
    def test_a_mergeable_preference_passes_through_untouched(self, user_format):
        config = config_helpers.get_composite_image_capture_config_from_settings(
            _settings(acquiring=('BF', 'Blue'), sequenced_format=user_format)
        )
        assert config.output_format_sequenced == user_format

    @pytest.mark.parametrize(
        'user_format',
        [OUTPUT_FORMAT_JPG, OUTPUT_FORMAT_HYPERSTACK],
        ids=['jpg', 'hyperstack'],
    )
    def test_a_format_the_merge_cannot_read_is_coerced(self, user_format):
        config = config_helpers.get_composite_image_capture_config_from_settings(
            _settings(acquiring=('BF', 'Blue'), sequenced_format=user_format)
        )
        assert config.output_format_sequenced == OUTPUT_FORMAT_OME_TIFF

    @pytest.mark.parametrize(
        'user_format',
        [OUTPUT_FORMAT_JPG, OUTPUT_FORMAT_HYPERSTACK],
        ids=['jpg', 'hyperstack'],
    )
    def test_the_coercion_is_logged(self, user_format, caplog):
        # The user picked a format and silently got another one; the run
        # record is where they find out why their composite is not JPG.
        import lvp_logger

        logged = []
        original = lvp_logger.logger.info
        lvp_logger.logger.info = lambda msg, *a, **kw: logged.append(str(msg))
        try:
            config_helpers.get_composite_image_capture_config_from_settings(
                _settings(acquiring=('BF', 'Blue'), sequenced_format=user_format)
            )
        finally:
            lvp_logger.logger.info = original
        assert any(user_format in line for line in logged), (
            f'the format override must name the rejected preference; got {logged}'
        )

    def test_the_hyperstack_build_gate_stays_shut(self):
        # The coercion is also what keeps a composite run from entering the
        # per-well stack build, which has no 2D frame for the merge to read.
        config = config_helpers.get_composite_image_capture_config_from_settings(
            _settings(acquiring=('BF', 'Blue'), sequenced_format=OUTPUT_FORMAT_HYPERSTACK)
        )
        assert config.output_format_sequenced != OUTPUT_FORMAT_HYPERSTACK

    def test_the_composite_captures_eight_bit(self):
        # build_composite downconverts to 8-bit, so a 12-bit capture would
        # cost acquisition time for depth the merged artifact discards.
        config = config_helpers.get_composite_image_capture_config_from_settings(
            _settings(acquiring=('BF', 'Blue'))
        )
        assert config.image_mode == '8bit'
        assert config.capture_depth == 8
