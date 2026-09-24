# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every position of the composite brightness slider expresses a real cutoff.

The slider offers 0..100 percent. The percentage was scaled onto 0..255 and
then compared with a strict ``>``, so the top of the user's own range was
unreachable: at 100 the cutoff was exactly 255, no 8-bit pixel could exceed it,
and the fluorescence layer vanished from the composite with a success status
and no log line. The bottom of the range had the mirror hazard -- a stored
value below zero put the cutoff under zero, every pixel including true black
counted as "above", and the transmitted base was erased frame-wide to black.

The conversion is now one named function producing an integer cutoff compared
with ``>=``, so 0 percent still means "any pixel carrying signal" and 100
percent means "only fully saturated pixels". A percentage outside the range is
a corrupt setting rather than a preference to be honoured approximately, and is
refused by name before the per-group try that would otherwise report a caller's
bad value as "every image group failed".
"""

import math

import numpy as np
import pandas as pd
import pytest

from modules.composite_builder import brightness_cutoff_from_percent, build_composite
from modules.composite_generation import CompositeGeneration
from modules.exceptions import ConfigError


class TestTheCutoffCoversTheWholeSlider:
    def test_the_top_of_the_slider_passes_saturated_pixels(self):
        """At 100 percent a fully saturated pixel still reaches the composite."""
        assert np.uint8(255) >= brightness_cutoff_from_percent(100)

    def test_the_bottom_of_the_slider_still_excludes_true_black(self):
        """At 0 percent every pixel carrying signal passes, and black does not."""
        cutoff = brightness_cutoff_from_percent(0)
        assert np.uint8(0) < cutoff
        assert np.uint8(1) >= cutoff

    def test_every_slider_position_is_a_distinct_reachable_cutoff(self):
        """No position is unreachable and none collapses onto its neighbour."""
        cutoffs = [brightness_cutoff_from_percent(p) for p in range(101)]
        pixel_values = np.arange(256, dtype=np.uint8)

        assert cutoffs == sorted(cutoffs), 'cutoffs must rise with the percentage'
        assert len(set(cutoffs)) == 101, 'two slider positions share one cutoff'
        for pct, cutoff in enumerate(cutoffs):
            passing = (pixel_values >= cutoff).sum()
            assert passing > 0, f'{pct}% admits no pixel value at all'


class TestTheLayerSurvivesTheTopOfTheSlider:
    """The end-to-end shape of the defect: a layer dropped from the output."""

    def _one_saturated_pixel(self):
        transmitted = np.full((2, 2), 40, dtype=np.uint8)
        red = np.zeros((2, 2), dtype=np.uint8)
        red[0, 0] = 255
        return transmitted, red

    def test_a_saturated_pixel_blends_at_the_top_of_the_slider(self):
        transmitted, red = self._one_saturated_pixel()

        img = build_composite(
            channel_images={'Red': red},
            significant_bits=8,
            transmitted_image=transmitted,
            brightness_thresholds={'Red': brightness_cutoff_from_percent(100)},
        )

        assert img[0, 0].tolist() == [255, 0, 0], 'the saturated pixel was dropped'
        assert img[1, 1].tolist() == [40, 40, 40], 'the transmitted base was disturbed'

    def test_an_unsaturated_pixel_does_not_blend_at_the_top_of_the_slider(self):
        transmitted = np.full((2, 2), 40, dtype=np.uint8)
        red = np.full((2, 2), 254, dtype=np.uint8)

        img = build_composite(
            channel_images={'Red': red},
            significant_bits=8,
            transmitted_image=transmitted,
            brightness_thresholds={'Red': brightness_cutoff_from_percent(100)},
        )

        assert img[0, 0].tolist() == [40, 40, 40]


class TestAPercentageOutsideTheRangeIsRefused:
    """A corrupt stored value is refused by name, not clamped into a guess."""

    def _group(self, tmp_path, threshold):
        transmitted = np.full((2, 2), 40, dtype=np.uint8)
        red = np.full((2, 2), 200, dtype=np.uint8)
        import imageio.v2 as imageio

        bf_path = tmp_path / 'bf.tiff'
        red_path = tmp_path / 'red.tiff'
        imageio.imwrite(bf_path, transmitted)
        imageio.imwrite(red_path, red)
        df = pd.DataFrame(
            [
                {'Color': 'BF', 'Filepath': bf_path},
                {'Color': 'Red', 'Filepath': red_path},
            ]
        )
        return df, {'Red': threshold}

    @pytest.mark.parametrize('bad', [-1, -25, 101, 500])
    def test_a_percentage_outside_the_range_is_refused_by_name(self, tmp_path, bad):
        df, thresholds = self._group(tmp_path, bad)

        with pytest.raises(ConfigError) as excinfo:
            CompositeGeneration._create_composite_image(
                path=tmp_path,
                df=df,
                brightness_thresholds_percent=thresholds,
                output_file_loc=None,
            )

        message = str(excinfo.value)
        assert 'Red' in message, 'the refusal must name the layer'
        assert str(bad) in message, 'the refusal must name the value'

    def test_a_non_numeric_percentage_is_refused_rather_than_raising_mid_merge(self, tmp_path):
        """A bad type is a caller error, reported as one, not as a failed group."""
        df, thresholds = self._group(tmp_path, 'bright')

        with pytest.raises(ConfigError):
            CompositeGeneration._create_composite_image(
                path=tmp_path,
                df=df,
                brightness_thresholds_percent=thresholds,
                output_file_loc=None,
            )

    @pytest.mark.parametrize('good', [0, 50, 100])
    def test_a_percentage_inside_the_range_is_accepted(self, tmp_path, good):
        df, thresholds = self._group(tmp_path, good)

        result = CompositeGeneration._create_composite_image(
            path=tmp_path,
            df=df,
            brightness_thresholds_percent=thresholds,
            output_file_loc=None,
        )

        assert result['status'] is True, result['error']


class TestTheConversionHasOneHome:
    """Production and tests share one mapping, so neither can drift."""

    def test_the_conversion_matches_its_stated_formula(self):
        for pct in range(101):
            assert brightness_cutoff_from_percent(pct) == max(1, math.ceil(pct * 255 / 100))
