# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The composite merge runs with no GUI and no settings global.

Blending fluorescence onto a transmitted base needs a per-layer
brightness threshold. That threshold used to be read from the module-level
``settings`` object in ``modules.settings_init``, bound at import time.
Only the app bootstrap ever publishes that global, so in a headless or
REST process it is still None when this module is imported and stays None
for the life of the process -- and the read crashed exactly when a
transmitted channel was present, while succeeding without one. A caller
got either a composite or a TypeError depending on which channels they
selected.

The threshold is now an argument, so the headless and GUI paths are the
same code with the same inputs, and a caller that omits it is refused
loudly rather than silently blended against an unstated value.
"""

import ast

import numpy as np
import pytest
import tifffile as tf

from modules.composite_generation import CompositeGeneration
from modules.exceptions import ConfigError
from tests import ast_seams

import pandas as pd


def _channel_files(tmp_path, colors):
    rows = []
    for color in colors:
        arr = np.full((8, 8), 120, dtype=np.uint8)
        tf.imwrite(str(tmp_path / f'{color}.tiff'), arr, compression='lzw')
        rows.append({'Color': color, 'Filepath': f'{color}.tiff'})
    return pd.DataFrame(rows)


class TestHeadlessCompositeWithTransmittedBase:
    def test_a_transmitted_composite_merges_without_any_app_context(self, tmp_path):
        # The exact shape that crashed: a transmitted channel present, so
        # the threshold is actually consulted, with no GUI anywhere.
        df = _channel_files(tmp_path, ('BF', 'Red'))
        out = tmp_path / 'composite.tiff'

        result = CompositeGeneration._create_composite_image(
            path=tmp_path,
            df=df,
            brightness_thresholds_percent={'Red': 25},
            output_file_loc=out,
        )

        assert result['status'] is True, f'headless merge failed: {result.get("error")}'
        assert out.exists(), 'the merged composite was never written'

    def test_a_missing_threshold_is_refused_loudly(self, tmp_path):
        df = _channel_files(tmp_path, ('BF', 'Red'))

        with pytest.raises(ConfigError) as excinfo:
            CompositeGeneration._create_composite_image(
                path=tmp_path,
                df=df,
                brightness_thresholds_percent={},
                output_file_loc=tmp_path / 'composite.tiff',
            )
        assert 'Red' in str(excinfo.value), (
            'the refusal must name the layer whose threshold is missing'
        )

    def test_fluorescence_only_needs_no_threshold(self, tmp_path):
        # Without a transmitted base there is nothing to blend onto, so no
        # threshold is consulted -- this is the case that used to succeed
        # while the transmitted one crashed.
        df = _channel_files(tmp_path, ('Red', 'Green'))
        out = tmp_path / 'composite.tiff'

        result = CompositeGeneration._create_composite_image(
            path=tmp_path,
            df=df,
            brightness_thresholds_percent={},
            output_file_loc=out,
        )

        assert result['status'] is True
        assert out.exists()


def test_the_merge_module_binds_no_settings_global():
    """Structural pin: the import that made this headless-broken is gone.

    A value binding of the settings global at module scope snapshots None
    in any process that is not the GUI bootstrap, and no later publish can
    reach it. Asserting the import's absence is what stops the pattern
    coming back, since restoring it would fail only on a headless run.
    """
    tree = ast_seams.parse_module('modules/composite_generation.py')

    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'modules.settings_init':
            offenders.append([alias.name for alias in node.names])
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith('modules.settings_init'):
                    offenders.append([alias.name])

    assert offenders == [], (
        f'modules/composite_generation.py imports the settings global again: '
        f'{offenders}. The merge runs unattended on a worker thread; every '
        f'value it needs is passed in by the caller that has the settings.'
    )
