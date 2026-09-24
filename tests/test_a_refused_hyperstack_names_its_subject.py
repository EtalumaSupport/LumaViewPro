# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The hyperstack builder's refusal names what it refused.

A protocol's frame set names its well, and the refusal did; a manual
recording's frame set has no well, and the same refusal indexed the
column anyway, so the one way a recording can be non-rectangular -- a
channel change while it recorded -- died as a KeyError instead of a
reason. Each entry point now names its own subject.
"""

import pathlib

import pandas as pd

from modules.stack_builder import StackBuilder


def _rows(*, well=None):
    rows = []
    for t in range(3):
        for color in ('Blue', 'Green'):
            if color == 'Green' and t == 0:
                continue  # one channel missing at one T: not rectangular
            row = {
                'Filepath': f'f_{t}_{color}.tiff',
                'Scan Count': t,
                'Z-Slice': 0,
                'Color': color,
                'X': 1.0,
                'Y': 2.0,
                'Z': 3.0,
            }
            if well is not None:
                row['Well'] = well
            rows.append(row)
    return pd.DataFrame(rows)


def test_a_protocol_frame_set_is_refused_naming_its_well(tmp_path):
    result = StackBuilder._create_stack(
        path=tmp_path, df=_rows(well='B3'), output_file_loc=pathlib.Path('out.ome.tiff')
    )
    assert result['status'] is False
    assert 'for well B3' in result['error']


def test_a_recordings_frame_set_is_refused_naming_the_recording(tmp_path):
    result = StackBuilder._create_stack(
        path=tmp_path,
        df=_rows(),
        output_file_loc=pathlib.Path('out.ome.tiff'),
        sort_order=['Scan Count'],
    )
    assert result['status'] is False
    assert 'for this recording' in result['error']
    assert '5 frames across 2 channels' in result['error']
