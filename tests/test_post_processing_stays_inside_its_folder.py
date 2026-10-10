# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Post-processing reads and writes only inside the run folder it was given.

A run folder's record and protocol are data in the folder, and a REST
caller can name any folder under the live folder. The record names the
protocol file it reads, and the protocol's steps name the outputs a build
writes; neither was checked, so a record naming '../..' read a file the run
never wrote, and a step whose well was '../../x' wrote its projection
outside the folder. Each is refused, and a refused build writes no output.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules import image_utils
from modules.exceptions import PostProcessingRefusedError
from modules.zprojector import ZProjector
from tests.test_capture_collision_policy import TILING_CONFIGS

_COLUMNS = [
    'Name', 'X', 'Y', 'Z', 'Auto_Focus', 'Color', 'False_Color', 'Illumination',
    'Gain', 'Auto_Gain', 'Exposure', 'Sum', 'Objective', 'Well', 'Tile', 'Z-Slice',
    'Custom Step', 'Tile Group ID', 'Z-Stack Group ID', 'Acquire', 'Video Config',
    'Stim_Config', 'Auto_Named', 'Label',
]  # fmt: skip


def _step(name, z, z_slice, well):
    return [
        name, '14.38', '11.24', z, 'False', 'BF', 'False', '5.0', '1.0', 'False',
        '2.0', '1', '20x w/collar', well, '', z_slice, 'False', '-1', '0', 'image',
        '"{""duration"": 5, ""fps"": 30}"', '"{}"', 'True', '',
    ]  # fmt: skip


def _write_tiff(path):
    image_utils.write_tiff(
        data=np.full((8, 8), 200, dtype=np.uint8),
        file_loc=path,
        metadata={
            'pixel_size_um': 0.5,
            'channel': 'BF',
            'objective': '20x',
            'exposure_time_ms': 2.0,
            'gain_db': 1.0,
            'illumination_ma': 5.0,
            'z_pos_um': 5000.0,
            'plate_pos_mm': {'x': 14.38, 'y': 11.24},
            'datetime': '2026:10:07 12:00:00',
            'camera_make': 'Test',
            'microscope': 'TestScope',
            'well_label': 'A1',
            'significant_bits': 8,
        },
        ome=False,
        color='BF',
        significant_bits=8,
        save_encoding='right_aligned',
    )


def _run_folder(tmp_path, *, well='A1', protocol_file='run.tsv'):
    """A two-slice Z-stack run, as a run of this release saves it."""
    folder = tmp_path / 'live' / 'ProtocolData' / '20261007_120000'
    folder.mkdir(parents=True)
    steps = [_step('A1_BF_Z0', '5000.0', '0', well), _step('A1_BF_Z1', '5010.0', '1', well)]
    (folder / 'run.tsv').write_text(
        'LumaViewPro Protocol\nVersion\t8\nPeriod\t1.0\nDuration\t0.5\n'
        'Labware\t96 well microplate\nCapture Root\t\n\nSteps\n'
        + '\t'.join(_COLUMNS)
        + '\n'
        + ''.join('\t'.join(step) + '\n' for step in steps)
    )
    (folder / 'protocol_record.tsv').write_text(
        'LumaViewPro Protocol Execution Record\n'
        'Version\t3\n'
        f'Protocol File\t{protocol_file}\n'
        'Filename\tStep Name\tStep Index\tScan Count\tTimestamp\tFrame Count\tDuration (s)\n'
        'A1_BF_Z0_0000.tiff\tA1_BF_Z0\t0\t0\t2026-10-07 12:00:00.000001\t1\t0.0\n'
        'A1_BF_Z1_0000.tiff\tA1_BF_Z1\t1\t0\t2026-10-07 12:00:01.000001\t1\t0.0\n'
    )
    _write_tiff(folder / 'A1_BF_Z0_0000.tiff')
    _write_tiff(folder / 'A1_BF_Z1_0000.tiff')
    return folder


def _project(folder):
    return ZProjector(has_turret=False).load_folder(
        path=folder, tiling_configs_file_loc=TILING_CONFIGS, method=ZProjector.methods()[0]
    )


def _tiffs(root):
    return sorted(p for p in root.rglob('*.tif*'))


def test_a_run_folder_is_projected_as_it_always_was(tmp_path):
    folder = _run_folder(tmp_path)

    result = _project(folder)

    made = [p for p in _tiffs(tmp_path) if 'A1_BF_Z' not in p.name]
    assert made and all(p.is_relative_to(folder) for p in made), result


def test_a_record_naming_a_protocol_outside_the_folder_is_refused(tmp_path):
    folder = _run_folder(tmp_path, protocol_file='../../elsewhere/p.tsv')
    elsewhere = tmp_path / 'live' / 'elsewhere'
    elsewhere.mkdir()
    (folder / 'run.tsv').rename(elsewhere / 'p.tsv')

    with pytest.raises(PostProcessingRefusedError) as refused:
        _project(folder)

    assert refused.value.reason == 'protocol_data_unreadable'
    assert 'outside the run folder' in str(refused.value)


@pytest.mark.parametrize(
    'record',
    [
        'not a run record\n',
        'LumaViewPro Protocol Execution Record\n',
        'LumaViewPro Protocol Execution Record\nVersion\tthree\n',
    ],
    ids=['wrong_header', 'cut_short', 'bad_version'],
)
def test_a_malformed_record_is_refused_not_raised(tmp_path, record):
    folder = _run_folder(tmp_path)
    (folder / 'protocol_record.tsv').write_text(record)

    with pytest.raises(PostProcessingRefusedError) as refused:
        _project(folder)

    assert refused.value.reason == 'protocol_data_unreadable'
    assert 'protocol_record.tsv could not be read' in str(refused.value)


def test_a_step_naming_an_output_outside_the_folder_is_refused_and_nothing_is_written(tmp_path):
    folder = _run_folder(tmp_path, well='../../../x')
    before = _tiffs(tmp_path)

    with pytest.raises(PostProcessingRefusedError) as refused:
        _project(folder)

    assert refused.value.reason == 'output_outside_folder'
    assert _tiffs(tmp_path) == before
