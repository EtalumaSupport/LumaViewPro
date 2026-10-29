# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A saved file's record is read the same way whatever container holds it.

``image_utils.read_file_record`` is the one reader: a shaped, ImageJ or OME
still, and every recorded frame LVP has shipped, read into the same record
keys. Every case writes through the writer the release used and reads back
from disk.
"""

from __future__ import annotations

import datetime
import pathlib
import re

import numpy as np
import pandas as pd
import pytest
import tifffile as tf

from modules import image_utils
from modules.exceptions import InputFramesUnusableError
from modules.layer_record import GREY, release_colors
from modules.stack_builder import StackBuilder
from tests.stock_channels import LED1

_A_TIME = datetime.datetime(2026, 10, 28, 12, 0, 0, 250000)

# What an OME file does not state until the OME plan writes its schema blocks:
# tifffile keeps only the schema's Image, Pixels, Channel and Plane.
_NOT_IN_AN_OME_FILE = {'instrument', 'plate', 'well_label', 'filterset', 'channel_display'}


def _a_record(**overrides) -> dict:
    """A still's record with its Plane facts set; override any key by name."""
    return {
        'pixel_size_um': 0.65,
        'channel': LED1,
        'channel_display': 'Blue',
        'excitation_nm': 470.0,
        'filterset': 'FS-STOCK',
        'datetime': _A_TIME,
        'objective': {'id': '20x Oly', 'magnification': 20, 'working_distance_mm': 8.1},
        'plate_pos_mm': {'x': 12.5, 'y': 30.25},
        'z_pos_um': 4321.0,
        'exposure_time_ms': 50.0,
        'gain_db': 3.0,
        'illumination_ma': 120.0,
        'frames_summed': 2,
        'binning_size': 1,
        'timestamp_iso': _A_TIME.isoformat(timespec='microseconds'),
        'timestamp_source': 'host_wallclock',
        'instrument': {'manufacturer': 'Etaluma', 'model': 'LS850', 'camera_model': 'cam'},
        'plate': {'name': '96 well microplate', 'rows': 8, 'columns': 12},
        'well_label': 'A1',
        **overrides,
    }


def _write(path, record, *, dtype=np.uint16, ome=False) -> pathlib.Path:
    image_utils.write_tiff(
        data=np.zeros((8, 8), dtype=dtype),
        file_loc=path,
        metadata=record,
        ome=ome,
        color=GREY,
        significant_bits=8 if dtype == np.uint8 else 12,
        save_encoding='8bit' if dtype == np.uint8 else 'right_aligned',
        white_bits=None,
    )
    return path


def test_one_frame_in_every_container_reads_back_as_one_record(tmp_path):
    """The same frame saved as a shaped, an ImageJ and an OME TIFF gives the
    same scale, time, objective, channel and Plane facts; the OME file lacks
    only what its container cannot yet hold."""
    record = _a_record()
    shaped = image_utils.read_file_record(_write(tmp_path / 's.tiff', record, dtype=np.uint8))
    imagej = image_utils.read_file_record(_write(tmp_path / 'ij.tiff', record))
    ome = image_utils.read_file_record(_write(tmp_path / 'o.ome.tiff', record, ome=True))

    assert shaped == imagej
    assert {k: v for k, v in shaped.items() if k not in _NOT_IN_AN_OME_FILE} == ome
    times = {
        name: image_utils.load_pixels_with_timestamp(path)[3]
        for name, path in (('imagej', tmp_path / 'ij.tiff'), ('ome', tmp_path / 'o.ome.tiff'))
    }
    assert times == {'imagej': _A_TIME, 'ome': _A_TIME}


def test_an_ome_still_with_no_position_or_exposure_states_its_scale_and_objective(tmp_path):
    """With no schema Plane attribute to write, tifffile writes no Plane
    element; the file still states its scale, channel and objective."""
    record = _a_record()
    for key in ('plate_pos_mm', 'z_pos_um', 'exposure_time_ms'):
        del record[key]
    path = _write(tmp_path / 'o.ome.tiff', record, ome=True)

    read = image_utils.read_file_record(path)

    assert read is not None
    assert (read['pixel_size_um'], read['channel'], read['objective']['id']) == (
        0.65,
        LED1,
        '20x Oly',
    )


def test_a_hyperstack_built_from_ome_stills_states_its_plane_times(tmp_path):
    """Each plane's DeltaT is its frame's time from the earliest; an OME
    still's time is read as an ImageJ still's is."""
    rows = []
    for t in range(2):
        when = _A_TIME + datetime.timedelta(seconds=5 * t)
        name = f'frame_t{t}.ome.tiff'
        _write(
            tmp_path / name,
            _a_record(datetime=when, timestamp_iso=when.isoformat(timespec='microseconds')),
            dtype=np.uint8,
            ome=True,
        )
        rows.append(
            {
                'Filepath': name,
                'Color': LED1,
                'Scan Count': t,
                'Z-Slice': 0,
                'X': 12.5,
                'Y': 30.25,
                'Z': 4321.0,
            }
        )

    StackBuilder._create_stack(
        path=tmp_path,
        df=pd.DataFrame(rows),
        output_file_loc=pathlib.Path('out.ome.tiff'),
        save_encoding='8bit',
        channel_colors=release_colors(),
    )

    with tf.TiffFile(str(tmp_path / 'out.ome.tiff')) as tif:
        xml = tif.ome_metadata
    delta_t = [float(v) for v in re.findall(r'DeltaT="([^"]+)"', xml)]
    assert delta_t == [0.0, 5.0]


def _legacy_frame(path, description, *, datetime_tag) -> pathlib.Path:
    """A recorded frame as the releases before v4.0.0-beta31 wrote it: its
    flat dict handed to tifffile as shaped metadata, with its DateTime tag."""
    with tf.TiffWriter(str(path)) as tif:
        tif.write(
            np.zeros((8, 8), dtype=np.uint16),
            metadata=description,
            datetime=datetime_tag,
            software='LumaViewPro',
        )
    return path


def test_a_recorded_frame_of_every_shipped_shape_reads_its_scale_and_time(tmp_path):
    """v2.3 and v3.0.0 wrote ``timestamp`` and ``frame_num``; v4.0.0-beta29
    and beta30 added ``timestamp_iso`` and the scale. Each reads back its
    time and its number, and its scale where it states one."""
    v2 = _legacy_frame(
        tmp_path / 'v2.tiff',
        {
            'datetime': _A_TIME.strftime('%Y:%m:%d %H:%M:%S'),
            'timestamp': _A_TIME.strftime('%Y:%m:%d %H:%M:%S.%f'),
            'frame_num': 3,
        },
        datetime_tag=_A_TIME.strftime('%Y:%m:%d %H:%M:%S'),
    )
    beta30 = _legacy_frame(
        tmp_path / 'beta30.tiff',
        {
            'datetime': _A_TIME.strftime('%Y:%m:%d %H:%M:%S'),
            'timestamp': _A_TIME.strftime('%Y:%m:%d %H:%M:%S.%f'),
            'timestamp_iso': _A_TIME.isoformat(timespec='microseconds'),
            'frame_num': 4,
            'pixel_size_um': 0.65,
            'timestamp_camera_ticks': 123,
            'frame_id': 9,
        },
        datetime_tag=_A_TIME.strftime('%Y:%m:%d %H:%M:%S'),
    )

    read = {
        name: image_utils.read_file_record(path) for name, path in (('v2', v2), ('b30', beta30))
    }

    assert {name: r['pixel_size_um'] for name, r in read.items()} == {'v2': None, 'b30': 0.65}
    assert {name: r['recording_frame'] for name, r in read.items()} == {'v2': 3, 'b30': 4}
    assert [image_utils.load_pixels_with_timestamp(p)[3] for p in (v2, beta30)] == [_A_TIME] * 2


def test_an_unreadable_file_is_refused_and_a_file_with_no_record_answers_none(tmp_path):
    """A file that cannot be decoded is not a file that states nothing: the
    first is refused as unreadable, the second has no record."""
    cut_short = tmp_path / 'cut.tiff'
    whole = _write(tmp_path / 'whole.tiff', _a_record())
    cut_short.write_bytes(whole.read_bytes()[:6])
    bare = tmp_path / 'bare.tiff'
    tf.imwrite(str(bare), np.zeros((8, 8), dtype=np.uint16))

    with pytest.raises(InputFramesUnusableError) as refused:
        image_utils.read_file_record(cut_short)

    assert refused.value.reason == 'unreadable'
    assert image_utils.read_file_record(bare) is None
