# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Headless OME round-trip: an independent parser validates our stacks.

tifffile serializes whatever unit and axis strings it is handed, so it
cannot catch its own output being schema-invalid -- Bio-Formats' lenient
parsing then hides the defect until a strict reader refuses the file.
The OME 2016-06 XSD (tests/ome_schema.py) is that strict reader, and
every assertion here is against the OME data model ome-types parses back
from the written XML, pinning the
ruled spec -- one OME-TIFF per (well, scan), T = frame capture order,
C = channel, DimensionOrder XYCZT, per-plane DeltaT from real
timestamps. The ImageJ/Fiji open and the 8/16-bit + BigTIFF permutation
sweep ride the bench gate; this is the headless half.
"""

import datetime
import pathlib

import numpy as np
import ome_types
import tifffile as tf

from modules import image_utils, recording_frames
from modules.stack_builder import StackBuilder
from tests.test_hyperstack_video_grouping import _frame_path, _stack_df
from modules.lumascope_api.frame_record import FrameTime, TimestampSource
from modules.layer_record import release_colors
from tests.frame_records import recorded_record, write_recorded
from tests.ome_schema import schema_errors
from tests.stock_channels import LED1, LED2


def _write_timestamped_frames(scope, tmp_path, df, base_s=1_755_000_000.0, spacing_s=0.5):
    """Write every df row via the production frame writer, with real
    per-frame capture timestamps spaced spacing_s apart."""
    for _, row in df.iterrows():
        loc = tmp_path / row['Filepath']
        loc.parent.mkdir(parents=True, exist_ok=True)
        n = recording_frames.frame_number(loc.name)
        write_recorded(
            scope,
            np.full((4, 4), n, dtype=np.uint8),
            loc,
            channel=row['Color'],
            record=recorded_record(
                captured_at=FrameTime(
                    at=datetime.datetime.fromtimestamp(base_s + n * spacing_s),
                    source=TimestampSource.HOST_ARRIVAL,
                ),
                frame_significant_bits=8,
            ),
            frame_number=n,
        )


def _build(tmp_path, df, out_name='out.ome.tiff'):
    builder = StackBuilder(has_turret=False)
    ((_, group),) = StackBuilder._get_groups(df)
    result = builder._group_algorithm(
        path=tmp_path,
        df=group.reset_index(drop=True),
        output_file_loc=pathlib.Path(out_name),
        save_encoding='right_aligned',
        channel_colors=release_colors(),
    )
    assert result.status, f'stack build failed: {result.error}'
    with tf.TiffFile(str(tmp_path / out_name)) as tif:
        return tif.ome_metadata


def test_video_well_stack_is_schema_valid_with_ruled_axes(sim_scope, tmp_path):
    rows = [(_frame_path('A1', 'BF', 0, n), 0, 'BF', 0) for n in range(3)]
    df = _stack_df(rows)
    _write_timestamped_frames(sim_scope, tmp_path, df)

    ome_xml = _build(tmp_path, df)

    assert schema_errors(ome_xml) == []
    ome = ome_types.from_xml(ome_xml)
    pixels = ome.images[0].pixels
    assert (pixels.size_t, pixels.size_z, pixels.size_c) == (3, 1, 1)
    assert pixels.dimension_order.value == 'XYCZT'
    assert len(pixels.planes) == 3, 'one OME Plane per source frame'
    deltas = sorted(p.delta_t for p in pixels.planes)
    assert deltas == [0.0, 0.5, 1.0], f'per-plane DeltaT must survive the round-trip, got {deltas}'


def test_two_channel_video_well_carries_both_channels(sim_scope, tmp_path):
    rows = []
    for color in (LED1, LED2):
        rows += [(_frame_path('A1', color, 0, n), 0, color, 0) for n in range(2)]
    df = _stack_df(rows)
    _write_timestamped_frames(sim_scope, tmp_path, df)

    ome_xml = _build(tmp_path, df)

    assert schema_errors(ome_xml) == []
    ome = ome_types.from_xml(ome_xml)
    pixels = ome.images[0].pixels
    assert (pixels.size_t, pixels.size_c) == (2, 2)
    assert sorted(c.name for c in pixels.channels) == [LED1, LED2]
    assert len(pixels.planes) == 4, 'T x C planes, one per source frame'


def test_hyperstack_container_is_ome_only(sim_scope, tmp_path, monkeypatch):
    """The stack writer must not pass tifffile's imagej flag: ome=True
    suppresses it wholesale, so its only observable product is a
    'nonconformant BigTIFF ImageJ' warning on >3.8 GiB writes. Color
    reaches FIJI through OME Channel.Color (Bio-Formats Importer,
    Composite mode), never through an ImageJ metadata block."""
    rows = [(_frame_path('A1', 'BF', 0, n), 0, 'BF', 0) for n in range(2)]
    df = _stack_df(rows)
    _write_timestamped_frames(sim_scope, tmp_path, df)

    captured = {}
    real_writer = image_utils.tf.TiffWriter

    class SpyWriter(real_writer):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(image_utils.tf, 'TiffWriter', SpyWriter)
    _build(tmp_path, df)

    assert captured.get('ome') is True, 'spy must have seen the stack write'
    assert 'imagej' not in captured, (
        'the hyperstack container is OME-only; tifffile ignores imagej '
        'under ome=True and the dead flag only buys the BigTIFF warning'
    )
