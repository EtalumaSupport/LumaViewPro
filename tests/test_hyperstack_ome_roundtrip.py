# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Headless OME round-trip: an independent parser validates our stacks.

tifffile serializes whatever unit and axis strings it is handed, so it
cannot catch its own output being schema-invalid -- Bio-Formats' lenient
parsing then hides the defect until a strict reader refuses the file.
ome-types (dev-only) is that strict reader: every assertion here is
against the OME data model parsed back from the written XML, pinning the
ruled spec -- one OME-TIFF per (well, scan), T = frame capture order,
C = channel, DimensionOrder XYCZT, per-plane DeltaT from real
timestamps. The ImageJ/Fiji open and the 8/16-bit + BigTIFF permutation
sweep ride the bench gate; this is the headless half.
"""

import pathlib

import numpy as np
import ome_types
import tifffile as tf

from modules import image_utils, recording_frames
from modules.stack_builder import StackBuilder
from tests.test_hyperstack_video_grouping import _frame_path, _stack_df


def _write_timestamped_frames(tmp_path, df, base_s=1_755_000_000.0, spacing_s=0.5):
    """Write every df row via the production frame writer, with real
    per-frame capture timestamps spaced spacing_s apart."""
    for _, row in df.iterrows():
        loc = tmp_path / row['Filepath']
        loc.parent.mkdir(parents=True, exist_ok=True)
        n = recording_frames.frame_number(loc.name)
        metadata, _ = recording_frames.tiff_frame_metadata(
            timestamp_s=base_s + n * spacing_s,
            frame_number=n,
            chunks=None,
            tick_freq_hz=None,
        )
        image_utils.write_tiff(
            data=np.full((4, 4), n, dtype=np.uint8),
            file_loc=loc,
            metadata=metadata,
            ome=False,
            color=row['Color'],
            significant_bits=8,
            save_encoding='8bit',
            video_frame=True,
        )


def _build(tmp_path, df, out_name='out.ome.tiff'):
    builder = StackBuilder(has_turret=False)
    ((_, group),) = StackBuilder._get_groups(df)
    result = builder._group_algorithm(
        path=tmp_path,
        df=group.reset_index(drop=True),
        output_file_loc=pathlib.Path(out_name),
    )
    assert result.status, f'stack build failed: {result.error}'
    with tf.TiffFile(str(tmp_path / out_name)) as tif:
        return tif.ome_metadata


def test_video_well_stack_is_schema_valid_with_ruled_axes(tmp_path):
    rows = [(_frame_path('A1', 'BF', 0, n), 0, 'BF', 0) for n in range(3)]
    df = _stack_df(rows)
    _write_timestamped_frames(tmp_path, df)

    ome_xml = _build(tmp_path, df)

    # from_xml IS the schema gate: an invalid unit or axis string raises.
    ome = ome_types.from_xml(ome_xml)
    pixels = ome.images[0].pixels
    assert (pixels.size_t, pixels.size_z, pixels.size_c) == (3, 1, 1)
    assert pixels.dimension_order.value == 'XYCZT'
    assert len(pixels.planes) == 3, 'one OME Plane per source frame'
    deltas = sorted(p.delta_t for p in pixels.planes)
    assert deltas == [0.0, 0.5, 1.0], f'per-plane DeltaT must survive the round-trip, got {deltas}'


def test_two_channel_video_well_carries_both_channels(tmp_path):
    rows = []
    for color in ('Green', 'Red'):
        rows += [(_frame_path('A1', color, 0, n), 0, color, 0) for n in range(2)]
    df = _stack_df(rows)
    _write_timestamped_frames(tmp_path, df)

    ome_xml = _build(tmp_path, df)

    ome = ome_types.from_xml(ome_xml)
    pixels = ome.images[0].pixels
    assert (pixels.size_t, pixels.size_c) == (2, 2)
    assert sorted(c.name for c in pixels.channels) == ['Green', 'Red']
    assert len(pixels.planes) == 4, 'T x C planes, one per source frame'


def test_hyperstack_container_is_ome_only(tmp_path, monkeypatch):
    """The stack writer must not pass tifffile's imagej flag: ome=True
    suppresses it wholesale, so its only observable product is a
    'nonconformant BigTIFF ImageJ' warning on >3.8 GiB writes. Color
    reaches FIJI through OME Channel.Color (Bio-Formats Importer,
    Composite mode), never through an ImageJ metadata block."""
    rows = [(_frame_path('A1', 'BF', 0, n), 0, 'BF', 0) for n in range(2)]
    df = _stack_df(rows)
    _write_timestamped_frames(tmp_path, df)

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
