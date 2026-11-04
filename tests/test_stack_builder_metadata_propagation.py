"""Regression: stack_builder propagates per-plane positions + acquisition
context into hyperstack OME-XML output.

Bug shape (pre-fix): StackBuilder._create_stack collected per-plane
PositionX/Y/Z lists from the dataframe and passed them to
``_generate_image_metadata`` as the ``plane_metadata`` parameter, but
the function ignored the parameter -- the positions were silently
dropped from the OME-XML <Plane> elements. Multi-tile T-series Z-stacks
saved with no per-plane provenance. Downstream FIJI / OME-aware readers
could not reconstruct plane positions for the hyperstack.

Additionally, ``_generate_image_metadata`` wrote only a minimal OME
block (axes + SignificantBits + Pixels (size+unit) + Channel.Name). It
did not propagate forward the source captures' objective, instrument,
or plate metadata -- so hyperstacks lost all acquisition provenance
that the per-frame TIFFs carry.

Fix: route through ``image_utils.build_hyperstack_output_metadata``
which (1) writes the plane_metadata lists into the OME Plane subdict
so they reach the <Plane> XML elements; (2) states what the input
frames agree on, the instrument, plate and objective among it.
"""

from __future__ import annotations

import datetime
import pathlib

import numpy as np
import pandas as pd
import pytest
import tifffile as tf

from modules import image_utils
from modules.stack_builder import StackBuilder
from modules.layer_record import release_color
from modules.layer_record import release_colors
from tests.ome_schema import q, valid_description
from tests.stock_channels import LED0, LED1, LED2


def _write_structured_input(
    path: pathlib.Path,
    *,
    channel: str,
    plate_pos_mm: dict,
    z_pos_um: float,
    value: int = 100,
) -> None:
    """Write a per-frame TIFF via image_utils.write_tiff so it carries
    structured acquisition metadata (the realistic input shape for
    stack_builder)."""
    arr = np.full((4, 4), value, dtype=np.uint8)
    image_utils.write_tiff(
        data=arr,
        file_loc=path,
        significant_bits=8,
        save_encoding='8bit',
        metadata={
            'datetime': datetime.datetime(2026, 5, 27, 12, 0, 0),
            'plate_pos_mm': plate_pos_mm,
            'z_pos_um': z_pos_um,
            'objective': {
                'id': '20x Oly',
                'magnification': 20,
                'numerical_aperture': 0.45,
                'working_distance_mm': 8.1,
            },
            'exposure_time_ms': 50.0,
            'gain_db': 3.0,
            'illumination_ma': 75.0,
            'pixel_size_um': 0.5,
            'channel': channel,
            'instrument': {
                'manufacturer': 'Etaluma',
                'model': 'LS720',
                'serial_number': 'SN12062',
                'camera_model': 'Basler a2A1920',
            },
            'plate': {
                'name': '96-well',
                'rows': 8,
                'columns': 12,
                'well_label': 'A1',
            },
        },
        ome=False,
        color=release_color(channel),
        white_bits=None,
    )


class TestStackBuilderPropagatesPlanePositions:
    """Per-plane PositionX/Y/Z must reach the OME-XML <Plane> elements.
    Pre-fix: positions were collected then ignored by
    _generate_image_metadata, silently dropped from the output."""

    def test_z_stack_plane_positions_land_in_ome_xml(self, tmp_path):
        # Build a single-channel Z-stack: 1 T x 3 Z x 1 C.
        # Plane positions distinguishable: Z = 100.0, 110.0, 120.0 um.
        z_positions = [100.0, 110.0, 120.0]
        plate_pos = {'x': 12.5, 'y': 8.25}
        rows = []
        for z_idx, z_val in enumerate(z_positions):
            fname = f'frame_t0_z{z_idx}_c0.tiff'
            _write_structured_input(
                tmp_path / fname,
                channel=LED1,
                plate_pos_mm=plate_pos,
                z_pos_um=z_val,
                value=50 + z_idx * 10,
            )
            rows.append(
                {
                    'Filepath': fname,
                    'Color': LED1,
                    'Scan Count': 0,
                    'Z-Slice': z_idx,
                    'X': plate_pos['x'],
                    'Y': plate_pos['y'],
                    'Z': z_val,
                }
            )
        df = pd.DataFrame(rows)

        output_file_loc = pathlib.Path('out.ome.tiff')
        result = StackBuilder._create_stack(
            path=tmp_path,
            df=df,
            output_file_loc=output_file_loc,
            save_encoding='right_aligned',
            channel_colors=release_colors(),
        )
        assert result['status'], f'_create_stack failed: {result.get("error")}'

        with tf.TiffFile(str(tmp_path / output_file_loc)) as tif:
            ome_xml = tif.ome_metadata or ''

        # OME-XML should contain one <Plane> per T*Z*C plane (1*3*1 = 3).
        assert ome_xml.count('<Plane ') == 3, (
            f'Hyperstack must emit one OME <Plane> per T*Z*C plane '
            f'(expected 3, got {ome_xml.count("<Plane ")}).'
        )

        # Each plane must carry its PositionZ -- the bug pre-fix dropped
        # these silently.
        for z_val in z_positions:
            assert f'PositionZ="{z_val}"' in ome_xml, (
                f'Expected PositionZ="{z_val}" in OME-XML <Plane> element; '
                f'missing means plane positions are not propagating into '
                f'the hyperstack output (regression of the plane_metadata-'
                f'ignored bug).'
            )

    def test_z_stack_plane_xy_positions_land_in_ome_xml(self, tmp_path):
        # Distinguishable X/Y across planes -- catches a regression that
        # only writes the first plane's position.
        positions = [
            (10.0, 5.0, 100.0),
            (11.0, 6.0, 110.0),
            (12.0, 7.0, 120.0),
        ]
        rows = []
        for z_idx, (x, y, z) in enumerate(positions):
            fname = f'frame_z{z_idx}.tiff'
            _write_structured_input(
                tmp_path / fname,
                channel=LED1,
                plate_pos_mm={'x': x, 'y': y},
                z_pos_um=z,
            )
            rows.append(
                {
                    'Filepath': fname,
                    'Color': LED1,
                    'Scan Count': 0,
                    'Z-Slice': z_idx,
                    'X': x,
                    'Y': y,
                    'Z': z,
                }
            )
        df = pd.DataFrame(rows)

        StackBuilder._create_stack(
            path=tmp_path,
            df=df,
            output_file_loc=pathlib.Path('out.ome.tiff'),
            save_encoding='right_aligned',
            channel_colors=release_colors(),
        )
        with tf.TiffFile(str(tmp_path / 'out.ome.tiff')) as tif:
            ome_xml = tif.ome_metadata or ''

        for x, y, _ in positions:
            assert f'PositionX="{x}"' in ome_xml, (
                f'PositionX="{x}" missing from OME-XML; per-plane X '
                f'positions not propagating to hyperstack output.'
            )
            assert f'PositionY="{y}"' in ome_xml, (
                f'PositionY="{y}" missing from OME-XML; per-plane Y '
                f'positions not propagating to hyperstack output.'
            )


class TestStackBuilderPropagatesPixelSizeAndChannels:
    """Hyperstack output must carry forward the schema fields tifffile's
    OME serializer accepts: PhysicalSizeX/Y, Channel.Name list, and
    per-plane positions. The Instrument, which tifffile drops, is spliced
    in after the write, and the plate and well ride in the LVP annotation
    (see TestAHyperstackStatesWhatItsFramesAgreeOn)."""

    def test_pixel_size_in_ome_xml(self, tmp_path):
        rows = []
        for z_idx in range(2):
            fname = f'frame_z{z_idx}.tiff'
            _write_structured_input(
                tmp_path / fname,
                channel=LED2,
                plate_pos_mm={'x': 0.0, 'y': 0.0},
                z_pos_um=float(z_idx) * 10,
            )
            rows.append(
                {
                    'Filepath': fname,
                    'Color': LED2,
                    'Scan Count': 0,
                    'Z-Slice': z_idx,
                    'X': 0.0,
                    'Y': 0.0,
                    'Z': float(z_idx) * 10,
                }
            )
        df = pd.DataFrame(rows)

        StackBuilder._create_stack(
            path=tmp_path,
            df=df,
            output_file_loc=pathlib.Path('out.ome.tiff'),
            save_encoding='right_aligned',
            channel_colors=release_colors(),
        )
        with tf.TiffFile(str(tmp_path / 'out.ome.tiff')) as tif:
            ome_xml = tif.ome_metadata or ''

        assert 'PhysicalSizeX=' in ome_xml, (
            'Hyperstack OME-XML must declare PhysicalSizeX so consumers '
            'can compute on-screen scale bars.'
        )
        # The unit is the OME schema's micro-sign token; the ASCII 'um'
        # shorthand is schema-invalid and strict readers refuse the file.
        assert f'PhysicalSizeXUnit="{image_utils.OME_UNIT_MICROMETER}"' in ome_xml, (
            'PhysicalSizeX must declare its unit (microns, OME token).'
        )

    def test_channel_names_in_ome_xml(self, tmp_path):
        rows = []
        for channel in (LED1, LED2):
            for z_idx in range(2):
                fname = f'frame_{channel}_z{z_idx}.tiff'
                _write_structured_input(
                    tmp_path / fname,
                    channel=channel,
                    plate_pos_mm={'x': 0.0, 'y': 0.0},
                    z_pos_um=float(z_idx) * 10,
                )
                rows.append(
                    {
                        'Filepath': fname,
                        'Color': channel,
                        'Scan Count': 0,
                        'Z-Slice': z_idx,
                        'X': 0.0,
                        'Y': 0.0,
                        'Z': float(z_idx) * 10,
                    }
                )
        df = pd.DataFrame(rows)

        StackBuilder._create_stack(
            path=tmp_path,
            df=df,
            output_file_loc=pathlib.Path('out.ome.tiff'),
            save_encoding='right_aligned',
            channel_colors=release_colors(),
        )
        with tf.TiffFile(str(tmp_path / 'out.ome.tiff')) as tif:
            ome_xml = tif.ome_metadata or ''

        assert f'Name="{LED1}"' in ome_xml, (
            f'{LED1} channel name must reach OME-XML <Channel> element.'
        )
        assert f'Name="{LED2}"' in ome_xml, (
            f'{LED2} channel name must reach OME-XML <Channel> element.'
        )


class TestStackBuilderHandlesInputsWithoutStructuredMetadata:
    """Inputs from bare tf.imwrite (test fixtures, external pipelines)
    have no structured metadata. Stack builder must still produce a
    valid hyperstack output -- fall back to minimal defaults rather
    than crashing."""

    def test_bare_input_falls_back_gracefully(self, tmp_path):
        rows = []
        for z_idx in range(2):
            fname = f'frame_z{z_idx}.tiff'
            # Bare tf.imwrite -- no structured metadata.
            tf.imwrite(
                str(tmp_path / fname),
                np.full((4, 4), 100, dtype=np.uint8),
                compression='lzw',
            )
            rows.append(
                {
                    'Filepath': fname,
                    'Color': LED1,
                    'Scan Count': 0,
                    'Z-Slice': z_idx,
                    'X': 0.0,
                    'Y': 0.0,
                    'Z': float(z_idx),
                }
            )
        df = pd.DataFrame(rows)

        result = StackBuilder._create_stack(
            path=tmp_path,
            df=df,
            output_file_loc=pathlib.Path('out.ome.tiff'),
            save_encoding='right_aligned',
            channel_colors=release_colors(),
        )
        assert result['status'], f'Bare-input fallback path must not crash: {result.get("error")}'
        # Output should still be a valid OME-TIFF.
        with tf.TiffFile(str(tmp_path / 'out.ome.tiff')) as tif:
            assert tif.is_ome, 'Hyperstack output must remain OME-tagged'


def _two_slice_stack(tmp_path, *, plate_pos_mm=None, bare=False) -> pathlib.Path:
    """A hyperstack built by the stack builder from two Z slices of LED1,
    written through write_tiff (or bare tf.imwrite: no record)."""
    plate_pos_mm = plate_pos_mm or {'x': 0.0, 'y': 0.0}
    rows = []
    for z_idx in range(2):
        fname = f'frame_z{z_idx}.tiff'
        if bare:
            tf.imwrite(str(tmp_path / fname), np.full((4, 4), 100, dtype=np.uint8))
        else:
            _write_structured_input(
                tmp_path / fname,
                channel=LED1,
                plate_pos_mm=plate_pos_mm,
                z_pos_um=float(z_idx) * 10,
            )
        rows.append(
            {
                'Filepath': fname,
                'Color': LED1,
                'Scan Count': 0,
                'Z-Slice': z_idx,
                'X': plate_pos_mm['x'],
                'Y': plate_pos_mm['y'],
                'Z': float(z_idx) * 10,
            }
        )
    result = StackBuilder._create_stack(
        path=tmp_path,
        df=pd.DataFrame(rows),
        output_file_loc=pathlib.Path('out.ome.tiff'),
        save_encoding='right_aligned',
        channel_colors=release_colors(),
    )
    assert result['status'], result.get('error')
    return tmp_path / 'out.ome.tiff'


class TestAHyperstackStatesWhatItsFramesAgreeOn:
    """A hyperstack names the scope, camera and objective its frames agree on
    in the OME schema's Instrument, and reads back, through the one reader,
    the facts a derived file states of them: the instrument, objective,
    plate and well, but no capture's own fact. The file writes no private
    sidecar tag and is valid OME."""

    def test_the_instrument_is_in_the_schema(self, tmp_path):
        root = valid_description(_two_slice_stack(tmp_path))
        microscope = root.find(f'{q("Instrument")}/{q("Microscope")}')
        assert dict(microscope.attrib) == {
            'Manufacturer': 'Etaluma',
            'Model': 'LS720',
            'SerialNumber': 'SN12062',
        }
        detector = root.find(f'{q("Instrument")}/{q("Detector")}')
        assert detector.get('Model') == 'Basler a2A1920'
        objective = root.find(f'{q("Instrument")}/{q("Objective")}')
        assert (objective.get('NominalMagnification'), objective.get('LensNA')) == ('20', '0.45')
        assert root.find(f'{q("Image")}/{q("InstrumentRef")}') is not None

    def test_the_record_reads_back_its_frames_shared_facts(self, tmp_path):
        out = _two_slice_stack(tmp_path, plate_pos_mm={'x': 12.5, 'y': 8.25})
        record = image_utils.read_file_record(out)
        assert record['instrument'] == {
            'manufacturer': 'Etaluma',
            'model': 'LS720',
            'serial_number': 'SN12062',
            'camera_model': 'Basler a2A1920',
        }
        assert record['plate'] == {'name': '96-well', 'rows': 8, 'columns': 12, 'well_label': 'A1'}
        assert record['objective']['magnification'] == 20
        assert record['plate_pos_mm'] == {'x': 12.5, 'y': 8.25}
        # One channel: its settings travel; never a capture's own fact.
        assert record['exposure_time_ms'] == 50.0
        assert [key for key in record if image_utils.FILE_FACTS[key].kind.value == 'capture'] == []
        with tf.TiffFile(str(out)) as tif:
            assert 51838 not in tif.pages[0].tags
        valid_description(out)

    def test_a_stack_of_frames_with_no_record_states_no_instrument(self, tmp_path):
        out = _two_slice_stack(tmp_path, bare=True)
        root = valid_description(out)
        assert root.find(q('Instrument')) is None
        record = image_utils.read_file_record(out)
        assert 'instrument' not in record
        assert 'plate' not in record


class TestHyperstackChannelColor:
    """Hyperstack output must carry OME Channel.Color so FIJI's
    Bioformats reader auto-opens in Composite view with the correct
    color per channel. Without Color, FIJI shows grayscale and the user
    has to manually set colors via Image > Color > Channels Tool every
    time. (LUTs in metadata['LUTs'] are dropped by tifffile when
    ome=True is set; Channel.Color is the OME-mode equivalent.)
    """

    def test_metadata_includes_color_per_channel(self, tmp_path):
        ref = tmp_path / 'ref.tiff'
        _write_structured_input(
            ref,
            channel=LED1,
            plate_pos_mm={'x': 0.0, 'y': 0.0},
            z_pos_um=0.0,
        )
        metadata = image_utils.build_hyperstack_output_metadata(
            input_records=[image_utils.read_file_record(ref)],
            channel_names=[LED1, 'BF', LED2, LED0],
            channel_colors=[release_color(name) for name in [LED1, 'BF', LED2, LED0]],
            plane_positions={
                'PositionX': [0.0, 0.0, 0.0, 0.0],
                'PositionY': [0.0, 0.0, 0.0, 0.0],
                'PositionZ': [0.0, 0.0, 0.0, 0.0],
                'DeltaT': None,
            },
            significant_bits=8,
            pixel_size_um=2.2,
        )
        assert 'Color' in metadata['Channel']
        colors = metadata['Channel']['Color']
        assert len(colors) == 4, 'One Color per channel'

        # Verify OME RGBA encoding via the helper -- spec encodes as
        # (R << 24) | (G << 16) | (B << 8) | A with two's-complement
        # int32. Green (0,255,0,255) -> 16711935 (positive int32).
        # Red (255,0,0,255) -> -16776961 (signed-folded). Blue
        # (0,0,255,255) -> 65535. White (BF: 255,255,255,255) -> -1.
        green_color, bf_color, red_color, blue_color = colors
        assert green_color == 0x00FF00FF, f'Green RGBA: {green_color}'
        assert blue_color == 0x0000FFFF, f'Blue RGBA: {blue_color}'
        assert red_color == 0xFF0000FF - (1 << 32), f'Red RGBA: {red_color}'
        assert bf_color == 0xFFFFFFFF - (1 << 32), f'BF (white) RGBA: {bf_color}'

    def test_color_reaches_ome_xml(self, tmp_path):
        # End-to-end: write through stack_builder, read OME-XML back,
        # confirm Color attribute on each Channel element. This is what
        # FIJI's Bioformats reader picks up to auto-color the hyperstack.
        rows = []
        for channel in (LED1, LED2):
            for z_idx in range(2):
                fname = f'frame_{channel}_z{z_idx}.tiff'
                _write_structured_input(
                    tmp_path / fname,
                    channel=channel,
                    plate_pos_mm={'x': 0.0, 'y': 0.0},
                    z_pos_um=float(z_idx) * 10,
                )
                rows.append(
                    {
                        'Filepath': fname,
                        'Color': channel,
                        'Scan Count': 0,
                        'Z-Slice': z_idx,
                        'X': 0.0,
                        'Y': 0.0,
                        'Z': float(z_idx) * 10,
                    }
                )
        df = pd.DataFrame(rows)

        StackBuilder._create_stack(
            path=tmp_path,
            df=df,
            output_file_loc=pathlib.Path('out.ome.tiff'),
            save_encoding='right_aligned',
            channel_colors=release_colors(),
        )

        with tf.TiffFile(str(tmp_path / 'out.ome.tiff')) as tif:
            ome_xml = tif.ome_metadata or ''

        assert 'Color=' in ome_xml, (
            'Hyperstack OME-XML must carry Channel.Color so FIJI / '
            'Bioformats auto-color the channels in Composite view.'
        )
        # Two channels -- two Color attrs.
        assert ome_xml.count('Color=') == 2, (
            f'Expected 2 Color attrs (one per channel), got {ome_xml.count("Color=")}'
        )


class TestHyperstackScaleComesFromTheInput:
    """The hyperstack's PhysicalSizeX must be the input frames' own scale.

    Re-deriving it from an objective focal length recomputed a value the input
    pixels already carry. The derived answer disagreed with the file whenever
    the capture's real optics differed from the caller's assumed focal length
    or binning, and on a scope with no objective selector the focal length is
    a default rather than a measurement -- so the output silently claimed a
    scale the pixels never had."""

    def test_hyperstack_carries_the_input_frames_pixel_size(self, tmp_path):
        # The fixture frames declare 0.5 um/px. The retired re-derive returned
        # 2.0 for these arguments, so this assertion separates the two.
        rows = []
        for z_idx in range(2):
            fname = f'frame_z{z_idx}.tiff'
            _write_structured_input(
                tmp_path / fname,
                channel=LED2,
                plate_pos_mm={'x': 0.0, 'y': 0.0},
                z_pos_um=float(z_idx) * 10,
            )
            rows.append(
                {
                    'Filepath': fname,
                    'Color': LED2,
                    'Scan Count': 0,
                    'Z-Slice': z_idx,
                    'X': 0.0,
                    'Y': 0.0,
                    'Z': float(z_idx) * 10,
                }
            )

        StackBuilder._create_stack(
            path=tmp_path,
            df=pd.DataFrame(rows),
            output_file_loc=pathlib.Path('out.ome.tiff'),
            save_encoding='right_aligned',
            channel_colors=release_colors(),
        )

        with tf.TiffFile(str(tmp_path / 'out.ome.tiff')) as tif:
            ome_xml = tif.ome_metadata or ''

        assert 'PhysicalSizeX="0.5"' in ome_xml, (
            "Hyperstack must carry the input frames' own PhysicalSizeX (0.5), not a "
            f'value re-derived from the objective. Got: {ome_xml[:400]}'
        )
        assert 'PhysicalSizeX="2.0"' not in ome_xml, (
            'PhysicalSizeX was re-derived from the focal length instead of read '
            'from the input frames.'
        )

    def test_hyperstack_makes_no_scale_claim_when_the_input_has_none(self, tmp_path):
        # Bare frames (third-party or pre-metadata captures) carry no
        # PhysicalSizeX, and binning is not recoverable to re-derive one. The
        # build must succeed and stay silent about scale rather than invent it.
        rows = []
        for z_idx in range(2):
            fname = f'frame_z{z_idx}.tiff'
            tf.imwrite(
                str(tmp_path / fname),
                np.full((4, 4), 100, dtype=np.uint8),
                compression='lzw',
            )
            rows.append(
                {
                    'Filepath': fname,
                    'Color': LED1,
                    'Scan Count': 0,
                    'Z-Slice': z_idx,
                    'X': 0.0,
                    'Y': 0.0,
                    'Z': float(z_idx),
                }
            )

        StackBuilder._create_stack(
            path=tmp_path,
            df=pd.DataFrame(rows),
            output_file_loc=pathlib.Path('out.ome.tiff'),
            save_encoding='right_aligned',
            channel_colors=release_colors(),
        )

        out = tmp_path / 'out.ome.tiff'
        assert out.exists(), 'A scale-less input must still produce a hyperstack.'
        with tf.TiffFile(str(out)) as tif:
            ome_xml = tif.ome_metadata or ''
            resolution_unit = tif.pages[0].tags['ResolutionUnit'].value

        assert 'PhysicalSizeX=' not in ome_xml, (
            'An input carrying no scale must yield no PhysicalSizeX claim; an '
            f'invented one is indistinguishable from a measured one. Got: {ome_xml[:400]}'
        )
        # tifffile always writes an XResolution tag and defaults it to 1/1.
        # Under CENTIMETER that reads as one pixel per centimetre, so the unit
        # is what decides whether the file claims an absolute scale.
        assert int(resolution_unit) == 1, (
            'With no known pixel size the resolution unit must be NONE (ratio '
            'only). CENTIMETER against the default 1/1 resolution claims a '
            f'1 cm pixel. Got unit {int(resolution_unit)}.'
        )


def test_write_hyperstack_streams_a_plane_iterator(tmp_path):
    """write_hyperstack_tiff takes an iterator of 2D planes plus the stack
    shape and dtype -- the stack never materializes in RAM, so the memory
    bound is one plane regardless of stack size (the cube-in-RAM build and
    its psutil guard are gone)."""
    plane_a = np.full((4, 4), 10, dtype=np.uint8)
    plane_b = np.full((4, 4), 20, dtype=np.uint8)

    reference = tmp_path / 'ref.tiff'
    tf.imwrite(reference, plane_a)
    metadata = image_utils.build_hyperstack_output_metadata(
        input_records=[image_utils.read_file_record(reference)],
        channel_names=[LED1],
        channel_colors=[release_color(LED1)],
        plane_positions={
            'PositionX': [1.0, 1.0],
            'PositionY': [2.0, 2.0],
            'PositionZ': [10.0, 10.0],
            'DeltaT': None,
        },
        significant_bits=8,
        pixel_size_um=None,
    )

    out = tmp_path / 'streamed.ome.tiff'
    image_utils.write_hyperstack_tiff(
        planes=iter([plane_a, plane_b]),
        file_loc=out,
        shape=(2, 1, 1, 4, 4),
        dtype=np.uint8,
        hyperstack_metadata=metadata,
        # photometric pins the trailing dims as YX; without it tifffile's
        # shape heuristics can read a <=4-wide last dimension as samples.
        hyperstack_options={'photometric': 'minisblack'},
    )

    with tf.TiffFile(str(out)) as tif:
        data = tif.asarray()
    assert data.reshape(2, 4, 4).tolist() == np.stack([plane_a, plane_b]).tolist(), (
        'Streamed planes must land in the file in iterator order.'
    )


def test_nondense_scan_counts_build_by_rank(tmp_path):
    """A rectangular well whose Scan Count values are not a dense 0..N-1
    range (e.g. only scans 0 and 2 recorded video) must still build, with T
    following sorted rank. The cube build indexed the T axis with the RAW
    Scan Count value and crashed on any non-dense range."""
    plate_pos = {'x': 1.0, 'y': 2.0}
    rows = []
    for scan, value in [(0, 30), (2, 60)]:
        fname = f'frame_s{scan}.tiff'
        _write_structured_input(
            tmp_path / fname,
            channel=LED1,
            plate_pos_mm=plate_pos,
            z_pos_um=10.0,
            value=value,
        )
        rows.append(
            {
                'Filepath': fname,
                'Color': LED1,
                'Scan Count': scan,
                'Z-Slice': 0,
                'X': plate_pos['x'],
                'Y': plate_pos['y'],
                'Z': 10.0,
            }
        )
    df = pd.DataFrame(rows)

    output_file_loc = pathlib.Path('out.ome.tiff')
    result = StackBuilder._create_stack(
        path=tmp_path,
        df=df,
        output_file_loc=output_file_loc,
        save_encoding='right_aligned',
        channel_colors=release_colors(),
    )
    assert result['status'], f'_create_stack failed: {result.get("error")}'

    with tf.TiffFile(str(tmp_path / output_file_loc)) as tif:
        data = tif.asarray().reshape(2, 4, 4)
    assert int(data[0, 0, 0]) == 30 and int(data[1, 0, 0]) == 60, (
        'T order must follow sorted Scan Count rank.'
    )


def test_mixed_dtype_inputs_refused_naming_the_file(tmp_path):
    """A well mixing pixel dtypes a stack cannot promote cannot share one
    hyperstack; the build fails with a typed error naming the offending frame.
    The cube build silently CAST the mismatched plane into the stack dtype,
    producing wrong data with a success status. The one mixture a stack
    promotes is uint8 beside uint16 (a sum beside an unsummed channel);
    anything else -- here a float32 frame beside a uint8 one -- is refused,
    and leaves no file behind (#722)."""
    from modules.exceptions import CaptureError

    plate_pos = {'x': 1.0, 'y': 2.0}
    _write_structured_input(
        tmp_path / 'frame_s0.tiff', channel=LED1, plate_pos_mm=plate_pos, z_pos_um=10.0
    )
    # A bare float32 TIFF simulates a foreign frame landing in the same well
    # folder.
    tf.imwrite(tmp_path / 'frame_s1.tiff', np.full((4, 4), 40.0, dtype=np.float32))

    rows = [
        {
            'Filepath': f'frame_s{scan}.tiff',
            'Color': LED1,
            'Scan Count': scan,
            'Z-Slice': 0,
            'X': plate_pos['x'],
            'Y': plate_pos['y'],
            'Z': 10.0,
        }
        for scan in (0, 1)
    ]
    df = pd.DataFrame(rows)

    with pytest.raises(CaptureError, match=r'frame_s1\.tiff') as refused:
        StackBuilder._create_stack(
            path=tmp_path,
            df=df,
            output_file_loc=pathlib.Path('out.ome.tiff'),
            save_encoding='right_aligned',
            channel_colors=release_colors(),
        )
    assert refused.value.reason == 'mixed_pixel_types'
    # Refused part-way through the write: nothing is left at the output name.
    assert not (tmp_path / 'out.ome.tiff').exists()
    assert not (tmp_path / 'out.ome.tiff.tmp').exists()


def test_hyperstack_output_uses_strips(tmp_path):
    """The hyperstack write uses strips, not tiles. Tiling forced ImageJ
    through Bio-Formats (the native reader cannot open tiled TIFFs), and its
    lenient colormap rescaling masked a colormap-scale defect in the 8-bit
    still path while breaking native open."""
    plate_pos = {'x': 1.0, 'y': 2.0}
    fname = 'frame_t0_z0_c0.tiff'
    _write_structured_input(tmp_path / fname, channel=LED1, plate_pos_mm=plate_pos, z_pos_um=10.0)
    df = pd.DataFrame(
        [
            {
                'Filepath': fname,
                'Color': LED1,
                'Scan Count': 0,
                'Z-Slice': 0,
                'X': plate_pos['x'],
                'Y': plate_pos['y'],
                'Z': 10.0,
            }
        ]
    )
    output_file_loc = pathlib.Path('out.ome.tiff')
    result = StackBuilder._create_stack(
        path=tmp_path,
        df=df,
        output_file_loc=output_file_loc,
        save_encoding='right_aligned',
        channel_colors=release_colors(),
    )
    assert result['status'], f'_create_stack failed: {result.get("error")}'

    with tf.TiffFile(str(tmp_path / output_file_loc)) as tif:
        assert not tif.pages[0].is_tiled, 'hyperstack output must use strips'
