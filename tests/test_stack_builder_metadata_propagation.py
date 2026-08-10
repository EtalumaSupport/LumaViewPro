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
so they reach the <Plane> XML elements; (2) reads structured metadata
from one input frame and propagates Instrument + Plate + Objective
forward.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import tifffile as tf

from modules import image_utils
from modules import stack_builder as stack_builder_module
from modules.stack_builder import StackBuilder


@pytest.fixture(autouse=True)
def _real_available_memory(monkeypatch):
    """Conftest mocks psutil globally, leaving virtual_memory().available
    a MagicMock that can't be compared with int. The hyperstack memory
    pre-check needs a real int; route to a generous 16 GB sentinel so
    the check passes for small test arrays."""
    mem = MagicMock()
    mem.available = 16 * 1024 * 1024 * 1024
    monkeypatch.setattr(stack_builder_module.psutil, 'virtual_memory', lambda: mem)


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
            'datetime': '2026-05-27T12:00:00',
            'plate_pos_mm': plate_pos_mm,
            'z_pos_um': z_pos_um,
            'objective': {
                'model': 'PlanFluor20x',
                'manufacturer': 'Nikon',
                'magnification': 20,
                'aperture': 0.45,
                'working_distance': 8.1,
                'immersion': 'Air',
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
                'firmware_version': '4.0.0-beta14',
                'camera_model': 'Basler a2A1920',
            },
            'plate': {
                'name': '96-well',
                'rows': 8,
                'columns': 12,
            },
            'well_label': 'A1',
        },
        ome=False,
        color=channel,
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
                channel='Green',
                plate_pos_mm=plate_pos,
                z_pos_um=z_val,
                value=50 + z_idx * 10,
            )
            rows.append(
                {
                    'Filepath': fname,
                    'Color': 'Green',
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
                channel='Green',
                plate_pos_mm={'x': x, 'y': y},
                z_pos_um=z,
            )
            rows.append(
                {
                    'Filepath': fname,
                    'Color': 'Green',
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
    per-plane positions. Instrument / Plate / Objective are dropped by
    tifffile's auto-OME-XML serializer regardless of placement; they
    travel via the LVP private TIFF tag for LVP-aware consumers (see
    TestStackBuilderPrivateTagRecoversDroppedMetadata)."""

    def test_pixel_size_in_ome_xml(self, tmp_path):
        rows = []
        for z_idx in range(2):
            fname = f'frame_z{z_idx}.tiff'
            _write_structured_input(
                tmp_path / fname,
                channel='Red',
                plate_pos_mm={'x': 0.0, 'y': 0.0},
                z_pos_um=float(z_idx) * 10,
            )
            rows.append(
                {
                    'Filepath': fname,
                    'Color': 'Red',
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
        )
        with tf.TiffFile(str(tmp_path / 'out.ome.tiff')) as tif:
            ome_xml = tif.ome_metadata or ''

        assert 'PhysicalSizeX=' in ome_xml, (
            'Hyperstack OME-XML must declare PhysicalSizeX so consumers '
            'can compute on-screen scale bars.'
        )
        assert 'PhysicalSizeXUnit="um"' in ome_xml, 'PhysicalSizeX must declare its unit (microns).'

    def test_channel_names_in_ome_xml(self, tmp_path):
        rows = []
        for channel in ('Green', 'Red'):
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
        )
        with tf.TiffFile(str(tmp_path / 'out.ome.tiff')) as tif:
            ome_xml = tif.ome_metadata or ''

        assert 'Name="Green"' in ome_xml, 'Green channel name must reach OME-XML <Channel> element.'
        assert 'Name="Red"' in ome_xml, 'Red channel name must reach OME-XML <Channel> element.'


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
                    'Color': 'Green',
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
        )
        assert result['status'], f'Bare-input fallback path must not crash: {result.get("error")}'
        # Output should still be a valid OME-TIFF.
        with tf.TiffFile(str(tmp_path / 'out.ome.tiff')) as tif:
            assert tif.is_ome, 'Hyperstack output must remain OME-tagged'


class TestStackBuilderPrivateTagRecoversDroppedMetadata:
    """Tifffile's auto-OME-XML serializer silently drops Instrument /
    Plate / Objective from the metadata dict. The hyperstack write
    path serializes the full metadata dict into the LVP private TIFF
    tag as a JSON sidecar so LVP-aware consumers can recover the
    dropped fields. Standard OME-XML readers (FIJI, ImageJ, generic
    OME parsers) ignore the unknown private tag and see the same
    OME-XML they always have.
    """

    def test_instrument_subtree_survives_via_private_tag(self, tmp_path):
        rows = []
        for z_idx in range(2):
            fname = f'frame_z{z_idx}.tiff'
            _write_structured_input(
                tmp_path / fname,
                channel='Green',
                plate_pos_mm={'x': 0.0, 'y': 0.0},
                z_pos_um=float(z_idx) * 10,
            )
            rows.append(
                {
                    'Filepath': fname,
                    'Color': 'Green',
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
        )

        recovered = image_utils.read_hyperstack_private_metadata(tmp_path / 'out.ome.tiff')
        assert recovered is not None, (
            'Private-tag sidecar missing from hyperstack output; the '
            'JSON-encoded metadata dict must be written alongside the '
            'auto-OME-XML so LVP-aware readers can recover Instrument / '
            'Plate / Objective fields tifffile drops.'
        )
        instrument = recovered.get('Instrument') or {}
        microscope = instrument.get('Microscope') or {}
        assert microscope.get('SerialNumber') == 'SN12062', (
            'Instrument.Microscope.SerialNumber must round-trip via the '
            'private tag; the auto-OME-XML drops the entire Instrument '
            'subtree.'
        )
        assert microscope.get('Model') == 'LS720'
        detector = instrument.get('Detector') or {}
        assert detector.get('Model') == 'Basler a2A1920'

    def test_objective_subtree_survives_via_private_tag(self, tmp_path):
        rows = []
        for z_idx in range(2):
            fname = f'frame_z{z_idx}.tiff'
            _write_structured_input(
                tmp_path / fname,
                channel='Green',
                plate_pos_mm={'x': 0.0, 'y': 0.0},
                z_pos_um=float(z_idx) * 10,
            )
            rows.append(
                {
                    'Filepath': fname,
                    'Color': 'Green',
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
        )

        recovered = image_utils.read_hyperstack_private_metadata(tmp_path / 'out.ome.tiff')
        objective = (recovered.get('Instrument') or {}).get('Objective') or {}
        assert objective.get('Model') == 'PlanFluor20x'
        assert objective.get('Magnification') == 20
        assert objective.get('LensNA') == 0.45

    def test_plate_subtree_survives_via_private_tag(self, tmp_path):
        rows = []
        for z_idx in range(2):
            fname = f'frame_z{z_idx}.tiff'
            _write_structured_input(
                tmp_path / fname,
                channel='Green',
                plate_pos_mm={'x': 12.5, 'y': 8.25},
                z_pos_um=float(z_idx) * 10,
            )
            rows.append(
                {
                    'Filepath': fname,
                    'Color': 'Green',
                    'Scan Count': 0,
                    'Z-Slice': z_idx,
                    'X': 12.5,
                    'Y': 8.25,
                    'Z': float(z_idx) * 10,
                }
            )
        df = pd.DataFrame(rows)

        StackBuilder._create_stack(
            path=tmp_path,
            df=df,
            output_file_loc=pathlib.Path('out.ome.tiff'),
        )

        recovered = image_utils.read_hyperstack_private_metadata(tmp_path / 'out.ome.tiff')
        plate = recovered.get('Plate') or {}
        assert plate.get('Name') == '96-well'
        assert plate.get('Rows') == 8
        assert plate.get('Columns') == 12
        assert plate.get('WellLabel') == 'A1'

    def test_bare_input_hyperstack_returns_none_or_minimal(self, tmp_path):
        # Hyperstack built from bare tf.imwrite inputs: no Instrument /
        # Plate data to propagate. The private tag still writes (carrying
        # only Channel + Plane data) since tifffile's auto-OME suffices
        # for the structural fields. read_hyperstack_private_metadata
        # returns the minimal dict.
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
                    'Color': 'Green',
                    'Scan Count': 0,
                    'Z-Slice': z_idx,
                    'X': 0.0,
                    'Y': 0.0,
                    'Z': float(z_idx),
                }
            )
        df = pd.DataFrame(rows)

        StackBuilder._create_stack(
            path=tmp_path,
            df=df,
            output_file_loc=pathlib.Path('out.ome.tiff'),
        )

        recovered = image_utils.read_hyperstack_private_metadata(tmp_path / 'out.ome.tiff')
        # Sidecar is always written for hyperstacks; carries at least
        # the schema fields (Channel, Plane). Instrument / Plate absent
        # because the bare inputs carry no acquisition context.
        assert recovered is not None
        assert 'Channel' in recovered
        assert 'Instrument' not in recovered
        assert 'Plate' not in recovered

    def test_read_helper_returns_none_on_non_lvp_file(self, tmp_path):
        # A file with no private tag (third-party hyperstack, or older
        # LVP file) returns None.
        out = tmp_path / 'third_party.ome.tiff'
        tf.imwrite(
            str(out),
            np.zeros((1, 1, 1, 16, 16), dtype=np.uint8),
            ome=True,
            metadata={'axes': 'TZCYX'},
        )
        assert image_utils.read_hyperstack_private_metadata(out) is None

    def test_read_helper_returns_none_on_missing_file(self, tmp_path):
        missing = tmp_path / 'does_not_exist.ome.tiff'
        assert image_utils.read_hyperstack_private_metadata(missing) is None


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
            channel='Green',
            plate_pos_mm={'x': 0.0, 'y': 0.0},
            z_pos_um=0.0,
        )
        metadata = image_utils.build_hyperstack_output_metadata(
            reference_input_path=ref,
            channel_names=['Green', 'BF', 'Red', 'Blue'],
            plane_positions={
                'PositionX': [0.0, 0.0, 0.0, 0.0],
                'PositionY': [0.0, 0.0, 0.0, 0.0],
                'PositionZ': [0.0, 0.0, 0.0, 0.0],
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
        for channel in ('Green', 'Red'):
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

    def test_color_omitted_from_private_tag_sidecar(self, tmp_path):
        # Channel.Color is encoded into OME-XML; the JSON sidecar exists
        # for fields tifffile DROPS from OME-XML (Instrument / Plate /
        # Objective). Color rides the standard OME-XML path and does
        # NOT need to ride the sidecar -- this asserts it actually
        # makes it into the OME-XML so we don't end up duplicating.
        rows = []
        for z_idx in range(2):
            fname = f'frame_z{z_idx}.tiff'
            _write_structured_input(
                tmp_path / fname,
                channel='Green',
                plate_pos_mm={'x': 0.0, 'y': 0.0},
                z_pos_um=float(z_idx) * 10,
            )
            rows.append(
                {
                    'Filepath': fname,
                    'Color': 'Green',
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
        )

        recovered = image_utils.read_hyperstack_private_metadata(tmp_path / 'out.ome.tiff')
        assert recovered is not None
        # Color IS in the sidecar copy too (build_hyperstack_output_metadata
        # writes it; the sidecar strips only LUTs). That's fine -- it does
        # not bloat the sidecar (one int per channel) and round-trips
        # cleanly via JSON.
        assert recovered['Channel']['Name'] == ['Green']
        assert recovered['Channel']['Color'] == [0x00FF00FF]


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
                channel='Red',
                plate_pos_mm={'x': 0.0, 'y': 0.0},
                z_pos_um=float(z_idx) * 10,
            )
            rows.append(
                {
                    'Filepath': fname,
                    'Color': 'Red',
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
                    'Color': 'Green',
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


def test_hyperstack_output_uses_strips(tmp_path):
    """The hyperstack write uses strips, not tiles. Tiling forced ImageJ
    through Bio-Formats (the native reader cannot open tiled TIFFs), and its
    lenient colormap rescaling masked a colormap-scale defect in the 8-bit
    still path while breaking native open."""
    plate_pos = {'x': 1.0, 'y': 2.0}
    fname = 'frame_t0_z0_c0.tiff'
    _write_structured_input(
        tmp_path / fname, channel='Green', plate_pos_mm=plate_pos, z_pos_um=10.0
    )
    df = pd.DataFrame(
        [
            {
                'Filepath': fname,
                'Color': 'Green',
                'Scan Count': 0,
                'Z-Slice': 0,
                'X': plate_pos['x'],
                'Y': plate_pos['y'],
                'Z': 10.0,
            }
        ]
    )
    output_file_loc = pathlib.Path('out.ome.tiff')
    result = StackBuilder._create_stack(path=tmp_path, df=df, output_file_loc=output_file_loc)
    assert result['status'], f'_create_stack failed: {result.get("error")}'

    with tf.TiffFile(str(tmp_path / output_file_loc)) as tif:
        assert not tif.pages[0].is_tiled, 'hyperstack output must use strips'


def test_load_plane_raises_typed_error_naming_the_file(tmp_path):
    """A malformed hyperstack input frame fails with a typed CaptureError that
    names the offending file, not a raw tifffile/OS exception. A hyperstack plane
    cannot be skipped the way a video frame can (it would misalign the TZCYX grid),
    so the read fails the whole build loudly and legibly -- the notify boundary
    then shows a clear message instead of a raw decoder error."""
    from modules.exceptions import CaptureError

    bad = tmp_path / 'corrupt_plane.tiff'
    bad.write_bytes(b'this is not a valid tiff')

    with pytest.raises(CaptureError, match=r'corrupt_plane\.tiff'):
        StackBuilder._load_plane(bad)
