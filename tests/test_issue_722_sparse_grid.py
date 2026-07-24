"""Regression: hyperstack build refuses a non-rectangular z/channel grid
instead of crashing with a cryptic tifffile IndexError.

Bug shape (pre-fix): StackBuilder._create_stack allocates a DENSE
(num_t, num_z, num_c, H, W) array sized by the unique count of each axis,
but appends per-plane OME PositionX/Y/Z metadata ONE ENTRY PER CAPTURED
FRAME. On a non-rectangular protocol -- one channel z-stacked, another
single-shot in the same well -- len(df) < num_t*num_z*num_c, so the
position lists are shorter than the dense plane count and tifffile's
OME-XML serializer runs off the end -> `IndexError: list index out of
range for attribute 'PositionX'`, aborting the whole post-processing run.

Fix: a hyperstack is a rectangular T x Z x C cube. Detect the sparse (or
duplicated) grid up front and refuse the well through the post-processor's
status=False failure path with an L1 message, so the other wells still
build. Per-well isolation itself (one failed group does not discard the
others) is the base loop's existing status=False contract, covered by
test_capture_collision_policy.py.
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
    """Conftest mocks psutil globally, leaving virtual_memory().available a
    MagicMock that cannot be compared with int. The hyperstack memory
    pre-check needs a real int; route to a generous 16 GB sentinel so the
    check passes for the small test arrays."""
    mem = MagicMock()
    mem.available = 16 * 1024 * 1024 * 1024
    monkeypatch.setattr(stack_builder_module.psutil, 'virtual_memory', lambda: mem)


def _write_frame(
    path: pathlib.Path, *, color: str = 'BF', z_um: float = 100.0, value: int = 100
) -> None:
    arr = np.full((4, 4), value, dtype=np.uint8)
    image_utils.write_tiff(
        data=arr,
        file_loc=path,
        significant_bits=8,
        save_encoding='8bit',
        metadata={
            'datetime': '2026-07-06T12:00:00',
            'plate_pos_mm': {'x': 10.0, 'y': 5.0},
            'z_pos_um': z_um,
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
            'channel': color,
            'instrument': {
                'manufacturer': 'Etaluma',
                'model': 'LS720',
                'serial_number': 'SN12062',
                'firmware_version': '4.0.0-beta14',
                'camera_model': 'Basler a2A1920',
            },
            'plate': {'name': '96-well', 'rows': 8, 'columns': 12},
            'well_label': 'A1',
        },
        ome=False,
        color=color,
    )


def _row(fname, *, color, z_idx, z_um, well='A1', scan=0):
    return {
        'Filepath': fname,
        'Color': color,
        'Scan Count': scan,
        'Z-Slice': z_idx,
        'X': 10.0,
        'Y': 5.0,
        'Z': z_um,
        'Well': well,
    }


def test_sparse_grid_refused_not_crashed(tmp_path):
    # One well: Brightfield z-stacked over 3 slices + Green single-shot.
    # Dense cube wants 1 x 3 x 2 = 6 planes; only 4 frames exist -> the
    # phantom (Green, z1/z2) cells are the pre-fix IndexError.
    rows = []
    for z_idx in range(3):
        fname = f'bf_z{z_idx}.tiff'
        _write_frame(tmp_path / fname, color='BF', value=50 + z_idx * 10)
        rows.append(_row(fname, color='BF', z_idx=z_idx, z_um=100.0 + z_idx * 10))
    _write_frame(tmp_path / 'green_z0.tiff', color='Green', value=200)
    rows.append(_row('green_z0.tiff', color='Green', z_idx=0, z_um=100.0))
    df = pd.DataFrame(rows)

    result = StackBuilder._create_stack(
        path=tmp_path,
        df=df,
        output_file_loc=pathlib.Path('out.ome.tiff'),
    )

    assert result['status'] is False, 'a non-rectangular grid must be refused, not built'
    assert 'A1' in result['error'], 'the refusal names the offending well'
    assert 'z-slices' in result['error']
    # No partial artifact left behind for the refused well.
    assert not (tmp_path / 'out.ome.tiff').exists()


def test_duplicate_cell_refused(tmp_path):
    # Two frames claim the same (scan, z, color) cell -- a duplicate would
    # silently overwrite in the dense array. len(df) == 2 but only 1 unique
    # cell for a 1 x 1 x 1 grid, so the guard must catch len != unique too.
    _write_frame(tmp_path / 'a.tiff', color='BF', value=10)
    _write_frame(tmp_path / 'b.tiff', color='BF', value=20)
    df = pd.DataFrame(
        [
            _row('a.tiff', color='BF', z_idx=0, z_um=100.0),
            _row('b.tiff', color='BF', z_idx=0, z_um=100.0),
        ]
    )

    result = StackBuilder._create_stack(
        path=tmp_path,
        df=df,
        output_file_loc=pathlib.Path('dup.ome.tiff'),
    )

    assert result['status'] is False
    assert 'A1' in result['error']


def test_rectangular_multichannel_still_builds(tmp_path):
    # Complete 1 x 2 x 2 grid (both channels captured at both z-slices):
    # the guard must NOT false-refuse a valid multi-channel z-stack.
    rows = []
    for color, base in (('BF', 10), ('Green', 100)):
        for z_idx in range(2):
            fname = f'{color}_z{z_idx}.tiff'
            _write_frame(tmp_path / fname, color=color, value=base + z_idx * 10)
            rows.append(_row(fname, color=color, z_idx=z_idx, z_um=100.0 + z_idx * 10))
    df = pd.DataFrame(rows)

    result = StackBuilder._create_stack(
        path=tmp_path,
        df=df,
        output_file_loc=pathlib.Path('good.ome.tiff'),
    )

    assert result['status'] is True, f'valid grid must build: {result.get("error")}'
    out = tmp_path / 'good.ome.tiff'
    assert out.exists()
    with tf.TiffFile(str(out)) as tif:
        ome_xml = tif.ome_metadata or ''
    # 1 T x 2 Z x 2 C = 4 planes, all real.
    assert ome_xml.count('<Plane ') == 4
