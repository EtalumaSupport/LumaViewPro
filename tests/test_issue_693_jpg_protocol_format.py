"""Regression for #693: JPG output for protocol (sequenced) capture.

Three parts, matching the fix:

  1. JPG is offered as a Sequenced Image Format and a single shared
     JPG-quality control covers both Live and Sequenced capture (one
     jpg_quality setting, not two). Kivy is not importable in the test env,
     so the kv parts are pinned with source-text structural locks -- the
     same approach #684 used for kv-only changes.
  2. The chosen jpg_quality is threaded through the image-capture config so
     the protocol writer hands it to save_image (pure functions, tested on
     the real path).
  3. Composite / stitch / z-projection re-read source frames via tifffile,
     which cannot decode JPG. The shared post-processor stops with a clear
     message for a JPG-only scan instead of letting tifffile raise partway
     through a group.
"""

import pathlib

import pandas as pd
import pytest

from modules import config_helpers
from tests.settings_fixtures import complete_settings
from modules.composite_generation import CompositeGeneration
from modules.exceptions import PostProcessingRefusedError
from modules.image_mode import OutputFormat
from tests.stock_channels import LED0, LED1, LED2
from tests.installation_fixtures import shipped_grids

# pin-justified: kv is declarative source with no headless seam; the kv
# text is the contract.
_UI = pathlib.Path(__file__).resolve().parents[1] / 'ui'
_KV = (_UI / 'lumaviewpro.kv').read_text()


def _jpg_row_block() -> str:
    start = _KV.index('id: jpg_quality_row')
    return _KV[start : start + 1600]


# --- Part 1: UI offers JPG for sequenced + one shared quality control -------


def test_sequenced_format_offers_jpg():
    # The sequenced spinner's values are every label of the panel's table,
    # built at the load rather than repeated in the kv.
    from ui.microscope_settings import OUTPUT_FORMAT_LABELS

    assert OUTPUT_FORMAT_LABELS[OutputFormat.JPG] == 'JPG'


def test_quality_row_shared_across_live_and_sequenced():
    block = _jpg_row_block()
    # The single quality row now reacts to EITHER format being JPG, so one
    # control serves manual + sequenced (no second slider).
    assert "live_image_output_format_spinner.text == 'JPG'" in block
    assert "sequenced_image_output_format_spinner.text == 'JPG'" in block


# --- Part 2: jpg_quality threads through the capture config ------------------


def test_settings_config_reads_jpg_quality():
    cfg = config_helpers.get_image_capture_config_from_settings(
        complete_settings(image_output_format={'live': 'JPG', 'sequenced': 'JPG'}, jpg_quality=40)
    )
    assert cfg.jpg_quality == 40


# --- Part 3: composite / stitch guard against JPG source ---------------------


class _RecordStub:
    def complete(self):
        pass

    def file_exists_in_records(self, filepath):
        return False


def _make_composite():
    return CompositeGeneration(has_turret=False)


def _stub_helper(post_processor, df, record=None):
    def _fake_load_folder(path, tiling_grids):
        return {
            'status': True,
            'images_df': df,
            'root_path': pathlib.Path('.'),
            'protocol_post_record': record,
            'protocol': None,
        }

    post_processor._post_processing_helper.load_folder = _fake_load_folder


def test_composite_rejects_jpg_source_with_clear_message():
    comp = _make_composite()
    df = pd.DataFrame({'Filepath': [f'A1_{LED1}_0000.jpg', f'A1_{LED2}_0000.jpg']})
    # No record stub needed: the JPG guard returns before the record is used.
    _stub_helper(comp, df)
    with pytest.raises(PostProcessingRefusedError) as refused:
        comp.load_folder(path='run', tiling_grids=shipped_grids())
    assert refused.value.reason == 'unsupported_source_format'
    assert 'JPG' in str(refused.value)
    assert 'TIFF' in str(refused.value)
    assert f'A1_{LED1}_0000.jpg' in str(refused.value)


def test_mixed_tiff_and_jpg_rejects_the_first_unsupported_source():
    comp = _make_composite()
    df = pd.DataFrame(
        {'Filepath': [f'A1_{LED1}_0000.tiff', f'A1_{LED2}_0000.jpg', f'A1_{LED0}_0000.jpg']}
    )
    _stub_helper(comp, df)

    with pytest.raises(PostProcessingRefusedError) as refused:
        comp.load_folder(path='run', tiling_grids=shipped_grids())

    assert refused.value.reason == 'unsupported_source_format'
    assert f'A1_{LED2}_0000.jpg' in str(refused.value)
    assert f'A1_{LED0}_0000.jpg' not in str(refused.value)


def test_composite_does_not_trip_guard_for_tiff_source():
    comp = _make_composite()
    df = pd.DataFrame({'Filepath': [f'A1_{LED1}_0000.tiff', f'A1_{LED2}_0000.tiff']})
    _stub_helper(comp, df, record=_RecordStub())
    # Isolate the format guard from downstream grouping: a TIFF scan must get
    # PAST the guard. Stub the grouping so the test asserts only that the
    # JPG-source rejection is not the answer: with no groups, the folder is
    # refused for having nothing to combine instead.
    comp._filter_ignored_types = lambda df: df
    comp._get_groups = lambda df: []
    with pytest.raises(PostProcessingRefusedError) as refused:
        comp.load_folder(path='run', tiling_grids=shipped_grids())
    assert refused.value.reason != 'unsupported_source_format'
