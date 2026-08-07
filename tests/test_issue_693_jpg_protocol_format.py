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

from modules import config_helpers
from modules.composite_generation import CompositeGeneration
from modules.protocol_runner import ProtocolRunner

# pin-justified: kv is declarative source with no headless seam; the kv
# text is the contract.
_UI = pathlib.Path(__file__).resolve().parents[1] / 'ui'
_KV = (_UI / 'lumaviewpro.kv').read_text()


def _jpg_row_block() -> str:
    start = _KV.index('id: jpg_quality_row')
    return _KV[start : start + 1600]


def _sequenced_spinner_block() -> str:
    start = _KV.index('id: sequenced_image_output_format_spinner')
    return _KV[start : start + 400]


# --- Part 1: UI offers JPG for sequenced + one shared quality control -------


def test_sequenced_format_offers_jpg():
    block = _sequenced_spinner_block()
    assert "'JPG'" in block
    assert 'values:' in block


def test_quality_row_shared_across_live_and_sequenced():
    block = _jpg_row_block()
    # The single quality row now reacts to EITHER format being JPG, so one
    # control serves manual + sequenced (no second slider).
    assert "live_image_output_format_spinner.text == 'JPG'" in block
    assert "sequenced_image_output_format_spinner.text == 'JPG'" in block


# --- Part 2: jpg_quality threads through the capture config ------------------


def test_build_image_capture_config_carries_jpg_quality_default():
    runner = ProtocolRunner.__new__(ProtocolRunner)
    cfg = runner.build_image_capture_config(image_mode='8bit', sequenced_format='JPG')
    assert cfg.output_format_sequenced == 'JPG'
    assert cfg.jpg_quality == 90


def test_build_image_capture_config_carries_custom_jpg_quality():
    runner = ProtocolRunner.__new__(ProtocolRunner)
    cfg = runner.build_image_capture_config(
        image_mode='8bit', sequenced_format='JPG', jpg_quality=55
    )
    assert cfg.jpg_quality == 55


def test_settings_config_reads_jpg_quality():
    cfg = config_helpers.get_image_capture_config_from_settings(
        {'image_output_format': {'live': 'JPG', 'sequenced': 'JPG'}, 'jpg_quality': 40}
    )
    assert cfg.jpg_quality == 40


def test_settings_config_defaults_jpg_quality_when_absent():
    cfg = config_helpers.get_image_capture_config_from_settings({})
    assert cfg.jpg_quality == 90


# --- Part 3: composite / stitch guard against JPG source ---------------------


class _RecordStub:
    def complete(self):
        pass

    def file_exists_in_records(self, filepath):
        return False


def _make_composite():
    return CompositeGeneration(has_turret=False)


def _stub_helper(post_processor, df, record=None):
    def _fake_load_folder(path, tiling_configs_file_loc):
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
    df = pd.DataFrame({'Filepath': ['A1_Green_0000.jpg', 'A1_Red_0000.jpg']})
    # No record stub needed: the JPG guard returns before the record is used.
    _stub_helper(comp, df)
    result = comp.load_folder(path='run', tiling_configs_file_loc=pathlib.Path('tiling.json'))
    assert result['status'] is False
    assert result['reason'] == 'unsupported_source_format'
    assert 'JPG' in result['message']
    assert 'TIFF' in result['message']
    assert 'A1_Green_0000.jpg' in result['message']


def test_mixed_tiff_and_jpg_rejects_the_first_unsupported_source():
    comp = _make_composite()
    df = pd.DataFrame({'Filepath': ['A1_Green_0000.tiff', 'A1_Red_0000.jpg', 'A1_Blue_0000.jpg']})
    _stub_helper(comp, df)

    result = comp.load_folder(path='run', tiling_configs_file_loc=pathlib.Path('tiling.json'))

    assert result['status'] is False
    assert result['reason'] == 'unsupported_source_format'
    assert 'A1_Red_0000.jpg' in result['message']
    assert 'A1_Blue_0000.jpg' not in result['message']


def test_composite_does_not_trip_guard_for_tiff_source():
    comp = _make_composite()
    df = pd.DataFrame({'Filepath': ['A1_Green_0000.tiff', 'A1_Red_0000.tiff']})
    _stub_helper(comp, df, record=_RecordStub())
    # Isolate the format guard from downstream grouping: a TIFF scan must get
    # PAST the guard. Stub the grouping so the test asserts only that the
    # JPG-source rejection is not returned.
    comp._filter_ignored_types = lambda df: df
    comp._get_groups = lambda df: []
    result = comp.load_folder(path='run', tiling_configs_file_loc=pathlib.Path('tiling.json'))
    assert 'saved as JPG' not in result.get('message', '')
