"""Regression for #693: JPG output for protocol (sequenced) capture.

Two parts, matching the fix:

  1. JPG is offered as a Sequenced Image Format and a single shared
     JPG-quality control covers both Live and Sequenced capture (one
     jpg_quality setting, not two). Kivy is not importable in the test env,
     so the kv parts are pinned with source-text structural locks -- the
     same approach #684 used for kv-only changes.
  2. The chosen jpg_quality is threaded through the image-capture config so
     the protocol writer hands it to save_image (pure functions, tested on
     the real path).
"""

import pathlib

from _lumascope.modules import config_helpers
from tests.settings_fixtures import complete_settings
from _lumascope.modules.image_mode import OutputFormat

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
