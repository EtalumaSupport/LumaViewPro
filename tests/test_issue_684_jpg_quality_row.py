"""Regression for #684: the JPG-quality control in Microscope Settings.

Two reported defects:

  1. The quality slider rendered larger than every other slider in the app
     because it omitted the standard cursor / track styling that the sibling
     ModSlider controls all carry. The fix gives it the same cursor_image +
     track_width + value_track so it matches.
  2. The quality row was always present (merely disabled) when the live format
     was not JPG, wasting space in an already-crowded panel. The fix collapses
     the row's height and hides it unless the Live Image Format spinner reads
     'JPG'.

Both fixes live in lumaviewpro.kv as declarative bindings, and the previously
imperative disable-toggle was removed from microscope_settings.py so the kv is
the single source of the row's visibility. Kivy is not importable in the test
env (the widget tree cannot be rendered here), so this pins the fix with a
source-text structural lock -- the same approach the repo uses for kv-only
changes. If a future edit drops a binding or re-introduces the imperative
toggle, these assertions fail.
"""

import pathlib

# pin-justified: kv is declarative source with no headless seam; the kv
# text is the contract.
_UI = pathlib.Path(__file__).resolve().parents[1] / 'ui'
_KV = (_UI / 'lumaviewpro.kv').read_text()
_MS = (_UI / 'microscope_settings.py').read_text()


def _jpg_row_block() -> str:
    start = _KV.index('id: jpg_quality_row')
    # The row block spans its own properties plus the slider and value label;
    # a generous window covers all three without reaching the next sibling.
    return _KV[start : start + 1400]


def _select_format_handler() -> str:
    start = _MS.index('def select_live_image_output_format')
    end = _MS.index('\n    def ', start + 1)
    return _MS[start:end]


def test_row_visibility_binds_to_jpg_selection():
    block = _jpg_row_block()
    # Part 2: height + opacity gate on JPG; disabled when not JPG.
    assert "live_image_output_format_spinner.text == 'JPG'" in block
    assert "live_image_output_format_spinner.text != 'JPG'" in block
    assert 'opacity:' in block


def test_slider_matches_sibling_styling():
    block = _jpg_row_block()
    # Part 1: the styling that makes it match every other ModSlider.
    assert 'cursor_image:' in block
    assert 'track_width:' in block
    assert 'value_track:' in block


def test_handler_no_longer_toggles_disabled_imperatively():
    handler = _select_format_handler()
    # Single source: the kv owns visibility/disabled now; the handler must not
    # also poke jpg_quality_slider.disabled (a second, conflicting source).
    assert 'jpg_quality_slider' not in handler
    assert '.disabled' not in handler
    # The handler still records the chosen format.
    assert "settings['image_output_format']['live'] = fmt" in handler
