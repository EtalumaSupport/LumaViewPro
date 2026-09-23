# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Regression: live_capture must pass the chosen LIVE output format to
save_image, not the whole image_output_format dict.

Bug
---
``CompositeCapture.live_capture`` saves two files (the original and an
overlay). Both save_image calls passed ``output_format=settings
['image_output_format']`` -- the dict ``{'live': ..., 'sequenced': ...}``
-- instead of ``settings['image_output_format']['live']``. Since the
dict never equals the strings "TIFF" / "OME-TIFF" / "JPG", the format
branch in save_image always fell through to the .tiff default, so the
user's Live Image Format choice was silently ignored on snapshots
(OME-TIFF was never honored; JPG could never be honored either).

Fix
---
The format is read once, from the live entry of the capture config, by the
one manual-capture path (``modules/manual_capture.py``), which the Capture
button, scripts and REST all call.

Test approach
-------------
Source-structural lock on that module: the save's output format is the
config's live format, and the bare settings dict never reaches it. The
behaviour itself -- a JPG live format writes a JPEG -- is pinned headless in
``tests/test_manual_capture_member.py``.
"""

from __future__ import annotations

import ast
import pathlib
import re


REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / 'modules' / 'manual_capture.py'


def test_source_parses():
    # Guards the regex assertions below against scanning a broken file.
    ast.parse(SRC.read_text())


def test_no_bare_image_output_format_dict_passed_as_format():
    text = SRC.read_text()
    assert "settings['image_output_format']" not in text, (
        'the manual capture must take its format from the capture config, not '
        'from the image_output_format dict'
    )


def test_live_format_is_the_output_format():
    text = SRC.read_text()
    keyed = re.findall(r'output_format=capture_config\.output_format_live', text)
    # The unmarked file and its overlay copy.
    assert len(keyed) == 2, f'expected both saves to take the live format; found {len(keyed)}'
