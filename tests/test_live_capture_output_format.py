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
Both call sites pass ``settings['image_output_format']['live']``.

Test approach
-------------
Source-structural lock on ui/composite_capture.py: the buggy bare-dict
pattern must be absent, and the save_image calls must subscript down to
'live'. A refactor that reintroduces the bare dict fails here.
"""

from __future__ import annotations

import ast
import pathlib
import re


REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / 'ui' / 'composite_capture.py'


def test_source_parses():
    # Guards the regex assertions below against scanning a broken file.
    ast.parse(SRC.read_text())


def test_no_bare_image_output_format_dict_passed_as_format():
    text = SRC.read_text()
    # The bug: output_format set to the bare dict (no ['live'] subscript).
    bare = re.findall(
        r"output_format=settings\['image_output_format'\]\s*[,)]", text
    )
    assert bare == [], (
        'live_capture must not pass the whole image_output_format dict as '
        "output_format; it must subscript ['live']. Found bare-dict "
        f'site(s): {len(bare)}'
    )


def test_live_format_subscript_present():
    text = SRC.read_text()
    keyed = re.findall(
        r"output_format=settings\['image_output_format'\]\['live'\]", text
    )
    # Two save_image calls in live_capture (original + overlay), plus the
    # save_live_image call earlier in the same path.
    assert len(keyed) >= 2, (
        "Expected the live save_image calls to pass ['live']; found "
        f'{len(keyed)}'
    )
