# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#830 regression: a spinner's selected text must fit inside the spinner.

The base ``<Spinner>:`` rule paints the dropdown arrow into the widget's own
right edge (a Triangle from ``self.right - dp(20)`` to ``self.right -
dp(10)``). Kivy draws a Label's texture at its full ``texture_size`` centred on
the widget center, in a plain canvas with no stencil, so a string wider than
the box spills past BOTH edges and over the arrow instead of clipping. Every
spinner in the app inherits that rule, so the guard belongs on the rule rather
than on any one option string.

Shortening the option strings could not have been the fix: most spinners
persist or compare their displayed text, and four value-sets are open-ended
(the graphing spinners take arbitrary CSV column headers; objectives, labware
and scopes load from user-writable JSON).

Four properties are load-bearing together, each verified by measurement
against Kivy 2.3.1 rather than by reading:

- ``text_size`` alone WRAPS the string onto a second line inside a
  single-line-high widget, which is worse than the overflow it replaces.
- ``shorten`` only applies when ``text_size[0]`` is set (Kivy's own contract,
  ``kivy/uix/label.py``), so neither works without the other.
- ``halign: 'center'`` is REQUIRED once ``text_size`` is set. Today's centring
  is an accident of ``texture_size`` equalling the glyph width; once a wrap
  width is given the texture becomes that wide and Kivy's default ``halign:
  'auto'`` renders flush left, which would visibly shift every short-text
  spinner ('TIFF', 'mp4', '1x1').
- The reserve must be at least twice the arrow gutter. The texture is centred,
  so a reserve of N leaves only N/2 of clearance on the right, and the arrow
  begins dp(20) in from the right edge.

This reads ``ui/lumaviewpro.kv`` as text because a kv rule has no AST seam to
assert against; the source-pin ratchet names exactly this case.
"""

from __future__ import annotations

import pathlib
import re

from modules import image_mode

REPO = pathlib.Path(__file__).resolve().parents[1]
KV = REPO / 'ui' / 'lumaviewpro.kv'

# The arrow occupies the rightmost dp(20) of the widget, so the text needs
# dp(20) of clearance on the right; the texture is centred, so the total
# reserve must be at least twice that.
_ARROW_GUTTER_DP = 20
_MIN_RESERVE_DP = 2 * _ARROW_GUTTER_DP


def _spinner_rule_body() -> str:
    """The body of the top-level ``<Spinner>:`` rule, without nested rules."""
    lines = KV.read_text(encoding='utf-8').splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.strip() == '<Spinner>:' and not line[:1].isspace()
    )
    body = []
    for line in lines[start + 1 :]:
        if line.strip() and not line[:1].isspace():
            break
        body.append(line)
    return '\n'.join(body)


def test_spinner_rule_constrains_text_width():
    """The rule gives the text a wrap width narrower than the widget."""
    body = _spinner_rule_body()
    assert 'text_size:' in body, (
        'the <Spinner> rule sets no text_size, so a long selection renders at '
        'full glyph width and spills outside the widget and over the arrow'
    )
    match = re.search(r'text_size:\s*self\.width\s*-\s*dp\((\d+)\)', body)
    assert match, (
        'text_size must reduce self.width by an explicit dp() reserve so the '
        f'text clears the dp({_ARROW_GUTTER_DP}) arrow gutter; found: '
        + repr(re.search(r'text_size:.*', body))
    )
    reserve = int(match.group(1))
    assert reserve >= _MIN_RESERVE_DP, (
        f'a reserve of dp({reserve}) leaves only dp({reserve // 2}) of clearance '
        f'on each side because the texture is centred, but the arrow starts '
        f'dp({_ARROW_GUTTER_DP}) in from the right edge'
    )


def test_spinner_rule_shortens_rather_than_wrapping():
    """shorten is mandatory: text_size alone wraps onto a second line."""
    body = _spinner_rule_body()
    assert re.search(r'shorten:\s*True', body), (
        'text_size without shorten wraps a long selection onto two lines '
        'inside a single-line-high spinner, which is worse than the overflow'
    )


def test_spinner_rule_keeps_text_centred():
    """halign is mandatory once text_size is set, or every spinner shifts left."""
    body = _spinner_rule_body()
    assert re.search(r"halign:\s*'center'", body), (
        "once text_size is set the texture is that wide and Kivy's default "
        "halign 'auto' renders flush left, shifting every short-text spinner"
    )


def test_twelve_bit_labels_keep_the_substring_the_kv_tests_for():
    """The JPG depth warning is driven by a substring test on the label.

    ``ui/lumaviewpro.kv`` gates the 'JPG saves 8-bit' warning row's height and
    opacity on ``'12-bit' in image_mode_spinner.text``. A 12-bit label that
    drops that substring silently stops warning the user that JPG discards
    their 12-bit capture, with nothing else failing.

    This guard protects live behaviour; it does not endorse the coupling. A
    substring test on a DISPLAY string is the GUI deciding something the API
    should own, and the right end state is a predicate exposed by the API that
    the kv merely renders. When that lands, this test should be REPLACED by one
    asserting the predicate -- not deleted, and not used as a reason to keep the
    substring test.
    """
    twelve_bit_modes = [
        image_mode.IMAGE_MODE_12BIT_SCIENTIFIC,
        image_mode.IMAGE_MODE_12BIT_SCALED,
        image_mode.IMAGE_MODE_12BIT_FALSE_COLOR_RGB,
    ]
    for mode in twelve_bit_modes:
        label = image_mode.IMAGE_MODE_LABELS[mode]
        assert '12-bit' in label, (
            f'{mode!r} has label {label!r}, which the kv depth-warning test '
            f"('12-bit' in image_mode_spinner.text) would not match"
        )
