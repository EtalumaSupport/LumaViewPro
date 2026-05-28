"""Tests for tools/check_rules.py rule_31b.

rule_31b blocks ``add_false_color`` callsites outside the display /
encode boundary. The boundary owners are ``ui/main_display.py``
(manual record path) and ``modules/video_capture.py`` (protocol
video capture).

The guard prevents regression on the mono-native pipeline: mono
fluorescence saves carry the layer color as TIFF metadata. Widening
to RGB at a save / process layer bakes false-color into the stored
file and breaks downstream consumers that expect mono + layer
metadata. False-color belongs at the display / encode edge only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.check_rules import check_source  # noqa: E402


def _violations(content: str, path: str) -> list:
    return [v for v in check_source(content, path) if v.rule == 'rule_31b']


class TestRule31bBlocksAddFalseColorOutsideBoundary:
    def test_attribute_call_in_modules_blocks(self):
        src = '''
from modules import image_utils

def save(arr):
    rgb = image_utils.add_false_color(array=arr, color='Blue')
    return rgb
'''
        violations = _violations(src, 'modules/some_processor.py')
        assert len(violations) == 1
        assert violations[0].rule == 'rule_31b'
        assert violations[0].line == 5

    def test_bare_name_call_in_modules_blocks(self):
        src = '''
from modules.image_utils import add_false_color

def save(arr):
    return add_false_color(arr, 'Red')
'''
        violations = _violations(src, 'modules/some_processor.py')
        assert len(violations) == 1

    def test_attribute_call_in_ui_blocks(self):
        src = '''
from modules import image_utils

def render(arr):
    return image_utils.add_false_color(arr, 'Green')
'''
        violations = _violations(src, 'ui/some_widget.py')
        assert len(violations) == 1


class TestRule31bExemptsBoundaryPaths:
    def test_video_capture_py_is_exempt(self):
        # Protocol video capture applies false-color at the encode
        # boundary -- the canonical use of the helper.
        src = '''
from modules import image_utils

def capture(arr, color):
    return image_utils.add_false_color(array=arr, color=color)
'''
        assert _violations(src, 'modules/video_capture.py') == []

    def test_main_display_py_is_exempt(self):
        # Manual record path applies false-color at the encode
        # boundary -- the canonical use of the helper.
        src = '''
from modules import image_utils

def record(arr, color):
    return image_utils.add_false_color(arr, color)
'''
        assert _violations(src, 'ui/main_display.py') == []


class TestRule31bScopedToProductionPaths:
    def test_test_file_does_not_fire(self):
        # Test files exercise the helper directly via the AST scan
        # fixtures or via fixture builders.
        src = '''
from modules import image_utils

def test_helper(arr):
    return image_utils.add_false_color(arr, 'Blue')
'''
        assert _violations(src, 'tests/test_image_utils.py') == []

    def test_image_utils_def_does_not_fire(self):
        # The def itself is a FunctionDef node, not a Call. The rule
        # walks Calls, so the def passes regardless of file location.
        src = '''
def add_false_color(array, color, output=None):
    pass
'''
        assert _violations(src, 'modules/image_utils.py') == []


class TestRule31bIgnoresUnrelatedNames:
    def test_get_false_color_bufs_does_not_fire(self):
        # Sibling methods that include 'false_color' in their name but
        # are not add_false_color are out of scope.
        src = '''
def _get_false_color_bufs(arr):
    return arr

def use():
    return _get_false_color_bufs(some_array)
'''
        assert _violations(src, 'modules/protocol_image_writer.py') == []

    def test_mono_to_rgb_falsecolor_does_not_fire(self):
        # The canonical boundary helper has a different name and is
        # allowed everywhere.
        src = '''
from modules import image_utils

def render(arr):
    return image_utils.mono_to_rgb_falsecolor(arr, 'Blue')
'''
        assert _violations(src, 'modules/some_processor.py') == []
