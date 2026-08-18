"""Tests for tools/check_rules.py cv2_channel_io.

cv2_channel_io blocks bare ``cv2.imread`` / ``cv2.imwrite`` / ``cv2.VideoWriter``
in production ``modules/`` and ``ui/`` outside the canonical owner
files. The canonical owners are ``modules/image_utils.py`` (which
holds the L1 file loader plus the capability-flag wrappers) and
``modules/video_writer.py`` (which holds the cv2 XVID fallback for the
canonical VideoWriter class).

The guard prevents regression on the mono-native pipeline: cv2 is
BGR-native, so a bare cv2 read or write outside the canonical wrappers
swaps channels at the file boundary and silently corrupts color order.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.check_rules import check_source


def _violations(content: str, path: str) -> list:
    return [v for v in check_source(content, path) if v.rule == 'cv2_channel_io']


class TestRule31aBlocksBareCv2InProductionPaths:
    def test_bare_cv2_imread_in_modules_blocks(self):
        src = """
import cv2

def load(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)
"""
        violations = _violations(src, 'modules/some_processor.py')
        assert len(violations) == 1
        assert violations[0].line == 5

    def test_bare_cv2_imwrite_in_modules_blocks(self):
        src = """
import cv2

def save(arr, path):
    cv2.imwrite(path, arr)
"""
        violations = _violations(src, 'modules/another_processor.py')
        assert len(violations) == 1
        assert violations[0].rule == 'cv2_channel_io'

    def test_bare_cv2_VideoWriter_in_modules_blocks(self):
        src = """
import cv2

def make_writer():
    return cv2.VideoWriter('out.mp4', 0, 30, (640, 480))
"""
        violations = _violations(src, 'modules/some_writer.py')
        assert len(violations) == 1

    def test_bare_cv2_imread_in_ui_blocks(self):
        src = """
import cv2

def load(path):
    return cv2.imread(path)
"""
        violations = _violations(src, 'ui/preview.py')
        assert len(violations) == 1


class TestRule31aExemptsCanonicalOwners:
    def test_image_utils_py_is_exempt(self):
        # image_utils.py owns image_file_to_image (the multi-format L1
        # reader) plus the capability-flag wrappers; cv2 use here is
        # boundary-correct by construction.
        src = """
import cv2

def image_file_to_image(image_file):
    return cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
"""
        assert _violations(src, 'modules/image_utils.py') == []

    def test_video_writer_py_is_exempt(self):
        # video_writer.py owns the cv2.VideoWriter XVID fallback.
        src = """
import cv2

class VideoWriter:
    def _init_cv2(self, width, height):
        return cv2.VideoWriter('out.avi', 0, 30, (width, height))
"""
        assert _violations(src, 'modules/video_writer.py') == []


class TestRule31aScopedToProductionPaths:
    def test_test_file_does_not_fire(self):
        # Test files routinely build synthetic cv2 fixtures.
        src = """
import cv2

def test_helper(path):
    return cv2.imread(path)
"""
        assert _violations(src, 'tests/test_image_utils.py') == []

    def test_top_level_script_does_not_fire(self):
        # Files outside modules/ and ui/ are not in scope -- e.g.
        # ad-hoc scripts at the repo root, tools/ helpers, lib/ files.
        src = """
import cv2

def main():
    return cv2.imread('foo.png')
"""
        assert _violations(src, 'tools/some_helper.py') == []
        assert _violations(src, 'lib/some_lib.py') == []


class TestRule31aIgnoresNonCv2Calls:
    def test_cv2_cvtColor_does_not_fire(self):
        # cv2_channel_io covers only imread / imwrite / VideoWriter --
        # cv2.cvtColor and other cv2 functions are out of scope (LAB
        # color transfer in stitch_algorithms uses cvtColor + split +
        # merge legitimately).
        src = """
import cv2

def to_gray(arr):
    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
"""
        assert _violations(src, 'modules/some_processor.py') == []

    def test_imread_from_other_module_does_not_fire(self):
        # tifffile.imread or some_module.imread are not bare cv2 calls.
        src = """
import tifffile

def load(path):
    return tifffile.imread(path)
"""
        assert _violations(src, 'modules/some_processor.py') == []
