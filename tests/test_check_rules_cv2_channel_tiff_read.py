"""Tests for tools/check_rules.py cv2_channel_tiff_read.

cv2_channel_tiff_read blocks bare ``tf.imread`` / ``tifffile.imread`` in production
``modules/`` and ``ui/`` outside the canonical reader. The one sanctioned
reader is ``modules/image_utils.load_pixels``, which returns the pixels
AND their significant-bit depth together; ``image_utils.py`` is the only
file allowed to call tifffile.imread directly (load_pixels and the
legacy-collapse helper live there).

The guard prevents regression on the depth-carrying I/O boundary: a bare
tifffile.imread hands back pixels with no depth, so the caller must
remember to read the depth separately. A right-aligned 12-bit frame read
without its depth scales ~16x dark. Routing every read through
load_pixels makes the depth inseparable from the pixels by construction.

Companion to cv2_channel_io (which bans bare cv2.imread / cv2.imwrite on the
same paths); together they close both the cv2 and tifffile read sides.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.check_rules import check_source


def _violations(content: str, path: str) -> list:
    return [v for v in check_source(content, path) if v.rule == 'cv2_channel_tiff_read']


class TestRule31dBlocksBareTiffileReadInProductionPaths:
    def test_bare_tf_imread_in_modules_blocks(self):
        src = """
import tifffile as tf

def load(path):
    return tf.imread(str(path))
"""
        violations = _violations(src, 'modules/some_processor.py')
        assert len(violations) == 1
        assert violations[0].line == 5

    def test_bare_tifffile_imread_in_modules_blocks(self):
        src = """
import tifffile

def load(path):
    return tifffile.imread(path)
"""
        violations = _violations(src, 'modules/another_processor.py')
        assert len(violations) == 1
        assert violations[0].rule == 'cv2_channel_tiff_read'

    def test_bare_tf_imread_in_ui_blocks(self):
        src = """
import tifffile as tf

def load(path):
    return tf.imread(path)
"""
        violations = _violations(src, 'ui/preview.py')
        assert len(violations) == 1

    def test_multiple_bare_reads_each_fire(self):
        src = """
import tifffile as tf

def load_two(a, b):
    return tf.imread(a), tf.imread(b)
"""
        violations = _violations(src, 'modules/stitcher.py')
        assert len(violations) == 2


class TestRule31dExemptsCanonicalReader:
    def test_image_utils_py_is_exempt(self):
        # image_utils.py owns load_pixels (the depth-carrying reader) plus
        # the legacy-collapse helper it calls; tifffile.imread here is the
        # boundary implementation by construction.
        src = """
import tifffile as tf

def read_tiff_with_legacy_collapse(path):
    return tf.imread(str(path))
"""
        assert _violations(src, 'modules/image_utils.py') == []


class TestRule31dScopedToProductionPaths:
    def test_test_file_does_not_fire(self):
        # Test files routinely build synthetic tifffile fixtures.
        src = """
import tifffile as tf

def test_helper(path):
    return tf.imread(path)
"""
        assert _violations(src, 'tests/test_image_utils.py') == []

    def test_top_level_script_does_not_fire(self):
        # Files outside modules/ and ui/ are not in scope.
        src = """
import tifffile as tf

def main():
    return tf.imread('foo.tif')
"""
        assert _violations(src, 'tools/some_helper.py') == []
        assert _violations(src, 'lib/some_lib.py') == []


class TestRule31dIgnoresNonTiffileReads:
    def test_tf_imwrite_does_not_fire(self):
        # cv2_channel_tiff_read covers only the read side; tifffile.imwrite is cv2_channel_tiff_write's
        # domain (false-color-aware write routing).
        src = """
import tifffile as tf

def save(arr, path):
    tf.imwrite(path, arr)
"""
        assert _violations(src, 'modules/some_processor.py') == []

    def test_cv2_imread_does_not_fire(self):
        # cv2.imread is cv2_channel_io's domain, not cv2_channel_tiff_read's.
        src = """
import cv2

def load(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)
"""
        assert _violations(src, 'modules/some_processor.py') == []

    def test_load_pixels_callsite_does_not_fire(self):
        # The sanctioned reader call carries depth with the pixels.
        src = """
import modules.image_utils as image_utils

def load(path):
    image, significant_bits = image_utils.load_pixels(path)
    return image
"""
        assert _violations(src, 'modules/some_processor.py') == []
