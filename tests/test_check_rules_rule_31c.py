"""Tests for tools/check_rules.py rule_31c.

rule_31c blocks bare ``tifffile.imwrite`` (aliased ``tf`` or
``tifffile``) calls in post-processor modules unless the same function
also calls one of the false-color helpers
(``image_utils.maybe_apply_false_color`` or ``image_utils.write_tiff``).

The guard prevents regression on the freshly-fixed sites in
``modules/zprojector.py`` and ``modules/stitcher.py`` (#669, #678).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.check_rules import check_source


def _violations(content: str, path: str) -> list:
    return [v for v in check_source(content, path) if v.rule == 'rule_31c']


class TestRule31cBlocksBareImwriteWithoutHelper:
    def test_bare_tf_imwrite_in_zprojector_with_no_helper_blocks(self):
        src = """
import tifffile as tf

def _zproject(self, path, df):
    tf.imwrite("out.tiff", data=some_array)
"""
        violations = _violations(src, 'modules/zprojector.py')
        assert len(violations) == 1
        assert violations[0].rule == 'rule_31c'
        assert violations[0].line == 5

    def test_bare_tifffile_imwrite_in_stitcher_with_no_helper_blocks(self):
        src = """
import tifffile

def _stitch(path):
    tifffile.imwrite(path / "out.tiff", arr)
"""
        violations = _violations(src, 'modules/stitcher.py')
        assert len(violations) == 1
        assert violations[0].rule == 'rule_31c'


class TestRule31cAllowsBareImwriteWithPairedHelper:
    def test_zprojector_with_maybe_apply_false_color_passes(self):
        src = """
import tifffile as tf
from modules import image_utils

def _zproject(self, path, df):
    data = image_utils.maybe_apply_false_color(data=arr, color=df['Color'].iloc[0])
    tf.imwrite("out.tiff", data=data)
"""
        assert _violations(src, 'modules/zprojector.py') == []

    def test_stitcher_with_write_tiff_passes(self):
        # If a future refactor routes the whole save through write_tiff
        # (no separate tf.imwrite), no bare-imwrite to flag at all --
        # but the regex still passes because write_tiff is in the
        # helper allowlist.
        src = """
from modules import image_utils

def _stitch(path):
    image_utils.write_tiff(data=arr, file_loc=path, ...)
"""
        assert _violations(src, 'modules/stitcher.py') == []


class TestRule31cScopedToPostProcessorPaths:
    def test_bare_imwrite_in_non_postprocessor_module_does_not_fire(self):
        # protocol_image_writer is NOT in the post-processor scope yet
        # (it is the canonical capture-side save path and already uses
        # write_tiff for fluorescence). A bare tf.imwrite somewhere in
        # protocol_image_writer is not in scope for rule_31c.
        src = """
import tifffile as tf

def helper(path, arr):
    tf.imwrite(path, arr)
"""
        assert _violations(src, 'modules/protocol_image_writer.py') == []

    def test_bare_imwrite_in_test_file_does_not_fire(self):
        # Test files build synthetic TIFFs via tf.imwrite all the time.
        src = """
import tifffile as tf

def _write_mono(p, value):
    tf.imwrite(str(p), value)
"""
        assert _violations(src, 'tests/test_zprojector.py') == []


class TestRule31cFunctionScopingHonored:
    def test_helper_in_one_function_does_not_satisfy_sibling_function(self):
        # Function A has the helper. Function B has a bare imwrite with
        # no helper. The pairing is per-function -- B must fire.
        src = """
import tifffile as tf
from modules import image_utils

def function_a(arr, color):
    return image_utils.maybe_apply_false_color(data=arr, color=color)

def function_b(arr, path):
    tf.imwrite(str(path), arr)
"""
        violations = _violations(src, 'modules/zprojector.py')
        assert len(violations) == 1
        assert violations[0].line == 9


class TestRule31cCompositeGenerationCovered:
    def test_bare_tf_imwrite_in_composite_generation_blocks(self):
        # composite_generation.py joined the post-processor write set
        # when the mono-native pipeline migration routed its outputs
        # through image_utils.write_tiff. A regression that reintroduces
        # a bare tifffile.imwrite there silently widens or strips the
        # false-color channel; the rule catches it.
        src = """
import tifffile as tf

def _build_composite(arr, path):
    tf.imwrite(str(path), arr)
"""
        violations = _violations(src, 'modules/composite_generation.py')
        assert len(violations) == 1
        assert violations[0].rule == 'rule_31c'

    def test_composite_with_write_tiff_helper_passes(self):
        src = """
from modules import image_utils

def _build_composite(arr, path):
    image_utils.write_tiff(data=arr, file_loc=path)
"""
        assert _violations(src, 'modules/composite_generation.py') == []

    def test_stack_builder_bare_tf_imwrite_blocks(self):
        # stack_builder joined the post-processor write path set when
        # the hyperstack write was routed through image_utils.write_tiff
        # (via the hyperstack_metadata override hook). A regression that
        # reintroduces a bare tifffile.imwrite in stack_builder bypasses
        # the canonical save path; the rule catches it.
        src = """
import tifffile as tf

def _save(arr, path):
    tf.imwrite(str(path), arr, ome=True)
"""
        violations = _violations(src, 'modules/stack_builder.py')
        assert len(violations) == 1
        assert violations[0].rule == 'rule_31c'

    def test_stack_builder_with_write_tiff_helper_passes(self):
        # The new canonical shape: stack_builder calls write_tiff with
        # the hyperstack_metadata override. The helper presence in the
        # same function satisfies the pairing requirement even if a
        # bare imwrite slipped in (belt-and-suspenders).
        src = """
from modules import image_utils

def _create_stack(arr, path):
    image_utils.write_tiff(
        data=arr,
        file_loc=path,
        metadata={},
        ome=True,
        color='',
        hyperstack_metadata={'axes': 'TZCYX'},
    )
"""
        assert _violations(src, 'modules/stack_builder.py') == []
