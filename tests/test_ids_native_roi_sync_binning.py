"""Regression: native ROI must reconstruct against the SYNC UI binning.

Bench repro (IDS U3-34Lx, 2x binning): the live frame came back non-square
(1056x950) instead of the expected square 950x950. Proof it was a
reconstructed native, not a stored one: 1x delivered 1900 (a stored 2112
native would deliver 2112 at 1x).

Root cause: ``MicroscopeSettings._native_roi`` reconstructed the native ROI
as ``displayed * imaging.get_binning_size()`` -- the hardware binning, which
the camera executor applies ASYNCHRONOUSLY. Right after a binning toggle the
driver still reported the previous factor, so ``displayed * stale_binning``
rebuilt a skewed (and, with a prior off-square displayed, non-square) native.

Fix:
  - The native ROI is reconstructed against the stored binning
    (``settings['binning']['size']``), never ``imaging.get_binning_size()``.
  - A binning change captures + stores the native ROI at the OLD binning the
    current displayed value corresponds to, before the new factor is stored.
  - The stored native pair is the unconditional source of truth.

The reconstruction now lives in ``ScopeSession`` (``set_binning_size`` /
``set_frame_size``) and is exercised behaviourally in
tests/test_the_session_applies_and_stores_camera_settings.py. What stays here
pins that the GUI frame handler reads no hardware binning, and the math
invariant, exercised with the pure binning functions the fixed code calls.
"""

import ast
import pathlib

import modules.binning as binning

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MS_PATH = REPO_ROOT / 'ui' / 'microscope_settings.py'

# The IDS driver crops to the exact request, so its deliverable granularity is
# even (2x2); a 1900 native stays 1900 at 1x and 950 at 2x.
IDS_ALIGN = {'width': 2, 'height': 2}


def _method_node(name: str) -> ast.FunctionDef:
    tree = ast.parse(MS_PATH.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f'{name} not found in {MS_PATH}')


def _calls_named(method: ast.FunctionDef, attr: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(method)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr
    ]


class TestProductionSourcesSyncBinning:
    """Pin the fix in production source so it cannot silently regress."""

    def test_frame_size_does_not_read_async_hardware_binning(self):
        method = _method_node('frame_size')
        assert not _calls_named(method, 'get_binning_size'), (
            'frame_size must reconstruct native against the sync UI binning '
            '(_ui_binning_size), not imaging.get_binning_size().'
        )

    def test_ui_binning_helper_reads_settings_binning(self):
        method = _method_node('_ui_binning_size')
        body = ast.dump(method)
        assert "'binning'" in body and "'size'" in body, (
            "_ui_binning_size must read settings['binning']['size'] (the "
            'synchronous UI binning), the SSOT the displayed value matches.'
        )


class TestSyncBinningMathInvariant:
    """Exercise the math the fixed code performs with the pure functions."""

    def test_square_native_toggle_1x_to_2x_stays_square(self):
        # The fixed select_binning_size sequence on a 1900-square frame, no
        # stored native: reconstruct at the OLD UI binning (1x), then derive
        # the new displayed at 2x.
        native_max = {'width': 1900, 'height': 1900}
        old_displayed = {'width': 1900, 'height': 1900}
        old_ui_binning = binning.binning_size_str_to_int('1x1')  # 1

        native = binning.displayed_to_native(old_displayed, old_ui_binning, native_max)
        assert native == {'width': 1900, 'height': 1900}  # square, stored SSOT

        new_frame = binning.native_to_displayed(native, 2, IDS_ALIGN)
        assert new_frame == {'width': 950, 'height': 950}  # NOT 1056x950

    def test_stored_native_is_unconditional_ssot_across_binning(self):
        # A stored square native delivers consistently at every binning level,
        # independent of any displayed value -- the SSOT guarantee.
        native = {'width': 1900, 'height': 1900}
        assert binning.native_to_displayed(native, 1, IDS_ALIGN) == {
            'width': 1900,
            'height': 1900,
        }
        assert binning.native_to_displayed(native, 2, IDS_ALIGN) == {
            'width': 950,
            'height': 950,
        }

    def test_mismatched_binning_axis_reproduces_nonsquare_bug(self):
        # Documents WHY the sync-binning fix matters: reconstructing one axis
        # against a stale 2x while the displayed value was a 1x value yields the
        # non-square native (2112x1900 -> 1056x950) seen on the bench. The fix
        # eliminates this by reading a single consistent UI binning.
        skewed_native = {
            'width': binning.displayed_to_native(
                {'width': 1056, 'height': 1056}, 2, {'width': 999999, 'height': 999999}
            )['width'],  # 1056 * stale 2 = 2112
            'height': 1900,  # height reconstructed at the correct factor
        }
        assert skewed_native == {'width': 2112, 'height': 1900}
        buggy_displayed = binning.native_to_displayed(skewed_native, 2, IDS_ALIGN)
        assert buggy_displayed == {'width': 1056, 'height': 950}  # the bench bug
