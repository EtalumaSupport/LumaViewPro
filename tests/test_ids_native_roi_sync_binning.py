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
  - ``_native_roi`` / ``frame_size`` reconstruct against the synchronous UI
    binning (``settings['binning']['size']`` via ``_ui_binning_size``), never
    ``imaging.get_binning_size()``.
  - ``select_binning_size`` captures + stores the native ROI BEFORE it
    overwrites ``settings['binning']['size']``, so the reconstruction reads
    the OLD binning the current displayed value corresponds to.
  - The stored native pair is the unconditional source of truth.

The UI methods touch Kivy widgets and cannot be imported under the test
mocks (see test_issue_683_binning_roundtrip), so the production change is
pinned structurally with AST guards, and the resulting math invariant is
exercised with the pure binning functions the fixed code calls.
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

    def test_native_roi_does_not_read_async_hardware_binning(self):
        # The whole bug: reconstructing against imaging.get_binning_size()
        # (applied async via the camera executor) instead of the UI binning.
        method = _method_node('_native_roi')
        assert not _calls_named(method, 'get_binning_size'), (
            '_native_roi must NOT read imaging.get_binning_size() (the async '
            'hardware binning); reconstruct against the sync UI binning.'
        )

    def test_frame_size_does_not_read_async_hardware_binning(self):
        method = _method_node('frame_size')
        assert not _calls_named(method, 'get_binning_size'), (
            'frame_size must reconstruct native against the sync UI binning '
            '(_ui_binning_size), not imaging.get_binning_size().'
        )

    def test_native_roi_uses_ui_binning_helper(self):
        method = _method_node('_native_roi')
        assert _calls_named(method, '_ui_binning_size'), (
            '_native_roi must source the reconstruction binning from '
            'self._ui_binning_size() (the settings SSOT).'
        )

    def test_ui_binning_helper_reads_settings_binning(self):
        method = _method_node('_ui_binning_size')
        body = ast.dump(method)
        assert "'binning'" in body and "'size'" in body, (
            "_ui_binning_size must read settings['binning']['size'] (the "
            'synchronous UI binning), the SSOT the displayed value matches.'
        )

    def test_select_binning_stores_native_before_overwriting_binning(self):
        # The ordering IS the fix for the toggle path: _native_roi reconstructs
        # against settings['binning']['size'], so native must be captured while
        # that still holds the OLD binning the current displayed value matches.
        method = _method_node('select_binning_size')
        store_calls = _calls_named(method, '_store_native_roi')
        assert store_calls, 'select_binning_size must persist the native ROI'
        store_line = min(c.lineno for c in store_calls)

        binning_assign_lines = [
            node.lineno
            for node in ast.walk(method)
            if isinstance(node, ast.Assign)
            and any(ast.unparse(t) == "settings['binning']['size']" for t in node.targets)
        ]
        assert binning_assign_lines, "select_binning_size must assign settings['binning']['size']"
        assert store_line < min(binning_assign_lines), (
            'select_binning_size must capture + store the native ROI BEFORE it '
            "overwrites settings['binning']['size'], or _native_roi rebuilds "
            'native against the new binning factor (the non-square 2x bug).'
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
