# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#551 regression: AG/AE BF stability -- ROI 50% centered + MinimizeGain.

Bug
---
Customer reported AutoGain/Exposure instability in BF -- the camera
controller was bouncing between gain and exposure values, producing
flickering brightness in the live preview. Two contributing factors
in drivers/pyloncamera.py::init_auto_gain_focus:

1. AutoFunction ROI was set to (Width.Max - 2*offsetX, Height.Max -
   2*offsetY) -- effectively full-frame. The controller sampled plate
   edges + dust + uneven illumination, driving oscillation.

2. AutoFunctionProfile was set to 'MinimizeExposureTime', which makes
   gain track noise; on BF where light is bright + consistent, this
   manifested as visible gain bouncing.

Fix
---
- ROI now 50%x50% centered (Width.Max/2, Height.Max/2 with centered
  offsets). 16-pixel aligned per Basler ace 2 / dart step granularity.
- AutoFunctionProfile switched to 'MinimizeGain' (pin gain low,
  adjust exposure first).

Test approach
-------------
AST + source-level structural lock on init_auto_gain_focus -- direct
exec is impractical (needs a real pypylon Camera() instance). Verifies
the two specific Basler parameter changes are present and the previous
full-frame / MinimizeExposureTime path is gone.

Bench verification gates the actual stability claim; this test is a
regression catch for the structural fix.
"""

from __future__ import annotations

import ast
import pathlib
import re


REPO = pathlib.Path(__file__).resolve().parent.parent
PYLON_SRC = REPO / 'drivers' / 'pyloncamera.py'


def _method_node(class_name: str, method_name: str) -> ast.FunctionDef:
    source = PYLON_SRC.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f'{class_name}.{method_name} not found in {PYLON_SRC}')


def test_init_auto_gain_focus_uses_minimize_gain_profile():
    """The AutoFunctionProfile SetValue call must pass 'MinimizeGain'."""
    method = _method_node('PylonCamera', 'init_auto_gain_focus')

    # Find every Call node that targets AutoFunctionProfile.SetValue and
    # inspect its first arg. AST inspection so explanatory comments
    # mentioning the old profile name don't trip a substring check.
    profile_setvalues = []
    for node in ast.walk(method):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr != 'SetValue':
                continue
            target = node.func.value
            if isinstance(target, ast.Attribute) and target.attr == 'AutoFunctionProfile':
                if node.args and isinstance(node.args[0], ast.Constant):
                    profile_setvalues.append(node.args[0].value)

    assert profile_setvalues, (
        'init_auto_gain_focus must call AutoFunctionProfile.SetValue '
        'with a string constant. (#551)'
    )
    assert all(v == 'MinimizeGain' for v in profile_setvalues), (
        f'AutoFunctionProfile.SetValue must use "MinimizeGain" for BF '
        f'stability; found {profile_setvalues}. (#551)'
    )
    assert 'MinimizeExposureTime' not in profile_setvalues, (
        'AutoFunctionProfile must NOT be set to "MinimizeExposureTime" '
        '-- that profile caused BF gain bouncing. (#551)'
    )


def test_init_auto_gain_focus_shrinks_roi_to_half():
    method = _method_node('PylonCamera', 'init_auto_gain_focus')
    src = ast.unparse(method)

    # ROI now uses Width.Max // 2 + Height.Max // 2 with centered offsets.
    # Quote-agnostic + whitespace-agnostic per source-style cluster.
    width_half = re.search(r'Width\.Max\s*//\s*2', src)
    height_half = re.search(r'Height\.Max\s*//\s*2', src)
    assert width_half is not None, (
        'AutoFunction ROI width must derive from Width.Max // 2 '
        '(centered 50% crop). (#551)'
    )
    assert height_half is not None, (
        'AutoFunction ROI height must derive from Height.Max // 2 '
        '(centered 50% crop). (#551)'
    )

    # The old full-frame ROI computation (Width.Max - 2*offsetX) must
    # be gone.
    assert not re.search(r'Width\.Max\s*-\s*2\s*\*\s*self\.active\.AutoFunctionROIOffsetX', src), (
        'init_auto_gain_focus must not use the old full-frame ROI '
        'derivation (Width.Max - 2*offsetX) which caused BF '
        'oscillation. (#551)'
    )


def test_init_auto_gain_focus_aligns_to_16():
    """Basler ace 2 / dart ROI step granularity is 16 px -- the new
    ROI computation must align down to 16."""
    method = _method_node('PylonCamera', 'init_auto_gain_focus')
    src = ast.unparse(method)
    assert '_align_down' in src or '// 16' in src or '* 16' in src, (
        'New ROI computation must align to the 16-px granularity Basler '
        'requires (use _align_down or explicit // 16 / * 16). (#551)'
    )


def test_init_auto_gain_focus_zeroes_offset_before_sizing():
    """Pylon node interdependency: AutoFunctionROIOffsetX/Y.Max equals
    (sensor bound) - (current AutoFunctionROIWidth/Height). A non-zero
    offset caps the achievable Width/Height; the centered-offset setpoint
    computed against sensor Max is rejected if OffsetX.Max is still
    constrained by the previous Width.

    The dart daA3840-45um reports AutoFunctionROIOffsetX.Max = 20 by
    default while ace 2 reports the full sensor extent; a centered
    offset of ~960 (half of (Width.Max - roi_width)) raises
    OutOfRangeException on the dart unless offsets are zeroed first.

    Regression test: AutoFunctionROIOffsetX/Y(0) must appear in the
    source BEFORE the centered SetValue on the same nodes, AND before
    AutoFunctionROIWidth/Height get set.
    """
    method = _method_node('PylonCamera', 'init_auto_gain_focus')
    src = ast.unparse(method)

    offset_zero_x = src.find('AutoFunctionROIOffsetX.SetValue(0)')
    offset_zero_y = src.find('AutoFunctionROIOffsetY.SetValue(0)')
    width_set = src.find('AutoFunctionROIWidth.SetValue(')
    height_set = src.find('AutoFunctionROIHeight.SetValue(')

    assert offset_zero_x != -1, (
        'init_auto_gain_focus must call AutoFunctionROIOffsetX.SetValue(0) '
        'before setting Width / Height -- non-zero offset caps the '
        'achievable Width and rejects the post-sizing centered offset '
        'on smaller-sensor cameras (dart daA3840-45um).'
    )
    assert offset_zero_y != -1, (
        'init_auto_gain_focus must call AutoFunctionROIOffsetY.SetValue(0) '
        'before setting Width / Height -- same constraint as X axis.'
    )
    assert width_set != -1 and height_set != -1, (
        'init_auto_gain_focus must set AutoFunctionROIWidth + '
        'AutoFunctionROIHeight (the 50% centered ROI per #551).'
    )
    assert offset_zero_x < width_set, (
        'Offset-zero must precede Width-set. Setting Width first while '
        'an existing OffsetX exceeds the post-sizing OffsetX.Max raises '
        'OutOfRangeException on the dart family.'
    )
    assert offset_zero_y < height_set, (
        'Offset-zero must precede Height-set. Same reasoning as X.'
    )


def test_init_auto_gain_focus_clamps_to_autofunction_roi_max():
    """Defensive clamp against the AutoFunctionROI* node's own Max --
    some cameras (dart family) report tighter bounds on these nodes
    than on the sensor's Width / Height proper. Without the clamp, a
    50% centered crop derived from Width.Max can exceed
    AutoFunctionROIWidth.Max and raise OutOfRangeException."""
    method = _method_node('PylonCamera', 'init_auto_gain_focus')
    src = ast.unparse(method)

    # The clamp uses min(...) against the per-node Max.
    assert re.search(r'AutoFunctionROIWidth\.Max', src), (
        'init_auto_gain_focus must read AutoFunctionROIWidth.Max for '
        'defensive clamp (dart family reports tighter bounds here than '
        'on sensor Width.Max).'
    )
    assert re.search(r'AutoFunctionROIHeight\.Max', src), (
        'init_auto_gain_focus must read AutoFunctionROIHeight.Max for '
        'defensive clamp.'
    )
    assert re.search(r'AutoFunctionROIOffsetX\.Max', src), (
        'init_auto_gain_focus must read AutoFunctionROIOffsetX.Max for '
        'defensive clamp on the centered offset.'
    )
    assert re.search(r'AutoFunctionROIOffsetY\.Max', src), (
        'init_auto_gain_focus must read AutoFunctionROIOffsetY.Max for '
        'defensive clamp.'
    )
