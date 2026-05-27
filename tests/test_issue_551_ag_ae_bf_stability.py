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
