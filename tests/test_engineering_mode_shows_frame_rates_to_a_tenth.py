# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Engineering mode shows the frame rates to a tenth; normal mode, whole.

The status line (the window title, then) showed `Capture: 4 | Display: 4 FPS`, which hides the
difference a bench is looking for (4.3 against 4.5 on a Classic camera).
Engineering mode now shows one decimal; normal mode stays whole numbers.
"""

import ast

from tests.ast_seams import find_def
from ui.shader import frame_rate_title
from ui.ui_helpers import FIGURE_SPACE as FS


def test_engineering_mode_shows_tenths():
    # Each rate is padded to the width of 99 so the status line holds still.
    assert (
        frame_rate_title(4.26, 4.31, engineering=True) == f'Capture: {FS}4.3 | Display: {FS}4.3 FPS'
    )
    assert frame_rate_title(29.7, 30.0, engineering=True) == 'Capture: 29.7 | Display: 30.0 FPS'


def test_normal_mode_shows_whole_numbers():
    assert frame_rate_title(4.26, 4.31, engineering=False) == f'Capture: {FS}4 | Display: {FS}4 FPS'
    assert frame_rate_title(29.7, 30.0, engineering=False) == 'Capture: 30 | Display: 30 FPS'


def test_the_title_passes_the_engineering_mode_it_runs_in():
    update = find_def('ui/shader.py', '_update_status_bar', class_name='ShaderViewer')
    calls = [
        node
        for node in ast.walk(update)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'frame_rate_title'
    ]
    assert len(calls) == 1
    engineering = next(kw.value for kw in calls[0].keywords if kw.arg == 'engineering')
    assert ast.unparse(engineering) == 'ctx.session.engineering_mode'
