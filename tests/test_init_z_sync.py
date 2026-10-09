# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""#639 regression: app init must sync the Z slider to the actual motor position.

Without this sync, ``ui/lumaviewpro.kv`` constructs ``obj_position`` with
``value: 0.`` and never updates it before the user can interact with it.
A user click on the slider then snaps Z to 0 regardless of where the
motor really is.

Static-source regression: assert ``complete_initialization`` in
``lumaviewpro.py`` invokes ``_handle_ui_update_for_axis('Z')``. The bug
was that the call was missing, so the test must fail when the call is
missing. (Functional coverage of ``_handle_ui_update_for_axis`` itself
lives elsewhere -- that helper is exercised on every motion end in
production and any breakage is caught by the existing motion tests.)
"""

import ast

from tests.ast_seams import find_def


def test_complete_initialization_calls_z_sync():
    """Read ``lumaviewpro.py`` source and assert the init sync is wired up.

    Read from the parsed function, so its body ends where the function
    does: the bug was that the call was missing, so the test must fail when
    the call is missing.
    """
    init = find_def('lumaviewpro.py', 'complete_initialization', class_name='LumaViewProApp')
    assert init is not None, 'complete_initialization() not found in lumaviewpro.py'
    body = ast.unparse(init)

    assert "_handle_ui_update_for_axis('Z')" in body, (
        "complete_initialization() must call _handle_ui_update_for_axis('Z') "
        'to sync the objective slider with the actual motor position on app '
        'startup. Without this, the .kv hardcoded obj_position.value=0 wins '
        'and the first user click snaps Z to 0. (#639)'
    )
