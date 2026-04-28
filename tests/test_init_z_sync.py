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
lives elsewhere — that helper is exercised on every motion end in
production and any breakage is caught by the existing motion tests.)
"""
from pathlib import Path


def test_complete_initialization_calls_z_sync():
    """Read ``lumaviewpro.py`` source and assert the init sync is wired up.

    A static-source test is brittle but precisely targets the regression:
    the bug was that the call was missing, so the test must fail when the
    call is missing.
    """
    repo_root = Path(__file__).resolve().parents[1]
    src = (repo_root / 'lumaviewpro.py').read_text()

    start = src.find('def complete_initialization')
    assert start != -1, "complete_initialization() not found in lumaviewpro.py"
    end = src.find('\n        Clock.schedule_once(complete_initialization', start)
    assert end != -1, "complete_initialization() body boundary not found"
    body = src[start:end]

    assert "_handle_ui_update_for_axis('Z')" in body, (
        "complete_initialization() must call _handle_ui_update_for_axis('Z') "
        "to sync the objective slider with the actual motor position on app "
        "startup. Without this, the .kv hardcoded obj_position.value=0 wins "
        "and the first user click snaps Z to 0. (#639)"
    )
