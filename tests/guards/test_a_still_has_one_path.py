# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A still is captured and saved by one path, and the GUI is not on it.

The Capture button used to decide every fact about a still itself: it
imported the save functions, drew the overlays with renderers of its own,
and reached the private capture body, so a script or REST caller could not
make the same file. `session.manual_capture` now makes the still, and the
button displays the outcome. These guards keep the GUI's route to the save
and to the overlay renderers from growing back; its reach into the private
capture body is pinned by the private-reach ratchet in
test_architecture_fixes.py.
"""

import ast

from tests.ast_seams import iter_package_modules, parse_module

_OVERLAY_RENDERERS = frozenset({'transform_to_bullseye', 'add_crosshairs'})


def _ui_modules():
    return list(iter_package_modules(['ui']))


def test_the_gui_does_not_import_the_save():
    offenders = []
    for rel, tree in _ui_modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = {f'{node.module}.{a.name}' for a in node.names} | {node.module}
            elif isinstance(node, ast.Import):
                imported = {a.name for a in node.names}
            else:
                continue
            if 'modules.image_save' in imported:
                offenders.append(f'{rel}:{node.lineno}')
    assert offenders == [], (
        'the GUI saves a still through session.manual_capture, never the save '
        f'functions: {offenders}'
    )


def test_the_save_module_does_not_capture():
    """A capture-and-save helper beside save_image was a second route for a
    still that skipped the camera lane, its run fence and the member's
    in-flight guard; the save module writes frames it is handed."""
    tree = parse_module('modules/image_save.py')
    reaches = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == '_capture_and_wait_impl'
    ]
    assert reaches == [], f'modules/image_save.py reaches the capture body at lines {reaches}'


def test_the_gui_defines_no_overlay_renderer():
    offenders = [
        f'{rel}:{node.lineno} {node.name}'
        for rel, tree in _ui_modules()
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name in _OVERLAY_RENDERERS
    ]
    assert offenders == [], (
        'the overlay a saved file carries is drawn by modules/capture_overlays.py; '
        f'a second renderer in the GUI can draw a different one: {offenders}'
    )
