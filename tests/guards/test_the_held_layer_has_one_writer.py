# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every camera write either keeps the camera's held-layer record true or is layer-neutral.

``CameraAPI`` records which layer's stored settings the camera holds, and a
capture is named and set by it. A write that changed the camera without
touching the record would leave the record naming a layer whose settings
the camera no longer holds. So every reference to a camera write body is
one of: a public door that hands it to ``_no_layers_write`` (the record
then says no layer), the record's own writers (``_hold_layer_impl``, the
restore), or a write that leaves the layer as it was (the in-capture lock
and resume of a standing arm, the arm's own ceiling clamp, the batched
apply's own steps). No module outside ``camera.py`` reaches a write body but
the capture's (``imaging.py``), whose lock and resume of a standing arm are
the in-capture ones above.
"""

from __future__ import annotations

import ast

from tests.ast_seams import REPO_ROOT, MemberNames

_WRITE_BODIES = MemberNames(
    'modules/lumascope_api/camera.py',
    {
        '_set_gain_db_impl',
        '_set_exposure_ms_impl',
        '_set_auto_gain_impl',
        '_set_auto_exposure_impl',
        '_update_auto_gain_target_brightness_impl',
        '_auto_gain_once_impl',
        '_lock_auto_gain_impl',
        '_apply_settings_impl',
        '_restore_camera_state_impl',
    },
)

# Where a write body may be called directly, each with why the record stays true.
_RECORD_KEPT_IN = {
    '_hold_layer_impl': 'the record writer',
    'restore_camera_state': "dispatches the restore, which sets the snapshot's record",
    '_restore_camera_state_impl': "clears the record, then sets the snapshot's once it is whole",
    '_apply_settings_impl': 'its own steps, under whichever door called it',
    '_lock_auto_gain_impl': 'its own steps, under whichever door called it',
    '_capture_still_impl': "a capture's lock of the standing arm, resumed after",
    '_resume_auto_gain_impl': 'the re-arm of the arm a capture locked',
    '_clamp_exposure_to_ceiling_before_arm': "the arm's ceiling, part of the arm",
}


def _write_body_references(tree: ast.AST):
    """(line, write body, enclosing function, whether handed to _no_layers_write)."""
    found = []

    def walk(node, function, wrapped):
        # The outermost function: a helper nested in a method is the method's.
        if function is None and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == '_no_layers_write'
        ):
            for arg in node.args:
                walk(arg, function, True)
            return
        if isinstance(node, ast.Attribute) and node.attr in _WRITE_BODIES:
            found.append((node.lineno, node.attr, function, wrapped))
        for child in ast.iter_child_nodes(node):
            walk(child, function, wrapped)

    walk(tree, None, False)
    return found


# The two modules a write body is named in: the camera's own, and the capture's.
_CAMERA_MODULES = ('camera.py', 'imaging.py')


def test_every_camera_write_keeps_the_held_layer_true():
    references = [
        (name, *reference)
        for name in _CAMERA_MODULES
        for reference in _write_body_references(
            ast.parse((REPO_ROOT / 'modules' / 'lumascope_api' / name).read_text())
        )
    ]
    assert {name for name, *_ in references} == set(_CAMERA_MODULES), (
        'the walk found no write body in one of the camera modules: it is not reading them'
    )
    offenders = [
        f'{name}:{line} {body} in {function}'
        for name, line, body, function, wrapped in references
        if not wrapped and function not in _RECORD_KEPT_IN
    ]
    assert offenders == []


def test_no_module_outside_the_camera_reaches_a_camera_write_body():
    roots = [REPO_ROOT / name for name in ('modules', 'ui', 'rest')]
    files = [REPO_ROOT / 'lumaviewpro.py']
    files += [path for root in roots for path in sorted(root.rglob('*.py'))]
    offenders = [
        f'{path.relative_to(REPO_ROOT)}:{line} {body}'
        for path in files
        if not (path.parent.name == 'lumascope_api' and path.name in _CAMERA_MODULES)
        for line, body, _function, _wrapped in _write_body_references(ast.parse(path.read_text()))
    ]
    assert offenders == []
