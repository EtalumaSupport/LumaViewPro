# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every write to the objective spinner is the objective the API derives.

The spinner's on_text calls select_objective, which on a turret scope
assigns the objective to the slot in the light path. So a write of anything
but the derived objective reassigns a slot: step navigation used to write
the step's objective before its turret move had landed, and a win of that
race put 10x on a slot holding 4x glass. The writers below each write the
derived objective (or, for the question's answer, the objective just
assigned to the live slot); a new writer must be one of those, and is
reviewed here before it can reassign a slot.
"""

import ast

from tests.ast_seams import iter_package_modules, walk_defs

_ALLOWED = {
    ('ui/microscope_settings.py', 'MicroscopeSettings.load_settings'),
    ('ui/vertical_control.py', 'VerticalControl._apply_objective_answer'),
    ('ui/vertical_control.py', 'VerticalControl._show_turret_outcome'),
}


def _writes_spinner_text(fn) -> bool:
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == 'text'
                and isinstance(target.value, ast.Subscript)
                and isinstance(target.value.slice, ast.Constant)
                and target.value.slice.value == 'objective_spinner2'
            ):
                return True
    return False


def _writers():
    return {
        (rel_path, qualname)
        for rel_path, tree in iter_package_modules(['ui'])
        for qualname, fn in walk_defs(tree.body)
        if _writes_spinner_text(fn)
    }


def test_only_the_reviewed_sites_write_the_spinner():
    assert _writers() == _ALLOWED


def test_the_racing_step_navigation_writer_is_gone():
    assert all('update_turret_gui' not in qualname for _, qualname in _writers())
