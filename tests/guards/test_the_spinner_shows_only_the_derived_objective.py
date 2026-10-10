# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The objective spinner is written in one place, with the API's answer.

The spinner used to call select_objective from on_text, so any write to it
reassigned the slot in the light path: step navigation wrote the step's
objective before its turret move had landed, and a win of that race put 10x
on a slot holding 4x glass. A pick now arrives as on_pick and the text is
display only, written by show_turret_state from the active objective the
API derives. A second writer would be a second display of it.
"""

import ast

from tests.ast_seams import iter_package_modules, walk_defs

_ALLOWED = {
    ('ui/vertical_control.py', 'VerticalControl.show_turret_state'),
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


def test_a_programmatic_write_reaches_no_session_member():
    """The kv binds the spinner's pick, not its text: a write to the text
    -- the display refreshing -- can assign nothing."""
    from tests.ast_seams import REPO_ROOT

    kv = (REPO_ROOT / 'ui' / 'lumaviewpro.kv').read_text()
    block = kv[kv.index('id: objective_spinner2') :]
    block = block[: block.index('RoundedButton:')]
    bindings = [line.strip() for line in block.splitlines() if not line.strip().startswith('#')]
    assert not any(line.startswith('on_text:') for line in bindings)
    assert 'on_pick: root.pick_objective(args[1])' in block
