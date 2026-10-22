# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The Objective Control panel chooses the objective by its turret slot (#861).

The panel offered two ways to say which objective is working: a slot and a
name. Only the slot is the hardware's truth, and the name control had to be
turned into a slot, refusing glass no slot held; two testers on two LS850Ts
picked an unassigned objective, were refused, and found the assignment only
by Reset and a slot press. The panel now has the four slot buttons and
Reset, or on a scope without a turret one Assign Objective button; no
control names an objective by itself, and each slot button wraps its
objective's catalogue name to two lines.
"""

from tests.ast_seams import REPO_ROOT


def _objective_control():
    kv = (REPO_ROOT / 'ui' / 'lumaviewpro.kv').read_text()
    start = kv.index('<VerticalControl>:')
    return kv[start : kv.index('# Z-Stack Settings', start)]


def test_no_control_names_an_objective_by_itself():
    block = _objective_control()
    assert 'PickSpinner' not in block
    assert 'objective_spinner' not in block
    assert 'root.set_turret_objective(' not in block


def test_the_panel_has_the_four_slot_buttons_reset_and_assign():
    block = _objective_control()
    for position in range(1, 5):
        assert f'on_release: root.turret_gesture({position})' in block
    assert 'on_release: root.reset_turret_objective()' in block
    assert 'on_release: root.assign_objective()' in block


def test_each_slot_button_wraps_its_name_to_its_width():
    block = _objective_control()
    buttons = block.split('RoundedToggleButton:')[1:]
    slots = [b for b in buttons if 'root.turret_gesture(' in b]
    assert len(slots) == 4
    assert all('text_size: self.width' in b for b in slots)
