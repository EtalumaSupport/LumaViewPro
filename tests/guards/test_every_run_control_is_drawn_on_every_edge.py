# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every run control is redrawn on every run-state edge, by the app's one listener.

A run control draws only the run it started, from the engine, and nothing
else redraws it when that run ends on its own: its completion callback
does not touch the button. The run-state edge -- publish_run_state, which
the Session's listener schedules onto the Kivy thread on every claim and
idle transition -- is the one place a naturally ended run is drawn idle.
Drop a control from it and that control stays "running" after its run is
over. The shared displays are drawn once, after the controls.
"""

import ast

from tests.ast_seams import find_def

# The redraw each run control owns, and the class it lives on.
RUN_CONTROL_REDRAWS = {
    'draw_composite_button': ('ui/composite_capture.py', 'CompositeCapture'),
    'draw_record_button': ('ui/main_display.py', 'MainDisplay'),
    'draw_protocol_buttons': ('ui/protocol_settings.py', 'ProtocolSettings'),
    'draw_zstack_button': ('ui/zstack.py', 'ZStack'),
    'draw_autofocus_button': ('ui/vertical_control.py', 'VerticalControl'),
}


def _called(node):
    return [
        sub.func.attr if isinstance(sub.func, ast.Attribute) else sub.func.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and isinstance(sub.func, (ast.Attribute, ast.Name))
    ]


def test_the_run_state_edge_draws_the_controls():
    publish = find_def('lumaviewpro.py', 'publish_run_state', class_name='LumaViewProApp')
    assert publish is not None, 'LumaViewProApp.publish_run_state is gone'
    assert '_draw_run_controls' in _called(publish)


def test_every_run_control_is_drawn_then_the_shared_displays_once():
    draw = find_def('lumaviewpro.py', '_draw_run_controls', class_name='LumaViewProApp')
    assert draw is not None, 'LumaViewProApp._draw_run_controls is gone'
    called = _called(draw)

    missing = set(RUN_CONTROL_REDRAWS) - set(called)
    assert not missing, f'run controls never redrawn on a run-state edge: {sorted(missing)}'
    assert called.count('draw_shared_run_displays') == 1
    assert called[-1] == 'draw_shared_run_displays', (
        'the shared displays are drawn after the controls'
    )


def test_each_redraw_is_its_controls_own():
    for name, (rel_path, class_name) in RUN_CONTROL_REDRAWS.items():
        assert find_def(rel_path, name, class_name=class_name) is not None, (
            f'{class_name}.{name} is gone from {rel_path}'
        )


# Each run button's kv id and the flag its own request holds while in flight.
IN_FLIGHT_FLAGS = {
    'composite_btn': 'root.composite_pending',
    'autofocus_id': 'root.autofocus_pending',
    'zstack_aqr_btn': 'root.zstack_pending',
    'run_autofocus_btn': 'root.autofocus_scan_pending',
    'run_scan_btn': 'root.scan_pending',
    'run_protocol_btn': 'root.protocol_pending',
}


def test_each_run_button_is_disabled_while_its_own_request_is_in_flight():
    """The Python flag is only half of it: the kv binding is what disables the button."""
    from tests.ast_seams import REPO_ROOT

    kv = (REPO_ROOT / 'ui' / 'lumaviewpro.kv').read_text()
    unbound = []
    for widget_id, flag in IN_FLIGHT_FLAGS.items():
        start = kv.find(f'id: {widget_id}')
        assert start > 0, f'{widget_id} is gone from the kv'
        block = kv[start : kv.find('on_release', start)]
        disabled = next((line for line in block.splitlines() if 'disabled:' in line), '')
        if flag not in disabled:
            unbound.append(widget_id)
    assert not unbound, f'run buttons not disabled by their in-flight flag: {unbound}'


# Each run button's kv id and the flag its draw sets from the Session's
# held_by_other(<the run this button started>): anything else holding the
# scope greys it, and its own run leaves it live as that run's Stop.
HELD_FLAGS = {
    'composite_btn': 'root.composite_held',
    'autofocus_id': 'root.autofocus_held',
    'zstack_aqr_btn': 'root.zstack_held',
    'run_autofocus_btn': 'root.autofocus_scan_held',
    'run_scan_btn': 'root.scan_held',
    'run_protocol_btn': 'root.protocol_held',
}


def _disabled_line(kv, widget_id):
    start = kv.find(f'id: {widget_id}')
    assert start > 0, f'{widget_id} is gone from the kv'
    block = kv[start : kv.find('on_release', start)]
    return next((line for line in block.splitlines() if 'disabled:' in line), '')


def test_each_run_button_greys_while_anything_else_holds_the_scope():
    from tests.ast_seams import REPO_ROOT

    kv = (REPO_ROOT / 'ui' / 'lumaviewpro.kv').read_text()
    unbound = [wid for wid, flag in HELD_FLAGS.items() if flag not in _disabled_line(kv, wid)]
    assert not unbound, f'run buttons not greyed by their held flag: {unbound}'


def test_no_run_button_decides_its_own_lock():
    """held_by_other already answers a recording, live or draining, and a
    finished run's drain is the API's to refuse: a second term for either on
    a toggle is a second answer to one question."""
    from tests.ast_seams import REPO_ROOT

    kv = (REPO_ROOT / 'ui' / 'lumaviewpro.kv').read_text()
    second_answers = {
        wid: line.strip()
        for wid in HELD_FLAGS
        if any(term in (line := _disabled_line(kv, wid)) for term in ('app.', 'files_draining'))
    }
    assert not second_answers, f'run buttons locked by a term of their own: {second_answers}'


# Where each held flag is declared: a kv binding to a property its class does
# not declare fails when the kv loads, which no headless test does.
HELD_FLAG_OWNERS = {
    'composite_held': ('ui/composite_capture.py', 'CompositeCapture'),
    'autofocus_held': ('ui/vertical_control.py', 'VerticalControl'),
    'zstack_held': ('ui/zstack.py', 'ZStack'),
    'autofocus_scan_held': ('ui/protocol_settings.py', 'ProtocolSettings'),
    'scan_held': ('ui/protocol_settings.py', 'ProtocolSettings'),
    'protocol_held': ('ui/protocol_settings.py', 'ProtocolSettings'),
}


def test_each_held_flag_is_a_property_of_the_widget_that_binds_it():
    from tests.ast_seams import parse_module

    assert {f'root.{name}' for name in HELD_FLAG_OWNERS} == set(HELD_FLAGS.values())
    undeclared = []
    for name, (rel_path, class_name) in HELD_FLAG_OWNERS.items():
        (cls,) = [
            node
            for node in ast.walk(parse_module(rel_path))
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        declared = any(
            isinstance(stmt, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets)
            and isinstance(stmt.value, ast.Call)
            and getattr(stmt.value.func, 'id', '') == 'BooleanProperty'
            for stmt in cls.body
        )
        if not declared:
            undeclared.append(f'{class_name}.{name}')
    assert not undeclared, f'held flags bound in kv but not declared: {undeclared}'
