# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Save Focus and Apply Focus write into the protocol the Protocol panel is editing.

Both handlers read the protocol from an application-context field that no
longer existed once the panel became the protocol's one holder; the read
defaulted to "no protocol loaded", so Save Focus saved only the layer focus
and Apply Focus changed no step, and neither said so. A test that handed the
handler a context carrying that field could not see it.

These tests drive the CLICK handlers -- where the protocol is taken, at click
time beside the selected step -- against a real ``AppContext``, which has no
protocol field, and a real ``Protocol`` held by a stand-in panel.

Kivy's widgets are MagicMock'd in the test environment, so the handler bodies
are compiled out of the source and run against a plain ``self``, as the #734
tests do.
"""

from __future__ import annotations

import ast
import pathlib
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from modules.app_context import AppContext
from modules.exceptions import ProtocolError
from tests.test_protocol_roundtrip import _build_protocol, _make_step

REPO = pathlib.Path(__file__).resolve().parent.parent
LAYER_CONTROL_SRC = REPO / 'ui' / 'layer_control.py'


def _compile(method_name: str, globals_: dict):
    tree = ast.parse(LAYER_CONTROL_SRC.read_text())
    cls = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'LayerControl'
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    ns = dict(globals_)
    exec(compile(ast.unparse(fn), f'<layer_control::{method_name}>', 'exec'), ns)
    return ns[method_name]


def _panel_with(protocol, selected_step):
    """A real AppContext whose Protocol panel holds ``protocol``, Z reading 5000."""
    panel = SimpleNamespace(_protocol=protocol, curr_step=selected_step)
    session = SimpleNamespace(
        settings={'BF': {'focus': 7000.0}, 'Blue': {'focus': 7000.0}},
        settings_lock=threading.Lock(),
    )
    scope = SimpleNamespace(
        motion=SimpleNamespace(
            refuse_unknown_positions=lambda axes, *, recording, then: None,
            get_current_position=lambda axis: 5000.0,
        )
    )
    ctx = AppContext(
        session=session,
        scope=scope,
        motion_settings=SimpleNamespace(ids={'protocol_settings_id': panel}),
        stage=MagicMock(),
    )
    globals_ = {
        '_app_ctx': SimpleNamespace(ctx=ctx),
        'gui_logger': MagicMock(),
        'logger': MagicMock(),
        'run_reported': lambda call, on_done, label: call(),
        'Clock': SimpleNamespace(schedule_once=lambda cb, dt=0: cb(0)),
        'ProtocolError': ProtocolError,
    }
    me = SimpleNamespace(layer='BF', _schedule_step_views_refresh=MagicMock())
    for name in ('execute_save_focus', 'execute_apply_focus_to_channel_steps'):
        body = _compile(name, globals_)
        setattr(me, name, lambda *a, _body=body, **kw: _body(me, *a, **kw))
    return me, globals_, ctx


def _two_bf_one_blue():
    return _build_protocol(
        [
            _make_step(name='A1_BF', color='BF', z=7000.0),
            _make_step(name='A2_BF', color='BF', z=7000.0, x=30.0),
            _make_step(name='A1_Blue', color='Blue', z=7000.0, x=50.0),
        ]
    )


def test_save_focus_writes_the_selected_step_of_the_panels_protocol():
    protocol = _two_bf_one_blue()
    me, globals_, ctx = _panel_with(protocol, selected_step=1)

    _compile('save_focus', globals_)(me)

    assert list(protocol.steps()['Z']) == [7000.0, 5000.0, 7000.0]
    assert ctx.settings['BF']['focus'] == 5000.0


def test_apply_focus_writes_every_step_of_the_channel_in_the_panels_protocol():
    protocol = _two_bf_one_blue()
    me, globals_, ctx = _panel_with(protocol, selected_step=0)

    _compile('apply_focus_to_channel_steps', globals_)(me)

    assert list(protocol.steps()['Z']) == [5000.0, 5000.0, 7000.0]
    assert ctx.settings['BF']['focus'] == 5000.0


def test_the_application_context_has_no_protocol_of_its_own():
    # The panel is the protocol's one holder; a second field here is how the
    # two came to disagree.
    assert 'protocol' not in AppContext.__dataclass_fields__
    assert not hasattr(AppContext(), 'protocol')
