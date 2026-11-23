# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Apply Focus writes into the protocol the Protocol panel is editing; Save Focus names no protocol.

The Apply Focus handler read the protocol from an application-context field
that no longer existed once the panel became the protocol's one holder; the
read defaulted to "no protocol loaded", so Apply Focus changed no step and
did not say so. A test that handed the handler a context carrying that field
could not see it. Save Focus saves the channel's focus and changes no step,
so its handler hands the Session the channel alone.

These tests drive the CLICK handlers -- where the protocol is taken, at click
time -- against a real ``AppContext``, which has no protocol field, and a
real ``Protocol`` held by a stand-in panel, and check what the handler hands
the Session; the writes themselves are the Session's, pinned by
``test_saving_a_focus_is_an_api_capability``.

Kivy's widgets are MagicMock'd in the test environment, so the handler bodies
are compiled out of the source and run against a plain ``self``, as the #734
tests do.
"""

from __future__ import annotations

import ast
import functools
import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock

from _lumascope.modules.app_context import AppContext
from tests.test_protocol_roundtrip import _build_protocol, _make_step
from tests.stock_channels import LED0

REPO = pathlib.Path(__file__).resolve().parent.parent
LAYER_CONTROL_SRC = REPO / 'ui' / 'channel_control.py'


def _compile(method_name: str, globals_: dict):
    tree = ast.parse(LAYER_CONTROL_SRC.read_text())
    cls = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'ChannelControl'
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    ns = dict(globals_)
    exec(compile(ast.unparse(fn), f'<channel_control::{method_name}>', 'exec'), ns)
    return ns[method_name]


def _panel_with(protocol, selected_step):
    """A real AppContext whose Protocol panel holds ``protocol``; the session records the call."""
    panel = SimpleNamespace(_protocol=protocol, curr_step=selected_step, update_step_ui=MagicMock())
    session = SimpleNamespace(save_focus=MagicMock(), apply_focus_to_channel_steps=MagicMock())
    ctx = AppContext(
        session=session,
        motion_settings=SimpleNamespace(ids={'protocol_settings_id': panel}),
        stage=MagicMock(),
    )
    globals_ = {
        '_app_ctx': SimpleNamespace(ctx=ctx),
        'gui_logger': MagicMock(),
        'logger': MagicMock(),
        'run_reported': lambda call, redraw, label: call(),
    }
    me = SimpleNamespace(channel=LED0, _refresh_step_views=MagicMock())
    return me, globals_, session


def _protocol():
    return _build_protocol([_make_step(name='A1_BF', color='BF', z=7000.0)])


def test_save_focus_hands_the_session_the_channel_alone():
    # #524: the handler handed the Session the panel's protocol and selected
    # step, and that step took the Z.
    me, globals_, session = _panel_with(_protocol(), selected_step=0)

    _compile('save_focus', globals_)(me)

    session.save_focus.assert_called_once_with(LED0)


def test_apply_focus_hands_the_session_the_panels_protocol():
    protocol = _protocol()
    me, globals_, session = _panel_with(protocol, selected_step=0)

    _compile('apply_focus_to_channel_steps', globals_)(me)

    session.apply_focus_to_channel_steps.assert_called_once_with(protocol, LED0)


def test_apply_focus_redraws_the_stage_and_step_editor_inside_the_boundary():
    # The redraw is run_reported's own, so anything it raises reaches the one
    # reporter; a clock hop would run it outside the boundary, where only a
    # catch of its own could answer for it.
    protocol = _protocol()
    me, globals_, _ = _panel_with(protocol, selected_step=0)
    redraws = []
    globals_['run_reported'] = lambda call, redraw, label: redraws.append(redraw)
    me._refresh_step_views = functools.partial(_compile('_refresh_step_views', globals_), me)
    ctx = globals_['_app_ctx'].ctx

    _compile('apply_focus_to_channel_steps', globals_)(me)
    (redraw,) = redraws
    redraw()

    (call,) = ctx.stage.set_protocol_steps.call_args_list
    assert call.args == (protocol,)
    ctx.motion_settings.ids['protocol_settings_id'].update_step_ui.assert_called_once_with()


def test_the_application_context_has_no_protocol_of_its_own():
    # The panel is the protocol's one holder; a second field here is how the
    # two came to disagree.
    assert 'protocol' not in AppContext.__dataclass_fields__
    assert not hasattr(AppContext(), 'protocol')
