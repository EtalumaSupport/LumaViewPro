# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Four text boxes that acted on an entry without recording it now report it.

Each of these boxes committed a value that reached settings, the camera or the
motor while ``gui_interactions.log`` said nothing about a user typing anything.
Reading a bundle afterwards, the setting simply changed by itself -- or worse,
was credited to whatever control the handler happened to touch on its way.

Two shapes are pinned here:

- the record carries the widget's OWN text, taken before the first transform.
  A handler that logs after parsing, clamping or sanitising reports what the
  app made of the entry and asserts the user typed it.
- a typed commit and a dragged slider on the same setting stay
  distinguishable. Z is the case that matters: one handler served both, so a
  keystroke reported itself as a drag.

The Z and frame-box cases are pinned through the AST because the suite mocks
Kivy rather than instantiating widgets; the acceleration box is driven through
its real handler.
"""

from __future__ import annotations

import ast
from typing import ClassVar

import pytest

from modules import gui_logger
from tests.ast_seams import find_def

# Handler -> the record it must write with the box's own text.
_TYPED_RECORDS = (
    ('ui/microscope_settings.py', 'MicroscopeSettings', 'frame_size'),
    ('ui/protocol_settings.py', 'ProtocolSettings', 'step_name_validation'),
    ('ui/vertical_control.py', 'VerticalControl', 'set_position_text'),
    ('ui/advanced_settings.py', 'AdvancedSettings', 'acceleration_pct_text'),
)


def _emitter_calls(fn, attr):
    """Every ``gui_logger.<attr>(...)`` call inside ``fn``, in source order."""
    return [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == 'gui_logger'
    ]


@pytest.mark.parametrize(('module', 'cls', 'handler'), _TYPED_RECORDS)
def test_each_handler_records_the_entry(module, cls, handler):
    """T1: the box that acts on an entry also reports it."""
    fn = find_def(module, handler, class_name=cls)
    assert fn is not None, f'{cls}.{handler} moved or was renamed'
    assert _emitter_calls(fn, 'text_input'), (
        f'{cls}.{handler} commits a typed value without recording it, so the '
        f'setting changes in the bundle with nothing saying a user set it'
    )


@pytest.mark.parametrize(('module', 'cls', 'handler'), _TYPED_RECORDS)
def test_the_entry_is_recorded_before_anything_else_it_does(module, cls, handler):
    """T1: the first emitter a handler reaches is the typed record.

    A handler that records after its own apply puts the consequence in the log
    ahead of the cause, and a freeze in between loses the entry entirely.
    """
    fn = find_def(module, handler, class_name=cls)
    emitters = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == 'gui_logger'
    ]
    assert emitters, f'{cls}.{handler} reaches no emitter at all'
    first = min(emitters, key=lambda n: n.lineno)
    assert first.func.attr == 'text_input', (
        f'{cls}.{handler} writes {first.func.attr.upper()} before it records '
        f'what was typed, so the bundle reports the consequence before the cause'
    )


class TestATypedZCommitIsNotADrag:
    """T2: the Z box and the Z slider share a setting, not a verb.

    One handler served both, and it emitted SLIDER unconditionally -- so
    typing a height into the box was indistinguishable in the bundle from
    dragging the slider to it.
    """

    def test_the_box_records_a_typed_entry(self):
        fn = find_def('ui/vertical_control.py', 'set_position_text', class_name='VerticalControl')
        assert fn is not None, 'the Z box lost its own handler'
        assert _emitter_calls(fn, 'text_input'), 'the Z box no longer records what was typed'
        assert not _emitter_calls(fn, 'slider'), (
            'a typed Z commit emits SLIDER, so a keystroke reads as a drag'
        )

    @staticmethod
    def _z_box(monkeypatch, typed, show_target):
        """The real commit and queue; the trigger, the target and the box stood in."""
        from types import SimpleNamespace

        import modules.app_context as _app_ctx
        from ui.vertical_control import VerticalControl

        lines = []
        monkeypatch.setattr(
            gui_logger, 'text_input', lambda name, value: lines.append((name, value))
        )
        monkeypatch.setattr(
            _app_ctx, 'ctx', SimpleNamespace(session=SimpleNamespace(controls_locked=False))
        )
        moves = []
        stand = SimpleNamespace(
            ids={'z_position_id': SimpleNamespace(text=typed)},
            _next_pos=None,
            queue_slider_position_trigger=lambda: moves.append(stand._next_pos),
        )
        stand._queue_z_move = lambda pos: VerticalControl._queue_z_move(stand, pos)
        stand._show_z_target = lambda: show_target(stand.ids['z_position_id'])
        VerticalControl.set_position_text(stand, typed)
        return stand, moves, lines

    @pytest.mark.parametrize('typed', ['', '-', '.', '-.'])
    def test_a_typed_non_number_moves_nothing_and_the_box_shows_the_target(
        self, monkeypatch, typed
    ):
        """The kv float filter lets these through. Nothing moves; the box
        shows the Z target again, and the record has what was typed, then
        what the box went back to."""
        stand, moves, lines = self._z_box(
            monkeypatch, typed, lambda box: setattr(box, 'text', '4950.00')
        )

        assert moves == []
        assert stand.ids['z_position_id'].text == '4950.00'
        assert lines == [('Z_POSITION', typed), ('Z_POSITION_APPLIED', '4950.00')]

    def test_a_typed_number_is_queued_as_one(self, monkeypatch):
        def _no_put_back(box):
            pytest.fail('a number does not put the box back')

        _stand, moves, lines = self._z_box(monkeypatch, '-3', _no_put_back)

        assert moves == [-3.0]
        assert lines == [('Z_POSITION', '-3')]

    def test_the_slider_keeps_its_own_verb(self):
        fn = find_def('ui/vertical_control.py', 'set_position', class_name='VerticalControl')
        assert fn is not None, 'the Z slider lost its handler'
        assert _emitter_calls(fn, 'slider'), 'the Z slider no longer records the drag'
        assert not _emitter_calls(fn, 'text_input'), (
            'the slider handler records a typed entry, so a drag reads as a keystroke'
        )


class TestTheAccelerationBoxReportsTheAttempt:
    """T4: the typed text, handed on as typed; the motion API owns the range."""

    @pytest.fixture
    def emitted(self, monkeypatch):
        lines = []
        monkeypatch.setattr(
            gui_logger, 'text_input', lambda name, value: lines.append((name, str(value)))
        )
        return lines

    def _panel(self, typed):
        from types import SimpleNamespace

        from ui.advanced_settings import AdvancedSettings

        applied = []

        class _Panel:
            ids: ClassVar[dict] = {
                'acceleration_pct_slider': SimpleNamespace(min=10, max=100, value=50),
                'acceleration_pct_text': SimpleNamespace(text=typed),
            }
            _show_acceleration_limit = AdvancedSettings._show_acceleration_limit

            def set_acceleration_limit(self, val_pct):
                applied.append(val_pct)

        return _Panel(), applied

    def test_an_in_range_entry_reports_no_correction(self, emitted):
        from ui.advanced_settings import AdvancedSettings

        panel, applied = self._panel('40')
        AdvancedSettings.acceleration_pct_text(panel)

        assert ('ACCELERATION', '40') in emitted, f'the typed limit was not recorded: {emitted}'
        assert not [n for n, _ in emitted if n == 'ACCELERATION_APPLIED'], (
            f'a value the clamp never moved was reported as a correction: {emitted}'
        )
        assert applied == [40]

    def test_an_out_of_range_entry_reaches_the_session_as_typed(self, emitted):
        """The box does not clamp: the API refuses, and the refusal is shown."""
        from ui.advanced_settings import AdvancedSettings

        panel, applied = self._panel('5000')
        AdvancedSettings.acceleration_pct_text(panel)

        assert emitted == [('ACCELERATION', '5000')], emitted
        assert applied == [5000]

    def test_the_redraw_shows_the_stored_limit_in_the_slider_and_the_box(self, monkeypatch):
        """After a refusal the slider has not moved, so its kv binding would
        not reset the box; the redraw sets both from the store."""
        from types import SimpleNamespace

        import modules.app_context as _app_ctx
        from ui.advanced_settings import AdvancedSettings

        monkeypatch.setattr(
            _app_ctx,
            'ctx',
            SimpleNamespace(settings={'motion': {'acceleration_max_pct': 60}}),
        )
        panel, _ = self._panel('5000')
        AdvancedSettings._show_stored_acceleration_limit(panel)

        assert panel.ids['acceleration_pct_slider'].value == 60
        assert panel.ids['acceleration_pct_text'].text == '60'

    @pytest.mark.parametrize('typed', ['', '-', 'abc'])
    def test_an_unparseable_entry_puts_the_box_back_and_reports_both(self, emitted, typed):
        """The kv int filter lets '' and '-' through. Nothing reaches the
        motor; the box shows the limit its slider holds again, and the
        record has the attempt, then what the box went back to."""
        from ui.advanced_settings import AdvancedSettings

        panel, applied = self._panel(typed)
        AdvancedSettings.acceleration_pct_text(panel)

        assert emitted == [('ACCELERATION', typed), ('ACCELERATION_APPLIED', '50')]
        assert panel.ids['acceleration_pct_text'].text == '50'
        assert applied == [], 'an unparseable entry must not reach the motor'
