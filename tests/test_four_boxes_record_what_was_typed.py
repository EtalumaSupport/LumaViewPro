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
import pathlib
from typing import ClassVar

import pytest

from modules import gui_logger
from tests.ast_seams import find_def

REPO = pathlib.Path(__file__).resolve().parent.parent

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

    def test_the_slider_keeps_its_own_verb(self):
        fn = find_def('ui/vertical_control.py', 'set_position', class_name='VerticalControl')
        assert fn is not None, 'the Z slider lost its handler'
        assert _emitter_calls(fn, 'slider'), 'the Z slider no longer records the drag'
        assert not _emitter_calls(fn, 'text_input'), (
            'the slider handler records a typed entry, so a drag reads as a keystroke'
        )

    def test_the_two_controls_bind_their_own_handlers(self):
        """The split is only real if the kv actually routes the box elsewhere."""
        kv = (REPO / 'ui' / 'lumaviewpro.kv').read_text()
        assert 'root.set_position_text(self.text)' in kv, (
            'the Z box no longer commits through its own handler, so it is '
            'back to reporting itself as a slider drag'
        )
        assert 'root.set_position(self.value)' in kv, 'the Z slider lost its binding'


class TestTheAccelerationBoxReportsTheAttemptAndTheClamp:
    """T4: the typed text, then the clamp -- and the clamp only when it moved."""

    @pytest.fixture
    def emitted(self, monkeypatch):
        lines = []
        monkeypatch.setattr(
            gui_logger, 'text_input', lambda name, value: lines.append((name, str(value)))
        )
        return lines

    def _panel(self, typed):
        from types import SimpleNamespace

        applied = []

        class _Panel:
            ids: ClassVar[dict] = {
                'acceleration_pct_slider': SimpleNamespace(min=10, max=100, value=50),
                'acceleration_pct_text': SimpleNamespace(text=typed),
            }

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

    def test_an_out_of_range_entry_reports_what_was_typed_and_what_took_effect(self, emitted):
        from ui.advanced_settings import AdvancedSettings

        panel, applied = self._panel('5000')
        AdvancedSettings.acceleration_pct_text(panel)

        assert ('ACCELERATION', '5000') in emitted, (
            f'the attempt is gone; the bundle would claim the user typed 100: {emitted}'
        )
        assert ('ACCELERATION_APPLIED', '100') in emitted, f'the clamp went unreported: {emitted}'
        assert applied == [100]

    def test_an_unparseable_entry_reports_the_attempt_only(self, emitted):
        from ui.advanced_settings import AdvancedSettings

        panel, applied = self._panel('abc')
        AdvancedSettings.acceleration_pct_text(panel)

        assert ('ACCELERATION', 'abc') in emitted, (
            f'a refused entry left no trace of the user action: {emitted}'
        )
        assert not [n for n, _ in emitted if n == 'ACCELERATION_APPLIED'], (
            'nothing took effect, so there is no correction to report'
        )
        assert applied == [], 'an unparseable entry must not reach the motor'
