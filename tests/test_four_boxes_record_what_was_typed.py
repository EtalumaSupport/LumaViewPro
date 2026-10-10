# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Four text boxes record what was typed, before the entry does anything.

Each of these boxes -- the frame width and height, the step name, Z, and the
acceleration limit -- commits a value that reaches settings, the protocol, the
camera or the motor. Each records ``TEXT_INPUT <NAME> <typed>`` with the box's
own text, taken before anything parses, clamps or sanitises it, and writes it
first, so a bundle reads in the order the person acted and a freeze between
the record and the apply cannot lose the entry.

A typed Z and a dragged Z slider share a record name and not a verb: the box
records ``TEXT_INPUT``, the slider ``SLIDER``, so a keystroke never reads as a
drag.

Kivy is stubbed in the test process, so each handler is driven on a stand-in
shaped like its panel, and what it recorded is read off the
``LVP.gui_interactions`` logger. Where order is the claim, the apply's catcher
takes a copy of the records already in the log at the moment it runs.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import ClassVar

import pytest

import modules.app_context as _app_ctx
from ui.advanced_settings import AdvancedSettings
from ui.microscope_settings import MicroscopeSettings, _CoalescingApplier
from ui.protocol_settings import ProtocolSettings
from ui.vertical_control import VerticalControl

_GUI_LOG = 'LVP.gui_interactions'


def _records(caplog):
    return [r.getMessage() for r in caplog.records if r.name == _GUI_LOG]


@pytest.fixture
def gui_log(caplog):
    caplog.set_level(logging.INFO, logger=_GUI_LOG)
    return caplog


# --------------------------------------------------------------------------
# The stand-in panels, each with a catcher on the apply its handler reaches.
# --------------------------------------------------------------------------


def _step_name_panel(monkeypatch, caplog, typed):
    """The protocol panel on step 0, labelled 'Step 1'; the step-name box holding ``typed``."""
    labels = ['Step 1']
    applies = []

    def rename_step(_protocol, idx, name):
        applies.append(_records(caplog))
        labels[idx] = name

    monkeypatch.setattr(
        _app_ctx, 'ctx', SimpleNamespace(session=SimpleNamespace(rename_step=rename_step))
    )
    shown = []
    # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler's record
    panel = SimpleNamespace(
        _protocol=SimpleNamespace(step=lambda idx: {'Label': labels[idx]}),
        curr_step=0,
        ids={'step_name_input': SimpleNamespace(text=typed)},
        generate_step_name_input=lambda: shown.append(labels[0]),
        _draw_protocol_steps=lambda: None,
    )
    panel.step_name_validation_ex = lambda name: ProtocolSettings.step_name_validation_ex(
        panel, name
    )
    return panel, applies, shown


def _z_panel(monkeypatch, caplog, typed='', show_target=None):
    """The real commit and queue; the move's trigger, the Z target and the box stood in.

    Returns the panel, the positions queued, and the records in the log at each queue.
    """
    monkeypatch.setattr(
        _app_ctx, 'ctx', SimpleNamespace(session=SimpleNamespace(controls_locked=False))
    )
    moves = []
    seen = []

    def _queued():
        moves.append(panel._next_pos)
        seen.append(_records(caplog))

    # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler's record
    panel = SimpleNamespace(
        ids={'z_position_id': SimpleNamespace(text=typed)},
        _next_pos=None,
        queue_slider_position_trigger=_queued,
    )
    panel._queue_z_move = lambda pos: VerticalControl._queue_z_move(panel, pos)
    panel._show_z_target = lambda: show_target(panel.ids['z_position_id'])
    return panel, moves, seen


def _acceleration_panel(typed, caplog):
    """The advanced panel, its slider at 50 and its box holding ``typed``.

    Returns the panel, the limits handed to the setter, and the records in the
    log at each hand-over.
    """
    applied = []
    seen = []

    # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler's record
    class _Panel:
        ids: ClassVar[dict] = {
            'acceleration_pct_slider': SimpleNamespace(min=10, max=100, value=50),
            'acceleration_pct_text': SimpleNamespace(text=typed),
        }
        _show_acceleration_limit = AdvancedSettings._show_acceleration_limit

        def set_acceleration_limit(self, val_pct):
            applied.append(val_pct)
            seen.append(_records(caplog))

    return _Panel(), applied, seen


class _InlineLane:
    """The camera lane, run inline: the task's call, then its redraw."""

    def put(self, task):
        task.action()
        task.callback()
        return True


def _frame_panel(monkeypatch, caplog, width_text):
    """The microscope panel at 768x1200, its width box holding ``width_text``."""
    settings = {'frame': {'width': 768, 'height': 1200}}
    applies = []

    def set_frame_size(width, height):
        applies.append(_records(caplog))
        settings['frame'] = {'width': width, 'height': height}

    monkeypatch.setattr(
        _app_ctx,
        'ctx',
        SimpleNamespace(
            settings=settings,
            session=SimpleNamespace(set_frame_size=set_frame_size),
            # a stand-in by design: the lane's threading is not the subject; the handler's record is
            camera_executor=_InlineLane(),
        ),
    )
    # a stand-in by design: Kivy is stubbed in the test process; the subject is the handler's record
    panel = SimpleNamespace(
        ids={
            'frame_width_id': SimpleNamespace(text=width_text, focus=False),
            'frame_height_id': SimpleNamespace(text='1200', focus=False),
        },
        _frame_size_applier=_CoalescingApplier(name='FRAME_SIZE'),
        _ui_binning_size=lambda: 1,
        _redraw_framing=lambda: None,
    )
    panel._typed_frame_dimensions = lambda: MicroscopeSettings._typed_frame_dimensions(panel)
    panel._framing_applied = lambda: MicroscopeSettings._framing_applied(panel)
    return panel, applies


# --------------------------------------------------------------------------
# The record comes first.
# --------------------------------------------------------------------------


def _commit_frame_width(monkeypatch, caplog):
    panel, applies = _frame_panel(monkeypatch, caplog, '800')
    MicroscopeSettings.frame_size(panel, 'frame_width_id')
    return 'TEXT_INPUT FRAME_WIDTH 800', applies


def _commit_step_name(monkeypatch, caplog):
    panel, applies, _shown = _step_name_panel(monkeypatch, caplog, 'well A1')
    ProtocolSettings.step_name_validation(panel, 'well A1')
    return 'TEXT_INPUT STEP_NAME well A1', applies


def _commit_z(monkeypatch, caplog):
    panel, _moves, seen = _z_panel(monkeypatch, caplog, '-3')
    VerticalControl.set_position_text(panel, '-3')
    return 'TEXT_INPUT Z_POSITION -3', seen


def _commit_acceleration(monkeypatch, caplog):
    panel, _applied, seen = _acceleration_panel('40', caplog)
    AdvancedSettings.acceleration_pct_text(panel)
    return 'TEXT_INPUT ACCELERATION 40', seen


@pytest.mark.parametrize(
    'commit',
    [_commit_frame_width, _commit_step_name, _commit_z, _commit_acceleration],
    ids=['frame_width', 'step_name', 'z', 'acceleration'],
)
def test_the_entry_is_recorded_before_anything_else_it_does(monkeypatch, gui_log, commit):
    """When the entry reaches its apply, its record -- and only it -- is already in the log.

    A handler that records after its own apply puts the consequence in the log
    ahead of the cause, and a freeze in between loses the entry. The layer
    panel's shared helper does not hold this today; its order is pinned in
    ``tests/test_layer_text_input_logging.py``.

    Fails if any of the four handlers moves its ``gui_logger.text_input`` call
    after the apply (the frame size, the rename, the Z move, the acceleration
    limit).
    """
    typed_record, seen_at_apply = commit(monkeypatch, gui_log)

    assert seen_at_apply == [[typed_record]]
    assert _records(gui_log)[0] == typed_record


def test_a_blank_step_name_is_recorded_though_nothing_is_renamed(monkeypatch, gui_log):
    """A blank step name keeps the step's name; the entry is still recorded, as typed.

    Fails if the step-name record moves below the branch that keeps the old
    name, which returns before a later record would be written.
    """
    panel, applies, shown = _step_name_panel(monkeypatch, gui_log, '  ')

    ProtocolSettings.step_name_validation(panel, '  ')

    assert applies == []
    assert shown == ['Step 1']
    assert _records(gui_log) == ['TEXT_INPUT STEP_NAME   ']


class TestATypedZCommitIsNotADrag:
    """T2: the Z box and the Z slider share a setting, not a verb.

    One handler served both, and it emitted SLIDER unconditionally -- so
    typing a height into the box was indistinguishable in the bundle from
    dragging the slider to it.
    """

    def test_a_typed_non_number_moves_nothing_and_the_box_shows_the_target(
        self, monkeypatch, gui_log
    ):
        """'-' passes the kv float filter. Nothing moves; the box shows the Z
        target again, and the record has what was typed, then what the box
        went back to."""
        panel, moves, _seen = _z_panel(
            monkeypatch, gui_log, '-', lambda box: setattr(box, 'text', '4950.00')
        )

        VerticalControl.set_position_text(panel, '-')

        assert moves == []
        assert panel.ids['z_position_id'].text == '4950.00'
        assert _records(gui_log) == [
            'TEXT_INPUT Z_POSITION -',
            'TEXT_INPUT Z_POSITION_APPLIED 4950.00',
        ]

    def test_a_typed_number_is_queued_as_one(self, monkeypatch, gui_log):
        """A typed number is queued as that number and recorded once, as typed, never as SLIDER.

        Fails if the box's handler records through ``gui_logger.slider``, or
        records anything but the typed text.
        """

        def _no_put_back(box):
            pytest.fail('a number does not put the box back')

        panel, moves, _seen = _z_panel(monkeypatch, gui_log, '-3', _no_put_back)

        VerticalControl.set_position_text(panel, '-3')

        assert moves == [-3.0]
        assert _records(gui_log) == ['TEXT_INPUT Z_POSITION -3']

    def test_the_slider_keeps_its_own_verb(self, monkeypatch, gui_log):
        """A slider release is queued and recorded as SLIDER with the value it resolved to.

        Fails if ``set_position`` records through ``gui_logger.text_input``, or
        the slider's verb changes.
        """
        panel, moves, _seen = _z_panel(monkeypatch, gui_log)

        VerticalControl.set_position(panel, 4950.0)

        assert moves == [4950.0]
        assert _records(gui_log) == ['SLIDER Z_POSITION 4950.0']


class TestTheAccelerationBoxReportsTheAttempt:
    """T4: the typed text, handed on as typed; the motion API owns the range."""

    def test_an_in_range_entry_reports_no_correction(self, gui_log):
        panel, applied, _seen = _acceleration_panel('40', gui_log)
        AdvancedSettings.acceleration_pct_text(panel)

        assert _records(gui_log) == ['TEXT_INPUT ACCELERATION 40'], (
            'an accepted limit is recorded as typed, with no correction'
        )
        assert applied == [40]

    def test_an_out_of_range_entry_reaches_the_session_as_typed(self, gui_log):
        """The box does not clamp: the API refuses, and the refusal is shown."""
        panel, applied, _seen = _acceleration_panel('5000', gui_log)
        AdvancedSettings.acceleration_pct_text(panel)

        assert _records(gui_log) == ['TEXT_INPUT ACCELERATION 5000']
        assert applied == [5000]

    def test_the_redraw_shows_the_stored_limit_in_the_slider_and_the_box(self, monkeypatch):
        """After a refusal the slider has not moved, so its kv binding would
        not reset the box; the redraw sets both from the store."""
        monkeypatch.setattr(
            _app_ctx,
            'ctx',
            SimpleNamespace(settings={'motion': {'acceleration_max_pct': 60}}),
        )
        panel, _applied, _seen = _acceleration_panel('5000', None)
        AdvancedSettings._show_stored_acceleration_limit(panel)

        assert panel.ids['acceleration_pct_slider'].value == 60
        assert panel.ids['acceleration_pct_text'].text == '60'

    def test_an_unparseable_entry_puts_the_box_back_and_reports_both(self, gui_log):
        """'-' passes the kv int filter. Nothing reaches the motor; the box
        shows the limit its slider holds again, and the record has the
        attempt, then what the box went back to."""
        panel, applied, _seen = _acceleration_panel('-', gui_log)
        AdvancedSettings.acceleration_pct_text(panel)

        assert _records(gui_log) == [
            'TEXT_INPUT ACCELERATION -',
            'TEXT_INPUT ACCELERATION_APPLIED 50',
        ]
        assert panel.ids['acceleration_pct_text'].text == '50'
        assert applied == [], 'an unparseable entry must not reach the motor'
