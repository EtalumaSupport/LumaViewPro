# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""New, Load and closing ask before an edited protocol is thrown away; Save writes only to its own file.

The protocol answers whether it has unsaved changes and which file is its
own (``Protocol.has_unsaved_changes``, ``Protocol.file_path``); the panel
asks from that answer, "<name> has changes that are not saved. Discard
them?", Discard / Cancel, and Cancel leaves everything as it was. New asks
before the build; Load once the file is picked; the close once the boxes
of the protocol's own the person typed in and left are committed. Save
writes to the protocol's own file, and with none opens Save As, which saves
the protocol it was opened for.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import ui.file_dialogs as file_dialogs
import ui.protocol_settings as ps
from _lumascope.modules.acquire_mode import AcquireMode
from tests.ast_seams import find_def
from tests.gesture_fakes import worker_submit_reported
from tests.scope_fakes import home_sim_scope
from tests.test_a_protocol_edit_is_the_apis_answer import _Panel


@pytest.fixture
def ctx(gui_ctx, monkeypatch):
    """The GUI's context over a homed simulated session with Brightfield acquiring."""
    gui_ctx.stage = MagicMock()
    home_sim_scope(gui_ctx.scope)
    gui_ctx.session.set_channel_settings('BF', acquire=AcquireMode.IMAGE)
    monkeypatch.setattr(ps.gui_logger, 'protocol_action', lambda *a, **kw: None)
    monkeypatch.setattr(ps.gui_logger, 'button', lambda *a, **kw: None)
    # The Load waits on the camera lane, so it runs off the inline boundary.
    monkeypatch.setattr(ps, 'submit_reported', worker_submit_reported)
    return gui_ctx


@pytest.fixture
def asked(monkeypatch):
    """Each question the panel puts to the person, its answer the test's to give."""
    questions = []
    # a stand-in by design: the popup is a window only a person answers.
    monkeypatch.setattr(
        'ui.notification_popup.show_confirmation_popup', lambda **kw: questions.append(kw)
    )
    return questions


def _panel(ctx, protocol):
    panel = _Panel(protocol)
    ctx.motion_settings = SimpleNamespace(ids={'protocol_settings_id': panel})
    return panel


def _opening(opened):
    """A Save As button that records the dialog it is asked to open."""

    # a stand-in by design: the dialog is a native window only a person answers.
    class _SaveAs:
        def choose(self, context):
            opened.append(context)

    return _SaveAs


def _edited(ctx, protocol):
    ctx.session.set_protocol_capture_root(protocol, 'edited')
    return protocol


class TestNew:
    def test_an_unchanged_protocol_is_replaced_unasked(self, ctx, asked):
        previous = ctx.session.new_protocol()
        panel = _panel(ctx, previous)

        panel.new_protocol()

        assert asked == []
        assert panel._protocol is not previous

    def test_an_edited_protocol_is_kept_on_cancel(self, ctx, asked):
        previous = _edited(ctx, ctx.session.new_protocol())
        panel = _panel(ctx, previous)

        panel.new_protocol()

        [question] = asked
        assert (question['title'], question['confirm_text'], question['cancel_text']) == (
            'Unsaved Changes',
            'Discard',
            'Cancel',
        )
        assert question['message'] == (
            'The new protocol has changes that are not saved. Discard them?'
        )
        assert panel._protocol is previous, 'nothing is built until the person answers'

    def test_an_edited_protocol_is_replaced_on_discard(self, ctx, asked):
        previous = _edited(ctx, ctx.session.new_protocol())
        panel = _panel(ctx, previous)

        panel.new_protocol()
        asked[0]['on_confirm']()

        assert panel._protocol is not previous
        assert panel._protocol.has_unsaved_changes is False


class TestLoad:
    def _pick(self, path):
        file_dialogs.FileChooseBTN.on_selection_function(
            SimpleNamespace(selection=[str(path)], context='load_protocol')
        )

    @pytest.fixture
    def other(self, ctx, tmp_path):
        return ctx.session.save_protocol(ctx.session.new_protocol(), tmp_path / 'other.tsv')

    def test_an_edited_loaded_protocol_is_named_and_kept_on_cancel(
        self, ctx, asked, other, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(file_dialogs.gui_logger, 'select', lambda *a, **kw: None)
        mine = ctx.session.load_protocol(
            ctx.session.save_protocol(ctx.session.new_protocol(), tmp_path / 'mine.tsv')
        )
        panel = _panel(ctx, _edited(ctx, mine))

        self._pick(other)

        [question] = asked
        assert question['message'] == 'mine.tsv has changes that are not saved. Discard them?'
        assert panel._protocol is mine

    def test_an_unchanged_protocol_is_replaced_by_the_pick_unasked(
        self, ctx, asked, other, monkeypatch
    ):
        monkeypatch.setattr(file_dialogs.gui_logger, 'select', lambda *a, **kw: None)
        monkeypatch.setattr(ps, 'reset_acquire_ui', lambda: None)
        previous = ctx.session.new_protocol()
        panel = _panel(ctx, previous)
        panel.ids['labware_spinner'] = SimpleNamespace(text='')

        self._pick(other)

        assert asked == []
        assert panel._protocol.file_path == other


class TestSave:
    def test_save_writes_the_protocols_own_file(self, ctx, tmp_path):
        mine = tmp_path / 'mine.tsv'
        protocol = _edited(
            ctx,
            ctx.session.load_protocol(ctx.session.save_protocol(ctx.session.new_protocol(), mine)),
        )
        panel = _panel(ctx, protocol)

        panel.save_protocol()

        assert protocol.has_unsaved_changes is False
        assert 'Capture Root\tedited' in mine.read_text()
        assert ctx.settings['protocol']['filepath'] == str(mine)

    def test_a_protocol_with_no_file_opens_save_as(self, ctx, monkeypatch):
        opened = []
        monkeypatch.setattr(file_dialogs, 'FileSaveBTN', _opening(opened))
        panel = _panel(ctx, ctx.session.new_protocol())

        panel.save_protocol()

        assert opened == ['saveas_protocol']

    def test_save_as_saves_the_protocol_it_opened_for(self, ctx, asked, tmp_path, monkeypatch):
        """A New while the dialog is open replaces the panel's protocol, not the one saved."""
        delivered = []
        monkeypatch.setattr(
            file_dialogs,
            '_run_native_dialog_async',
            lambda btn, open_, deliver, *, on_cancel: delivered.append(deliver),
        )
        monkeypatch.setattr(file_dialogs.gui_logger, 'select', lambda *a, **kw: None)
        monkeypatch.setattr(file_dialogs.gui_logger, 'button', lambda *a, **kw: None)
        opened_for = _edited(ctx, ctx.session.new_protocol())
        panel = _panel(ctx, opened_for)
        file_dialogs.FileSaveBTN.choose(SimpleNamespace(context=''), 'saveas_protocol')
        asked.clear()
        panel.new_protocol()
        asked[0]['on_confirm']()
        replaced_by = panel._protocol

        delivered[0](str(tmp_path / 'saved.tsv'))

        assert opened_for.file_path == tmp_path / 'saved.tsv'
        assert 'Capture Root\tedited' in (tmp_path / 'saved.tsv').read_text()
        assert replaced_by.file_path is None
        assert ctx.settings['protocol']['filepath'] != str(tmp_path / 'saved.tsv'), (
            'the next start opens the panel protocol, not one it no longer holds'
        )

    def test_save_after_a_refused_startup_load_leaves_the_kept_file_alone(
        self, ctx, centre_posts, tmp_path, monkeypatch
    ):
        """The startup load is refused and the path kept; the panel holds an empty protocol.

        Refused as the plate change a recording forbids: the file is on the
        6 well plate, the scope on the 96.
        """
        ctx.session.select_labware('6 well microplate')
        kept = ctx.session.save_protocol(ctx.session.new_protocol(), tmp_path / 'kept.tsv')
        ctx.session.select_labware('96 well microplate')
        before = kept.read_text()
        ctx.set_protocol_filepath(str(kept))
        opened = []
        monkeypatch.setattr(file_dialogs, 'FileSaveBTN', _opening(opened))
        monkeypatch.setattr(ps.ProtocolSettings, 'update_step_ui', lambda self: None)
        panel = _panel(ctx, ctx.session.create_empty_protocol())
        recording = ctx.session._activity_claim.try_claim('recording')
        try:
            panel.load_persisted_protocol()
        finally:
            recording.release()
        assert [n.reason for n in centre_posts] == ['exclusive_activity_running']
        assert ctx.settings['protocol']['filepath'] == str(kept), 'precondition: the path kept'
        assert panel._protocol.num_steps() == 0, 'precondition: the panel holds the empty protocol'

        panel.save_protocol()

        assert kept.read_text() == before
        assert opened == ['saveas_protocol']


class TestTheClose:
    class _Box:
        """A box the person typed in and has not left: leaving it commits, as ``CommitInput`` does."""

        def __init__(self, on_commit, text):
            self._on_commit = on_commit
            self._focus = True
            self.text = text

        @property
        def focus(self):
            return self._focus

        @focus.setter
        def focus(self, focused):
            # a stand-in by design: the real box's commit needs a Kivy window
            # (tests/test_a_box_commits_an_edit.py runs it in a child process).
            if self._focus and not focused:
                self._focus = False
                self._on_commit()

    def _typing(self, panel, box, text):
        commit = {'capture_period': panel.update_period, 'capture_dur': panel.update_duration}
        panel.ids[box] = self._Box(commit[box], text)

    def _idle(self, panel):
        for box in ('capture_period', 'capture_dur', 'capture_root', 'step_name_input'):
            if not hasattr(panel.ids[box], 'focus'):
                panel.ids[box].focus = False

    def test_a_typed_period_is_committed_and_then_counts(self, ctx):
        protocol = ctx.session.new_protocol()
        panel = _panel(ctx, protocol)
        self._typing(panel, 'capture_period', '3')
        self._idle(panel)

        assert panel.commit_typed_edits() is True
        assert protocol.period_s() == 180
        assert panel.unsaved_changes_text() == 'The new protocol has changes that are not saved.'

    def test_a_refused_typed_period_stops_the_close(self, ctx, centre_posts, monkeypatch):
        import ui.ui_helpers as ui_helpers

        monkeypatch.setattr(ui_helpers, '_input_frame', lambda: 41)
        protocol = ctx.session.new_protocol()
        panel = _panel(ctx, protocol)
        self._typing(panel, 'capture_period', '0.005')
        self._idle(panel)

        assert panel.commit_typed_edits() is False
        assert len(centre_posts) == 1, 'the refusal is shown by its commit'

    def test_the_close_commits_then_asks_the_protocol(self):
        """``lumaviewpro.py`` is out of the suite's reach (it builds a Kivy tree at
        import): its close commits the boxes, stops on a refusal, and names the
        unsaved protocol in Confirm Exit."""
        close = ast.unparse(
            find_def('lumaviewpro.py', 'on_request_close', class_name='LumaViewProApp')
        )

        committed = close.index('commit_typed_edits()')
        assert (
            committed
            < close.index('unsaved_changes_text()')
            < close.index('self._close_the_session(session)')
        )
        assert 'Exiting discards them.' in close
