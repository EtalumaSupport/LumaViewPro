# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A protocol edit hands its request to the API and shows the answer once.

New Protocol, Modify and Add ask the API, which refuses a step or a
protocol it cannot perform (the objective is not on the turret, the
z-stack has no range) in its own words. The panel shows that refusal once,
through the boundary, and changes nothing it did not get: a refused
protocol is not adopted, and a refused step is not navigated to. What the
panel does after an accepted edit -- the move to the new step, the file
name cleared for a new protocol -- is reached only on acceptance.
"""

import ast
import logging
import pathlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _StubWidget:
    def __init__(self, **kwargs):
        pass


for _name in (
    'kivy.app',
    'kivy.properties',
    'kivy.uix',
    'kivy.uix.label',
    'kivy.uix.popup',
    'kivy.lang',
    'kivy.metrics',
    'kivy.graphics',
):
    sys.modules.setdefault(_name, MagicMock())

for _name, _attr in (
    ('kivy.uix.floatlayout', 'FloatLayout'),
    ('kivy.uix.boxlayout', 'BoxLayout'),
    ('kivy.uix.scrollview', 'ScrollView'),
    ('kivy.uix.widget', 'Widget'),
):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        setattr(_mod, _attr, _StubWidget)
        sys.modules[_name] = _mod

import modules.app_context as _app_ctx
import ui.protocol_settings as ps
from modules.exceptions import (
    HardwareCommandRefusedError,
    ProtocolNotLoadedError,
    ProtocolNotSavedError,
    ProtocolRunRefusedError,
)
from modules.protocol import ProtocolFormatError
from tests.ast_seams import find_def
from tests.scope_fakes import spec_scope
from tests.settings_fixtures import protocol_filepath_writer, settings_writer


def _refusal():
    return ProtocolRunRefusedError(
        reason='turret_objectives_unassigned',
        title='Objective Not Available',
        message='That objective is not on the turret.',
    )


class _Panel(ps.ProtocolSettings):
    """The real class, with only the widget tree and the navigation stubbed."""

    def __init__(self, protocol):
        self.ids = {
            'protocol_filename': SimpleNamespace(text='plate.tsv'),
            'capture_root': SimpleNamespace(text='root'),
            'step_name_input': SimpleNamespace(text=''),
            'tiling_size_spinner': SimpleNamespace(text='1x1'),
            'acquire_zstack_id': SimpleNamespace(active=False),
            'capture_period': SimpleNamespace(text=''),
            'capture_dur': SimpleNamespace(text=''),
        }
        self._protocol = protocol
        self.curr_step = 0
        self.moves = []
        self.redraws = 0

    def go_to_step(self, step_idx, protocol=True):
        self.moves.append(step_idx)

    def update_step_ui(self):
        self.redraws += 1


def _protocol(num_steps=3):
    protocol = MagicMock()
    protocol.num_steps.return_value = num_steps
    protocol.validate_steps.return_value = []
    return protocol


@pytest.fixture
def ctx(monkeypatch):
    session = MagicMock()
    scope = spec_scope()
    settings = {'protocol': {'filepath': 'plate.tsv'}}
    context = SimpleNamespace(
        session=session,
        scope=scope,
        lumaview=SimpleNamespace(scope=scope),
        stage=MagicMock(),
        settings=settings,
        update_settings=settings_writer(settings),
        set_protocol_filepath=protocol_filepath_writer(settings),
        image_settings=MagicMock(),
    )
    monkeypatch.setattr(_app_ctx, 'ctx', context)
    monkeypatch.setattr(ps.common_utils, 'get_opened_layer', lambda _settings: 'Green')
    monkeypatch.setattr(ps, 'get_active_layer_config', lambda layer: (layer, {}))
    monkeypatch.setattr(ps.gui_logger, 'protocol_action', lambda *a, **kw: None)
    monkeypatch.setattr(ps.gui_logger, 'button', lambda *a, **kw: None)
    return context


@pytest.fixture
def shown(monkeypatch):
    from tests.shown_outcomes import capture_shown

    return capture_shown(monkeypatch)


class TestAddStep:
    def test_an_accepted_step_is_gone_to_last(self, ctx):
        ctx.session.add_step.return_value = ['A1_Green']
        panel = _Panel(_protocol())

        panel.insert_step(after_current_step=True)

        assert panel.curr_step == 1
        assert panel.moves == [1]
        ctx.stage.set_protocol_steps.assert_called_once()

    def test_a_refused_step_is_shown_once_and_not_navigated_to(self, ctx, shown):
        ctx.session.add_step.side_effect = _refusal()
        panel = _Panel(_protocol())

        panel.insert_step(after_current_step=True)

        assert [n.title for n in shown] == ['Objective Not Available']
        assert panel.moves == [], 'a refused step has nowhere to go'
        assert panel.curr_step == 0
        assert panel.redraws == 1, 'the panel still shows its protocol as the API has it'


class TestModifyStep:
    def test_the_name_field_and_the_open_layer_reach_the_api(self, ctx):
        panel = _Panel(_protocol())
        panel.ids['step_name_input'].text = 'my_step'

        panel.modify_step()

        kwargs = ctx.session.update_step.call_args.kwargs
        assert kwargs['layer'] == 'Green'
        assert kwargs['label'] == 'my_step'

    def test_a_refused_modify_is_shown_once(self, ctx, shown):
        ctx.session.update_step.side_effect = _refusal()
        panel = _Panel(_protocol())

        panel.modify_step()

        assert [n.title for n in shown] == ['Objective Not Available']
        assert panel.redraws == 1


class TestNewProtocol:
    def test_an_accepted_protocol_is_adopted_and_has_no_file(self, ctx):
        built = _protocol()
        ctx.session.new_protocol.return_value = built
        panel = _Panel(_protocol())

        panel.new_protocol()

        assert panel._protocol is built
        assert ctx.settings['protocol']['filepath'] == ''
        assert (panel.ids['protocol_filename'].text, panel.ids['capture_root'].text) == ('', '')
        assert panel.moves == [0]

    def test_a_refused_build_is_shown_once_and_adopts_nothing(self, ctx, shown):
        ctx.session.new_protocol.side_effect = _refusal()
        previous = _protocol()
        panel = _Panel(previous)

        panel.new_protocol()

        assert [n.title for n in shown] == ['Objective Not Available']
        assert panel._protocol is previous


class TestSave:
    @pytest.fixture(autouse=True)
    def own_popups(self, monkeypatch):
        popups = []
        monkeypatch.setattr(
            'ui.notification_popup.show_notification_popup',
            lambda **kw: popups.append(kw['title']),
        )
        return popups

    def _panel(self):
        return _Panel(_protocol())

    def test_a_saved_protocol_is_remembered_under_its_new_name(self, ctx, own_popups):
        panel = self._panel()
        ctx.session.save_protocol.return_value = pathlib.Path('/data/other.tsv')

        panel.save_protocol(filepath='/data/other')

        ctx.session.save_protocol.assert_called_once_with(panel._protocol, '/data/other')
        assert ctx.settings['protocol']['filepath'] == '/data/other.tsv'
        assert panel.ids['protocol_filename'].text == 'other.tsv'
        assert own_popups == []

    def test_a_failed_save_is_shown_once_and_the_previous_name_stays(self, ctx, shown, own_popups):
        panel = self._panel()
        ctx.session.save_protocol.side_effect = ProtocolNotSavedError(
            file='/data/other.tsv', cause=PermissionError(13, 'Permission denied')
        )

        panel.save_protocol(filepath='/data/other')

        assert own_popups == [], 'the outcome is shown by the one reporter'
        assert [n.title for n in shown] == ['Protocol Not Saved']
        assert ctx.settings['protocol']['filepath'] == 'plate.tsv'
        assert panel.ids['protocol_filename'].text == 'plate.tsv'


class TestLoad:
    """A clicked Load asks the Session, and its answer is shown by the reporter."""

    @pytest.fixture(autouse=True)
    def own_popups(self, monkeypatch):
        popups = []
        monkeypatch.setattr(
            'ui.notification_popup.show_notification_popup',
            lambda **kw: popups.append(kw['title']),
        )
        return popups

    @pytest.fixture
    def tsv(self, tmp_path):
        path = tmp_path / 'other_plate.tsv'
        path.write_text('LumaViewPro Protocol\n')
        return path

    def _refused_load(self, ctx, tsv, error):
        ctx.session.open_protocol.side_effect = error
        # The scope's own load is not the GUI's to call: a protocol loaded
        # there is adopted without the Session putting the scope on its plate.
        ctx.scope.protocols.load_protocol.side_effect = AssertionError('loaded past the Session')
        previous = _protocol()
        panel = _Panel(previous)

        loaded = panel.load_protocol(filepath=str(tsv), navigate=True)

        assert loaded is False
        assert panel._protocol is previous
        assert ctx.settings['protocol']['filepath'] == 'plate.tsv'
        assert panel.ids['protocol_filename'].text == 'plate.tsv'
        assert panel.moves == []

    def test_a_plate_refused_under_a_recording_is_shown_once_and_adopts_nothing(
        self, ctx, shown, own_popups, tsv
    ):
        refusal = HardwareCommandRefusedError(
            'exclusive_activity_running', 'select_labware', 'recording'
        )

        self._refused_load(ctx, tsv, refusal)

        assert own_popups == [], 'the outcome is shown by the one reporter'
        assert [n.title for n in shown] == [refusal.title]

    def test_a_file_that_cannot_be_read_says_so(self, ctx, shown, own_popups, tsv):
        error = ProtocolNotLoadedError(file=tsv, cause=PermissionError(13, 'Permission denied'))

        self._refused_load(ctx, tsv, error)

        assert own_popups == [], 'the outcome is shown by the one reporter'
        assert [(n.title, n.message) for n in shown] == [('Protocol Not Loaded', str(error))], (
            'an unreadable file was a silent no-op'
        )

    def test_a_file_that_is_not_a_protocol_is_shown_in_its_words(self, ctx, shown, own_popups, tsv):
        error = ProtocolFormatError('Not a valid LumaViewPro Protocol', file=tsv)

        self._refused_load(ctx, tsv, error)

        assert own_popups == [], 'the outcome is shown by the one reporter'
        assert [(n.title, n.message) for n in shown] == [('Protocol Refused', str(error))]

    def test_a_failure_drawing_the_panel_is_reported_and_leaves_the_sessions_protocol(
        self, ctx, shown, own_popups, tsv
    ):
        """The file dialog's callback has no reporter of its own: a failure
        drawing the panel after the Session opened the file was raised out of
        it. The panel's protocol is then the one the Session opened, whose
        plate, layer settings and path are already in place, not the one it
        had beside a path naming another file."""
        opened = _protocol()
        ctx.session.open_protocol.return_value = opened
        ctx.stage.set_protocol_steps.side_effect = RuntimeError('the stage could not draw')
        panel = _Panel(_protocol())
        panel.ids['labware_spinner'] = SimpleNamespace(text='')

        loaded = panel.load_protocol(filepath=str(tsv), navigate=True)

        assert loaded is False
        assert panel._protocol is opened
        ctx.session.open_protocol.assert_called_once_with(str(tsv))
        assert ctx.settings['protocol']['filepath'] == 'plate.tsv', 'the panel writes no path'
        [outcome] = shown
        assert (outcome.category, outcome.kind.value) == ('UI:LOAD_PROTOCOL', 'fault')

    @pytest.mark.parametrize('shown_plate', ['6 well microplate', '96 well microplate'])
    def test_an_adoption_draws_the_stage_and_declares_only_a_spinner_write_that_dispatches(
        self, ctx, shown, tsv, monkeypatch, shown_plate
    ):
        """The stage redraws only on XY motion outside a run, so the adoption
        draws the new steps itself. The spinner's write is declared to the GUI
        log only when the text changes: an equal assignment dispatches no
        record, and a declaration left pending would swallow the person's
        next pick of that plate."""
        opened = _protocol()
        opened.labware.return_value = '6 well microplate'
        ctx.session.open_protocol.return_value = opened
        monkeypatch.setattr(ps, 'reset_acquire_ui', lambda: None)
        monkeypatch.setattr(ps, 'reset_stim_ui', lambda: None)
        monkeypatch.setattr(ps.gui_logger, '_write_backs', {})
        panel = _Panel(_protocol())
        panel.ids['labware_spinner'] = SimpleNamespace(text=shown_plate)

        assert panel.load_protocol(filepath=str(tsv), navigate=True) is True

        assert panel.ids['labware_spinner'].text == '6 well microplate'
        ctx.stage.full_redraw.assert_called_once_with()
        declared = ps.gui_logger.consume_write_back('LABWARE', '6 well microplate')
        assert declared is (shown_plate != '6 well microplate')

    def test_the_adoption_only_draws(self):
        """Every write a Load makes is the Session's (``open_protocol``); the
        adoption that follows draws the panel and writes nothing a failure
        partway through it could leave half-done."""
        fn = find_def('ui/protocol_settings.py', '_adopt_protocol', class_name='ProtocolSettings')
        src = ast.unparse(fn)

        for write in ('session.', 'set_protocol_filepath', 'select_labware(', 'update_settings'):
            assert write not in src, f'the adoption calls {write!r}'


class TestTheStartupLoad:
    def test_a_refusal_already_reported_is_not_logged_again(
        self, ctx, shown, tmp_path, caplog, monkeypatch
    ):
        from modules.notification_center import notifications

        refusal = _refusal()
        # The API's funnel reports a refusal as it raises it.
        notifications.report_outcome(refusal, solicited=True, category='Protocol')
        shown.clear()
        caplog.clear()
        saved = tmp_path / 'saved.tsv'
        saved.write_text('LumaViewPro Protocol\n')
        ctx.settings['protocol']['filepath'] = str(saved)
        # The Session keeps a refused file's path (open_remembered_protocol's
        # own tests); the panel shows the refusal's outcome and writes nothing.
        ctx.session.open_remembered_protocol.side_effect = refusal
        monkeypatch.setattr(ps.ProtocolSettings, 'update_step_ui', lambda self: None)
        panel = _Panel(_protocol())

        with caplog.at_level(logging.DEBUG):
            panel.load_persisted_protocol()

        assert shown == [], 'nobody asked for the startup load'
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
        assert ctx.settings['protocol']['filepath'] == str(saved), 'the panel writes no path'
