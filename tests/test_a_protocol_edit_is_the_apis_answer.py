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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import modules.app_context as _app_ctx
import ui.protocol_settings as ps
from modules.acquire_mode import AcquireMode
from modules.caller import Caller, calling
from tests.ast_seams import find_def
from tests.scope_fakes import home_sim_scope


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

    def go_to_step(self, step_index, protocol=True):
        self.moves.append(step_index)

    def update_step_ui(self):
        self.redraws += 1


@pytest.fixture
def ctx(gui_ctx, monkeypatch):
    """The GUI's context over an unhomed simulated session, remembering ``plate.tsv``.

    The stage is the test's own widget: the Kivy tree never runs here.
    """
    gui_ctx.stage = MagicMock()
    gui_ctx.set_protocol_filepath('plate.tsv')
    monkeypatch.setattr(ps.gui_logger, 'protocol_action', lambda *a, **kw: None)
    monkeypatch.setattr(ps.gui_logger, 'button', lambda *a, **kw: None)
    return gui_ctx


@pytest.fixture
def homed(ctx):
    """``ctx`` homed, with Brightfield acquiring: a scope the Session lets build and edit."""
    home_sim_scope(ctx.scope)
    ctx.session.set_channel_acquire('BF', AcquireMode.IMAGE)
    return ctx


@pytest.fixture
def tsv(homed, tmp_path):
    """A protocol file the Session wrote on the 6 well plate, the scope back on the 96."""
    homed.session.select_labware('6 well microplate')
    path = tmp_path / 'other_plate.tsv'
    homed.session.save_protocol(homed.session.new_protocol(), path)
    homed.session.select_labware('96 well microplate')
    return path


def _titles(centre_posts):
    return [n.title for n in centre_posts]


class TestAddStep:
    def test_an_accepted_step_is_gone_to_last(self, homed):
        protocol = homed.session.new_protocol()
        before = protocol.num_steps()
        panel = _Panel(protocol)

        panel.insert_step(after_current_step=True)

        assert protocol.num_steps() == before + 1
        assert panel.curr_step == 1
        assert panel.moves == [1]
        homed.stage.set_protocol_steps.assert_called_once_with(protocol)

    def test_a_refused_step_is_shown_once_and_not_navigated_to(self, ctx, centre_posts):
        # Unhomed, the scope does not know where its axes stand, so the
        # Session refuses the step (step_position_unknown).
        protocol = ctx.session.create_empty_protocol()
        panel = _Panel(protocol)

        panel.insert_step(after_current_step=True)

        assert _titles(centre_posts) == ['Step Not Saved']
        assert protocol.num_steps() == 0
        assert panel.moves == [], 'a refused step has nowhere to go'
        assert panel.curr_step == 0
        assert panel.redraws == 1, 'the panel still shows its protocol as the API has it'


class TestModifyStep:
    def test_the_name_field_and_the_shown_layer_reach_the_api(self, homed):
        protocol = homed.session.new_protocol()
        panel = _Panel(protocol)
        panel.ids['step_name_input'].text = 'my_step'

        panel.modify_step()

        row = protocol.step_row(step_index=0)
        assert row['Color'] == homed.session.shown_channel.channel
        assert 'my_step' in row['Name']

    def test_a_refused_modify_is_shown_once(self, homed, centre_posts):
        protocol = homed.session.new_protocol()
        named = protocol.step_row(step_index=0)['Name']
        panel = _Panel(protocol)
        panel.ids['step_name_input'].text = 'my_step'
        with calling(Caller.REST):
            homed.session.take_control('plate reader')

        panel.modify_step()

        assert _titles(centre_posts) == ['Under Remote Control']
        assert protocol.step_row(step_index=0)['Name'] == named
        assert panel.redraws == 1


class TestNewProtocol:
    def test_an_accepted_protocol_is_adopted_and_has_no_file(self, homed):
        previous = homed.session.create_empty_protocol()
        panel = _Panel(previous)

        panel.new_protocol()

        assert panel._protocol is not previous
        assert panel._protocol.num_steps() > 0
        assert homed.settings['protocol']['filepath'] == ''
        assert (panel.ids['protocol_filename'].text, panel.ids['capture_root'].text) == ('', '')
        assert panel.moves == [0]

    def test_a_refused_build_is_shown_once_and_adopts_nothing(self, ctx, centre_posts):
        # No layer acquires on the template, so the Session refuses the build.
        previous = ctx.session.create_empty_protocol()
        panel = _Panel(previous)

        panel.new_protocol()

        assert len(centre_posts) == 1, _titles(centre_posts)
        assert panel._protocol is previous
        assert ctx.settings['protocol']['filepath'] == 'plate.tsv'


class TestSave:
    @pytest.fixture(autouse=True)
    def own_popups(self, monkeypatch):
        popups = []
        monkeypatch.setattr(
            'ui.notification_popup.show_notification_popup',
            lambda **kw: popups.append(kw['title']),
        )
        return popups

    def test_a_saved_protocol_is_remembered_under_its_new_name(self, ctx, own_popups, tmp_path):
        panel = _Panel(ctx.session.create_empty_protocol())
        target = tmp_path / 'other.tsv'

        panel.save_protocol(filepath=str(target))

        assert target.read_text().startswith('LumaViewPro Protocol')
        assert ctx.settings['protocol']['filepath'] == str(target)
        assert panel.ids['protocol_filename'].text == 'other.tsv'
        assert own_popups == []

    def test_a_failed_save_is_shown_once_and_the_previous_name_stays(
        self, ctx, centre_posts, own_popups, tmp_path
    ):
        panel = _Panel(ctx.session.create_empty_protocol())
        locked = tmp_path / 'locked'
        locked.mkdir()
        locked.chmod(0o500)
        try:
            panel.save_protocol(filepath=str(locked / 'other.tsv'))
        finally:
            locked.chmod(0o700)

        assert own_popups == [], 'the outcome is shown by the one reporter'
        assert _titles(centre_posts) == ['Protocol Not Saved']
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

    def _refused_load(self, ctx, path):
        previous = ctx.session.create_empty_protocol()
        panel = _Panel(previous)

        loaded = panel.load_protocol(filepath=str(path), navigate=True)

        assert loaded is False
        assert panel._protocol is previous
        assert ctx.settings['protocol']['filepath'] == 'plate.tsv'
        assert panel.ids['protocol_filename'].text == 'plate.tsv'
        assert panel.moves == []

    def test_a_plate_refused_under_a_recording_is_shown_once_and_adopts_nothing(
        self, homed, centre_posts, own_popups, tsv
    ):
        recording = homed.session.activity_claim.try_claim('recording')
        try:
            self._refused_load(homed, tsv)
        finally:
            recording.release()

        assert own_popups == [], 'the outcome is shown by the one reporter'
        assert [(n.title, n.reason) for n in centre_posts] == [
            ('Microscope Busy', 'exclusive_activity_running')
        ]

    def test_a_file_that_cannot_be_read_says_so(self, homed, centre_posts, own_popups, tsv):
        tsv.chmod(0o000)
        try:
            self._refused_load(homed, tsv)
        finally:
            tsv.chmod(0o600)

        assert own_popups == [], 'the outcome is shown by the one reporter'
        assert _titles(centre_posts) == ['Protocol Not Loaded'], (
            'an unreadable file was a silent no-op'
        )

    def test_a_file_that_is_not_a_protocol_is_shown_in_its_words(
        self, ctx, centre_posts, own_popups, tmp_path
    ):
        path = tmp_path / 'not_a_protocol.tsv'
        path.write_text('LumaViewPro Protocol\n')

        self._refused_load(ctx, path)

        assert own_popups == [], 'the outcome is shown by the one reporter'
        assert _titles(centre_posts) == ['Protocol Refused']

    def test_a_failure_drawing_the_panel_is_reported_and_leaves_the_sessions_protocol(
        self, homed, centre_posts, own_popups, tsv
    ):
        """The file dialog's callback has no reporter of its own: a failure
        drawing the panel after the Session opened the file was raised out of
        it. The panel's protocol is then the one the Session opened, whose
        plate, layer settings and path are already in place, not the one it
        had beside a path naming another file."""
        homed.stage.set_protocol_steps.side_effect = RuntimeError('the stage could not draw')
        previous = homed.session.create_empty_protocol()
        panel = _Panel(previous)
        panel.ids['labware_spinner'] = SimpleNamespace(text='')

        loaded = panel.load_protocol(filepath=str(tsv), navigate=True)

        assert loaded is False
        assert panel._protocol is not previous
        assert panel._protocol.labware_id() == '6 well microplate'
        assert homed.settings['protocol']['filepath'] == str(tsv), 'the Session remembered it'
        [outcome] = centre_posts
        assert (outcome.category, outcome.kind.value) == ('UI:LOAD_PROTOCOL', 'fault')

    @pytest.mark.parametrize('shown_plate', ['6 well microplate', '96 well microplate'])
    def test_an_adoption_draws_the_stage_and_its_spinner_write_is_not_a_pick(
        self, homed, centre_posts, tsv, monkeypatch, caplog, shown_plate
    ):
        """The stage redraws only on XY motion outside a run, so the adoption
        draws the new steps itself. The spinner showing the protocol's plate
        is the app's write, and records no pick: a person's pick is recorded
        by the spinner itself, from the pick."""
        monkeypatch.setattr(ps, 'reset_acquire_ui', lambda: None)
        monkeypatch.setattr(ps, 'reset_stim_ui', lambda: None)
        panel = _Panel(homed.session.create_empty_protocol())
        panel.ids['labware_spinner'] = SimpleNamespace(text=shown_plate)

        with caplog.at_level(logging.INFO, logger='LVP.gui_interactions'):
            assert panel.load_protocol(filepath=str(tsv), navigate=True) is True

        assert panel.ids['labware_spinner'].text == '6 well microplate'
        homed.stage.full_redraw.assert_called_once_with()
        recorded = [r.getMessage() for r in caplog.records if r.name == 'LVP.gui_interactions']
        assert not [line for line in recorded if line.startswith('SELECT')], recorded

    def test_the_adoption_only_draws(self):
        """Every write a Load makes is the Session's (``open_protocol``); the
        adoption that follows draws the panel and writes nothing a failure
        partway through it could leave half-done."""
        fn = find_def('ui/protocol_settings.py', '_adopt_protocol', class_name='ProtocolSettings')
        src = ast.unparse(fn)

        for write in ('session.', 'set_protocol_filepath', 'select_labware(', 'update_settings'):
            assert write not in src, f'the adoption calls {write!r}'


class TestTheStartupLoad:
    def test_a_refused_startup_load_is_shown_once_and_keeps_its_path(
        self, homed, centre_posts, tsv, caplog, monkeypatch
    ):
        """Nobody asked for the startup load, but its refusal is shown: the
        panel shows only the kept file's name, and this is where the person
        learns why it did not load. The Session keeps a refused file's path
        (open_remembered_protocol's own tests); the panel writes none."""
        homed.set_protocol_filepath(str(tsv))
        recording = homed.session.activity_claim.try_claim('recording')
        monkeypatch.setattr(ps.ProtocolSettings, 'update_step_ui', lambda self: None)
        panel = _Panel(homed.session.create_empty_protocol())
        centre_posts.clear()
        caplog.clear()
        try:
            with caplog.at_level(logging.DEBUG):
                panel.load_persisted_protocol()
        finally:
            recording.release()

        assert [(n.category, n.reason) for n in centre_posts] == [
            ('UI:LOAD_PROTOCOL', 'exclusive_activity_running')
        ]
        assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1
        assert panel.ids['protocol_filename'].text == tsv.name
        assert homed.settings['protocol']['filepath'] == str(tsv), 'the panel writes no path'


class TestCaptureRoot:
    """The capture root's field is an edit the API may refuse, shown once like any other."""

    @pytest.fixture
    def held(self, tmp_path, monkeypatch):
        """A simulated session whose control a REST client holds, as the GUI's session."""
        from modules.caller import Caller, calling
        from modules.scope_session import ScopeSession
        from tests.settings_fixtures import complete_settings

        session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
        try:
            with calling(Caller.REST):
                session.take_control('plate reader')
            monkeypatch.setattr(_app_ctx, 'ctx', SimpleNamespace(session=session))
            yield session
        finally:
            session.shutdown()

    def test_a_root_refused_under_another_clients_control_is_shown_once_and_the_field_keeps_the_protocols(
        self, held, centre_posts
    ):
        protocol = held.create_empty_protocol()
        kept = protocol.capture_root()
        panel = _Panel(protocol)
        panel.ids['capture_root'].text = 'typed'

        panel.update_capture_root('typed')

        assert [n.title for n in centre_posts] == ['Under Remote Control']
        assert protocol.capture_root() == kept
        assert panel.ids['capture_root'].text == kept, 'the field shows the root the protocol kept'
