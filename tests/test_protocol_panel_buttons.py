# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The protocol panel's three run buttons hand a press to the engine and draw its answer.

Scan, Full Protocol and Autofocus Scan decide nothing. A press is a Stop
only when the engine says the run this button started is still live;
otherwise it is a start, handed to the boundary with every panel value
read first. Every refusal is the engine's, shown once by the one
reporter; a fault is one error. One redraw is the only code that styles
the three buttons, from four states the API reports: running, stopping,
writing the finished run's files, idle.
"""

import sys
import types
from datetime import timedelta
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
import ui.ui_helpers as ui_helpers
from modules.exceptions import ProtocolRunRefusedError, RunAlreadyEndedError
from modules.run_outcome import PendingRunOutcome
from modules.protocol_image_writer import RunWriteBatch
from modules.sequenced_capture_runner import RunHandle
from tests.pool_fakes import run_task_now


class _Button:
    def __init__(self, text):
        self.state = 'normal'
        self.text = text
        self.background_down = ''


class _Panel(ps.ProtocolSettings):
    """The real class, with only the widget tree stubbed."""

    def __init__(self):
        self.ids = {
            'run_scan_btn': _Button('Run One Scan'),
            'run_protocol_btn': _Button('Run Full Protocol'),
            'run_autofocus_btn': _Button('Autofocus All Steps'),
            'protocol_filename': SimpleNamespace(text='plate'),
        }
        self._protocol = MagicMock()
        self._runs_started_here = {}
        self._drain_tick_trigger = MagicMock()
        self.scan_pending = False
        self.protocol_pending = False
        self.autofocus_scan_pending = False


@pytest.fixture
def engine():
    e = MagicMock()
    e._is_live_run.return_value = False
    e._is_stopping.return_value = False
    e._last_run.return_value = None
    e.run_dir.return_value = None
    # A handle's progress: the engine's reading while the handle is live.
    e._live_run_value.side_effect = lambda run, read: read() if e._is_live_run(run) else None
    return e


@pytest.fixture
def session(engine):
    # The three buttons run through ProtocolRunner, whose run is the
    # engine's prepare() then start(); this member does the same with the
    # engine stand-in, so one stand-in answers all three buttons.
    member = MagicMock()
    for name in ('run_single_scan', 'run_protocol', 'run_autofocus_all_steps'):
        getattr(member, name).side_effect = lambda protocol, **kw: engine.start(
            engine.prepare(protocol=protocol, **kw)
        )
    member.run_dir.side_effect = engine.run_dir
    return SimpleNamespace(
        protocol_files_draining=False,
        is_protocol_running=False,
        protocol_files_stalled=False,
        protocol_files_pending=0,
        # Nothing holds the scope.
        held_by_other=lambda run: False,
        create_protocol_runner=lambda: member,
    )


@pytest.fixture
def held():
    """Pool tasks, when a test holds them instead of running them at once."""
    return []


@pytest.fixture
def app_ctx(engine, session, held, tmp_path, monkeypatch):
    from modules.sequential_io_executor import ENQUEUED

    saved = getattr(_app_ctx, 'ctx', None)
    pool = MagicMock()

    def _put(task):
        if held is not None and held == ['hold']:
            held.append(task)
        else:
            run_task_now(task)
        return ENQUEUED

    pool.put.side_effect = _put
    monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda fn, timeout=0: fn(0))
    monkeypatch.setattr(ps, 'is_image_saving_enabled', lambda: True)
    _app_ctx.ctx = SimpleNamespace(
        session=session,
        settings={'live_folder': str(tmp_path)},
        settings_lock=MagicMock(),
        worker_pool=pool,
        sequenced_capture_runner=engine,
    )
    yield _app_ctx.ctx
    _app_ctx.ctx = saved


def _handle(engine):
    """A run's handle over the engine stand-in, as the engine's start() makes it."""
    return RunHandle(engine, PendingRunOutcome(), RunWriteBatch(MagicMock()))


def _live(engine, *runs):
    engine._is_live_run.side_effect = lambda run: run is not None and any(run is r for r in runs)


@pytest.mark.parametrize(
    'press, member_name, trigger',
    [
        ('run_scan_from_ui', 'run_single_scan', 'scan'),
        ('run_protocol_from_ui', 'run_protocol', 'protocol'),
    ],
)
def test_a_press_starts_its_run_through_the_apis_runner(
    app_ctx, engine, session, press, member_name, trigger
):
    """Run and Scan are the calls a script makes, with what only the panel knows."""
    panel = _Panel()
    handle = _handle(engine)
    engine.start.return_value = handle

    getattr(panel, press)()

    member = getattr(session.create_protocol_runner(), member_name)
    member.assert_called_once()
    kwargs = member.call_args.kwargs
    assert kwargs['run_trigger_source'] == trigger
    assert 'engineering_mode' not in kwargs, "the mode is the session's; the panel states none"
    assert kwargs['sequence_name'] == 'plate', 'the file name is read on the GUI thread'
    assert member.call_args.args[0] is panel._protocol.copy_for_execution.return_value, (
        "the run gets the copy taken at the click; the panel's protocol is the person's"
    )
    assert panel._runs_started_here[trigger] is handle, 'the handle is what its Stop names'
    assert not engine._reset.called, 'a start is not a stop'


def test_a_refused_start_is_shown_once_and_the_button_draws_idle(app_ctx, engine, centre_posts):
    engine.prepare.side_effect = ProtocolRunRefusedError(
        'already_running', 'Already Running', 'Another run is already in progress.'
    )
    panel = _Panel()
    panel.ids['run_scan_btn'].state = 'down'  # Kivy flips the toggle at touch-down

    panel.run_scan_from_ui()

    assert [n.title for n in centre_posts] == ['Already Running']
    assert panel.ids['run_scan_btn'].state == 'normal'
    assert panel.ids['run_scan_btn'].text == 'Run One Scan'
    assert panel.scan_pending is False, 'the button must come back for the next press'
    assert not engine.start.called


def test_an_empty_protocol_is_the_engines_refusal_not_the_panels(app_ctx, engine, centre_posts):
    engine.prepare.side_effect = ProtocolRunRefusedError(
        'empty_protocol', 'No Steps', 'Protocol has no steps.'
    )
    panel = _Panel()
    panel._protocol.num_steps.return_value = 0

    panel.run_protocol_from_ui()

    assert [n.title for n in centre_posts] == ['No Steps'], "one refusal, the engine's words"


def test_an_unexpected_failure_is_one_fault_and_the_button_draws_idle(
    app_ctx, engine, centre_posts
):
    engine.prepare.side_effect = TypeError('bad call')
    panel = _Panel()
    panel.ids['run_autofocus_btn'].state = 'down'

    panel.run_autofocus_scan_from_ui()

    assert len(centre_posts) == 1
    assert panel.ids['run_autofocus_btn'].state == 'normal'
    assert panel.autofocus_scan_pending is False


def test_a_started_run_draws_running(app_ctx, engine):
    panel = _Panel()
    handle = _handle(engine)
    engine.start.return_value = handle
    _live(engine, handle)

    panel.run_scan_from_ui()

    button = panel.ids['run_scan_btn']
    assert (button.state, button.text) == ('down', 'Abort One Scan')


def test_the_full_protocol_button_counts_the_live_runs_own_scans(app_ctx, engine):
    panel = _Panel()
    handle = _handle(engine)
    engine.start.return_value = handle
    engine._remaining_scans.return_value = 3
    engine._protocol_interval.return_value = timedelta(minutes=20)
    _live(engine, handle)

    panel.run_protocol_from_ui()

    assert panel.ids['run_protocol_btn'].text == '3 scans (1h 0m) remaining.\nPress to ABORT'


def test_a_run_that_ended_between_the_reads_draws_idle(app_ctx, engine):
    """Live at the first read, ended by the progress read: the button
    draws what it read -- idle -- not a running label for an ended run."""
    panel = _Panel()
    handle = _handle(engine)
    panel._runs_started_here = {'protocol': handle}
    _live(engine, handle)
    engine._live_run_value.side_effect = lambda run, read: None

    panel.draw_protocol_buttons()

    button = panel.ids['run_protocol_btn']
    assert (button.state, button.text) == ('normal', 'Run Full Protocol')


def test_a_second_press_stops_its_own_run_ahead_of_queued_work(app_ctx, engine):
    from modules.sequential_io_executor import PRIORITY_HIGH

    panel = _Panel()
    handle = _handle(engine)
    panel._runs_started_here['protocol'] = handle
    _live(engine, handle)
    engine._is_stopping.side_effect = lambda run: run is handle and engine._reset.called

    panel.run_protocol_from_ui()

    task = app_ctx.worker_pool.put.call_args.args[0]
    assert task.priority == PRIORITY_HIGH, 'a Stop must not wait behind queued work'
    engine._reset.assert_called_once_with(handle)
    assert not engine.prepare.called, 'a Stop is not a start'
    assert panel.ids['run_protocol_btn'].text == 'Stopping...'


def test_a_stop_that_finds_its_run_already_ended_shows_nothing(app_ctx, engine, centre_posts):
    panel = _Panel()
    handle = _handle(engine)
    panel._runs_started_here['scan'] = handle
    _live(engine, handle)

    def _ended(run):
        _live(engine)  # the run ended between the press and the pool
        raise RunAlreadyEndedError('the run already ended')

    engine._reset.side_effect = _ended

    panel.run_scan_from_ui()

    assert centre_posts == []
    assert panel.ids['run_scan_btn'].state == 'normal'


def test_a_refused_stop_leaves_every_button_showing_its_own_run(app_ctx, engine, centre_posts):
    panel = _Panel()
    mine, theirs = _handle(engine), _handle(engine)
    panel._runs_started_here['scan'] = mine
    panel._runs_started_here['protocol'] = theirs
    _live(engine, mine, theirs)
    engine._reset.side_effect = ProtocolRunRefusedError(
        'run_not_live', 'Not Running', 'That run is not the one running.'
    )

    panel.run_scan_from_ui()

    assert [n.title for n in centre_posts] == ['Not Running']
    assert panel.ids['run_scan_btn'].state == 'down'
    assert panel.ids['run_protocol_btn'].state == 'down'


def test_the_button_is_disabled_while_its_own_request_is_in_flight(app_ctx, held, engine):
    held.append('hold')
    panel = _Panel()

    panel.run_scan_from_ui()

    assert panel.scan_pending is True, 'a second press must not race the first to the pool'
    task = held[1]
    run_task_now(task)
    assert panel.scan_pending is False, "the request's own redraw brings the button back"


def test_a_finished_runs_drain_shows_its_count(app_ctx, engine, session):
    panel = _Panel()
    finished = _handle(engine)
    panel._runs_started_here['protocol'] = finished
    engine._last_run.return_value = finished
    session.protocol_files_draining = True
    session.protocol_files_pending = 7

    panel.draw_protocol_buttons()

    assert panel.ids['run_protocol_btn'].text == 'Writing Files... (7)'
    assert panel.ids['run_scan_btn'].text == 'Run One Scan'
    panel._drain_tick_trigger.assert_called_once()

    session.protocol_files_stalled = True
    panel.draw_protocol_buttons()
    assert panel.ids['run_protocol_btn'].text == 'File writer stalled'

    session.protocol_files_draining = False
    session.protocol_files_stalled = False
    panel.draw_protocol_buttons()
    assert panel.ids['run_protocol_btn'].text == 'Run Full Protocol'


def test_a_live_runs_own_writes_do_not_disable_its_stop(app_ctx, engine, session):
    panel = _Panel()
    handle = _handle(engine)
    panel._runs_started_here['scan'] = handle
    _live(engine, handle)
    session.protocol_files_draining = True
    session.is_protocol_running = True

    panel.draw_protocol_buttons()

    assert panel.ids['run_scan_btn'].state == 'down'


def test_a_stalled_drain_opens_no_offer_of_its_own(app_ctx, session, monkeypatch):
    """The run engine reports a stalled writer, with its recovery; the drain tick only redraws."""
    import ui.notification_popup as notification_popup

    offers = []
    monkeypatch.setattr(
        notification_popup, 'show_confirmation_popup', lambda **kw: offers.append(kw)
    )
    panel = _Panel()
    session.protocol_files_draining = True
    session.protocol_files_stalled = True

    panel._drain_tick(0)
    panel._drain_tick(0)

    assert offers == []


def test_a_scan_between_iterations_redraws_rather_than_drawing_idle(app_ctx, engine):
    panel = _Panel()
    handle = _handle(engine)
    engine.start.return_value = handle
    _live(engine, handle)
    panel.run_scan_from_ui()
    scan_ended = engine.prepare.call_args.kwargs['events'].scan_ended

    scan_ended(1, 1, timedelta(minutes=5))

    button = panel.ids['run_scan_btn']
    assert (button.state, button.text) == ('down', 'Abort One Scan'), (
        'between scans the run is still live; the button shows it, not idle'
    )


def test_each_panel_press_is_recorded_as_what_it_did(app_ctx, engine, monkeypatch):
    """A Stop press is recorded as a Stop: the interaction log is what a
    support bundle reads to learn what the person did."""
    recorded = []
    monkeypatch.setattr(
        ps.gui_logger, 'protocol_action', lambda action, *a: recorded.append(action)
    )
    panel = _Panel()
    handle = _handle(engine)
    engine.start.return_value = handle

    panel.run_autofocus_scan_from_ui()
    assert recorded == ['AF_SCAN'], 'a start is not a stop'

    recorded.clear()
    _live(engine, handle)
    panel.run_autofocus_scan_from_ui()
    assert recorded == ['AF_SCAN', 'ABORT_AF_SCAN']


def test_a_press_during_a_finished_runs_drain_is_the_engines_to_refuse(
    app_ctx, engine, session, centre_posts
):
    """A finished run's files draining is no reason for the panel to say no:
    the press reaches the engine, whose refusal every client gets, and it is
    shown once."""
    session.protocol_files_draining = True
    session.protocol_files_pending = 3
    engine.prepare.side_effect = ProtocolRunRefusedError(
        'files_writing', 'Files Still Writing', 'The last run is still writing its files.'
    )
    panel = _Panel()

    panel.run_scan_from_ui()

    assert engine.prepare.called, 'the press never reached the engine'
    assert [n.title for n in centre_posts] == ['Files Still Writing']
    assert not engine.start.called


def test_each_button_greys_while_anything_else_holds_the_scope(app_ctx, session):
    """Each of the three asks the Session about the run it started: another
    holder greys it, its own run leaves it live as that run's Stop."""
    panel = _Panel()
    engine = app_ctx.sequenced_capture_runner
    runs = {trigger: _handle(engine) for trigger in ('scan', 'protocol', 'autofocus_scan')}
    panel._runs_started_here = dict(runs)
    own_run = runs['protocol']
    session.held_by_other = lambda run: run is not own_run

    panel.draw_protocol_buttons()

    assert (panel.scan_held, panel.protocol_held, panel.autofocus_scan_held) == (
        True,
        False,
        True,
    )
