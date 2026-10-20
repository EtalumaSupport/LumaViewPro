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
from modules import live_work
from modules.exceptions import ProtocolRunRefusedError
from modules.live_work import LiveWork, WorkItem
from modules.run_outcome import PendingRunOutcome
from modules.protocol_image_writer import RunWriteBatch
from modules.sequenced_capture_runner import EngineRun, RunProgress
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
    e.run_dir.return_value = None
    _live(e)
    return e


@pytest.fixture
def session(engine):
    # The three buttons run through ProtocolRunner, whose run is the
    # engine's prepare() then start(); this member does the same with the
    # engine stand-in, so one stand-in answers all three buttons.
    member = MagicMock()
    for name in ('start_single_scan', 'start_protocol', 'start_autofocus_all_steps'):
        getattr(member, name).side_effect = lambda protocol, **kw: engine.start(
            engine.prepare(protocol=protocol, **kw)
        )
    member.run_dir.side_effect = engine.run_dir
    return _Session(member)


class _Session(SimpleNamespace):
    """The session the panel reads: a finished run's drain as the tests set
    it, answered in the live work as the Session's ``RUN_FILES`` item."""

    def __init__(self, member):
        super().__init__(
            protocol_files_draining=False,
            is_protocol_running=False,
            protocol_files_stalled=False,
            protocol_files_pending=0,
            # Nothing holds the scope.
            held_by_other=lambda run: False,
            create_protocol_runner=lambda: member,
        )

    @property
    def live_work(self):
        files = (
            (WorkItem(live_work.RUN_FILES, "A run's images", left=self.protocol_files_pending),)
            if self.protocol_files_draining
            else ()
        )
        return LiveWork(work=files, closing=False, closed=False)


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


def _run(engine):
    """A run over the engine stand-in, as start() makes it; a button holds its ``operation``."""
    return EngineRun(engine, PendingRunOutcome(), RunWriteBatch(MagicMock()))


def _live(
    engine,
    *runs,
    stopping=lambda: False,
    last=None,
    remaining_scans=None,
    interval=None,
):
    """The engine's progress record of each run: live for *runs*, ended for every other.

    *stopping* is asked at each read, so a Stop the test makes shows on the
    next redraw; *last* is the engine's most recent run, live or finished.
    """

    def _progress_of(run):
        live = any(run is r for r in runs)
        return RunProgress(
            is_live=live,
            is_stopping=live and stopping(),
            is_last_run=run is last,
            run_dir=None,
            step_number=1 if live else None,
            num_steps=1 if live else None,
            remaining_scans=remaining_scans if live else None,
            interval=interval if live else None,
        )

    engine._progress_of.side_effect = _progress_of


@pytest.mark.parametrize(
    'press, member_name, trigger',
    [
        ('run_scan_from_ui', 'start_single_scan', 'scan'),
        ('run_protocol_from_ui', 'start_protocol', 'protocol'),
    ],
)
def test_a_press_starts_its_run_through_the_apis_runner(
    app_ctx, engine, session, press, member_name, trigger
):
    """Run and Scan are the calls a script makes, with what only the panel knows."""
    panel = _Panel()
    run = _run(engine)
    engine.start.return_value = run.operation

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
    assert panel._runs_started_here[trigger] is run.operation, (
        'the operation is what its Stop names'
    )
    assert not engine._stop_run.called, 'a start is not a stop'


def test_a_refused_start_is_shown_once_and_the_button_draws_idle(app_ctx, engine, centre_posts):
    engine.prepare.side_effect = ProtocolRunRefusedError(
        'already_running', 'Another run is already in progress.'
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
    engine.prepare.side_effect = ProtocolRunRefusedError('empty_protocol', 'Protocol has no steps.')
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
    run = _run(engine)
    engine.start.return_value = run.operation
    _live(engine, run)

    panel.run_scan_from_ui()

    button = panel.ids['run_scan_btn']
    assert (button.state, button.text) == ('down', 'Abort One Scan')


def test_the_full_protocol_button_counts_the_live_runs_own_scans(app_ctx, engine):
    panel = _Panel()
    run = _run(engine)
    engine.start.return_value = run.operation
    _live(engine, run, remaining_scans=3, interval=timedelta(minutes=20))

    panel.run_protocol_from_ui()

    assert panel.ids['run_protocol_btn'].text == '3 scans (1h 0m) remaining.\nPress to ABORT'


def test_a_second_press_stops_its_own_run_ahead_of_queued_work(app_ctx, engine):
    from modules.sequential_io_executor import PRIORITY_HIGH

    panel = _Panel()
    run = _run(engine)
    panel._runs_started_here['protocol'] = run.operation
    _live(engine, run, stopping=lambda: engine._stop_run.called)

    panel.run_protocol_from_ui()

    task = app_ctx.worker_pool.put.call_args.args[0]
    assert task.priority == PRIORITY_HIGH, 'a Stop must not wait behind queued work'
    engine._stop_run.assert_called_once_with(run)
    assert not engine.prepare.called, 'a Stop is not a start'
    assert panel.ids['run_protocol_btn'].text == 'Stopping...'


def test_a_stop_that_finds_its_run_already_ended_shows_nothing(app_ctx, held, engine, centre_posts):
    # The press reads its run live and hands a Stop to the pool; the run
    # ends before the pool runs it. A Stop of an ended run does nothing and
    # raises nothing, so there is nothing to show and nothing is stopped.
    held.append('hold')
    panel = _Panel()
    run = _run(engine)
    panel._runs_started_here['scan'] = run.operation
    _live(engine, run)

    panel.run_scan_from_ui()
    _live(engine)
    run.operation.set_result(None)
    run_task_now(held[1])

    assert centre_posts == []
    assert not engine._stop_run.called, 'a Stop of an ended run reached the engine'
    assert panel.ids['run_scan_btn'].state == 'normal'


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
    finished = _run(engine)
    panel._runs_started_here['protocol'] = finished.operation
    _live(engine, last=finished)
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


def test_each_button_records_what_it_shows_and_whether_it_is_held(app_ctx, engine, session, caplog):
    """The GUI record carries each button's words and its greying, once per stage.

    A held button is grey whatever the controls lock says: another activity
    -- a recording's files still being written among them -- holds the scope.
    The drain's count is progress, so a falling count is one line.
    """
    import logging

    from modules import gui_logger

    gui_logger._shown.clear()
    caplog.set_level(logging.INFO, logger='LVP.gui_interactions')
    panel = _Panel()
    finished = _run(engine)
    panel._runs_started_here['protocol'] = finished.operation
    _live(engine, last=finished)
    session.held_by_other = lambda run: run is None
    session.protocol_files_draining = True

    for pending in (12, 7, 3):
        session.protocol_files_pending = pending
        panel.draw_protocol_buttons()

    lines = [r.getMessage() for r in caplog.records if r.name == 'LVP.gui_interactions']
    gui_logger._shown.clear()
    assert lines == [
        'DISPLAY RUN_BUTTON_SCAN Run One Scan held=True',
        'DISPLAY RUN_BUTTON_PROTOCOL Writing Files... (12) held=False',
        'DISPLAY RUN_BUTTON_AUTOFOCUS_SCAN Autofocus All Steps held=True',
    ]


def test_a_live_runs_own_writes_do_not_disable_its_stop(app_ctx, engine, session):
    panel = _Panel()
    run = _run(engine)
    panel._runs_started_here['scan'] = run.operation
    _live(engine, run)
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
    run = _run(engine)
    engine.start.return_value = run.operation
    _live(engine, run)
    panel.run_scan_from_ui()
    scan_ended = engine.prepare.call_args.kwargs['events'].scan_ended

    scan_ended(1, 1, timedelta(minutes=5))

    button = panel.ids['run_scan_btn']
    assert (button.state, button.text) == ('down', 'Abort One Scan'), (
        'between scans the run is still live; the button shows it, not idle'
    )


# Each panel run button: its press handler, and what a start and a stop record.
_PRESSES = (
    ('run_scan_from_ui', 'SCAN', 'ABORT_SCAN'),
    ('run_protocol_from_ui', 'RUN', 'ABORT_PROTOCOL'),
    ('run_autofocus_scan_from_ui', 'AF_SCAN', 'ABORT_AF_SCAN'),
)


def test_each_panel_press_is_recorded_once_as_what_it_did(app_ctx, engine, monkeypatch):
    """The interaction log is what a support bundle reads to learn what the
    person did: a Stop press is recorded as a Stop and nothing else, a start
    as a start, and a press the panel dropped as not started."""
    recorded = []
    monkeypatch.setattr(
        ps.gui_logger,
        'protocol_action',
        lambda action, detail='': recorded.append(f'{action} {detail}'.strip()),
    )
    wrong = {}
    for press, start, stop in _PRESSES:
        panel = _Panel()
        run = _run(engine)
        engine.start.return_value = run.operation

        recorded.clear()
        getattr(panel, press)()
        started = list(recorded)

        recorded.clear()
        _live(engine, run)
        getattr(panel, press)()
        stopped = list(recorded)

        # a stand-in by design: the frame test that makes a real refused edit
        # is tests/test_a_click_off_a_refused_edit_does_not_act.py; here the
        # subject is what the panel records for the press it drops.
        monkeypatch.setattr(ps, 'refused_in_this_input', lambda: True)
        recorded.clear()
        getattr(_Panel(), press)()
        dropped = list(recorded)
        monkeypatch.setattr(ps, 'refused_in_this_input', lambda: False)

        got = (started, stopped, dropped)
        want = ([start], [stop], [f'{start} not started: the press left a refused edit'])
        if got != want:
            wrong[press] = got
    assert wrong == {}


def test_a_press_during_a_finished_runs_drain_is_the_engines_to_refuse(
    app_ctx, engine, session, centre_posts
):
    """A finished run's files draining is no reason for the panel to say no:
    the press reaches the engine, whose refusal every client gets, and it is
    shown once."""
    session.protocol_files_draining = True
    session.protocol_files_pending = 3
    engine.prepare.side_effect = ProtocolRunRefusedError(
        'files_writing', 'The last run is still writing its files.'
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
    runs = {trigger: _run(engine).operation for trigger in ('scan', 'protocol', 'autofocus_scan')}
    panel._runs_started_here = dict(runs)
    own_run = runs['protocol']
    session.held_by_other = lambda run: run is not own_run

    panel.draw_protocol_buttons()

    assert (panel.scan_held, panel.protocol_held, panel.autofocus_scan_held) == (
        True,
        False,
        True,
    )
