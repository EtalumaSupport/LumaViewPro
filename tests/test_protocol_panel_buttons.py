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
        self.files_draining = False


@pytest.fixture
def engine():
    e = MagicMock()
    e.is_live_run.return_value = False
    e.is_stopping.return_value = False
    e.run_outcome.return_value = None
    e.run_dir.return_value = None
    return e


@pytest.fixture
def session():
    return SimpleNamespace(
        protocol_files_draining=False,
        is_protocol_running=False,
        protocol_files_stalled=False,
        protocol_files_pending=0,
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
            task.action(*task.args, **task.kwargs)
        return ENQUEUED

    pool.put.side_effect = _put
    monkeypatch.setattr(ui_helpers, '_schedule_ui', lambda fn, timeout=0: fn(0))
    for name, value in (
        ('require_file_writes_idle', lambda operation: True),
        ('get_image_capture_config_from_ui', lambda: {}),
        ('get_auto_gain_settings', lambda: {}),
        ('is_image_saving_enabled', lambda: True),
        ('live_display_callbacks', lambda: {}),
        (
            'get_protocol_time_params',
            lambda: {'period': timedelta(minutes=5), 'duration': timedelta(hours=1)},
        ),
    ):
        monkeypatch.setattr(ps, name, value)
    monkeypatch.setattr(ps.config_helpers, 'autofocus_snapshot_from_settings', lambda *a: {})
    monkeypatch.setattr(ps.config_helpers, 'get_sequenced_run_settings', lambda *a, **k: {})
    _app_ctx.ctx = SimpleNamespace(
        session=session,
        settings={'live_folder': str(tmp_path)},
        settings_lock=MagicMock(),
        worker_pool=pool,
        sequenced_capture_runner=engine,
        engineering_mode=False,
    )
    yield _app_ctx.ctx
    _app_ctx.ctx = saved


@pytest.fixture
def shown(monkeypatch):
    from tests.shown_outcomes import capture_shown

    return capture_shown(monkeypatch)


def _live(engine, *runs):
    engine.is_live_run.side_effect = lambda run: run is not None and any(run is r for r in runs)


def test_a_press_hands_the_start_to_the_engine_with_the_panels_inputs(app_ctx, engine):
    panel = _Panel()
    handle = PendingRunOutcome()
    engine.start.return_value = handle

    panel.run_scan_from_ui()

    kwargs = engine.prepare.call_args.kwargs
    assert kwargs['run_trigger_source'] == 'scan'
    assert kwargs['sequence_name'] == 'plate', 'the file name is read on the GUI thread'
    assert kwargs['protocol'] is panel._protocol.copy_for_execution.return_value, (
        "the run gets its own copy; the panel's protocol is the person's"
    )
    assert panel._runs_started_here['scan'] is handle, 'the handle is what its Stop names'
    assert not engine.reset.called, 'a start is not a stop'


def test_a_refused_start_is_shown_once_and_the_button_draws_idle(app_ctx, engine, shown):
    engine.prepare.side_effect = ProtocolRunRefusedError(
        'already_running', 'Already Running', 'Another run is already in progress.'
    )
    panel = _Panel()
    panel.ids['run_scan_btn'].state = 'down'  # Kivy flips the toggle at touch-down

    panel.run_scan_from_ui()

    assert [n.title for n in shown] == ['Already Running']
    assert panel.ids['run_scan_btn'].state == 'normal'
    assert panel.ids['run_scan_btn'].text == 'Run One Scan'
    assert panel.scan_pending is False, 'the button must come back for the next press'
    assert not engine.start.called


def test_an_empty_protocol_is_the_engines_refusal_not_the_panels(app_ctx, engine, shown):
    engine.prepare.side_effect = ProtocolRunRefusedError(
        'empty_protocol', 'No Steps', 'Protocol has no steps.'
    )
    panel = _Panel()
    panel._protocol.num_steps.return_value = 0

    panel.run_protocol_from_ui()

    assert [n.title for n in shown] == ['No Steps'], "one refusal, the engine's words"


def test_an_unexpected_failure_is_one_fault_and_the_button_draws_idle(app_ctx, engine, shown):
    engine.prepare.side_effect = TypeError('bad call')
    panel = _Panel()
    panel.ids['run_autofocus_btn'].state = 'down'

    panel.run_autofocus_scan_from_ui()

    assert len(shown) == 1
    assert panel.ids['run_autofocus_btn'].state == 'normal'
    assert panel.autofocus_scan_pending is False


def test_a_started_run_draws_running(app_ctx, engine):
    panel = _Panel()
    handle = PendingRunOutcome()
    engine.start.return_value = handle
    _live(engine, handle)

    panel.run_scan_from_ui()

    button = panel.ids['run_scan_btn']
    assert (button.state, button.text) == ('down', 'Abort One Scan')


def test_the_full_protocol_button_counts_the_live_runs_own_scans(app_ctx, engine):
    panel = _Panel()
    handle = PendingRunOutcome()
    engine.start.return_value = handle
    engine.remaining_scans.return_value = 3
    engine.protocol_interval.return_value = timedelta(minutes=20)
    _live(engine, handle)

    panel.run_protocol_from_ui()

    assert panel.ids['run_protocol_btn'].text == '3 scans (1h 0m) remaining.\nPress to ABORT'


def test_a_second_press_stops_its_own_run_ahead_of_queued_work(app_ctx, engine):
    from modules.sequential_io_executor import PRIORITY_HIGH

    panel = _Panel()
    handle = PendingRunOutcome()
    panel._runs_started_here['protocol'] = handle
    _live(engine, handle)
    engine.is_stopping.side_effect = lambda run: run is handle and engine.reset.called

    panel.run_protocol_from_ui()

    task = app_ctx.worker_pool.put.call_args.args[0]
    assert task.priority == PRIORITY_HIGH, 'a Stop must not wait behind queued work'
    engine.reset.assert_called_once_with(handle)
    assert not engine.prepare.called, 'a Stop is not a start'
    assert panel.ids['run_protocol_btn'].text == 'Stopping...'


def test_a_stop_that_finds_its_run_already_ended_shows_nothing(app_ctx, engine, shown):
    panel = _Panel()
    handle = PendingRunOutcome()
    panel._runs_started_here['scan'] = handle
    _live(engine, handle)

    def _ended(run):
        _live(engine)  # the run ended between the press and the pool
        raise RunAlreadyEndedError('the run already ended')

    engine.reset.side_effect = _ended

    panel.run_scan_from_ui()

    assert shown == []
    assert panel.ids['run_scan_btn'].state == 'normal'


def test_a_refused_stop_leaves_every_button_showing_its_own_run(app_ctx, engine, shown):
    panel = _Panel()
    mine, theirs = PendingRunOutcome(), PendingRunOutcome()
    panel._runs_started_here['scan'] = mine
    panel._runs_started_here['protocol'] = theirs
    _live(engine, mine, theirs)
    engine.reset.side_effect = ProtocolRunRefusedError(
        'run_not_live', 'Not Running', 'That run is not the one running.'
    )

    panel.run_scan_from_ui()

    assert [n.title for n in shown] == ['Not Running']
    assert panel.ids['run_scan_btn'].state == 'down'
    assert panel.ids['run_protocol_btn'].state == 'down'


def test_the_button_is_disabled_while_its_own_request_is_in_flight(app_ctx, held, engine):
    held.append('hold')
    panel = _Panel()

    panel.run_scan_from_ui()

    assert panel.scan_pending is True, 'a second press must not race the first to the pool'
    task = held[1]
    task.action(*task.args, **task.kwargs)
    assert panel.scan_pending is False, "the request's own redraw brings the button back"


def test_a_finished_runs_drain_shows_its_count_and_disables_all_three(app_ctx, engine, session):
    panel = _Panel()
    finished = PendingRunOutcome()
    panel._runs_started_here['protocol'] = finished
    engine.run_outcome.return_value = finished
    session.protocol_files_draining = True
    session.protocol_files_pending = 7

    panel.draw_protocol_buttons()

    assert panel.ids['run_protocol_btn'].text == 'Writing Files... (7)'
    assert panel.ids['run_scan_btn'].text == 'Run One Scan'
    assert panel.files_draining is True, 'every start would be refused while the files drain'
    panel._drain_tick_trigger.assert_called_once()

    session.protocol_files_stalled = True
    panel.draw_protocol_buttons()
    assert panel.ids['run_protocol_btn'].text == 'File writer stalled'

    session.protocol_files_draining = False
    session.protocol_files_stalled = False
    panel.draw_protocol_buttons()
    assert panel.ids['run_protocol_btn'].text == 'Run Full Protocol'
    assert panel.files_draining is False


def test_a_live_runs_own_writes_do_not_disable_its_stop(app_ctx, engine, session):
    panel = _Panel()
    handle = PendingRunOutcome()
    panel._runs_started_here['scan'] = handle
    _live(engine, handle)
    session.protocol_files_draining = True
    session.is_protocol_running = True

    panel.draw_protocol_buttons()

    assert panel.files_draining is False
    assert panel.ids['run_scan_btn'].state == 'down'


def test_a_stalled_drain_offers_recovery_once(app_ctx, session, monkeypatch):
    offers = []
    monkeypatch.setattr(ps, '_offer_wedged_writer_recovery', lambda: offers.append(1))
    panel = _Panel()
    panel._wedge_recovery_offered = False
    session.protocol_files_draining = True
    session.protocol_files_stalled = True

    panel._drain_tick(0)
    panel._drain_tick(0)

    assert offers == [1]
