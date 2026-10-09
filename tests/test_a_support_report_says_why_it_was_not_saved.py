# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A support report is a Session member that answers with its ZIP or says why there is none.

The report was a module class only the GUI drove: it caught every failure,
logged it and returned ``None``, and the panel then showed 'Report Failed'
or 'Zip failed' with the reason dropped, on threads of its own. Now
``ScopeSession.make_support_report`` and ``make_logs_zip`` return where the
ZIP is, or raise ``SupportReportNotSavedError`` in the failure's own words,
which the reporter shows; and they run on an executor of their own, so a
Stop never waits behind one. No ``ui`` import here.
"""

from __future__ import annotations

import pathlib
import threading

import pytest

from modules.exceptions import SupportReportNotSavedError
from modules.notification_center import notifications
from modules.tech_support_report import SupportReportSaved, TechSupportReport

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def session(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    yield session
    session.shutdown()


def test_a_logs_zip_that_cannot_be_written_says_why(session, tmp_path):
    # A folder cannot be made under a file: the failure is the OS's own.
    blocker = tmp_path / 'not_a_folder'
    blocker.write_text('')
    with pytest.raises(SupportReportNotSavedError) as raised:
        session.make_logs_zip(output_dir=blocker / 'out')
    assert isinstance(raised.value.__cause__, OSError)
    assert str(raised.value.__cause__) in str(raised.value)


def test_a_support_report_that_fails_says_why(session, monkeypatch, tmp_path):
    def _fails(self, *args):
        raise OSError('the disk is full')

    monkeypatch.setattr(TechSupportReport, '_generate', _fails)
    with pytest.raises(SupportReportNotSavedError, match='the disk is full') as raised:
        session.make_support_report(output_dir=tmp_path / 'out')
    assert raised.value.report == 'support report'


def test_the_person_is_shown_the_reports_own_words(centre_posts):
    error = SupportReportNotSavedError('logs zip', OSError('the disk is full'))
    notifications.report_outcome(error, solicited=True, category='ZIP_LOGS')
    shown = [post for post in centre_posts if post.title == 'Support Report Not Saved']
    assert len(shown) == 1
    assert 'the disk is full' in shown[0].message


def test_a_saved_logs_zip_says_where_it_is(session, tmp_path):
    saved = session.make_logs_zip(output_dir=tmp_path / 'out')
    assert saved.path.is_file()
    assert saved.title == 'Logs Zip Saved'
    assert str(tmp_path / 'out') in saved.message


def test_a_saved_report_names_the_folder_it_is_in_not_the_desktop():
    saved = SupportReportSaved(pathlib.Path('/home/someone/SN1-TSR.zip'), 'support report')
    assert saved.message.startswith('Saved to /home/someone:')
    assert 'Desktop' not in saved.message


def test_a_stop_does_not_wait_behind_a_report(session):
    # The report holds the diagnostics executor for minutes; a Stop goes
    # through the worker pool, which it does not hold.
    started, release = threading.Event(), threading.Event()

    def _report():
        started.set()
        release.wait(timeout=15)

    bundle = session.executor_bundle
    bundle.diagnostics_executor.put(_task(_report))
    assert started.wait(timeout=5)
    try:
        stopped = threading.Event()
        bundle.worker_pool.put(_task(stopped.set))
        assert stopped.wait(timeout=5), 'a Stop waited behind the support report'
    finally:
        release.set()


def _task(action):
    from modules.sequential_io_executor import IOTask

    return IOTask(action=action)


@pytest.mark.parametrize('words', ['TechSupportReport(', "'Report Failed'", "'Zip failed'"])
def test_the_gui_neither_builds_nor_words_a_report(words):
    # The panel calls the Session's members and shows what they answer; it
    # neither constructs the report nor writes its own failure.
    hits = [
        str(path.relative_to(REPO))
        for path in (REPO / 'ui').rglob('*.py')
        if words in path.read_text(encoding='utf-8')
    ]
    assert hits == []


def test_the_diagnostics_executor_is_reported_and_stopped_with_the_bundle(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    session = ScopeSession.create(complete_settings(live_folder=str(tmp_path)), simulate=True)
    bundle = session.executor_bundle
    try:
        assert 'DIAGNOSTICS' in bundle.snapshot()
    finally:
        session.shutdown()
    assert bundle.diagnostics_executor.pending_shutdown


def test_the_panel_submits_the_report_with_the_members_budget(monkeypatch):
    # The panel names the member it calls, so the task carries that member's
    # declared budget and a report that takes minutes is not a slow task.
    import types

    import modules.app_context as app_context
    import ui.microscope_settings as panel
    import ui.progress_popup as progress_popup

    class _Popup:
        def __init__(self, **kwargs):
            pass

        def open(self):
            pass

        def dismiss(self):
            pass

    submitted = {}
    session = types.SimpleNamespace(
        make_support_report=lambda **kwargs: None,
        executor_bundle=types.SimpleNamespace(diagnostics_executor=object()),
    )
    monkeypatch.setattr(app_context, 'ctx', types.SimpleNamespace(session=session))
    monkeypatch.setattr(progress_popup, 'CustomPopup', _Popup)
    monkeypatch.setattr(
        panel, 'submit_reported', lambda call, redraw, label, **kwargs: submitted.update(kwargs)
    )
    host = types.SimpleNamespace()
    host._make_zip = types.MethodType(panel.MicroscopeSettings._make_zip, host)
    panel.MicroscopeSettings._start_support_report(host)

    assert submitted['budget_of'] is session.make_support_report
    assert submitted['lane'] is session.executor_bundle.diagnostics_executor
