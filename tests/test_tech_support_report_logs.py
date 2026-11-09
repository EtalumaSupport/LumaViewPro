# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A support report says when the logs it carries are not its process's.

A process that declined LumaViewPro's log files (``start_logging(log_files=False)``)
wrote none, so the log folder a report collects, if there is one, is another
LumaViewPro process's. The report states it beside the logs instead of
passing them off as the reporting process's own; a process that started its
files states nothing.
"""

from _lumascope.modules import tech_support_report
from _lumascope.modules.path_utils import get_log_dir
from _lumascope.modules.tech_support_report import TechSupportReport
from tests.installation_fixtures import process_logs_at


def _logs_step(session, tmp_path, monkeypatch, started):
    # a stand-in by design: the process's logging state, which the suite's
    # conftest replaces with a mock and which a test process cannot start twice
    monkeypatch.setattr(tech_support_report, 'logging_started', lambda: started)
    process_log = process_logs_at(monkeypatch, tmp_path / 'process') / 'lumaviewpro.log'
    process_log.write_text('another process\n')
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    TechSupportReport(session=session)._step_logs(bundle)
    return bundle


def test_a_report_from_a_declined_process_says_its_logs_are_another_processs(
    sim_session_on_its_own_installation, tmp_path, monkeypatch
):
    logs = _logs_step(sim_session_on_its_own_installation, tmp_path, monkeypatch, 'declined')
    logs = logs / 'logs'

    assert (logs / 'LVP_Log' / 'lumaviewpro.log').read_text() == 'another process\n'
    assert (logs / 'NOT_THIS_PROCESS.txt').read_text() == (
        'This process started no log files (start_logging(log_files=False)). '
        "Any logs here are another LumaViewPro process's.\n"
    )


def test_a_report_from_a_process_that_started_its_files_says_nothing_of_it(
    sim_session_on_its_own_installation, tmp_path, monkeypatch
):
    logs = _logs_step(sim_session_on_its_own_installation, tmp_path, monkeypatch, 'lumascope')

    assert not (logs / 'logs' / 'NOT_THIS_PROCESS.txt').exists()


def test_a_session_on_another_installation_brings_that_installations_logs(
    sim_session_on_its_own_installation, tmp_path, monkeypatch
):
    # A characterization run or a probe writes under the session's
    # installation; a script's session can run on one that is not the
    # process's, and the report carries both.
    session = sim_session_on_its_own_installation
    result = get_log_dir(session.scope.source_path) / 'characterization_results' / 'run.csv'
    result.parent.mkdir(parents=True)
    result.write_text('result\n')

    bundle = _logs_step(session, tmp_path, monkeypatch, 'lumascope')

    carried = bundle / 'session_installation_logs' / 'LVP_Log' / 'characterization_results'
    assert (carried / 'run.csv').read_text() == 'result\n'
    assert (bundle / 'logs' / 'LVP_Log' / 'lumaviewpro.log').read_text() == 'another process\n'
