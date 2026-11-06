# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A support report says when the logs it carries are not its process's.

A process that declined LumaViewPro's log files (``start_logging(log_files=False)``)
wrote none, so the log folder a report collects, if there is one, is another
LumaViewPro process's. The report states it beside the logs instead of
passing them off as the reporting process's own; a process that started its
files states nothing.
"""

from _lumascope.modules import tech_support_report
from _lumascope.modules.tech_support_report import TechSupportReport


def _logs_step(sim_session, tmp_path, monkeypatch, started):
    # a stand-in by design: the process's logging state, which the suite's
    # conftest replaces with a mock and which a test process cannot start twice
    monkeypatch.setattr(tech_support_report, 'logging_started', lambda: started)
    another_process = tmp_path / 'logs'
    (another_process / 'LVP_Log').mkdir(parents=True)
    (another_process / 'LVP_Log' / 'lumaviewpro.log').write_text('another process\n')
    monkeypatch.setattr(tech_support_report, '_get_lvp_logs_dir', lambda: another_process)
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    TechSupportReport(session=sim_session)._step_logs(bundle)
    return bundle / 'logs'


def test_a_report_from_a_declined_process_says_its_logs_are_another_processs(
    sim_session, tmp_path, monkeypatch
):
    logs = _logs_step(sim_session, tmp_path, monkeypatch, 'declined')

    assert (logs / 'LVP_Log' / 'lumaviewpro.log').read_text() == 'another process\n'
    assert (logs / 'NOT_THIS_PROCESS.txt').read_text() == (
        'This process started no log files (start_logging(log_files=False)). '
        "Any logs here are another LumaViewPro process's.\n"
    )


def test_a_report_from_a_process_that_started_its_files_says_nothing_of_it(
    sim_session, tmp_path, monkeypatch
):
    logs = _logs_step(sim_session, tmp_path, monkeypatch, 'lumascope')

    assert not (logs / 'NOT_THIS_PROCESS.txt').exists()
