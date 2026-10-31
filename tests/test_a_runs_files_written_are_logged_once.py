# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The log says when a run's files are written, once, and what was not.

A run's images keep landing after the run itself ends, and nothing said
when the last one had: a bundle could not tell a run whose files were all
written from one whose last write never landed. When the run's last write
lands, one INFO line names the run's folder, the outcome, how many images
were written and how many were not, and why.
"""

from modules.run_events import RunEvents
import logging
import time

import modules.protocol_image_writer as protocol_image_writer
from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session

WAIT_S = 30.0


def _files_lines(caplog, levels=(logging.INFO, logging.WARNING)):
    """The main log's lines about a run's files: INFO, or a WARNING when a
    file did not land."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == 'lvp_logger' and r.levelno in levels and "run's files" in r.getMessage()
    ]


def _one_run(tmp_path, caplog):
    files = []
    with open_composite_session(headless_settings(tmp_path)) as (_session, runner):
        caplog.set_level(logging.INFO, logger='lvp_logger')
        outcome = runner.start_single_scan(
            protocol=_protocol(
                [_step('C1', 0, x=20.0, gain=1.0), _step('C2', 1, x=20.0, gain=1.0)]
            ),
            parent_dir=str(tmp_path / 'runs'),
            events=RunEvents(files_written=lambda run_dir, _files: files.append(run_dir)),
        )
        assert outcome.result(timeout=WAIT_S) is not None
        deadline = time.monotonic() + WAIT_S
        while not files and time.monotonic() < deadline:
            time.sleep(0.02)
    assert files, 'files_complete never came'
    return _files_lines(caplog), str(files[0])


def test_a_run_whose_files_all_landed_says_so_once(tmp_path, caplog):
    lines, run_dir = _one_run(tmp_path, caplog)

    assert len(lines) == 1, lines
    [message] = lines
    assert run_dir in message and 'are written' in message
    assert '2 written, 0 not written' in message, message


def test_a_run_with_a_failed_save_says_which_and_why(tmp_path, monkeypatch, caplog):
    real_save = protocol_image_writer.save_image
    saves = []

    def _save(scope, **kwargs):
        saves.append(kwargs)
        if len(saves) == 2:
            raise OSError('the save drive went away')
        return real_save(scope, **kwargs)

    monkeypatch.setattr(protocol_image_writer, 'save_image', _save)
    lines, run_dir = _one_run(tmp_path, caplog)

    assert len(lines) == 1, lines
    [message] = lines
    assert run_dir in message and 'are incomplete' in message
    assert '1 written, 1 not written (write_batch_save_failed)' in message, message


def test_a_run_that_saves_no_images_says_it_has_no_folder(tmp_path, caplog):
    """An autofocus-only run saves no images, so it never makes a run folder;
    the line says so rather than naming a folder called None."""
    from tests.test_run_outcome_reports_autofocus_data import _AfRig

    caplog.set_level(logging.INFO, logger='lvp_logger')
    rig = _AfRig()
    try:
        rig.run_autofocus(tmp_path / 'af', save_data=False)
    finally:
        rig.close()

    lines = _files_lines(caplog, levels=(logging.INFO,))
    assert len(lines) == 1, lines
    [message] = lines
    assert 'None' not in message, message
    assert 'no run folder' in message, message
    assert '0 written, 0 not written' in message, message
