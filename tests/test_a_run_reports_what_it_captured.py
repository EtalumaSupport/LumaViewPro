# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""A run's ending says what the instrument did.

Driven end to end on the simulated scope: a real session, runner, run
loop, writer and file lane; only the camera driver's grab is replaced
where a test needs a capture to produce no frame.
"""

import csv
import pathlib

from tests.test_a_late_write_records_its_frame import _protocol, _step
from tests.test_composite_run_e2e import headless_settings, open_composite_session
from tests.test_composite_run_failures import _info_lines

WAIT_S = 60.0


def _run(runner, run_parent, steps, callbacks=None):
    pending = runner.run_single_scan(
        protocol=_protocol(steps),
        parent_dir=str(run_parent),
        image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
        callbacks=callbacks or {},
    )
    return pending.wait(timeout_s=WAIT_S)


def _two_steps():
    return [_step('C1', 0, x=20.0, gain=1.0), _step('C2', 1, x=30.0, gain=1.0)]


class TestANormalRunLogsNoCrash:
    def test_the_late_cleanup_passes_name_no_crash(self, tmp_path):
        # Cleanup is asked three times on every run; the safety net's pass
        # carries a 'run_loop_crashed' ending it never uses. Naming that
        # ending in the log recorded a crash on every normal run.
        with (
            _info_lines() as info,
            open_composite_session(headless_settings(tmp_path)) as (
                _session,
                runner,
            ),
        ):
            outcome = _run(runner, tmp_path / 'runs', _two_steps())
        assert outcome.status == 'completed', outcome
        crash_lines = [line for line in info if 'crash' in line.lower()]
        assert crash_lines == [], crash_lines
        assert any('no longer live' in line for line in info), (
            'the late passes stopped logging at all; this test no longer sees them'
        )


def _record_rows(run_parent):
    records = list(pathlib.Path(run_parent).rglob('protocol_record.tsv'))
    assert len(records) == 1, records
    with open(records[0], newline='') as fp:
        return [row[0] for row in list(csv.reader(fp, delimiter='\t'))[4:]]


def _no_frame(timeout_s):
    return (False, None, 0)


class TestARunWhoseCameraIsGoneEndsAtOnce:
    def test_the_first_failure_after_a_removal_ends_the_run_failed(self, tmp_path):
        # The bench case: an unplugged camera, a run too short for three
        # strikes. The driver had already declared the camera removed, so
        # the first failed capture ends the run, and in the word that
        # refuses the next run.
        run_parent = tmp_path / 'runs'
        with open_composite_session(headless_settings(tmp_path)) as (session, runner):
            camera = session.scope._camera_driver
            imaging = session.scope.imaging
            # Every camera write cleanup makes after the removal. The
            # simulated camera accepts a write whatever its latch says, so
            # the writes themselves are what is counted, not their reports.
            writes_after_removal = []
            for setter in ('_set_gain_db_impl', '_set_exposure_ms_impl'):
                real = getattr(imaging, setter)

                def counted(value, *a, _real=real, _name=setter, **kw):
                    if camera.is_device_removed():
                        writes_after_removal.append((_name, value))
                    return _real(value, *a, **kw)

                setattr(imaging, setter, counted)

            def unplugged(timeout_s):
                camera._mark_disconnected()
                return _no_frame(timeout_s)

            camera.grab_new_capture = unplugged
            steps = [_step(f'C{i}', i, x=20.0 + 10 * i, gain=1.0) for i in range(3)]
            outcome = _run(runner, run_parent, steps)
            rows = _record_rows(run_parent)
        assert (outcome.status, outcome.reason) == ('failed', 'hardware_disconnected'), outcome
        assert rows == ['capture_failed'], (
            f'the run went on capturing after its camera was removed: {rows}'
        )
        # Cleanup's camera restore writes nothing to a removed camera; each
        # write would report the camera absent again after the ending did.
        assert writes_after_removal == [], writes_after_removal
