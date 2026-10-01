# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""A run's ending says what the instrument did.

Driven end to end on the simulated scope: a real session, runner, run
loop, writer and file lane; only the camera driver's grab is replaced
where a test needs a capture to produce no frame.
"""

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
