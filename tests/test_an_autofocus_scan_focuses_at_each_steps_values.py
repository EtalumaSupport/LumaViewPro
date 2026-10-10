"""An autofocus scan focuses each step at that step's gain and exposure.

The run takes a standing live-view auto-gain arm for its duration, so a
protocol's autofocus step scans at the step's values, not at what the
live view's auto-gain achieved. An autofocus scan is a protocol's steps
too and gets the same takeover; it was skipped with the standalone
autofocus, so every step of the scan swept at the live arm's lock. The
standalone autofocus keeps the arm: it focuses the field the user is
watching, and its lock scans at what that arm achieved.
"""

from modules.protocol_state_machine import SequencedCaptureRunMode
from tests.test_auto_gain_lock import AG_SETTINGS_TRANSMITTED
from tests.test_run_outcome_reports_autofocus_data import COMPLETION_TIMEOUT, _AfRig

# The one step _AfRig's protocol holds asks for these.
STEP_GAIN_DB = 1.0
STEP_EXPOSURE_MS = 10.0


def _sweep_targets(tmp_path, run_mode):
    """Run one autofocus of the given mode under a live arm; each sweep's (source, gain, exposure)."""
    rig = _AfRig()
    sweeps = []
    choose = rig.af_runner._apply_sweep_camera_targets

    def record(lock):
        source = choose(lock)
        sweeps.append((source, rig.af_runner._camera_gain, rig.af_runner._camera_exposure))
        return source

    rig.af_runner._apply_sweep_camera_targets = record
    rig.scope.imaging._apply_layer_camera_settings_impl(
        layer='BF',
        gain_db=7.0,
        exposure_ms=40.0,
        auto_gain=True,
        auto_gain_settings=AG_SETTINGS_TRANSMITTED,
        resume_after_capture=True,
    )
    try:
        plan = rig.prepare_autofocus(
            tmp_path / 'af', save_data=False, run_trigger_source='autofocus', run_mode=run_mode
        )
        pending = rig.runner.start(plan)
        assert rig._done.wait(timeout=COMPLETION_TIMEOUT), 'the autofocus run did not complete'
        pending.wait(timeout_s=COMPLETION_TIMEOUT)
    finally:
        rig.close()
    return sweeps


def test_an_autofocus_scan_sweeps_at_the_steps_values_under_a_live_arm(tmp_path):
    sweeps = _sweep_targets(tmp_path, SequencedCaptureRunMode.SINGLE_AUTOFOCUS_SCAN)
    assert sweeps == [('step', STEP_GAIN_DB, STEP_EXPOSURE_MS)]


def test_the_standalone_autofocus_sweeps_at_the_live_arms_lock(tmp_path):
    sweeps = _sweep_targets(tmp_path, SequencedCaptureRunMode.SINGLE_AUTOFOCUS)
    assert [source for source, _, _ in sweeps] == ['lock']
    assert sweeps[0][1:] != (STEP_GAIN_DB, STEP_EXPOSURE_MS)
