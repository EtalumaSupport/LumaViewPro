# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A run's own camera writes are admitted by the camera lane it holds.

Bug shape: the public imaging members are dispatchers, and while a run held
the camera lane they refused work -- including the run's own. The step-end
auto-gain disarm once bound the public member: the refusal raised out of the
step, the run loop classified it as a transient scan failure, and the whole
scan was retried from step 0 -- observed on a field unit as a run that
stuttered and duplicated its files. Run code then bound the private bodies
to get past its own lane.

Now the lane asks the activity claim: the run calls the public members under
its taking, and its own writes go through the run's door on the camera lane.
Driven through the Session on the simulator, an auto-gain step arms, captures
and disarms, and the run completes.
"""

from __future__ import annotations

import ast
import threading

from tests import ast_seams

STEP_RUNNER = 'modules/protocol_step_runner.py'


def test_an_auto_gain_step_arms_captures_and_disarms_inside_its_run(tmp_path):
    from modules.scope_session import ScopeSession
    from tests.scope_fakes import home_sim_scope
    from tests.settings_fixtures import complete_settings
    from tests.test_a_run_needs_every_axis_position import COMPLETION_TIMEOUT, _settings
    from tests.test_run_refusal_contract import _build_real_protocol, _make_single_step_protocol

    session = ScopeSession.create(complete_settings(**_settings(tmp_path)), simulate=True)
    try:
        home_sim_scope(session.scope)
        step = {**_make_single_step_protocol().step(idx=0), 'Auto_Gain': True}
        runner = session.create_protocol_runner()
        files_written = threading.Event()
        runner.run_single_scan(
            protocol=_build_real_protocol([step]),
            sequence_name='auto_gain',
            parent_dir=str(tmp_path),
            image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
            callbacks={
                'run_complete': lambda **kw: None,
                'files_complete': lambda **kw: files_written.set(),
            },
        )
        assert files_written.wait(COMPLETION_TIMEOUT), 'the run never finished its files'
        outcome = runner.wait_for_completion(timeout=COMPLETION_TIMEOUT)
        assert outcome.status == 'completed', outcome
        images = [p for p in tmp_path.rglob('*') if p.suffix.lower() in ('.tif', '.tiff')]
        assert len(images) == 1, (
            f'a one-step scan wrote {len(images)} images; a refused write retries the scan '
            'and duplicates its files'
        )
    finally:
        session.shutdown()


def _arm_call_kwargs(func: ast.FunctionDef) -> dict:
    """The keywords of the call that arms auto-gain for a step: the
    layer-settings apply with ``auto_gain`` True."""
    for node in ast.walk(func):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'apply_layer_camera_settings'
        ):
            continue
        kws = {kw.arg: kw.value for kw in node.keywords}
        auto_gain = kws.get('auto_gain')
        if isinstance(auto_gain, ast.Constant) and auto_gain.value is True:
            return kws
    raise AssertionError(
        'scan_iterate has no call arming auto-gain through the layer-settings apply'
    )


def test_step_runner_arm_does_not_resume():
    """A protocol step's arm is unattended: the lock that consumes it logs
    and records the state, shows no notice, and does not re-arm after the
    capture. The API half (a non-resuming arm neither re-arms nor
    notifies) is proven in test_auto_gain_lock; this pins the wiring --
    the step runner must actually pass the flag, which the bench showed it
    did not (a re-arm and an info popup after every protocol capture)."""
    func = ast_seams.find_def(STEP_RUNNER, 'scan_iterate', class_name='ProtocolStepRunner')
    assert func is not None
    flag = _arm_call_kwargs(func).get('resume_after_capture')
    assert isinstance(flag, ast.Constant) and flag.value is False, (
        "the step runner's arm must pass resume_after_capture=False; without it the "
        'arm is recorded as a live-view arm and every protocol capture re-arms and notifies'
    )
