# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Every camera write the protocol step runner queues binds the API's
non-dispatching body, never the public dispatcher.

Bug shape: the public imaging members are dispatchers over a private
``_impl`` that run the body on the camera worker and refuse work while a
run holds the camera lane. The step runner already sits inside the run's
own serialization, so a public binding there is refused by the run it is
part of. The step-end auto-gain disarm carried exactly that binding: the
refusal raised out of the step, the run loop classified it as a transient
scan failure, and the whole scan was retried from step 0 -- observed on a
field unit as a run that stuttered and duplicated its files.
"""

from __future__ import annotations

import ast

from tests import ast_seams

STEP_RUNNER = 'modules/protocol_step_runner.py'


def _imaging_actions_in(func: ast.FunctionDef) -> list[str]:
    """The ``.imaging.<member>`` names bound as ``IOTask(action=...)``."""
    names = []
    for node in ast.walk(func):
        if not (isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'IOTask'):
            continue
        for kw in node.keywords:
            if kw.arg != 'action' or not isinstance(kw.value, ast.Attribute):
                continue
            owner = kw.value.value
            if isinstance(owner, ast.Attribute) and owner.attr == 'imaging':
                names.append(kw.value.attr)
    return names


def test_scan_iterate_binds_only_imaging_impls():
    func = ast_seams.find_def(STEP_RUNNER, 'scan_iterate', class_name='ProtocolStepRunner')
    assert func is not None
    names = _imaging_actions_in(func)
    assert names, 'scan_iterate queues no imaging IOTask -- the survey found nothing'
    public = [n for n in names if not n.startswith('_')]
    assert not public, (
        f'scan_iterate binds public imaging dispatcher(s) {public} inside the run; '
        'bind the _impl body -- the dispatcher refuses its own run'
    )


def test_step_end_disarm_binds_the_impl():
    func = ast_seams.find_def(STEP_RUNNER, 'scan_iterate', class_name='ProtocolStepRunner')
    names = _imaging_actions_in(func)
    assert '_set_auto_gain_impl' in names
    assert 'set_auto_gain' not in names


def _arm_iotask_kwargs(func: ast.FunctionDef) -> dict:
    """The literal kwargs dict of the IOTask that arms auto-gain for a step:
    the one whose action is the layer-settings apply with ``auto_gain``
    True. The step-end disarm names an auto-gain action too, so the
    selection is on the action AND the flag."""
    for node in ast.walk(func):
        if not (isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'IOTask'):
            continue
        kws = {kw.arg: kw.value for kw in node.keywords}
        action = kws.get('action')
        if not (
            isinstance(action, ast.Attribute) and action.attr == '_apply_layer_camera_settings_impl'
        ):
            continue
        literal = kws.get('kwargs')
        assert isinstance(literal, ast.Dict)
        pairs = {
            k.value: v
            for k, v in zip(literal.keys, literal.values, strict=True)
            if isinstance(k, ast.Constant)
        }
        auto_gain = pairs.get('auto_gain')
        if isinstance(auto_gain, ast.Constant) and auto_gain.value is True:
            return pairs
    raise AssertionError(
        'scan_iterate has no IOTask arming auto-gain through the layer-settings apply'
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
    pairs = _arm_iotask_kwargs(func)
    flag = pairs.get('resume_after_capture')
    assert isinstance(flag, ast.Constant) and flag.value is False, (
        "the step runner's arm must pass resume_after_capture=False; without it the "
        'arm is recorded as a live-view arm and every protocol capture re-arms and notifies'
    )
