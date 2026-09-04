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
