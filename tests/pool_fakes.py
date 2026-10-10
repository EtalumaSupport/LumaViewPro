# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Run a queued IOTask at once, the way the executor's worker would.

A test that fakes the GUI's worker pool (``put`` runs the task as it is
handed over) must run everything the executor runs for that task: the
action, then its callback. ``submit_reported`` puts its redraw on the
callback so that every outcome the executor answers is redrawn exactly
once; a fake that runs only the action loses the redraw and leaves a
button showing its request as pending.
"""

from __future__ import annotations


def run_task_now(task) -> None:
    """Run *task*'s action, then its callback, with the executor's arguments."""
    result = exception = None
    try:
        result = task.action(*task.args, **task.kwargs)
    except Exception as e:  # the executor returns it to the callback, it does not raise
        exception = e
    if task.callback is None:
        return
    cb_kwargs = dict(task.cb_kwargs)
    if task.pass_result:
        cb_kwargs['result'] = result
        cb_kwargs['exception'] = exception
    task.callback(*task.cb_args, **cb_kwargs)
