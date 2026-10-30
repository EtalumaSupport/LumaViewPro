# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Builders that drive a real AutofocusRunner.run() headlessly.

``af_runner_on`` and ``sweep`` run it on a simulated scope, a real sweep
over the simulated camera's focus target. A test chooses the curve's
shape by the field it shows (``'white'`` scores zero everywhere) or by
patching modules.autofocus_functions.autofocus_score, a score of the frame
or of the Z it was made at.
"""

from __future__ import annotations

import contextlib
import threading
import time


def park_z(scope, z: float) -> None:
    """Move a simulated scope's Z to ``z`` and wait for it to arrive.

    A built or homed simulated scope stands at the bottom of Z travel,
    where every autofocus window reaches past it and is refused. A test
    that wants a sweep (or a later exit path) starts it where its window
    fits.
    """
    scope.motion.move_z(z)
    while scope.motion.is_any_axis_moving():
        time.sleep(0.01)


@contextlib.contextmanager
def raises_the_failure_from(original: type[BaseException]):
    """The run raises the one failure it reported, with *original* chained as its cause.

    An unexpected error inside a sweep is reported as an
    ``AutofocusFailedError('unexpected_error')`` and that same object is
    raised, so the caller holds what the report named; the error that
    stopped the sweep travels with it, unmasked.
    """
    import pytest

    from modules.exceptions import AutofocusFailedError

    with pytest.raises(AutofocusFailedError) as raised:
        yield raised
    assert raised.value.reason == 'unexpected_error'
    assert isinstance(raised.value.__cause__, original), repr(raised.value.__cause__)


# Where the simulated scope's focus target is in focus, and where Z is
# parked: inside the first window of AF_OBJECTIVE, above the focus.
FOCAL_Z_UM = 2950.0
PARKED_Z_UM = 3000.0
AF_OBJECTIVE = '10x Oly'


def af_runner_on(scope, pattern: str = 'focus_target'):
    """A real AutofocusRunner over a simulated scope a sweep can run on.

    A built scope is unhomed, stands at the bottom of Z travel where every
    window reaches past it, and shows a specimen no sweep is aimed at.
    This one is homed, parked at PARKED_Z_UM, and shows ``pattern`` in
    focus at FOCAL_Z_UM: the focus target, where a sweep finds the focus
    within a few um, or ``'white'``, a field with nothing in it, where
    every frame scores zero. The 2 ms exposure keeps BF from saturating
    the target, which would move the focus it finds.
    """
    from modules.autofocus_runner import AutofocusRunner
    from tests.scope_fakes import home_sim_scope

    home_sim_scope(scope)
    camera = scope._camera_driver
    camera.set_test_pattern(enabled=True, pattern=pattern)
    camera.set_focal_z(FOCAL_Z_UM)
    scope.camera.set_exposure_ms(2.0)
    park_z(scope, PARKED_Z_UM)
    return AutofocusRunner(scope=scope)


def sweep(runner, scope, **overrides):
    """Call runner.run() on ``scope`` as a run calls it: BF lit, under the run's lease.

    A lease the caller does not pass is taken here and released when the
    sweep ends, leaving lit what the sweep left lit, as a run's end does;
    so a later sweep on the same scope can take its own.
    """
    from tests.protocol_drives import held_run_claim

    kwargs = {
        'objective_id': AF_OBJECTIVE,
        'abort_event': threading.Event(),
        'led_color': 'BF',
        'led_illumination': 5.0,
    }
    kwargs.update(overrides)
    if 'led_lease' in kwargs:
        return runner.run(**kwargs)
    lease = scope.illumination.acquire_led_lease('protocol', claim=held_run_claim())
    try:
        return runner.run(led_lease=lease, **kwargs)
    finally:
        lease.release(leave_on=True)


def z_asked(scope, monkeypatch) -> list[float]:
    """Every absolute Z target the scope is asked for from here on, in order.

    A sweep is judged by the targets asked, not by where Z ends: the
    restore and the last move of a success both end near the start.
    """
    asked = []
    for member in ('start_move_z', 'move_z'):
        real = getattr(scope.motion, member)

        def spy(position, *args, _real=real, **kwargs):
            asked.append(position)
            return _real(position, *args, **kwargs)

        monkeypatch.setattr(scope.motion, member, spy)
    return asked


def settled_z(scope) -> float:
    """Z once every axis has arrived."""
    while scope.motion.is_any_axis_moving():
        time.sleep(0.01)
    return scope.motion.get_current_position('Z')
