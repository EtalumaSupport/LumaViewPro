"""P10 -- an autofocus whose search would leave Z's travel is refused, by the API.

GUI entry: ui/vertical_control.py run_autofocus_from_ui -> the standalone
autofocus run, the one `ProtocolRunner.run_autofocus` starts.

Three starts, each through `run_autofocus` and the Session's outcome
listener, which hears the failure no caller waits on:

  top of travel     the first window reaches past the maximum: refused before
                    the stage moves, Z unchanged.
  bottom of travel  the first window reaches below the minimum: refused the
                    same way (it used to be clamped to 0 and searched).
  mid-travel        the window fits: no travel refusal. Whether a focus is
                    found depends on what is on the stage, so the focus is a
                    figure, not a check.

Run with --hardware for the connected scope; the simulator otherwise.
"""

import sys

from harness import HARDWARE, check, figure, hardware_session, headless_session, live_dir, report

LAYER = 'BF'
TIMEOUT_S = 300


def _autofocus_from(session, runner, z, heard):
    """Autofocus once from ``z``; return (outcome, travel refusals heard, Z after)."""
    motion = session.scope.motion
    motion.move_absolute('Z', z)
    heard.clear()
    outcome = runner.run_autofocus(layer=LAYER).wait(timeout_s=TIMEOUT_S)
    while motion.is_moving():
        pass
    refused = [n for n in heard if getattr(n, 'reason', None) == 'out_of_travel']
    return outcome, refused, motion.get_current_position('Z')


def body(session, runner):
    heard = []
    session.add_outcome_listener(heard.append)
    limits = session.scope.motion.get_axis_limits('Z')
    objective = session.scope.runtime_state.get_current_objective()
    af_range = objective['AF_range']
    figure('Z travel', limits)
    figure('objective', session.scope.runtime_state.get_current_objective_id())
    figure('AF_range', af_range)

    for name, start in (('top', limits['max']), ('bottom', limits['min'])):
        outcome, refused, z_after = _autofocus_from(session, runner, start, heard)
        figure(f'{name}: status', outcome.status)
        figure(f'{name}: focus', outcome.af_focus_z_um)
        figure(f'{name}: Z after', z_after)
        check(
            f'{name} of travel: the autofocus is refused as out_of_travel, reported once',
            len(refused) == 1,
            f'out_of_travel reports heard: {len(refused)}; all reasons: '
            f'{[getattr(n, "reason", None) for n in heard]}',
        )
        check(f'{name} of travel: no focus is reported', outcome.af_focus_z_um is None)
        check(
            f'{name} of travel: Z is where the autofocus started',
            abs(z_after - start) < 1.0,
            f'start {start}, after {z_after}',
        )

    mid = (limits['min'] + limits['max']) / 2.0
    outcome, refused, z_after = _autofocus_from(session, runner, mid, heard)
    figure('mid: status', outcome.status)
    figure('mid: focus', outcome.af_focus_z_um)
    figure('mid: reasons heard', [getattr(n, 'reason', None) for n in heard])
    check('mid-travel: the autofocus is not refused for travel', not refused)
    check(
        'mid-travel: Z stays inside the travel',
        limits['min'] <= z_after <= limits['max'],
        f'Z after {z_after}',
    )


def main():
    if HARDWARE:
        with hardware_session() as (session, runner):
            body(session, runner)
    else:
        with headless_session(live_dir('p10')) as (session, runner):
            body(session, runner)
    sys.exit(report())


main()
