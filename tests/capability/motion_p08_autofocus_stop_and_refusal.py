"""P08 -- the STOP half of the autofocus toggle, and the stale stop that does nothing.

GUI entry: ui/vertical_control.py run_autofocus_from_ui (a press while the
engine says the button's own run is live) -> _stop_autofocus -> ui_helpers
submit_reported -> run.stop(), where run is the handle the button's own
start returned.
"""

from harness import REPO, check, run
from modules.caller import Caller
from modules.protocol_runner import ProtocolRunner
from modules.run_events import RunEvents


def body(s):
    m = s.scope.motion
    m.home('ALL')
    s.select_labware('96 well microplate')
    m.move_z(3000.0)
    runner = ProtocolRunner(s)

    # A first autofocus, run to its end: its handle is the stale one below.
    earlier = runner.start_autofocus(layer='BF')
    check('the earlier run finished', earlier.result(timeout=300) is not None)

    # --- a run in flight names who started it: the host stamped on the
    #     starting thread, a script ('api') where none did.
    seen = {}

    def on_ended(outcome, run_dir, protocol):
        seen['ended'] = True

    pending = runner.start_autofocus(layer='BF', events=RunEvents(run_ended=on_ended))
    holder = s.activity_claim.holder
    src = holder.run_started_by if holder is not None else None
    check(
        "an unstamped caller's run is started by the API, not the GUI",
        src in (Caller.API, None),
        f'run_started_by={src!r}',
    )

    # --- a STOP through a handle whose run is not the live one does nothing
    #     to the live run, and raises nothing (a raise fails the probe) ---
    stale_stop = earlier.stop()
    still_live = pending.progress().is_live
    print('stale-handle stop ->', stale_stop, 'live run still live:', still_live, flush=True)

    outcome = pending.result(timeout=300)
    check('run finished', outcome is not None, f'status={outcome.status}')
    check('a caller-supplied run_ended handler fires', seen.get('ended') is True)

    check(
        'a STOP naming another run returns None and leaves the live run live',
        stale_stop is None and still_live is True,
        f'<the earlier run>.stop() -> {stale_stop!r}; live run is_live={still_live!r}',
    )

    # --- a STOP after the run ended does nothing and raises nothing ---
    check('the run let go of the scope when its wait returned', s.activity_claim.run_holder is None)
    late_stop = pending.stop()
    check(
        'a STOP naming a run that has ended returns None',
        late_stop is None,
        f'<the finished run>.stop() -> {late_stop!r}',
    )

    # --- no clock of its own: the GUI's 15 s stuck-AF timer is gone, and
    #     nothing under modules/ or ui/ carries one. Every wait inside a
    #     sweep is a move the motion monitor judges or a bounded capture.
    #     Source only: a stale __pycache__ entry compiled from the file that
    #     carried the timer still holds its name (a 3.12 .pyc beside the 3.13
    #     one, 2026-10-16) and is not the tree under test.
    import subprocess

    hits = subprocess.run(
        [
            'grep',
            '-rln',
            '--include=*.py',
            'AF_SAFETY_TIMEOUT_S',
            str(REPO / 'modules'),
            str(REPO / 'ui'),
        ],
        capture_output=True,
        text=True,
    ).stdout.split()
    check(
        'no stuck-AF timer under modules/ or ui/: a sweep has no clock of its own',
        not hits,
        f'hits: {hits}',
    )


run(body)
