"""P08 -- the STOP half of the autofocus toggle, and the stale stop that does nothing.

GUI entry: ui/vertical_control.py run_autofocus_from_ui (a press while the
engine says the button's own run is live) -> _stop_autofocus -> ui_helpers
submit_reported -> run.stop(), where run is the handle the button's own
start returned.
"""

from harness import REPO, check, run
from modules.protocol_runner import ProtocolRunner
from modules.run_events import RunEvents


def body(s):
    m = s.scope.motion
    m.home('ALL')
    s.select_labware('96 well microplate')
    m.move_absolute('Z', 3000.0)
    runner = ProtocolRunner(s)

    # A first autofocus, run to its end: its handle is the stale one below.
    earlier = runner.start_autofocus(layer='BF')
    check('the earlier run finished', earlier.result(timeout=300) is not None)

    # --- a run in flight names its trigger source; the GUI's is 'autofocus',
    #     start_autofocus's is 'api_autofocus'.
    seen = {}

    def on_ended(outcome, run_dir, protocol):
        seen['ended'] = True

    pending = runner.start_autofocus(layer='BF', events=RunEvents(run_ended=on_ended))
    holder = s.activity_claim.holder
    src = holder.run_trigger_source if holder is not None else None
    check(
        "the API member's trigger source is 'api_autofocus', not the GUI's 'autofocus'",
        src in ('api_autofocus', None),
        f'run_trigger_source={src!r}',
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
    check('the run let go of the scope when its wait returned', not s.is_protocol_running)
    late_stop = pending.stop()
    check(
        'a STOP naming a run that has ended returns None',
        late_stop is None,
        f'<the finished run>.stop() -> {late_stop!r}',
    )

    # --- the stuck-AF bound: the GUI arms a 15 s Clock timer
    #     (ui/vertical_control.py:382, AF_SAFETY_TIMEOUT_S). Nothing under
    #     modules/ carries one, so a headless autofocus that stops
    #     progressing is bounded only by the run pipeline's motion timeout.
    import subprocess

    hits = subprocess.run(
        ['grep', '-rln', 'AF_SAFETY_TIMEOUT_S', str(REPO / 'modules')],
        capture_output=True,
        text=True,
    ).stdout.split()
    check(
        'the 15 s stuck-AF bound has no home under modules/ -- it is GUI-only',
        not hits,
        f'modules/ hits: {hits}',
    )


run(body)
