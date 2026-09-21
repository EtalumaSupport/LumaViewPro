"""P08 -- the STOP half of the autofocus toggle, and the ownership refusal.

GUI entry: ui/vertical_control.py:446 (second click / 'normal' state) ->
_cleanup_at_end_of_autofocus (vertical_control.py:346) -> ui_helpers
reset_with_refusal_boundary -> SequencedCaptureRunner.reset(requester='autofocus').
"""

from harness import check, run
from modules.protocol_runner import ProtocolRunner
from modules.exceptions import ProtocolRunRefusedError


def body(s):
    m = s.scope.motion
    m.home('ALL')
    s.select_labware('96 well microplate')
    m.move_absolute('Z', 3000.0, wait_until_complete=True)
    runner = ProtocolRunner(s)

    check('ProtocolRunner exposes reset(requester)', callable(runner.reset))
    check('ProtocolRunner exposes abort(requester)', callable(runner.abort))
    check('ProtocolRunner exposes run_trigger_source()', callable(runner.run_trigger_source))

    # --- a run in flight names its trigger source; the GUI's is 'autofocus',
    #     run_autofocus's is 'api_autofocus'.
    seen = {}

    def on_complete(**kw):
        seen['complete'] = True

    pending = runner.run_autofocus(layer='BF', callbacks={'run_complete': on_complete})
    src = runner.run_trigger_source()
    check(
        "the API member's trigger source is 'api_autofocus', not the GUI's 'autofocus'",
        src in ('api_autofocus', None),
        f'run_trigger_source={src!r}',
    )

    # --- a STOP from a non-owner is refused ---
    refused = None
    try:
        runner.reset(requester='somebody_else')
        refused = False
    except ProtocolRunRefusedError:
        refused = True
    except Exception as e:
        refused = f'{type(e).__name__}'
    print('non-owner reset ->', refused, flush=True)

    outcome = pending.wait(timeout_s=300)
    check('run finished', outcome is not None, f'status={outcome.status}')
    check('a caller-supplied run_complete callback fires', seen.get('complete') is True)

    check(
        'a non-owner STOP is refused at the API (ProtocolRunRefusedError)',
        refused is True,
        f'reset(requester="somebody_else") -> {refused}',
    )

    # --- the stuck-AF bound: the GUI arms a 15 s Clock timer
    #     (ui/vertical_control.py:382, AF_SAFETY_TIMEOUT_S). Nothing under
    #     modules/ carries one, so a headless autofocus that stops
    #     progressing is bounded only by the run pipeline's motion timeout.
    import subprocess

    hits = subprocess.run(
        ['grep', '-rln', 'AF_SAFETY_TIMEOUT_S', '/Users/ericweiner/Projects/LumaViewPro/modules'],
        capture_output=True,
        text=True,
    ).stdout.split()
    check(
        'the 15 s stuck-AF bound has no home under modules/ -- it is GUI-only',
        not hits,
        f'modules/ hits: {hits}',
    )


run(body)
