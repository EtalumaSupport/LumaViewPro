"""P06 -- the autofocus button, headless.

GUI entry: ui/vertical_control.py:446 run_autofocus_from_ui (kv line 537).
API member under test: modules/protocol_runner.py:281 ProtocolRunner.run_autofocus.
"""

from harness import check, run, void
from modules.protocol_runner import ProtocolRunner
from modules.exceptions import ConfigError


def body(s):
    m = s.scope.motion
    m.home('ALL')
    s.select_labware('96 well microplate')
    m.move_absolute('Z', 3000.0, wait_until_complete=True)
    m.move_absolute('X', 40.0, frame='plate', wait_until_complete=True)
    m.move_absolute('Y', 30.0, frame='plate', wait_until_complete=True)

    runner = ProtocolRunner(s)
    z_before = m.get_current_position('Z')

    pending = runner.run_autofocus(layer='BF')
    outcome = pending.wait(timeout_s=300)
    z_after = m.get_current_position('Z')
    print('outcome:', outcome.status, '|', outcome.reason, '|', outcome.message, flush=True)
    check(
        'run_autofocus completes',
        str(outcome.status).lower().find('fail') < 0,
        f'status={outcome.status} reason={outcome.reason}',
    )
    check(
        'run_autofocus moved Z (stage left at the focus it found)',
        abs(z_after - z_before) > 0.5,
        f'{z_before} -> {z_after}',
    )

    # does it ALSO write the layer's focus the way the button does?
    focus_after = s.get_settings_snapshot()['BF']['focus']
    void(
        "run_autofocus wrote settings['BF']['focus'] to what it found",
        abs(focus_after - z_after) < 5.0,
        f'settings BF focus={focus_after}, stage Z={z_after}; '
        'a headless caller can only infer the focus from the stage position',
    )

    # a bad layer is refused
    try:
        runner.run_autofocus(layer='Purple')
        check('unknown layer refused', False, 'NO RAISE')
    except ConfigError as e:
        check('unknown layer refused', True, str(e)[:80])

    # second run works (no lock left behind)
    p2 = runner.run_autofocus(layer='BF')
    o2 = p2.wait(timeout_s=300)
    check(
        'a second run_autofocus is accepted',
        str(o2.status).lower().find('fail') < 0,
        f'status={o2.status}',
    )


run(body)
