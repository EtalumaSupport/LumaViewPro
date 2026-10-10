"""Probe 03 -- the GUI's z-stack pre-check vs the API's refusal.

ui/zstack.py:195-210 pre-checks get_zstack_positions() and pops a
notification when range/step are zero. Does the API refuse the same run
on its own, so a script and REST get the same FAIL?
"""

from harness import make_session, banner

session, live = make_session(
    'zszero',
    home=True,
    zstack={'step_size': 0, 'range': 0, 'position': 'Current Position at Center'},
)
try:
    from modules.protocol_runner import ProtocolRunner

    runner = ProtocolRunner(session)
    banner('run_zstack with range=0 step_size=0')
    try:
        outcome = runner.run_zstack(layer='BF')
        settled = outcome.wait(timeout_s=120)
        print('NO RAISE. status:', settled.status, settled.reason, '|', settled.message)
    except Exception as e:
        print(f'RAISED {type(e).__name__}: {e}')
        print('reason attr:', getattr(e, 'reason', None))
finally:
    session.shutdown()
print('\nPROBE 03 DONE')
