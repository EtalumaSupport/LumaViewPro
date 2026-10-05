"""Serial P01: after a cable pull and a replug, do both boards answer again?

The firmware simulator says no (K68): a command while the cable is out
makes `SerialBoard._open_serial` rescan, find nothing and set the port to
None, and from then on `connect()` raises "No port found" without scanning,
so a replugged board is lost until LumaViewPro restarts. This probe asks the
same question of the connected scope, where the operator pulls and replugs
the cable.

Each stage reads both boards two ways: through the API (`motor_connected`,
`led_connected`, `get_actual_position('Z')`) and with one INFO exchange
through the production driver, the command the reconnect rides on. The
driver's port and open state are reported beside them.

    python tests/capability/serial_p01_cable_pull.py             # a simulated LS850T
    python tests/capability/serial_p01_cable_pull.py --hardware  # the connected scope

Under --hardware the probe waits for the boards' ports to leave and return
in the OS's port list, so the operator pulls and replugs the one cable when
it says PULL and REPLUG.
"""

import sys
import time
import traceback

from harness import HARDWARE, banner, check, figure, hardware_session, make_session, report, void

# Gap between the two commands at each stage, as K68's simulator run had it.
_COMMAND_GAP_S = 2.0
# How long the operator has for each cable step.
_OPERATOR_S = 180.0
# Enumeration settles after the port reappears before the first command.
_SETTLE_S = 3.0


def _boards(scope):
    return {'motor': scope._motion_driver, 'led': scope._led_driver}


def _emulated(scope):
    """The simulator's two boards, which stand in for the operator's cable."""
    return (
        scope._motion_driver._backend.motor_board,
        scope._led_driver._backend.led_board,
    )


def _listed(board):
    import serial.tools.list_ports

    return any(
        p.vid == board._vid and p.pid == board._pid for p in serial.tools.list_ports.comports()
    )


def _read(scope, stage):
    out = {}
    for name, board in _boards(scope).items():
        answer = board.exchange_command('INFO')
        out[name] = answer
        figure(f'{stage}: {name} INFO', answer)
        figure(f'{stage}: {name} port, open', (board.port, board.driver is not None))
    figure(f'{stage}: API motor_connected', scope.motor_connected)
    figure(f'{stage}: API led_connected', scope.led_connected)
    figure(f'{stage}: API Z actual', scope.motion.get_actual_position('Z'))
    return out


def _wait_for(boards, present, what):
    print(f'\n>>> {what} <<<\n', flush=True)
    deadline = time.monotonic() + _OPERATOR_S
    while time.monotonic() < deadline:
        if all(_listed(b) == present for b in boards):
            return True
        time.sleep(0.25)
    return False


def _pull(scope):
    boards = list(_boards(scope).values())
    if HARDWARE:
        return _wait_for(boards, False, 'PULL the cable now')
    for emulated in _emulated(scope):
        emulated.unplug()
    return True


def _replug(scope):
    boards = list(_boards(scope).values())
    if HARDWARE:
        back = _wait_for(boards, True, 'REPLUG the cable now')
        time.sleep(_SETTLE_S)
        return back
    for emulated in _emulated(scope):
        emulated.replug()
    return True


def _probe(session):
    scope = session.scope
    banner('before')
    before = _read(scope, 'before')
    check('both boards answer INFO before the pull', all(before.values()))

    banner('cable out')
    if not check('the boards left the port list', _pull(scope)):
        # With the cable never out, the replug reads would only say a
        # connected board answers -- no verdict on K68 at all.
        return
    for n in (1, 2):
        _read(scope, f'out {n}')
        time.sleep(_COMMAND_GAP_S)

    banner('replugged')
    check('the boards are back in the port list', _replug(scope))
    after = [_read(scope, f'back {n}') for n in (1, 2)]
    time.sleep(_COMMAND_GAP_S)
    for name in ('motor', 'led'):
        void(
            f'the {name} board answers INFO after the replug',
            any(a[name] for a in after),
            'K68: a command while the cable was out left the port None; connect() never rescans',
        )


def main():
    if HARDWARE:
        with hardware_session() as (session, _runner):
            _probe(session)
        return
    session, _live = make_session(
        'probe_serial', home=True, microscope='LS850T', simulator_tier='firmware'
    )
    try:
        _probe(session)
    finally:
        session.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
