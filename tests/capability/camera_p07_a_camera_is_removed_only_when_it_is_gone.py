"""Camera P07: a camera is declared removed when it is gone, idle or streaming, and only then.

The bench half of PYLON_TRANSIENT_LATCH_PLAN_2026-10-16: a Basler camera is
declared removed only on a definite signal (the SDK's removal callback, a
grab that finds no device, a run of failed grabs), never on one failed
write. What the code cannot show is that a real unplug still produces one of
those signals in each state the camera can be in:

    idle       streaming stopped: only the SDK's removal callback can see it
    streaming  live frames: the grab path sees it as well
    writes     gain writes back to back: the write that meets the unplug is
               rejected or refused, and the camera is declared removed

Each phase is its own run on hardware, since the camera is plugged back in
between; on the simulator a run with no phase runs all three:

    python tests/capability/camera_p07_a_camera_is_removed_only_when_it_is_gone.py
    python tests/capability/camera_p07_a_camera_is_removed_only_when_it_is_gone.py idle --hardware
    python tests/capability/camera_p07_a_camera_is_removed_only_when_it_is_gone.py streaming --hardware
    python tests/capability/camera_p07_a_camera_is_removed_only_when_it_is_gone.py writes --hardware

On hardware the probe prints `UNPLUG NOW` and watches for up to WINDOW_S;
the camera's own USB cable is pulled at any moment in that window. On the
simulator `simulation.remove_camera()` stands in for the hand.
"""

import contextlib
import random
import sys
import threading
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report

WINDOW_S = 300.0
# How long after the unplug the camera may take to be declared removed. The
# SDK's removal callback is asynchronous; a USB detach is seen by the host
# within about a second.
DECLARED_WITHIN_S = 10.0
POLL_S = 0.05


def _the_hand(session):
    """Remove the camera: the operator's hand on hardware, the simulator's at random."""
    if HARDWARE:
        print(f'UNPLUG NOW: pull the camera USB cable within {WINDOW_S:.0f} s', flush=True)
        return time.monotonic()
    delay = random.uniform(0.5, 2.0)
    figure('simulated removal after s', round(delay, 2))
    threading.Timer(delay, session.scope.simulation.remove_camera).start()
    return time.monotonic() + delay


def _until_removed(imaging, writes=None):
    """Wait for the removal flag; with ``writes``, keep writing gain meanwhile.

    Answers when the flag was seen (monotonic), and each write's answer.
    """
    answers = []
    until = time.monotonic() + WINDOW_S
    gain = 0.0
    while time.monotonic() < until:
        if imaging.camera_removed:
            return time.monotonic(), answers
        if writes is None:
            time.sleep(POLL_S)
            continue
        gain = 6.0 if gain == 0.0 else 0.0
        try:
            imaging.set_gain_db(gain)
            answers.append('applied')
        except Exception as e:
            answers.append(f'{type(e).__name__}:{getattr(e, "reason", e)}')
        # Paced, so a 300 s window logs thousands of writes, not millions.
        time.sleep(POLL_S)
    return None, answers


def _next_command(imaging):
    try:
        imaging.set_gain_db(3.0)
        return 'applied'
    except Exception as e:
        return f'{type(e).__name__}:{getattr(e, "reason", e)}'


def _phase(session, phase):
    imaging = session.scope.imaging
    if phase == 'idle':
        imaging.disable_streaming()
    else:
        imaging.enable_streaming()
    figure('streaming when the hand comes', imaging.is_streaming())
    check('the camera starts connected', not imaging.camera_removed)
    if phase == 'writes':
        imaging.set_gain_db(0.0)
    unplugged = _the_hand(session)
    seen, answers = _until_removed(imaging, writes=phase == 'writes' or None)
    figure('declared removed', seen is not None)
    if seen is not None and HARDWARE:
        # On hardware the window opens at the prompt, not at the pull; the
        # operator's hand is the delay, so the gap is reported, not judged.
        figure('declared s after the prompt', round(seen - unplugged, 2))
    elif seen is not None:
        figure('declared s after the removal', round(seen - unplugged, 2))
        check(
            f'declared within {DECLARED_WITHIN_S:.0f} s of the removal',
            seen - unplugged < DECLARED_WITHIN_S,
        )
    check('the camera was declared removed', seen is not None)
    if phase == 'writes':
        failed = [a for a in answers if a != 'applied']
        figure('gain writes applied before the removal', answers.count('applied'))
        figure('the writes that did not apply', failed)
        check(
            'each write that did not apply was rejected or refused not_connected',
            all(
                a.endswith(':not_connected') or a.startswith('CameraSettingRejected:')
                for a in failed
            ),
            failed,
        )
    after = _next_command(imaging)
    figure('the next command answered', after)
    check('the next command is refused not_connected', after.endswith(':not_connected'), after)


PHASES = ('idle', 'streaming', 'writes')


def main():
    asked = [a for a in sys.argv[1:] if a in PHASES]
    if HARDWARE and len(asked) != 1:
        print(f'usage: {sys.argv[0]} {{{"|".join(PHASES)}}} --hardware')
        sys.exit(2)
    for phase in asked or PHASES:
        figure('phase', phase)
        if HARDWARE:
            session_cm = hardware_session()
        else:

            @contextlib.contextmanager
            def _sim():
                session, _live = make_session('probe_camera_removal', microscope='LS850T')
                try:
                    yield session, None
                finally:
                    session.shutdown()

            session_cm = _sim()
        with session_cm as (session, _runner):
            _phase(session, phase)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
