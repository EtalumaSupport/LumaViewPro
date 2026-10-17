"""Camera P03: a camera removed during a capture under a live-view auto-gain arm is told once, as the removal.

W2d's bench row B3. A manual still and an autofocus each meet the camera
leaving partway through, with live-view auto-gain armed. A still raises the
`not_connected` refusal and the API posts nothing beside it: the raise is
the report, made once by the caller's boundary. An autofocus is a run: it
ends `failed`, reported once as the `not_connected` refusal, in words naming
the camera. Neither is told again by the auto-gain arm's re-arm.

On hardware each phase is its own run, since the camera has to be plugged
back in between; on the simulator a run with no phase runs both, each in its
own session:

    python tests/capability/camera_p03_a_removed_camera_is_reported_once.py                   # simulated LS850T
    python tests/capability/camera_p03_a_removed_camera_is_reported_once.py still
    python tests/capability/camera_p03_a_removed_camera_is_reported_once.py autofocus
    python tests/capability/camera_p03_a_removed_camera_is_reported_once.py autofocus blank  # nothing to focus on
    python tests/capability/camera_p03_a_removed_camera_is_reported_once.py still --hardware  # the connected scope
    python tests/capability/camera_p03_a_removed_camera_is_reported_once.py autofocus --hardware

On hardware the probe prints `UNPLUG NOW` and captures back to back for up
to WINDOW_S; the camera's USB cable is pulled at any moment in that window.
The camera's own cable: on a scope whose main cable carries the motor board
too, pulling that removes the stage as well, and the autofocus meets both.
On the simulator `simulation.remove_camera()` stands in for the hand: at a
random moment among the stills, and partway into an autofocus run's sweep,
since a hand that lands between two runs tests nothing. A still that is refused at its entry, before it reached the
camera, is recorded as such: the question is what a capture under way does.
An autofocus moves only Z, at the stage's home position.
"""

import contextlib
import random
import sys
import threading
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report

WINDOW_S = 300.0
AUTO_GAIN = {
    'target_brightness': 0.5,
    'min_gain_db': 0.0,
    'max_gain_db': 20.0,
    'max_exposure_ms': 100.0,
    'min_exposure_ms': 0.1,
}
BF_MA = 50.0
# A capture refused this soon after it was asked never reached the camera.
AT_ENTRY_S = 0.005


@contextlib.contextmanager
def _told(session):
    """Every notification posted and every outcome reported while the block runs."""
    from modules.notification_center import Severity, notifications

    posted, outcomes = [], []
    lock = threading.Lock()

    def post(notification):
        with lock:
            posted.append(f'{notification.severity.name} {notification.title}')

    def outcome(notification):
        with lock:
            outcomes.append(f'{notification.reason} shown={notification.shown}')

    notifications.add_listener(post, min_severity=Severity.DEBUG)
    session.add_outcome_listener(outcome)
    try:
        yield posted, outcomes
    finally:
        notifications.remove_listener(post)


# How far into an autofocus run the simulator removes the camera: inside
# its sweep, which takes over a second on the simulated scope.
SIM_REMOVAL_INTO_RUN_S = 0.5


def _the_hand(session, delay=None):
    """Remove the camera: the operator's hand on hardware, the simulator's after ``delay`` s."""
    if HARDWARE:
        print(f'UNPLUG NOW: pull the camera USB cable within {WINDOW_S:.0f} s', flush=True)
        return
    if delay is None:
        delay = random.uniform(0.5, 3.0)
    figure('simulated removal after s', round(delay, 2))
    threading.Timer(delay, session.scope.simulation.remove_camera).start()


def _arm(session):
    scope = session.scope
    scope.illumination.led_on('BF', BF_MA)
    scope.imaging.set_auto_gain(True, AUTO_GAIN)
    scope.imaging.capture_and_wait()


def _still(session, runner):
    _arm(session)
    with _told(session) as (posted, outcomes):
        _the_hand(session)
        until = time.monotonic() + WINDOW_S
        stills = 0
        while time.monotonic() < until:
            asked = time.monotonic()
            try:
                session.manual_capture.capture(false_color_on=False).result(timeout=60)
                stills += 1
            except Exception as e:
                took = time.monotonic() - asked
                answer = f'{type(e).__name__}: {getattr(e, "reason", e)}'
                break
        else:
            answer, took = 'no removal in the window', 0.0
        time.sleep(1.0)
    figure('stills saved before the removal', stills)
    figure('the still answered', answer)
    figure('the failing still took s', round(took, 3))
    figure('met the removal', 'at its entry' if took < AT_ENTRY_S else 'under way')
    figure('posted', posted)
    figure('outcomes reported', outcomes)
    check('the still raised the not_connected refusal', answer.endswith('not_connected'), answer)
    check('the API posted nothing beside the raise', posted == [], posted)


def _autofocus(session, runner):
    # The home leaves Z at the end of its travel, where an autofocus refuses
    # to search; from the middle each run sweeps, capturing as it goes.
    motion = session.scope.motion
    z = motion.get_axis_limits('Z')
    motion.start_move_absolute('Z', (z.min_um + z.max_um) / 2).result(timeout=60)
    _arm(session)
    runs = 0
    outcome, met = None, 'no removal in the window'
    # A run before the removal may end on its own (a field with nothing to
    # focus on), so runs follow one another with no pause until one meets the
    # removal, and only what is told from that run's start on is judged.
    with _told(session) as (all_posted, all_outcomes):
        if HARDWARE:
            _the_hand(session)
        until = time.monotonic() + WINDOW_S
        while time.monotonic() < until:
            seen = (len(all_posted), len(all_outcomes))
            if not HARDWARE and runs == 0:
                _the_hand(session, SIM_REMOVAL_INTO_RUN_S)
            try:
                outcome = runner.start_autofocus('BF').result(timeout=120)
            except Exception as e:
                # The removal landed between two runs: the start is refused.
                met = f'between runs: {type(e).__name__}: {getattr(e, "reason", e)}'
                outcome = None
                break
            if any(o.split()[0] == 'not_connected' for o in all_outcomes[seen[1] :]):
                met = 'under way'
                break
            runs += 1
        time.sleep(1.0)
        posted, outcomes = all_posted[seen[0] :], all_outcomes[seen[1] :]
    figure('autofocus runs before the removal', runs)
    figure('met the removal', met)
    if outcome is not None:
        figure('the run ended', (outcome.status, outcome.reason))
        figure('its message', outcome.message)
    figure('posted', posted)
    figure('outcomes reported', outcomes)
    check('the removal met a run under way', met == 'under way', met)
    check('the autofocus ended failed', outcome is not None and outcome.status == 'failed')
    check(
        'the message names the camera',
        outcome is not None and 'camera' in outcome.message.lower(),
        outcome and outcome.message,
    )
    # What the run told before the removal is its own (a field too dark for
    # the auto-gain arm reports exposure_at_maximum at each run's start); the
    # removal is told once, and nothing is told after it.
    reasons = [o.split()[0] for o in outcomes]
    check(
        'the removal was reported once, as the not_connected refusal, and nothing after it',
        reasons.count('not_connected') == 1 and reasons[-1] == 'not_connected',
        outcomes,
    )


PHASES = {'still': _still, 'autofocus': _autofocus}


def main():
    asked = [a for a in sys.argv[1:] if a in PHASES]
    if not asked and HARDWARE:
        print(f'usage: {sys.argv[0]} {{{"|".join(PHASES)}}} --hardware')
        sys.exit(2)
    for phase in asked or list(PHASES):
        _phase(phase)


def _phase(phase):
    figure('phase', phase)
    # A simulated field with nothing to focus on, as a scope with no sample sees.
    blank = 'blank' in sys.argv[1:]
    if HARDWARE:
        session_cm = hardware_session()
    else:

        @contextlib.contextmanager
        def _sim():
            from tests.scope_fakes import TEST_TURRET_OBJECTIVES

            session, _live = make_session(
                'probe_removed_camera',
                home=True,
                microscope='LS850T',
                turret_objectives=dict(TEST_TURRET_OBJECTIVES),
            )
            # The hardware bring-up turns the turret to slot 1; the simulated
            # session's home leaves it in no known slot.
            session.scope.motion.home('T')
            session.scope.motion.move_turret(1)
            if blank:
                session.scope._camera_driver.set_test_pattern(enabled=True, pattern='white')
            try:
                yield session, session.create_protocol_runner()
            finally:
                session.shutdown()

        session_cm = _sim()
    with session_cm as (session, runner):
        PHASES[phase](session, runner)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
