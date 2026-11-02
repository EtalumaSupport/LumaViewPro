"""Motion P12: a capture asked as a stage move starts waits the move out, and how long a full-travel move would make it wait.

A move leaves frame validity pending until the stage settles, so a
`capture_still` asked as a move starts drains frames for the whole move.
The capture's wait on the camera lane is a wall-time bound (at least
`_CAPTURE_WAIT_TIMEOUT_S`, 30 s), so a move longer than that bound would
end the capture `call_overran` rather than with a frame (W2d plan E4,
bench row B2).

For X and Y the probe starts a move across SPAN_UM, for Z a rise of
Z_SPAN_UM from 0, and on a scope with a turret one slot change; it asks a
capture at once and records when the move settled, when the capture
returned and what it returned. Whether the capture waited is its frame
judged against the move's settle, as frame validity recorded it. Each
axis's speed over its span gives the time a move across its whole range
would take: derived from the span, not measured over the range.

    python tests/capability/motion_p12_a_capture_behind_a_move.py             # a simulated LS850T
    python tests/capability/motion_p12_a_capture_behind_a_move.py --hardware  # the connected scope

On hardware the stage homes at bring-up; then Z goes to 0, its lowest, and
stays there for every X and Y move, which stay within SPAN_UM of where the
home left the stage, nowhere near an end: an objective raised while the
stage crosses its travel runs into the stage (an LS850T, 2026-10-14). Z
rises only after every X and Y move, at the stage's home position, as an
autofocus does, and goes back to 0 before the turret turns. A move that has
not arrived within MOVE_TIMEOUT_S is stopped at once.
"""

import concurrent.futures
import contextlib
import sys
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report

from drivers import host_clock
from modules.frame_validity import FrameValidity, FrameValiditySource
from tests.motion_moves import start_axis_move

MOVE_TIMEOUT_S = 10.0
SPAN_UM = 20000.0
Z_SPAN_UM = 3000.0
# Where the home left X and Y; every move stays within SPAN_UM of it.
CENTRE: dict[str, float] = {}


def _refuse_unless_z_is_down(motion, axis):
    z = motion.get_current_position('Z')
    if axis in ('X', 'Y') and not (z is not None and abs(z) < 1.0):
        raise RuntimeError(f'refusing to move {axis}: Z is at {z}, not 0')


def _arrive(motion, handle, timeout_s=MOVE_TIMEOUT_S):
    """Wait for a move; one that has not arrived in ``timeout_s`` is stopped at once."""
    try:
        return handle.result(timeout=timeout_s)
    except TimeoutError:
        motion.stop_motion()
        raise


def _capture_behind(session, source, start_move):
    """Start a move of ``source``, capture at once; return (move s, capture s, answer, waited).

    The move's end is the motion API's own stamp: once the stage settles,
    ``FrameValidity.motion_settled`` puts the change in force, its
    ``in_force_ns`` the settle and its ``at_seq`` the frames delivered then,
    before the move's operation is done. ``waited`` is the frame judged
    against that stamp: after it, the capture waited the move out; no later
    than the command's first motion change, the frame shows the stage before
    anything moved, which is true. A capture asked as the command starts can
    return that frame before the move does; that is not a capture that
    failed to wait. A move with no settle stamp has no known end: its move
    time is None and ``waited`` says why.
    """
    motion = session.scope.motion
    fv = session.scope.frame_validity
    first = []

    def observed(change_source, *, began_ns):
        change = FrameValidity.invalidate(fv, change_source, began_ns=began_ns)
        if change_source in FrameValidity.MOTION_SOURCES and not first:
            first.append(change.at_seq)
        return change

    # An observer on this probe's own scope, passing each call through.
    fv.invalidate = observed
    try:
        start_ns = host_clock.steady_ns()
        handle = start_move()
        try:
            still = session.scope.imaging.capture_still()
            answer = f'frame {still.image.shape}'
        except Exception as e:
            answer = f'{type(e).__name__}: {getattr(e, "reason", e)}'
        captured = (host_clock.steady_ns() - start_ns) / 1e9
        _arrive(motion, handle)
    finally:
        del fv.invalidate
    settled = fv.changes().get(source)
    if settled is None or settled.in_force_ns is None or settled.in_force_ns < start_ns:
        return None, captured, answer, f'the {source.value} move has no settle stamp'
    moved = (settled.in_force_ns - start_ns) / 1e9
    waited = 'no frame'
    if answer.startswith('frame'):
        frame_seq = still.record.judgments[0].frame_seq
        if frame_seq > settled.at_seq:
            waited = 'after the settle'
        elif not first or frame_seq <= first[0]:
            waited = 'before the move'
        else:
            waited = f'frame {frame_seq} came during the move (settled at frame {settled.at_seq})'
    return moved, captured, answer, waited


def _capture_behind_a_move(session, axis):
    """Move ``axis`` across its span with a capture behind it, then back; return its figures."""
    motion = session.scope.motion
    if axis == 'Z':
        low, high, home = 0.0, Z_SPAN_UM, 0.0
        source = FrameValiditySource.Z_MOVE
    else:
        low, high, home = CENTRE[axis] - SPAN_UM, CENTRE[axis] + SPAN_UM, CENTRE[axis]
        source = FrameValiditySource.XY_MOVE
    _refuse_unless_z_is_down(motion, axis)
    _arrive(motion, start_axis_move(motion, axis, low))
    result = _capture_behind(session, source, lambda: start_axis_move(motion, axis, high))
    _arrive(motion, start_axis_move(motion, axis, home))
    return result


def _judge(name, moved, captured, answer, waited):
    """Grade one capture behind a move; True when the move's settle is known."""
    figure(
        f'{name} move s / capture s',
        (None if moved is None else round(moved, 2), round(captured, 2)),
    )
    figure(f'{name} capture answered', answer)
    check(f'{name}: the capture returned a frame', answer.startswith('frame'), answer)
    check(f'{name}: the move settled, stamped by the motion API', moved is not None, waited)
    check(
        f'{name}: the capture waited the move out, or its frame predates the move',
        waited in ('after the settle', 'before the move'),
        waited,
    )
    return moved is not None


def main():
    if HARDWARE:
        session_cm = hardware_session()
    else:

        @contextlib.contextmanager
        def _sim():
            session, _live = make_session(
                'probe_capture_move', home=True, microscope='LS850T', simulator_tier='realistic'
            )
            try:
                yield session, None
            finally:
                session.shutdown()

        session_cm = _sim()

    with session_cm as (session, _runner):
        motion = session.scope.motion
        _arrive(motion, motion.start_move_z(0.0), timeout_s=30.0)
        check('Z is at 0 before any X or Y move', abs(motion.get_current_position('Z')) < 1.0)
        CENTRE.update({a: motion.get_current_position(a) for a in ('X', 'Y')})
        figure('X, Y where the home left them um', (round(CENTRE['X']), round(CENTRE['Y'])))
        bound_s = session.scope.imaging._CAPTURE_WAIT_TIMEOUT_S
        figure('the capture wait bound floor s', bound_s)
        for axis in ('X', 'Y', 'Z'):
            limits = motion.get_axis_limits(axis)
            if limits is None:
                continue
            moved, captured, answer, waited = _capture_behind_a_move(session, axis)
            if not _judge(axis, moved, captured, answer, waited):
                continue
            span = Z_SPAN_UM if axis == 'Z' else 2 * SPAN_UM
            speed = span / moved
            full_s = (limits.max_um - limits.min_um) / speed
            figure(f'{axis} um/s over the span', round(speed))
            figure(f'{axis} whole range s (derived)', round(full_s, 1))
            check(
                f'{axis}: a move across the whole range is inside the capture bound (derived)',
                full_s < bound_s,
                f'{full_s:.1f} s against {bound_s:.0f} s',
            )
        check('Z is back at 0 before the turret turns', abs(motion.get_current_position('Z')) < 1.0)
        slot = motion.turret_status().slot
        if slot is not None:
            other = 2 if slot == 1 else 1
            # move_turret waits for the change, so it runs on its own thread
            # and the capture is asked as it starts.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                moved, captured, answer, waited = _capture_behind(
                    session,
                    FrameValiditySource.TURRET,
                    lambda: pool.submit(motion.move_turret, other),
                )
            if _judge(f'turret {slot} to {other}', moved, captured, answer, waited):
                check(
                    'a slot change is inside the capture bound',
                    moved < bound_s,
                    f'{moved:.1f} s against {bound_s:.0f} s',
                )
            motion.move_turret(slot)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
