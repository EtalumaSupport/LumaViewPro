"""P16: an objective chosen by name turns the turret to the one slot holding it, or is refused before anything moves.

Asked of the API alone, as REST or a script would: no GUI control chooses an
objective by name, so `ScopeSession.start_select_objective` is reached only
here and through its route. Four picks, each read through the public surface
(the operation, `MotionAPI.turret_status`, `ScopeSession.objective_id`):

1. an objective two slots hold is refused `objective_in_several_slots`
   when the call is made, and the turret stays where it was;
2. the objective already in the light path is a pick that changes nothing:
   the operation is done at once with False, and the turret stays;
3. an objective one other slot holds turns the turret there: the operation
   ends with True, the slot is that one and the active objective is it;
4. that pick back, stopped while the turret turns, ends `stopped`, with no
   slot claimed and no objective active; a pick after it lands on the slot.

    python tests/capability/motion_p16_an_objective_chosen_by_name.py             # a simulated LS850T
    python tests/capability/motion_p16_an_objective_chosen_by_name.py --hardware  # the connected scope

The hardware run needs, in the installation's current.json, the light path's
slot assigned, one objective assigned to two other slots, and another
assigned to one slot alone. Z goes to 0, its lowest, before the first pick
and stays there: a pick that turns lowers Z to 0 and puts it back, so no
slot's glass, whatever is mounted, turns or rises toward the sample. Nothing
is assigned or saved.
"""

import contextlib
import sys
import threading
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report

# The simulated layout: slot 1 in the light path, one objective in two slots,
# another in one.
SIM_SLOTS = {1: '20x w/collar', 2: '4x Oly', 3: '40x w/collar', 4: '40x w/collar'}
TURN_TIMEOUT_S = 60.0
# How long the turret turns before the stop, once it is seen moving.
INTO_THE_TURN_S = 0.3


def _active(session):
    try:
        return session.objective_id
    except Exception as e:
        return f'unknown ({getattr(e, "reason", type(e).__name__)})'


def _turret(session):
    status = session.scope.motion.turret_status()
    return status.state, status.slot


def _layout(slots, here):
    """The objective two slots hold, and one held by one slot other than the light path's."""
    held = {}
    for slot, objective in slots.items():
        if objective:
            held.setdefault(objective, []).append(slot)
    several = next((o for o, at in held.items() if len(at) > 1 and o != here), None)
    alone = next((o for o, at in held.items() if len(at) == 1 and o != here), None)
    return several, alone


def _ending(op):
    try:
        return op.result(timeout=TURN_TIMEOUT_S), None
    except Exception as e:
        return None, getattr(e, 'reason', type(e).__name__)


def _stopped_mid_turn(session, objective):
    """Pick ``objective`` and stop the pick once the turret has been turning ``INTO_THE_TURN_S``."""
    from _lumascope.modules.axes import AxisState

    motion = session.scope.motion
    turning = threading.Event()

    def heard(status):
        if status.state is AxisState.MOVING:
            turning.set()

    motion.add_turret_listener(heard)
    try:
        op = session.start_select_objective(objective)
        seen = turning.wait(TURN_TIMEOUT_S)
        if seen:
            time.sleep(INTO_THE_TURN_S)
        op.stop()
    finally:
        motion.remove_turret_listener(heard)
    check('the turret was seen turning before the stop', seen)
    return _ending(op)


@contextlib.contextmanager
def _sim():
    session, _live = make_session(
        'probe_select_objective',
        home=True,
        microscope='LS850T',
        simulator_tier='realistic',
        turret_objectives=dict(SIM_SLOTS),
    )
    try:
        # The hardware bring-up turns the turret to slot 1; the simulated
        # home leaves it in no known slot.
        session.scope.motion.move_turret(1)
        yield session
    finally:
        session.shutdown()


def main():
    from _lumascope.modules.exceptions import ArgumentRefusedError

    if HARDWARE:

        @contextlib.contextmanager
        def _scope():
            with hardware_session() as (session, _runner):
                yield session

        session_cm = _scope()
    else:
        session_cm = _sim()

    with session_cm as session:
        motion = session.scope.motion
        motion.start_move_z(0.0).result(timeout=TURN_TIMEOUT_S)
        if not check('Z is at 0 before any pick', abs(motion.get_current_position('Z')) < 1.0):
            return
        slots = session.scope.runtime_state.get_turret_config()
        start_state, start_slot = _turret(session)
        here = slots.get(start_slot)
        figure('turret slots assigned', slots)
        figure('turret at start (state, slot)', (start_state, start_slot))
        figure('active objective at start', _active(session))
        several, alone = _layout(slots, here)
        if not check('the light path holds an assigned objective', here is not None):
            return
        if not check('one objective is assigned to two slots', several is not None, str(slots)):
            return
        if not check('another objective is assigned to one slot alone', alone is not None):
            return
        alone_slot = next(s for s, o in slots.items() if o == alone)

        # 1. Two slots hold it: refused at the call, nothing moved.
        try:
            session.start_select_objective(several)
            refused = None
        except ArgumentRefusedError as e:
            refused = e.reason
        figure(f'pick {several}', refused)
        check(
            'an objective in two slots is refused at the call',
            refused == 'objective_in_several_slots',
            str(refused),
        )
        check('the refused pick left the turret where it was', _turret(session)[1] == start_slot)

        # 2. The one in the light path: done at once, nothing changed.
        op = session.start_select_objective(here)
        figure(f'pick {here} (done at once, result)', (op.done(), _ending(op)))
        check('picking the active objective is done at once', op.done())
        check('and changes nothing', _ending(op) == (False, None))
        check('the turret stayed', _turret(session)[1] == start_slot)

        # 3. One slot holds it: the turret turns there.
        result, ended = _ending(session.start_select_objective(alone))
        figure(f'pick {alone} (result, ending)', (result, ended))
        check('the pick ended True', result is True and ended is None, str(ended))
        check(f'the turret is in slot {alone_slot}', _turret(session)[1] == alone_slot)
        check(f'the active objective is {alone}', _active(session) == alone, _active(session))

        # 4. Back, stopped while it turns; then picked again.
        result, ended = _stopped_mid_turn(session, here)
        figure(f'pick {here}, stopped (result, ending)', (result, ended))
        check('the stopped pick ended stopped', ended == 'stopped', str(ended))
        state, slot = _turret(session)
        figure('turret after the stop (state, slot)', (state, slot))
        check('no slot is claimed after the stop', slot is None, str(slot))
        figure('active objective after the stop', _active(session))
        check('no objective is active after the stop', _active(session) != here)

        result, ended = _ending(session.start_select_objective(here))
        figure(f'pick {here} again (result, ending)', (result, ended))
        check('the pick after the stop ended True', result is True and ended is None, str(ended))
        check(f'the turret is back in slot {start_slot}', _turret(session)[1] == start_slot)
        check(f'the active objective is {here} again', _active(session) == here)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
