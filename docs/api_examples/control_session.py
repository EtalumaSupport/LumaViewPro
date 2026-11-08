#!/usr/bin/env python3
# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Holding the scope in a control session, on the simulator.

A client that must not be interfered with takes the control session: while
it is held, only work in it writes to the scope, and every other caller's
write is refused. This takes it, moves and saves a still in it, shows a
write outside it refused, then ends it and shows its id no longer acts.

Run it in a LumaViewPro source folder (how, step by step: README.md here):

    python3 docs/api_examples/control_session.py

The same session over REST is ``control_session.sh``.
"""

from lumascope import ControlSessionRefusedError, ScopeSession


def refused(attempt):
    """The control session's refusal of ``attempt``, printed; it is expected to raise one."""
    try:
        attempt()
    except ControlSessionRefusedError as refusal:
        print(f'{refusal.reason}: {refusal}')
        return refusal
    raise AssertionError('the attempt was not refused')


def move_in(session, session_id):
    """A move made in the control session ``session_id``."""
    with session.acting_in_control(session_id):
        session.scope.motion.move_z(4000.0)


def main():
    session = ScopeSession.create(ScopeSession.load_user_settings('.'), simulate=True)
    try:
        session.run_application_session()
        # A new install has not been told which objective is mounted, and a
        # saved still and a run are refused until it has. Answer with the
        # objective in the light path (on the simulator, the one it proposes).
        question = session.objective_question()
        if question is not None:
            session.confirm_objective(question.proposed, slot=question.turret_slot)
        scope = session.scope

        # The id goes to the taker alone: work in the session names it.
        session_id = session.take_control('plate reader')
        print(f'Held: {session.control}')
        with session.acting_in_control(session_id):
            scope.motion.move_to_well('B2')
            scope.illumination.led_on('BF', 5)
            saving = session.manual_capture.capture(false_color_on=False)
            still = saving.result(timeout=30)[0]
            scope.illumination.leds_off()
        print(f'Saved in the session: {still.file.name}')

        # Any other caller's write is refused while the session is held.
        held = refused(lambda: scope.motion.move_z(4000.0))

        # Ending it stops a run or recording taken in it; its id then acts
        # no more.
        session.end_control(session_id)
        ended = refused(lambda: move_in(session, session_id))
        return still, held, ended
    finally:
        session.shutdown()


if __name__ == '__main__':
    main()
