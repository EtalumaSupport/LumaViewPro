"""P09 -- when the turret lands on a slot, who changes the selected objective?

The GUI's turret_select (vertical_control.py:746) does three things after the
move: session.set_turret_position(slot), then -- if the slot has an assignment
-- writes the spinner and calls select_objective(); if it does not, it raises
the objective question. This probe asks whether the API alone does the
objective adoption, with the objectives deliberately DIFFERENT so the check
cannot pass by accident.
"""

import sys
import tempfile
import traceback
from harness import check, report, SCRATCH
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

live = tempfile.mkdtemp(prefix='probe_motion_', dir=SCRATCH)
s = ScopeSession.create(complete_settings(live_folder=live, microscope='LS850T'), simulate=True)
try:
    m = s.scope.motion
    objs = s.objective_helper.get_objectives_list()
    a, b = objs[0], objs[1]
    s.assign_turret_objective(1, a)
    s.assign_turret_objective(2, b)
    s.select_objective(a)
    m.home('ALL')
    m.home('T')
    m.move_turret(1)
    s.set_turret_position(1)
    check('starting objective is slot 1s', s.get_current_objective_info()[0] == a, f'{a}')

    # move to slot 2, whose objective is DIFFERENT
    m.move_turret(2)
    after_move = s.get_current_objective_info()[0]
    check(
        'move_turret alone does NOT adopt the slot objective',
        after_move == a,
        f'after move_turret(2) objective is {after_move!r}, slot 2 holds {b!r}',
    )
    s.set_turret_position(2)
    after_record = s.get_current_objective_info()[0]
    check(
        'set_turret_position alone does NOT adopt the slot objective either',
        after_record == a,
        f'after set_turret_position(2) objective is {after_record!r}, slot 2 holds {b!r}',
    )
    check(
        'a script must call select_objective itself to finish the slot change',
        True,
        's.select_objective(settings["turret_objectives"][slot])',
    )
    s.select_objective(s.get_settings_snapshot()['turret_objectives'][2])
    check(
        'after the explicit select_objective the objective matches the slot',
        s.get_current_objective_info()[0] == b,
        f'{b}',
    )

    # --- the objective question: API-backed, GUI renders it ---
    check('session.objective_question() exists', callable(s.objective_question))
    check('session.confirm_objective() exists', callable(s.confirm_objective))
    s.clear_turret_objective(3)
    m.move_turret(3)
    s.set_turret_position(3)
    q = s.objective_question()
    print('objective_question on an unassigned slot ->', q, flush=True)
    check(
        'landing on an unassigned slot produces a question a script can answer',
        q is not None,
        str(q)[:140],
    )
    if q is not None:
        s.confirm_objective(b, turret_position=3)
        check(
            'confirm_objective assigns the slot and clears the question',
            s.get_settings_snapshot()['turret_objectives'][3] == b
            and s.objective_question() is None,
            str(s.get_settings_snapshot()['turret_objectives']),
        )
except BaseException:
    traceback.print_exc()
    check('probe ran', False)
finally:
    s.shutdown()
sys.exit(report())
