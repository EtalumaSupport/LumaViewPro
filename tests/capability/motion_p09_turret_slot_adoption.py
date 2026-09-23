"""P09 -- when the turret lands on a slot, who changes the active objective?

The move. On a turret scope the active objective IS the objective assigned
to the slot in the light path, derived on every read, so nothing adopts it
and a script does not select it after a move. The GUI's turret_select
(vertical_control.py) only displays the outcome, or asks when the slot has
no assignment. The objectives are deliberately DIFFERENT so the check
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
    m.home('ALL')
    m.home('T')
    m.move_turret(1)
    check(
        'starting objective is slot 1s',
        s.scope.runtime_state.resolve_current_objective()[0] == a,
        f'{a}',
    )

    # move to slot 2, whose objective is DIFFERENT
    m.move_turret(2)
    after_move = s.scope.runtime_state.resolve_current_objective()[0]
    check(
        'move_turret alone makes the slot objective active',
        after_move == b,
        f'after move_turret(2) objective is {after_move!r}, slot 2 holds {b!r}',
    )

    # --- the objective question: API-backed, GUI renders it ---
    check('session.objective_question() exists', callable(s.objective_question))
    check('session.confirm_objective() exists', callable(s.confirm_objective))
    s.clear_turret_objective(3)
    m.move_turret(3)
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
            'confirm_objective assigns the slot, clears the question, and is active',
            s.get_settings_snapshot()['turret_objectives'][3] == b
            and s.objective_question() is None
            and s.scope.runtime_state.resolve_current_objective()[0] == b,
            str(s.get_settings_snapshot()['turret_objectives']),
        )
except BaseException:
    traceback.print_exc()
    check('probe ran', False)
finally:
    s.shutdown()
sys.exit(report())
