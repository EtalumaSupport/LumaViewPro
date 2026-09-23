"""P03 -- turret / objective selection, headless.

GUI capabilities covered: pick an objective from the spinner
(vertical_control.py:284 select_objective -- on a turret scope it assigns
the slot in the light path), rotate the turret to a slot
(vertical_control.py:746 turret_select, via the four turret buttons),
assign/clear an objective to a turret slot (vertical_control.py:627/647),
home the turret (vertical_control.py:598).
"""

import sys
import tempfile
from harness import check, report, SCRATCH


def main():
    from modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    live = tempfile.mkdtemp(prefix='probe_motion_', dir=SCRATCH)
    s = ScopeSession.create(complete_settings(live_folder=live, microscope='LS850T'), simulate=True)
    try:
        m = s.scope.motion
        caps = s.scope.capabilities
        check('turret-model session reports has_turret', caps.has_turret, 'model=LS850T')

        # --- before any turret command the slot, so the objective, is unknown ---
        from modules.exceptions import ObjectiveUnknownError

        try:
            s.get_current_objective_info()
            check('the objective is unknown before any turret command', False, 'NO RAISE')
        except ObjectiveUnknownError as e:
            check('the objective is unknown before any turret command', True, e.reason)

        m.home('ALL')
        check('turret homes', m.home('T') is True)
        check('has_turret_homed after home', m.has_turret_homed())
        m.move_turret(1, restore_z=True)

        # --- objective selection (the spinner): assigns the slot in the light path ---
        objectives = s.objective_helper.get_objectives_list()
        first, other = objectives[0], objectives[1]
        s.select_objective(first)
        changed = s.select_objective(other)
        now_id, info = s.get_current_objective_info()
        check(
            'select_objective changes the live objective by assigning the slot',
            changed and now_id == other and s.settings['turret_objectives'][1] == other,
            f'{first} -> {now_id}',
        )
        check(
            'objective info carries the jog steps the GUI jog reads',
            all(k in info for k in ('z_coarse', 'z_fine', 'xy_coarse', 'xy_fine')),
            str(sorted(k for k in info if 'coarse' in k or 'fine' in k)),
        )

        # --- assign an objective to a turret slot, then select that slot ---
        s.assign_turret_objective(2, first)
        s.assign_turret_objective(3, other)
        slots = s.get_settings_snapshot().get('turret_objectives')
        check('turret slot assignment is stored', slots is not None, str(slots))

        # --- rotate the turret (what the 4 turret buttons do) ---
        m.move_turret(2, restore_z=True)
        t2 = m.get_current_position('T')
        at2 = s.get_current_objective_info()[0]
        m.move_turret(3, restore_z=True)
        t3 = m.get_current_position('T')
        at3 = s.get_current_objective_info()[0]
        check('turret position changes on move_turret', t2 != t3, f'slot2={t2} slot3={t3}')
        check(
            'the active objective follows the slot',
            (at2, at3) == (first, other),
            f'slot2={at2} slot3={at3}',
        )

        # --- the slot a person last turned to is the preferred slot, which
        # the one slot lookup reads first; nothing else records it.
        check(
            'the last turret move is the preferred slot',
            m.get_preferred_turret_slot() == 3,
            str(m.get_preferred_turret_slot()),
        )

        # --- out-of-range slot: the MOVE must refuse.
        for bad in (0, 5, 99):
            try:
                m.move_turret(bad)
                check(f'move_turret({bad}) refused', False, 'NO RAISE')
            except Exception as e:
                check(f'move_turret({bad}) refused', True, f'{type(e).__name__}: {str(e)[:60]}')
        for bad in (0, 5):
            try:
                s.assign_turret_objective(bad, other)
                check(f'assign_turret_objective({bad}) refused', False, 'NO RAISE')
            except Exception as e:
                check(f'assign_turret_objective({bad}) refused', True, type(e).__name__)

        s.clear_turret_objective(3)
        check('clear_turret_objective runs', True)
    except BaseException:
        import traceback

        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    finally:
        s.shutdown()
    sys.exit(report())


main()
