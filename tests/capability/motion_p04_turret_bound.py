"""P04 -- what does an out-of-range turret slot actually do?"""

import sys
import traceback
from harness import check, make_session, report
from modules.exceptions import PositionOutOfRangeError

s, _live = make_session('probe_motion', microscope='LS850T')
try:
    m = s.scope.motion
    m.home('ALL')
    print('T axis limits:', m.get_axis_limits('T'), flush=True)
    m.move_turret(1)
    # Z sits off its floor so a refusal that parked Z first would show.
    m.move_absolute('Z', 1000.0)
    z_before = m.get_current_position('Z')
    for bad in (0, 5, 99, -3):
        try:
            m.move_turret(bad)
            landed = m.get_current_position('T')
            check(f'move_turret({bad}) refused', False, f'accepted, T landed at {landed}')
        except PositionOutOfRangeError as e:
            z_after = m.get_current_position('Z')
            check(
                f'move_turret({bad}) refused naming the slots, before the Z park',
                e.bound == 'turret slots' and abs(z_after - z_before) < 5.0,
                f'{str(e)[:60]} Z before={z_before} after={z_after}',
            )
        m.move_turret(1)
except BaseException:
    traceback.print_exc()
    check('probe ran', False)
finally:
    s.shutdown()
sys.exit(report())
