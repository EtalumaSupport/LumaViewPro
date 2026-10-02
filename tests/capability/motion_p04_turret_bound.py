"""P04 -- what does an out-of-range turret slot actually do?"""

import sys
import traceback
from harness import check, make_session, report

s, _live = make_session('probe_motion', microscope='LS850T')
try:
    m = s.scope.motion
    m.home('ALL')
    print('T axis limits:', m.get_axis_limits('T'), flush=True)
    m.move_turret(1)
    for bad in (0, 5, 99, -3):
        try:
            m.move_turret(bad)
            landed = m.get_current_position('T')
            check(f'move_turret({bad}) refused', False, f'accepted, T landed at {landed}')
        except Exception as e:
            check(f'move_turret({bad}) refused', True, f'{type(e).__name__}: {str(e)[:70]}')
        m.move_turret(1)
except BaseException:
    traceback.print_exc()
    check('probe ran', False)
finally:
    s.shutdown()
sys.exit(report())
