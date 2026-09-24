"""P04 -- what does an out-of-range turret slot actually do?"""

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
