"""P00 -- session bring-up smoke: what the motion API reports headless."""

import tempfile

import harness

LIVE = tempfile.mkdtemp(prefix='probe_motion_', dir=str(harness.SCRATCH))

from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

s = ScopeSession.create(complete_settings(live_folder=LIVE), simulate=True)
try:
    m = s.scope.motion
    print('motor_connected:', s.scope.motor_connected)
    print('has_xy_stage:', s.scope.capabilities.has_xy_stage)
    print('has_turret:', getattr(s.scope.capabilities, 'has_turret', 'n/a'))
    print('limits X:', m.get_axis_limits('X'))
    print('limits Y:', m.get_axis_limits('Y'))
    print('limits Z:', m.get_axis_limits('Z'))
    print('current:', m.get_current_position())
    print('plate position:', s.get_current_plate_position())
    print('objective:', s.scope.runtime_state.resolve_current_objective()[0])
finally:
    s.shutdown()
