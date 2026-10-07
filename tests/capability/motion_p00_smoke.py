"""P00 -- session bring-up smoke: what the motion API reports headless."""

import harness

s, _live = harness.make_session('probe_motion')
try:
    m = s.scope.motion
    print('motor_connected:', s.scope.motor_connected)
    print('has_xy_stage:', s.scope.capabilities.has_xy_stage)
    print('has_turret:', getattr(s.scope.capabilities, 'has_turret', 'n/a'))
    print('limits X:', m.get_axis_limits('X'))
    print('limits Y:', m.get_axis_limits('Y'))
    print('limits Z:', m.get_axis_limits('Z'))
    print('current:', m.get_current_position())
    # Before a home the plate position is refused; the axes say why.
    print('axis positions:', m.axis_positions())
    print('objective:', s.scope.runtime_state.resolve_current_objective()[0])
finally:
    s.shutdown()
