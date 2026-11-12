# Motion

Homing, moves, stops, interlocks, jogging and the turret.

**Homing is required.** An axis's position is unknown until it is homed, and
again after a fault loses it. A move of any axis, and a run, is refused
`'position_unknown'` while any axis of the scope has no position; a kept
position only while one of its own axes has none.
`axes_without_position()` names each axis that has none, and why (`unknown` or
`homing`). A session starts unhomed; the
startup motion ([Bringing a session up](concepts.md#bringing-a-session-up)) homes it.

```python
scope.motion.home()                         # every axis; 'Z' or 'T' homes one
assert scope.motion.axes_without_position() == {}
```

`start_home(axis)` starts the same home and returns its `Operation` at once. A
home holds the scope like a run: other work is refused
`'exclusive_activity_running'`, its `holder` `home`, and a second home
`'home_in_flight'`. Only the LS720's board can stop a home; on every other, a
stop while the home drives is refused (`'home_not_stoppable'`) and the home runs
to its end.

A home leaves Z at the bottom, the turret on slot 1 and the stage at the middle
of its travel. With the setting `motion.restore_after_home` on (off as shipped;
Advanced Settings' *Restore Focus and Objective After Home*), it then puts the
turret back on its slot and Z back at its target, each only when its position
was known before the home, and nothing while any axis has none; X and Y stay
where the home left them. `home(axis, put_back=True)` or `put_back=False`
decides one home whatever the setting says. A put-back that fails raises its
move's error, with every homed axis known.

```python
scope.motion.home(put_back=True)            # back on its objective, Z where it was
```

**A position to keep is read at rest.** A position you will act on later -- a
step, a bookmark, a sweep's centre -- is read while its axes are still, by
`scope.motion.positions_to_keep`. It is refused `AxisMovingError` (`'axis_moving'`)
while an axis it names moves, and `AxisStateUnknownError` (`'position_unknown'`)
while one of them has no position: a number is never
answered for a position the controller cannot vouch for.

```python
kept = scope.motion.positions_to_keep(('X', 'Y', 'Z'), then='saving a step')
print(kept['Z'])
```


**Moves.** A move is asked for by what it moves, in stage micrometres:
`move_xy(x_um, y_um)` moves X and Y as one request, an axis given as None left
where it is; `move_z(z_um)` moves the focus; `move_xy_by` and `move_z_by` move
by a distance. Each returns once the move has ended. A target outside an axis's
limits is refused before anything drives (`PositionOutOfRangeError`); the
limits are `get_axis_limits(axis)`.

```python
scope.motion.move_xy(x_um=60000.0, y_um=40000.0)
scope.motion.move_z(4000.0)
scope.motion.move_z_by(-50.0)
print(scope.motion.axis_positions())        # each axis's position and state
print(scope.motion.get_axis_limits('Z'))
```

A Z move to a place approaches from below, so the gears' backlash is taken up
the same way every time; a move by a distance does not, unless asked
(`overshoot_enabled`).

**Started moves.** Each X, Y and Z move has a `start_` form that returns at
once with an [`Operation`](python.md#operations): wait on `result()`, or
`stop()` it. A move's `stop()` is `stop_motion()` while the move still runs: it
halts every axis and ends the live run. A stop of an operation that has ended
does nothing, so a late stop never halts what took the axis after it.

```python
moving = scope.motion.start_move_z(3000.0)
moving.result(timeout=30)                   # None once the move has ended
```

**How a move ends.** A move that does not end where asked raises a
`MoveNotCompletedError`: `MoveStoppedError`, a notice, when someone stopped it
or a newer move superseded it (`'stopped'`, `'superseded'`); `MoveFaultError`, a
fault, when the board failed, stalled or lost the axis, which leaves the axis
unknown until homed again.

**Stopping.** `stop_motion()` halts every axis at once, whoever started the
move, and ends the live run (`'stage_stopped'`). No holder and no control
session refuses a stop. It is refused on a scope with no motor controller
(`'axis_absent'`, `'not_connected'`), and while a home drives on a board that
cannot stop one (`'home_not_stoppable'`). A board that does not take the STOP
raises `MotorStopFailedError`: the stage may still be moving.

**Motion you did not start.** `wait_until_finished_moving()` returns once every
axis moving now has stopped, and raises if one did not stop well;
`get_target_positions()` answers where each axis was last commanded to go, and
None for an axis whose position is not known. A move you started is judged by
its own operation's `result()`.

**Interlocks.** The LS720's stage refuses to move with its lid open or its
stage unpowered (`'lid_open_before_motion'`, `'stage_unpowered_before_motion'`);
`interlocks()` answers which are tripped now. No other model has one.

**The turret.** A scope with a turret (`scope.capabilities.has_turret`) moves
it by slot, 1 to 4: `move_turret(slot)`. `turret_status()` answers its slot,
or `TurretStatus('unknown', None)` while it is not known; None means the scope
has no turret. A turret move is three moves: Z lowered, the turret turned, and
Z put back (`restore_z=False` leaves Z down, for a caller that sets Z next);
a recording refuses it. The objective is the slot's assignment
([Bringing a session up](concepts.md#bringing-a-session-up), `session.objective_id`).

```python model=LS850T
session.assign_turret_objective(2, '10x Oly')
scope.motion.move_turret(2)
print(scope.motion.turret_status())
print(session.objective_id)                 # '10x Oly', slot 2's objective
```

**Jogging.** `start_jog(axis, steps, coarse)` moves X, Y or Z by `steps` of
the active objective's fine or coarse jog step: `1` or `-1` for one step, a
fraction or a multiple for a gesture that scales it. It is refused while any
axis of the scope has no known position, before the objective is asked, so its
refusal says to home the scope. `jog_step_um(axis,
coarse)` answers the step itself, for showing it.

**Centering on a point.** `start_center_on(x_fraction, y_fraction)` moves X
and Y so a point of the camera's frame comes to its centre: the point is a
fraction of the frame's width from column 0 and of its height from row 0, in
the frame as `capture_still().image` holds it (row 0 is the top of the
picture, as in the live view and a saved file). A fraction is the same place on the sample at
any binning. It refuses as a jog does, then with `ObjectiveUnknownError` or
`PixelSizeUnknownError` when no distance can be measured.
`runtime_state.frame_point_distance(x_fraction, y_fraction)` answers the
distance without moving, refusing as `start_center_on` does when it cannot be
measured.
`runtime_state.frame_point_plate(x_fraction, y_fraction)` answers the point's
position on the plate, or the refusal that says why it is not known (no X or
Y, X or Y not homed, no distance measurable), raised to no one.
