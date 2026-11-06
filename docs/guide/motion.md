# Motion

Homing, moves, stops, interlocks, jogging and the turret.

**Homing is required.** An axis's position is unknown until it is homed, and
again after a fault loses it. A move, a run or a kept position on an unhomed
axis is refused `'position_unknown'`; `axes_without_position()` names each axis
that has none, and why (`unknown` or `homing`). A session starts unhomed; the
startup motion ([Bringing a session up](concepts.md#bringing-a-session-up)) homes it.

```python
scope.motion.home()                         # every axis; 'Z' or 'T' homes one
assert scope.motion.axes_without_position() == {}
```

A home holds the scope like a run: other requests are refused naming it
(`'home_in_flight'`). On the EL-0940 a home cannot be stopped once its board has
begun it (`'home_not_stoppable'`); it runs to its end.

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

**Started moves.** Each move has a `start_` form that returns at once with an
[`Operation`](python.md#operations): wait on `result()`, or `stop()` it. A stop
of an operation that has ended does nothing, so a late stop never halts what
took the axis after it.

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
move, and ends the live run (`'stage_stopped'`). No holder and no control session
refuses a stop; only a scope with no motor controller does (`'axis_absent'`,
`'not_connected'`).

**Interlocks.** A Classic scope's stage refuses to move with its lid open or
its stage unpowered (`'lid_open_before_motion'`, `'stage_unpowered_before_motion'`);
`interlocks()` answers which are tripped now. The EL-0940 has none.

**The turret.** A scope with a turret (`scope.capabilities.has_turret`) moves
it by slot, 1 to 4: `move_turret(slot)`. `turret_status()` answers its slot,
or `TurretStatus('unknown', None)` while it is not known; None means the scope
has no turret. The objective is the slot's assignment
([Bringing a session up](concepts.md#bringing-a-session-up), `session.objective_id`).

```python model=LS850T
scope.motion.move_turret(2)
print(scope.motion.turret_status())
print(session.objective_id)                 # slot 2's objective
```

**Jogging.** `start_jog(axis, steps, coarse)` moves X, Y or Z by `steps` of
the active objective's fine or coarse jog step: `1` or `-1` for one step, a
fraction or a multiple for a gesture that scales it. Like every move it is
refused while any axis of the scope has no known position, before the
objective is asked, so its refusal says to home the scope. `jog_step_um(axis,
coarse)` answers the step itself, for showing it.
