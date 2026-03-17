# IO Executor Polling Design — Single Source of Truth

## Status: DESIGN (not yet implemented)

## Problem

Multiple consumers independently poll firmware for motion state via serial:

| Consumer | What it polls | Rate | Serial cost |
|----------|--------------|------|-------------|
| Protocol executor `_scan_iterate()` | 3x `get_target_status` + `get_overshoot` | ~1000 Hz (1ms sleep) | 4000 cmds/sec |
| AF executor `_iterate()` | `get_target_status('Z')` + `get_overshoot` | IOTask throughput | 2 per tick |
| `wait_until_finished_moving()` | 4x `get_target_status` + `get_overshoot` | 20 Hz (50ms sleep) | 100 cmds/sec |
| GUI (was, now cached) | `get_current_position` | 10 Hz | 0 (fixed) |

Total during a protocol move: potentially **4000+ serial commands/sec** from the protocol executor alone.

## Solution: IO Executor Owns the Poll

The IO executor already owns all hardware commands. Extend it to own **state polling** too.

### Architecture

```
IO Executor Thread
├── Process IOTask queue (move commands, LED, capture, etc.)
└── State poll loop (runs between tasks when any axis is MOVING)
    ├── For each MOVING axis: get_target_status() → update _axis_state
    ├── For each MOVING axis: get_overshoot() check
    ├── Update _pos_cache (interpolated or from firmware)
    └── Signal move_complete Events when axis arrives
```

### Poll Rate

- **50 Hz** while any axis is MOVING (20ms interval)
- **0 Hz** when all axes are IDLE (no polling at all)
- Total serial cost: 1-4 `get_target_status` + 1 `get_overshoot` per tick = **50-250 cmds/sec max**

This is a 16-80x reduction from the current protocol executor behavior.

### Position Interpolation

While an axis is MOVING, interpolate position in the cache for smooth UI:

```python
# Known at move command time:
self._move_start_time[axis] = time.monotonic()
self._move_start_pos[axis] = current_pos
self._move_target_pos[axis] = target_pos
self._move_velocity[axis] = velocity_from_motorconfig(axis)  # um/sec

# Interpolation (called from get_target_position):
def _interpolated_position(self, axis):
    if self._axis_state[axis] != AxisState.MOVING:
        return self._pos_cache[axis]

    elapsed = time.monotonic() - self._move_start_time[axis]
    distance = self._move_velocity[axis] * elapsed
    start = self._move_start_pos[axis]
    target = self._move_target_pos[axis]

    # Clamp to target (don't overshoot in interpolation)
    if target > start:
        return min(start + distance, target)
    else:
        return max(start - distance, target)
```

The TMC5072 uses trapezoidal velocity profiles (accel → cruise → decel), so linear interpolation won't be exact. But for UI display purposes it's good enough — the real position snaps in every 20ms from the firmware poll.

**Refinement (optional):** Use trapezoidal model if we know AMAX/VMAX from motorconfig:
```
Phase 1: accelerating — pos = start + 0.5 * amax * t^2
Phase 2: cruising    — pos = p1 + vmax * (t - t1)
Phase 3: decelerating — pos = p2 + vmax*(t-t2) - 0.5*amax*(t-t2)^2
```

### Consumer Changes

| Consumer | Before | After |
|----------|--------|-------|
| Protocol executor | 4 serial calls per 1ms tick | Read `_axis_state` dict |
| AF executor | 2 serial calls per tick | Read `_axis_state['Z']` |
| `wait_until_finished_moving()` | 5 serial calls per 50ms | Wait on `threading.Event` |
| `is_moving()` | 5 serial calls | Read `_axis_state` dict |
| GUI position display | Cache (already done) | Cache + interpolation |

### Event-Based Wait (replaces polling in wait_until_finished_moving)

```python
# Per-axis completion events
self._move_complete = {ax: threading.Event() for ax in axes}

def wait_until_finished_moving(self):
    """Block until all MOVING axes arrive. Zero serial I/O from caller."""
    for ax, state in self._axis_state.items():
        if state == AxisState.MOVING:
            self._move_complete[ax].wait()

# In the IO executor poll loop:
def _poll_axis_state(self):
    for ax, state in self._axis_state.items():
        if state == AxisState.MOVING:
            if self.motion.target_status(ax):
                self._axis_state[ax] = AxisState.IDLE
                self._pos_cache[ax] = self.motion.target_pos(ax)  # snap to real pos
                self._move_complete[ax].set()
```

### Implementation Steps

1. **Add poll loop to IO executor** — runs at 50Hz between IOTasks when any axis is MOVING. Uses a threading.Timer or checks elapsed time in the task processing loop.

2. **Add move metadata** — store start_time, start_pos, target_pos, velocity per axis when move is commanded. Used for interpolation and poll scheduling.

3. **Add move_complete Events** — per-axis threading.Events, set by poll loop on arrival.

4. **Replace `wait_until_finished_moving()`** — wait on Events instead of polling serial.

5. **Replace protocol executor polling** — `_scan_iterate` lines 701-706 read `get_axis_state()` instead of `get_target_status()`. Remove `get_overshoot()` call (overshoot state tracked in axis model).

6. **Replace AF executor polling** — `_iterate` line 250 reads `get_axis_state('Z')` instead of `get_target_status('Z')`.

7. **Add interpolated position** — `get_target_position()` returns interpolated value during MOVING, snapped value when IDLE.

### Overshoot Handling

Z overshoot (backlash compensation) is currently tracked by `motion.overshoot` which is a serial read. The IO executor should track this:

```python
# When move_abs_pos is called with overshoot_enabled:
self._axis_state['Z'] = AxisState.MOVING  # stays MOVING through overshoot
# The motion board internally does: move past target → move back to target
# Both sub-moves complete before target_status returns True
# So the poll loop naturally handles this — Z stays MOVING until truly arrived
```

### Velocity Data Source

Motor velocities come from motorconfig INI files:
- `VMAX` register value per axis
- Conversion: `um/sec = VMAX * (um_per_ustep) * (clock_freq / 2^24)`
- Already available in `motorconfig` — just need an accessor like `velocity_um_per_sec(axis)`

### What This Does NOT Change

- Move command flow — still goes through `scope.move_absolute_position()` etc.
- LED commands — no polling needed, already push-based
- Camera — separate executor, separate state model
- Homing — blocking calls, state set before/after (already implemented)

### Risks

- **Poll loop timing**: Must not starve IOTask processing. The poll should yield to queued tasks.
- **Interpolation accuracy**: Trapezoidal vs linear — UI might show jumpy position during accel/decel. Acceptable for display.
- **Overshoot edge case**: If overshoot timing changes in firmware, the poll loop still handles it correctly (waits for target_status=True).

### Verification

1. Profile serial traffic during protocol execution — should see ~50 cmds/sec max instead of 4000+
2. UI position display should update smoothly during moves (interpolation)
3. Protocol step timing should be unchanged or faster (less serial contention)
4. AF convergence should be unchanged
5. All existing tests pass
