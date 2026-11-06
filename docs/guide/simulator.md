# The simulator

The simulated scope: its tiers, its models, and the faults it injects.

`simulate=True` builds a scope whose boards and camera are simulated, and every
member works on it as on the hardware: a change is proven there before it
reaches a bench. The camera images a specimen fixed to the stage, so moving the
stage moves the image.

The simulator's own surface, `scope.simulation`, is outside what `4.0.0`
freezes: it may change in any 4.x release. It carries no version of its own; it
ships with the release.

**Tiers.** `settings['simulator_tier']` in the settings handed to `create`
chooses the fidelity:

| Tier | What is simulated |
|---|---|
| `fast` | A stand-in for each driver, with no timing: moves and homes end at once. For tests. |
| `firmware` | The production drivers, talking to the boards' real firmware run in a MicroPython process; the motors move at once. |
| `realistic` | The same firmware, its motors moving at the speed fitted to bench moves, so a move or a home takes seconds. Shipped. |

A Classic model is simulated as its own devices behind the production drivers:
the FX2 camera and LED board, and on the LS720 the TMCM-6110 motor board.

**Faults.** `scope.simulation` injects what the hardware can do wrong, through
the same production driver path a real fault takes: a board that drops replies,
a camera unplugged, a home that never trips its switch, an open lid. Each is a
guarded write, refused on a real scope (`'needs_simulated_scope'`) and on a board
that cannot show it (`'not_injectable_on_board'`). `injected()` answers what is
injected, and `clear()` removes it all.

Which board can show a fault depends on how the model is simulated. Each
member's own fault, and its arguments, are in the
[reference](../api-reference.md#SimulationAPI):

| Member | Fast | Firmware, realistic | Classic |
|---|---|---|---|
| `drop_replies` | motor | motor, LED | refused |
| `delay_reply`, `reboot` | refused | motor, LED | refused |
| `silence`, `unsilence` | motor | refused | TMCM-6110; FX2 (no `unsilence`) |
| `unplug`, `replug` | motor | motor, LED | TMCM-6110; FX2 (no `replug`) |
| `fail_home` | refused | every way | TMCM-6110, `switch_never_trips` |
| `hold_axis` (in-process) | motor | refused | refused |
| `interlock` | refused | refused | TMCM-6110 |
| `answer_model` | motor | refused | refused |
| `remove_camera` | camera | camera | FX2 |
| `refuse_camera_write`, `refuse_camera_read`, `stall_stream`, `delay_delivery`, `light_outside` | camera | camera | refused |
| `refuse_stream_start`, `refuse_led_write`, `bind_camera_usb_driver` | refused | refused | FX2 |

A fault is armed for the next command, not the one in flight. A removed camera,
an unplugged or silenced FX2 and a reboot under way stay as they are after
`clear()`, as on a real scope.

```python
scope.simulation.drop_replies('motor', 1)   # the motor board misses its next reply
print(scope.simulation.injected())
```

```python raises=HardwareCommandRefusedError:position_unread
scope.motion.move_z(4000.0)                 # position_unread: the move was not sent
```

```python
scope.simulation.clear()
scope.simulation.remove_camera()            # as if its cable were pulled
assert 'camera' in scope.unconnected_parts()
```

```python raises=HardwareCommandRefusedError:not_connected
scope.camera.set_gain_db(2.0)               # not_connected, until a restart
```

```python model=LS720
scope.simulation.interlock(lid_open=True)
print(scope.motion.interlocks())
```

```python raises=HardwareCommandRefusedError:lid_open_before_motion
scope.motion.move_xy(x_um=1000.0)           # lid_open_before_motion
```

`create` also arms a fault before bring-up, so what bring-up does with it is
what a test sees. The simulated scope is one record, `SimulatedScope`, passed
as `create(simulate=True, simulated=...)`: its `model` builds the scope as
another model than the settings declare, its `camera_stall` stops the camera's
frames for a while, and its `file_stall` holds the save drive. Over REST the
faults are at `/api/v1/scope/simulation/<member>`.
