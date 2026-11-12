# The simulator

The simulated scope: its tiers, its models, and the faults it injects.

`simulate=True` builds a scope whose boards and camera are simulated, and every
member works on it as on the hardware, so a program can be written and tested
before it meets a scope. The camera images a specimen fixed to the stage, so moving the
stage moves the image. It is the model the settings declare (`microscope`,
shipped `LS850`), so a source copy's simulated scope is an LS850, its camera
`SimulatedCamera-3840x2160` on an EL-0940 model and the FX2 camera on a
Classic one.

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

The tiers are the EL-0940 board's. `firmware` and `realistic` need a
MicroPython runtime on the machine; where there is none, the session runs
`fast` and says why in the log. A Classic model is simulated as its own devices
behind the production drivers, the same on every tier: the FX2 camera and LED
board, and on the LS720 the TMCM-6110 motor board.

**Faults.** `scope.simulation` injects what the hardware can do wrong, through
the same production driver path a real fault takes: a board that drops replies,
a camera unplugged, a home that never trips its switch, an open lid. Each is a
write, refused under another's control session, refused on a real scope
(`'needs_simulated_scope'`), on a board the scope lacks (`'fault_unknown'`) and
on a board that cannot show it (`'not_injectable_on_board'`). `injected()` answers what is
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
| `hold_axis` | motor | refused | refused |
| `interlock` | refused | refused | TMCM-6110 |
| `answer_model` | motor | refused | refused |
| `remove_camera` | camera | camera | FX2 |
| `refuse_camera_write`, `refuse_camera_read`, `stall_stream`, `delay_delivery`, `light_outside`, `withhold_chunk` | camera | camera | refused |
| `refuse_stream_start`, `refuse_led_write`, `bind_camera_usb_driver` | refused | refused | FX2 |

`hold_axis` halts the axis's next move part of the way and returns an
operation done once it has halted there (over REST, a job), so a caller acts
on a move while it travels without a clock; the move travels on after
`clear()`, and a hold that a stop, a new target, a home or `clear()` ends first
raises `HoldEndedError`.

A fault is armed for the next command, not the one in flight. A removed camera,
an unplugged or silenced FX2 and a reboot under way stay as they are after
`clear()`, as on a real scope, and a board that found its cable pulled is
plugged back in but stays disconnected until it is reconnected.
`exposed_under(frame)`, in-process, is a read, not a fault: the settings the
simulated camera's sensor exposed a frame under, so a test judges a frame by
what made it, never by its pixels.

```python
scope.simulation.drop_replies('motor', 1)   # the motor board misses its next reply
print(scope.simulation.injected())
```

```python raises=MoveFaultError:driver_failed
scope.motion.move_z(4000.0)                 # driver_failed: the board missed the target; home Z
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
what a test sees. The simulated scope is one record,
[`SimulatedScope`](../api-reference.md#SimulatedScope), passed as
`create(simulate=True, simulated=...)`: among its fields, `model` builds the
scope as another model than the settings declare, `camera_present=False` brings
it up with no camera, `slide` puts a Ronchi ruling on the stage in place of the
cells, `camera_stall` stops the camera's frames for a while, and `file_stall`
holds the save drive. On a Classic, `camera_usb_bindings` starts the FX2 on
another driver, on none, or on LumaViewPro's with a Windows problem code, so
bring-up refuses its camera as on a PC whose driver went wrong, and tells it
once, by the camera's notice with Repair's cause. `unit_config` takes a unit's own `motorconfig.json`, an
OEM unit's included, and brings the scope up as that unit: under the name its
board reports, on the axes the file marks present, with its optics and
travel. The simulated motor firmware boots the file, so it runs on the
`firmware` and `realistic` simulator tiers, on a model with an EL-0940 motor
board, and is refused anywhere else. `python -m lumascope.rest` takes only `--simulate`, so
a REST client has none of them; over REST the faults are at
`/api/v1/scope/simulation/<member>`.
