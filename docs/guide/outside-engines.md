# Driving the scope from another acquisition engine

What an acquisition engine of your own -- Micro-Manager, pymmcore-plus, a lab's own sequencer -- needs to drive the scope as it sets it.

An acquisition engine of your own -- Micro-Manager, pymmcore-plus, a lab's own
sequencer -- decides what each frame is: which light, which exposure, where the
stage is. LumaViewPro's channels and runs make those decisions for its own
acquisitions; an outside engine makes them itself, through the members that
take the scope as set and record what it was when each frame was taken.

**The as-set members.** `scope.camera.set_exposure_ms` and `set_gain_db` write
the camera now ([Camera](light-and-camera.md#camera)); `scope.illumination.led_on(channel,
illumination_ma)` and `led_off(channel)` light and darken a channel's LEDs at
the current the engine names ([Illumination](light-and-camera.md#illumination)); `start_move_xy`,
`start_move_z` and `move_turret` move the stage ([Motion](motion.md#motion)); and
`session.capture_still()`, with no channel, takes a frame with the camera and
the LEDs as they are ([A still](python.md#a-still)). None of
them reads or writes a channel's stored settings, so the engine's exposures and
currents never change what the person at LumaViewPro's window has stored.

**The channel layer is LumaViewPro's.** `session.show_channel`, a still named
for a channel, and a run's step put a channel's stored gain, exposure and
auto-gain on the camera. An engine that holds its own settings calls none of
them, or its next frame is taken at the channel's settings instead of its own;
after a raw camera write a still's record says the camera holds no channel's
(`held_channel` None).

**A still exposed after the call.** Frame validity waits out every change
LumaViewPro recorded. A change it did not make -- another program's light or
shutter, an illuminator on its own driver -- is not recorded, so a still taken
just after it can keep a frame exposed before it. An engine that changed what
the camera sees outside LumaViewPro, and waited for its device to settle, says
so with `exposed_after_call=True`: the still keeps only a frame exposed after
the call, and its record's `judgments` include the `caller` change. The
simulator's `light_outside` stands in for such a light.

```python
from lumascope import FrameValiditySource

scope.illumination.leds_off()
scope.camera.set_exposure_ms(10.0)                 # the engine's exposure
scope.simulation.light_outside(40)                 # a light LumaViewPro does not switch
still = session.capture_still(exposed_after_call=True, force_to_8bit=False)
record = still.record
assert FrameValiditySource.CALLER in {j.change.source for j in record.judgments}
assert record.held_channel is None                 # the camera as the engine set it
print(still.image.dtype, still.significant_bits)   # the camera's own depth
scope.simulation.clear()
```

Over REST the same still is `POST /api/v1/still` with
`{"exposed_after_call": true}`: its record, then its pixels at the camera's own
depth, answered when the still is taken, never as a job
([The live view and files](rest.md#the-live-view-and-files)).

**A move's outcome comes before the next frame's position.** A started move
returns at once with its `Operation`; it ends arrived, or raises
`MoveStoppedError` when someone stopped it or a newer move superseded it, or
`MoveFaultError` when the board failed, which leaves the axis unknown until it
is homed ([How a move ends](motion.md#motion)). The person at LumaViewPro's window can
stop a move the engine started. An engine reads the move's outcome before it
records a frame at the position it asked for; over REST that is the move's job,
`completed`, or `failed` with its notice or fault ([Jobs](rest.md#jobs)).

```python
held = scope.simulation.hold_axis('Z')             # the next Z move halts on its way
moving = scope.motion.start_move_z(5000.0)
assert held.wait(timeout=10)
moving.stop()                                      # as the person at the window can
```

```python raises=MoveStoppedError:stopped
moving.result(timeout=30)                          # stopped: Z is not at 5000 um
```

**A host that has not homed refuses moves.** A session's axes have no position
until its host's startup motion homes them, and again after a fault loses one;
until then every move is refused `'position_unknown'`
([Bringing a session up](concepts.md#bringing-a-session-up)). An engine that connects to a
host which has not homed, or after a fault, homes the scope (`home()`, over REST
`POST /api/v1/scope/motion/home`) before its first move.

**Binning and frame size are saved.** `scope.camera.set_binning` and
`set_frame_size` are the one writer of each, and save what they set as
LumaViewPro's own `binning` and `frame` settings: the person at LumaViewPro's
window sees the engine's geometry, and the next session opens with it.

```python
scope.simulation.clear()
before = session.get_settings_snapshot()['binning']
scope.camera.set_binning(2)
assert session.get_settings_snapshot()['binning'] != before   # saved, as the window's own
scope.camera.set_binning(1)
```

**Control, and its end by silence.** An engine that must not be interfered with
takes the control session ([Who may act](concepts.md#who-may-act)) and names it on every
call, as the `Control-Session` header over REST
([The control session](rest.md#the-control-session)). A session in which nothing has
acted for 180 s, with nothing it started still running, is silent: the next
caller it would refuse ends it, and the engine's next call is refused
`'control_session_ended'`. An engine idle between time points longer than that
renews it -- any request naming it does, a read or a job poll included, and
`POST /api/v1/renew_control` with `{"session_id": ...}` renews it with no other
call. A refused `take_control` and an ended session are the engine's to report;
neither is retried behind its back.
