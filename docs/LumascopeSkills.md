# LumaViewPro -- the Lumascope API guide

How the API that drives an Etaluma microscope fits together, and how to use it
from Python and over REST. Every member, field, closed set and outcome class is
in the [API reference](API_REFERENCE.md), generated from the code; this guide
links to it rather than repeating it. Every Python block below runs, in order,
on the simulator in LumaViewPro's test suite, and so does the REST walkthrough.

## PRE-RELEASE API

The Lumascope SDK API is **subject to breaking changes** until `4.0.0` ships.
Members, routes, records and outcome reasons may still be renamed, moved or
removed without a deprecation cycle. What is most likely to move:

- The REST surface: its routes, handles, jobs, problems and event stream.
- The protocol members and the protocol file's columns, as their wire forms
  settle.

If you build on this API before `4.0.0`, **contact Etaluma support**, so we
can consult you before a structural change.

When the first LumaViewPro release that is not a beta ships (`4.0.0`), the API
freezes for the 4.x versions, and every change after it is recorded in the
[Changelog](#changelog) at the end of this guide.

## What LumaViewPro is

LumaViewPro controls Etaluma's Lumascope microscopes: the LEDs that light the
sample, the stage, focus and objective turret that move it, and the camera that
images it. It runs as a desktop application, and the same API it is built on is
open to your own programs:

- **Python**, in the same process: a script, a notebook, MATLAB calling Python,
  or a plugin loaded into LumaViewPro.
- **REST**, over HTTP from any language: a server LumaViewPro starts beside its
  window, or one you start without a window at all.

Every shipped Lumascope is supported. The models differ in what they have -- an
XY stage, a focus axis, a turret, which LEDs, which camera -- and the API tells
your program what the attached scope has rather than leaving it to know models
by name (see [Who may act](#who-may-act)):

```python
caps = scope.capabilities                   # read once, when the scope came up
print(caps.axes, caps.has_xy_stage, caps.has_turret)
print(caps.camera_model, caps.pixel_size_um, caps.led_max_ma)
```

This guide is for researchers who script an experiment and for developers who
integrate a Lumascope into a larger system. It runs on Python 3.12 and 3.13, on
Windows, macOS and Linux.

## Architecture

```text
 Your program                       LumaViewPro's window
 (Python script, REST client)       (one client of the same API)
        |                                      |
        v                                      |
   REST server  --- one route per member ---   |
        |                                      |
        v                                      v
 ScopeSession -- settings, protocols, runs, the control session, outcomes
   |  manual_capture   manual_recording   post_processing
   v
 Lumascope (session.scope) -- the instrument
   |  motion   illumination   camera   imaging   capabilities
   |  channel_identity   runtime_state   diagnostics   simulation
   v
 Drivers -- the motor board, the LED board, the camera
   v
 Hardware (or the simulator, which stands in for every board)
```

**The session.** `ScopeSession` is the entry point: one session per
microscope. It owns the user's settings, builds and runs protocols, takes stills
and recordings, post-processes what was saved, decides who may write to the
scope, and reports outcomes. `ScopeSession.create` builds the scope beneath it,
so every host -- the GUI, the REST server, your script -- brings a scope up the
same way.

**The scope.** `session.scope` is the `Lumascope`: the instrument, split into
sub-APIs, one per part. `scope.motion` moves the stage, the focus and the
turret; `scope.illumination` lights the LEDs; `scope.camera` holds the camera's
settings; `scope.imaging` takes frames; `scope.capabilities` says what this
scope has; `scope.channel_identity` names its channels; `scope.runtime_state`
answers derived state (the objective, the pixel size, plate coordinates);
`scope.diagnostics` reads the boards for support; `scope.simulation` injects
faults into a simulated scope.

**The lanes.** Each bus is driven by one thread, its lane: the IO lane carries
every LED and motion command, the CAMERA lane every camera command. Commands from
every caller -- your script, the window, a REST client -- run one at a time per
bus, in the order they arrived. A command waits until the hardware has it, so
calling the API from several threads is safe. The session adds a FILE lane for
saving images, a worker pool, and the thread that runs protocols.

**The holders.** Some work takes the whole scope for its length: a protocol run,
an autofocus, a manual recording, a diagnostic, a home. While one holds the
scope, a command from anyone else that would change what it is doing is
refused, naming the holder (see [Who may act](#who-may-act)).

**What is API.** A member is API when it carries the `@api` mark, and a field
when its class names it in `@api_fields` (`modules.api_surface`); the
[reference](API_REFERENCE.md) lists exactly those. A member marked *in-process*
is for a Python caller only -- it takes a callback, returns a live object, or
blocks -- and has no route. Anything unmarked is not API, whatever its name: it
may change in any release.

**REST is the same API.** Each route is one `@api` member, at a path that
follows how you reach it from the session: `session.select_labware` is
`POST /api/v1/select_labware`, `scope.motion.move_z` is
`POST /api/v1/scope/motion/move_z`. A read is a `GET`, a setting is a `PUT` on
its read's URL, and every other call is a `POST`. The same refusals, faults and
events reach a REST client as reach a Python caller ([REST](#rest)).

## Bringing a session up

**Settings.** A LumaViewPro installation keeps the user's configuration in
`data/current.json`, the live file, and ships `data/settings.json`, the template
a new key's value comes from. `ScopeSession.load_user_settings` reads the user's
configuration as LumaViewPro does: the live file, checked against the template,
repaired, and completed with keys newer releases added. Settings are an argument
of `create`, never read behind your back.

```python
from modules.scope_session import ScopeSession

root = '.'                                   # the folder LumaViewPro is installed in
settings = ScopeSession.load_user_settings(root)
session = ScopeSession.create(
    settings,
    simulate=True,                           # False (the default) opens the real hardware
    settings_file=ScopeSession.user_settings_file(root),   # where accepted writes are saved
)
session.run_application_session()           # the startup motion: home, then the turret
scope = session.scope
```

`create` returns a session whose scope is connected and configured -- the
stored objective, plate, frame, binning and image mode applied -- and whose
camera is streaming, so a capture works at once. It does **not** move anything.
The scope's position is unknown until it is homed, and a move or a run before
that is refused (`position_unknown`). `run_application_session()` is the
startup motion LumaViewPro itself runs at launch: every axis homed, then the
turret moved to slot 1. `start_application_session()` starts the same motion
and returns its [`Operation`](#protocols-and-runs) without waiting. While the
home runs, it holds the scope like a run: other requests are refused naming the
home (`home_in_flight`).

**What `create` reads from the settings.** A dict you build yourself is
completed from the template, and each of its values is judged as a write of it
would be: a value outside its range is refused (`SettingRefusedError`, naming
the path), never clamped. When the motor board reports a model the catalogue
knows and it differs from `settings['microscope']`, bring-up writes the board's
model into the dict you passed: the hardware outranks the stored selection.
`settings['simulator_tier']` picks the simulator's fidelity
([The simulator](#the-simulator)).

**Saving.** Given `settings_file`, the session saves every accepted write
itself, within a second and at shutdown. Without it nothing is saved, which is
what a test or a one-off script usually wants.

**Hearing bring-up.** Bring-up has happened by the time `create` returns. What
it found -- a camera not found, a stored value replaced, a part missing -- is
reported as outcomes; to hear them, pass your listener to `create`
(`outcome_listener=`). A client that comes later reads the record instead:

```python
record = session.bring_up_record()
for part in record.parts:
    print(part.part, 'up' if part.up else 'not up')
assert record.missing == ()                # the simulator has every part
```

**The plate, the objective and the model.** The plate is chosen by
`select_labware`, which answers whether it changed. The objective is
`select_objective` on a scope without a turret; on one with a turret it is the
slot's assignment (`assign_turret_objective`), and `session.objective_id` is
None while the slot in the light path holds none. Where nobody has said which
objective is in the light path, the session asks rather than guesses:
`objective_question()` answers the question owed, or None, and
`confirm_objective` answers it. The pixel size and field of view follow from
the objective, and are None while it is unknown. `select_model` stores the
model the scope is brought up as next time; the running model never changes.

```python model=LS850T
print(session.select_labware('384 well microplate'))    # True: the plate changed
question = session.objective_question()
if question is not None:                                # which objective is in slot 1?
    session.confirm_objective(question.proposed, slot=question.turret_slot)
print(session.objective_id, scope.runtime_state.get_pixel_size_um())
```

**One session per scope.** A real scope is opened by one process at a time:
while LumaViewPro, a REST server or another script drives the microscope,
`create(..., simulate=False)` is refused `ScopeInUseError` (`scope_in_use`)
with nothing opened. A simulated scope takes no such lock.

**Shutting down.** `session.shutdown()` finishes what the session is doing --
it stops a live run and recording, then waits for their files and any
post-processing to be written -- and then turns the LEDs off, stops motion and
disconnects the scope. From its first step, new work is refused
(`session_closing`). A second call does nothing.

```python
session.shutdown()
assert session.live_work.closed
```

## Outcomes

Everything the API does ends in a return or an outcome, and every outcome is an
exception class under `LumascopeError` (`modules.exceptions`). An outcome
raised to you is yours to report; the API has not shown or logged it as a
failure on your behalf. An outcome nobody asked for -- a camera stream that
stalls, a run's file writer that stops -- reaches the outcome listeners
([below](#hearing-outcomes)).

**Kinds.** Each outcome class is of one kind:

| Kind | Means | REST |
|---|---|---|
| refusal | Nothing was done: the request was wrong, or the scope's state did not allow it. | 409, 422 or 503, by its cause |
| fault | Something broke while it was being done. | 500 |
| notice | It ended differently from what was asked, by someone's choice: a move stopped, or superseded by a newer one. | by its cause |
| quiet | An expected answer that needs no report: a wait that ran out. | by its cause |

**Reasons and causes.** Every outcome carries a `reason`, the code your
program branches on, and every refusal a `cause`: whether the same request
could succeed later.

```python
from modules.exceptions import PositionOutOfRangeError, RefusalCause

try:
    scope.motion.move_z(1_000_000)          # far beyond Z's travel
except PositionOutOfRangeError as refused:
    assert refused.cause is RefusalCause.REQUEST
    print(refused.reason, '--', refused)    # the reason, then the words a person reads
```

| Cause | Means | Retry? | REST |
|---|---|---|---|
| `RefusalCause.STATE` | The scope's state refused it: a run is live, an axis is not homed. | Yes, once the state has changed. | 409 |
| `RefusalCause.REQUEST` | The request cannot succeed as sent: a bad argument, hardware this scope lacks. | No; change the request. | 422 |
| `RefusalCause.RESTART` | Nothing succeeds until the application restarts: a part is disconnected, the session is closing. | No. | 503 |

**`Raises:` is the contract.** Each member's docstring lists what it may raise,
by class and reason; a member that raises nothing says `Nothing.`. The reference
shows each member's list, and for a route the REST status of each reason. Every
outcome class, with each reason's cause and status, is under
[Outcomes in the reference](API_REFERENCE.md#outcomes).

**Refusals every member shares.** Three refusals are made at every member's
door, so no member lists them:

- An argument of another type than the member declares is refused
  `ArgumentTypeRefusedError` (`wrong_argument_type`).
- A value outside a closed set -- an axis name, an image mode, an output
  format -- is refused `ArgumentRefusedError` (`not_offered`), its `offered`
  listing what is.
- A number that is not a number (NaN) is refused (`not_a_number`; a setting
  written by path, `wrong_kind`).

```python raises=ArgumentTypeRefusedError
scope.motion.move_z('5000')                 # a str where a float is declared
```

```python raises=ArgumentRefusedError
scope.motion.get_axis_limits('W')           # no axis W: offered X, Y, Z
```

**Remedies.** A refusal that one action answers carries it as `remedy`: the
member to call and the words of the offer. Take it through
`session.apply_remedy(refusal.remedy)`, the one place a remedy's name becomes
an action; the member still decides whether it applies now.

### Hearing outcomes

An outcome nobody asked for is delivered to every outcome listener in the
process, as a [`Notification`](API_REFERENCE.md#Notification). `shown` is
whether it is for display now: an outcome muted when it happened -- during an
unattended run, inside the duplicate window, at shutdown -- arrives with `shown`
False, and again with the same `outcome_id` and `shown` True if it is later
shown. A program that displays opens only the shown ones; one that records keeps
them all.

```python
heard = []

def on_outcome(note):
    heard.append(note)
    if note.shown:
        print(note.kind, note.title, '--', note.message)

session.add_outcome_listener(on_outcome)
```

Over REST every outcome is an `outcome` event on the event stream
([REST](#rest)).

## Who may act

**The control session.** A client that must not be interfered with -- a plate
reader running a long experiment, an integration that owns the scope -- takes
the control session. While it is held, only work in it writes to the scope:
every other caller's command, setting, file, run, recording, home or diagnostic
is refused `ControlSessionRefusedError` (`control_session_held`), naming the
session. A stop is never refused, and the display preferences stay open to
everyone. The session is taken only while nothing of anyone else's is under
way.

```python
session_id = session.take_control('plate reader')     # the id goes to the taker alone
with session.acting_in_control(session_id):
    scope.motion.move_z(100)                          # work in the session
print(session.control)                                # who holds it, and since when
```

```python raises=ControlSessionRefusedError
scope.motion.move_z(200)                    # outside the session: refused while it is held
```

```python
ended = session.end_control(session_id)     # ends it, and stops a run or recording taken in it
assert session.control is None
```

Over REST, work in the session is a request carrying the `Control-Session`
header with the id. A request naming a session that has ended, or was never
issued, is refused `control_session_ended`. The person at the scope can take it
back without the id (`take_back_control`, in-process only).

**The holders.** A run, an autofocus, a recording, a diagnostic and a home each
hold the scope for their length. A command that would change what one is doing
is refused `HardwareCommandRefusedError` (`exclusive_activity_running`), its
`holder` naming the work and `holder_trigger` the host that started it (`gui`,
`rest` or `api`). A recording refuses only what would falsify its frames -- a
change of frame size, binning or image mode -- and leaves gain and exposure
open. What holds the scope now is `session.controls_lock` and, with a finished
run's files still being written, `session.live_work`; over REST both are in
`GET /api/v1/status`.

**Parts a scope does not have.** A command for hardware the scope lacks is
refused before anything moves, lights or is written, and its `missing`
(a `MissingPart`) names the part:

| Reason | Means |
|---|---|
| `axis_absent` | The scope has no such motion part: an axis, the turret, or no motors at all on a manual model. |
| `not_on_this_scope` | The scope has no such part: an LED, a channel, fan control. |
| `not_connected` | The part exists but is not connected: never came up, or its cable was pulled. |

```python raises=HardwareCommandRefusedError
scope.motion.move_turret(2)                 # this scope has no turret: axis_absent
```

**A lost part stays lost.** Nothing reconnects a part during a session. A part
that is not connected is named by `scope.unconnected_parts()` until LumaViewPro
restarts, so `not_connected` is a restart refusal (REST 503), and a run that
loses a part ends at once (`hardware_disconnected`).

## State and settings

**One store per fact.** Each piece of the scope's state has one owner, and
every reader asks it: the user's settings are the session's, which LED is lit
is the illumination API's, where an axis is the motion API's. Nothing keeps a
copy to fall out of step.

**Settings by path.** A setting is named by its dotted path into the settings
-- `BF.exposure_ms`, `video.max_fps`, `zstack.range` -- and read and written
through the session. A write is judged before it is stored: a value outside its
range is refused, never clamped, and the refusal names the path and the range.

```python
session.update_settings('BF.illumination_ma', 5)
assert session.get_setting('BF.illumination_ma') == 5
print(session.get_setting('BF'))            # a whole block, as a dict
```

```python raises=SettingRefusedError
session.update_settings('video.max_fps', 300)    # out_of_range: at most 200
```

A setting that a member owns is written only through that member, since the
member also applies it: `binning` is the camera's (`scope.camera.set_binning`),
`microscope` the session's `select_model`. Writing it by path is refused
`has_member`, its `member` naming the door. The reasons a path write is refused
-- `has_member`, `host_only`, `installation_only`, `derived`, `not_a_setting`,
`block`, `wrong_kind`, `out_of_range`, `not_offered` -- are
[`SettingRefusedError`'s](API_REFERENCE.md#SettingRefusedError).

```python raises=SettingRefusedError
session.update_settings('binning', 2)       # has_member: scope.camera.set_binning
```

A listener hears every accepted write, with the paths it changed:

```python
changed = []
session.add_settings_listener(changed.append)
session.update_settings('BF.gain_db', 2.0)
assert ('BF.gain_db',) in changed
session.remove_settings_listener(changed.append)
```

**Intent and what was delivered.** A setting keeps the value asked for, even
when the hardware delivers less; the member that drives the hardware answers
what it delivered. An LED board drives in steps -- whole milliamps on the
EL-0940, about 3.29 mA on the Classic's FX2 -- so `led_on` answers the current
it commanded, the nearest step. A camera snaps or quantizes a gain or an
exposure, so `set_gain_db` and `set_exposure_ms` answer the value now in
effect. A stored gain above the camera's ceiling stays stored; what the camera
would be given for it is `scope.camera.applied_gain_db_for(gain)`, and a gain
asked of the camera directly outside its range is refused
(`gain_db_out_of_range`).

```python
assert scope.illumination.led_on('BF', 10.37) == 10.0      # the board's step
scope.illumination.leds_off()
applied = scope.camera.applied_gain_db_for(999.0)
assert applied.capped and applied.applied_db == scope.capabilities.camera_limits.gain_max_db
```

**What the settings hold, and what they do not.** `get_settings_snapshot()`
answers a copy of every setting, for work that must not see a write made while
it runs. Positions, the lit LED and the camera's live values are not settings:
each is read from its owner.

## Units and coordinate frames

A quantity's unit is the last part of its name: `z_um`, `exposure_ms`,
`gain_db`, `illumination_ma`, `period_s`. Times the API takes and answers are
seconds; the protocol file and the settings keep a run's period in minutes and
its duration in hours, as LumaViewPro's window shows them.

**Two frames.** The stage moves in **stage micrometres**, each axis measured
from where it homes. A plate is addressed
in **plate millimetres**, X and Y from the plate's top-left corner. The two are
related by the plate's geometry and the **stage offset**, where the plate sits
on the stage.

```python
print(session.labware_id)                                   # the plate selected
print(scope.runtime_state.get_stage_offset())               # where it sits on the stage
plate = scope.runtime_state.stage_to_plate(60000.0, 40000.0)   # stage um -> plate mm
stage = scope.runtime_state.plate_to_stage(plate.x_mm, plate.y_mm)
assert round(stage.x_um) == 60000 and round(stage.y_um) == 40000
```

Both conversions use the selected plate and the live offset unless you name
another (`labware_id=`, `stage_offset=`). A protocol carries its own plate, and
its steps' X and Y are in plate millimetres on that plate; a run keeps the
offset it started with.

**Moving in plate terms.** `scope.motion.move_to_well` and `move_to_plate`
move X and Y on the selected plate; Z stays in stage micrometres, so a
`PlateLocation` for `move_to_plate` gives Z as None.

```python
from modules.lumascope_api.motion import PlateLocation

scope.motion.move_to_well('B2')
scope.motion.move_to_plate(PlateLocation(x_mm=23.38, y_mm=20.24, z_um=None))
print(session.get_current_plate_position())                 # X, Y in plate mm; Z in um
```

**A position to keep is read at rest.** A position you will act on later -- a
step, a bookmark, a sweep's centre -- is read while its axes are still, by
`scope.motion.positions_to_keep`. It is refused `axis_moving` while an axis it
names moves, and `position_unknown` while one is unhomed: a number is never
answered for a position the controller cannot vouch for.

```python
kept = scope.motion.positions_to_keep(('X', 'Y', 'Z'), then='saving a step')
print(kept['Z'])
```

## Motion

**Homing is required.** An axis's position is unknown until it is homed, and
again after a fault loses it. A move, a run or a kept position on an unhomed
axis is refused `position_unknown`; `axes_without_position()` names each axis
that has none, and why (`unknown` or `homing`). A session starts unhomed; the
startup motion ([Bringing a session up](#bringing-a-session-up)) homes it.

```python
scope.motion.home()                         # every axis; 'Z' or 'T' homes one
assert scope.motion.axes_without_position() == {}
```

A home holds the scope like a run: other requests are refused naming it
(`home_in_flight`). On the EL-0940 a home cannot be stopped once its board has
begun it (`home_not_stoppable`); it runs to its end.

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
[`Operation`](#protocols-and-runs): wait on `result()`, or `stop()` it. A stop
of an operation that has ended does nothing, so a late stop never halts what
took the axis after it.

```python
moving = scope.motion.start_move_z(3000.0)
moving.result(timeout=30)                   # None once the move has ended
```

**How a move ends.** A move that does not end where asked raises a
`MoveNotCompletedError`: `MoveStoppedError`, a notice, when someone stopped it
or a newer move superseded it (`stopped`, `superseded`); `MoveFaultError`, a
fault, when the board failed, stalled or lost the axis, which leaves the axis
unknown until homed again.

**Stopping.** `stop_motion()` halts every axis at once, whoever started the
move, and ends the live run (`stage_stopped`). No holder and no control session
refuses a stop; only a scope with no motor controller does (`axis_absent`,
`not_connected`).

**Interlocks.** A Classic scope's stage refuses to move with its lid open or
its stage unpowered (`lid_open_before_motion`, `stage_unpowered_before_motion`);
`interlocks()` answers which are tripped now. The EL-0940 has none.

**The turret.** A scope with a turret (`scope.capabilities.has_turret`) moves
it by slot, 1 to 4: `move_turret(slot)`. `turret_status()` answers its slot,
or `TurretStatus('unknown', None)` while it is not known; None means the scope
has no turret. The objective is the slot's assignment
([Bringing a session up](#bringing-a-session-up), `session.objective_id`).

```python model=LS850T
scope.motion.move_turret(2)
print(scope.motion.turret_status())
print(session.objective_id)                 # slot 2's objective
```

**Jogging.** `jog_step_um(axis, coarse)` answers the step a jog of that axis
takes under the active objective.

## Illumination

**Channels.** A channel is a way of imaging the sample: brightfield (`BF`),
phase contrast (`PC`), darkfield (`DF`), a fluorescence channel named for its
excitation (`Ex405`, `Ex488`, `Ex550`, `Ex589`, `Ex640`), or luminescence
(`Lumi`), which lights nothing. Which channels this scope has, and which LEDs
each drives, is `scope.channel_identity`. A channel is always named, never
numbered: a number where a channel is taken is refused.

```python
for channel in scope.channel_identity.channels:
    print(channel.key_name, channel.label, channel.led_addresses)
```

A file written before fluorescence channels were named for their excitation
(`Blue`, `Green`, `Red`) is read by the LED each old name lit; a caller's old
name is refused, as any unknown one is (`channel_unknown`, or `not_a_setting`
in a settings path).

**LEDs.** `led_on(channel, illumination_ma)` lights a channel's LEDs and
answers the current commanded ([State and settings](#state-and-settings));
`led_off(channel)` and `leds_off()` turn them off. A channel this scope lacks
is refused `not_on_this_scope`; one the release does not know,
`channel_unknown`. The ceiling is `scope.capabilities.led_max_ma`.

```python
scope.illumination.led_on('Ex405', 50)
print(scope.illumination.get_led_states())  # one LedReading per board address
scope.illumination.leds_off()
```

**The API holds the truth.** Which LED is lit, at what current, and who holds
it is held by the illumination API; the drivers hold nothing. A run, an
autofocus, a diagnostic or a home holds the LEDs for its length, and a write
from anyone else is refused `exclusive_activity_running`. Work that borrows the
light and gives it back saves it first and restores it after
(`save_led_state` / `restore_led_state`).

**The shown channel.** LumaViewPro shows one channel at a time, and so does
the session: `session.show_channel(channel)` puts that channel's stored gain,
exposure and auto-gain on the camera, and `session.light_shown_channel(lit)`
lights or darkens it at its stored current. This is the one way a channel's
settings reach the camera; a run and a go-to-step move it too. Over REST each
change is a `shown_channel` event.

```python
print(session.show_channel('BF'))           # ShownChannel(channel='BF', lit=False)
print(session.light_shown_channel(True))    # lit at BF.illumination_ma
session.light_shown_channel(False)
```

## Camera

**Where things are.** The camera's settings are `scope.camera`; taking frames
is `scope.imaging`. One member writes each setting, and the settings LumaViewPro
keeps (binning, frame size, image mode) are saved by that member, whoever calls
it.

**A write applies or raises.** Every camera write either took effect or raises:
a value outside the camera's declared range is refused before it reaches the
camera (`CameraSettingOutOfRangeError`, its `limits` the range), and a value
the camera refused or did not answer is the fault `CameraSettingRejected`
(`camera_rejected`, `camera_no_answer`). Each setter answers what the camera
now runs, which may differ from what was asked.

```python
gain = scope.camera.set_gain_db(3.3)
exposure = scope.camera.set_exposure_ms(5.0)
print(gain, exposure)
print(scope.capabilities.camera_limits)     # the declared ranges, read once at connect
```

```python raises=CameraSettingOutOfRangeError
scope.camera.set_gain_db(999.0)             # gain_db_out_of_range
```

**Held and fresh reads.** Each setting has two reads. The held read, a
property named for the setting (`gain_db`, `exposure_ms`, `binning`,
`frame_size`, ...), answers what the camera last took or reported, with no call
to the camera, and is None whenever that is unknown. The fresh read, `read_*`
(`read_gain_db`, `read_exposure_ms`, ...), asks the camera now and raises
`CameraReadFailedError` when it does not answer.

```python
print(scope.camera.gain_db, scope.camera.read_gain_db())
```

**Auto gain and exposure.** `set_auto_gain(state, settings)` turns the
camera's auto-gain on or off with its target settings, and `lock_auto_gain()`
fixes the gain and exposure it reached. `set_auto_exposure(mode)` takes a
`CameraAutoMode` (`OFF`, `CONTINUOUS`). A camera without either answers `OFF`.

**Geometry and image mode.** `set_binning(size)` and
`set_frame_size(width, height)` answer the `FrameSize` the camera delivers;
the frame size is floored to even sides and refused outside
`camera_limits.frame_min` to `frame_max`. `set_image_mode(mode)` chooses the
`ImageMode`: 8-bit, or 12-bit scientific, scaled or false-colour. A camera that
lacks a binning factor or a feature refuses it `CameraSettingUnsupportedError`.

```python
from modules.image_mode import ImageMode

print(scope.camera.set_binning(2))
print(scope.camera.set_frame_size(960, 600))
scope.camera.set_binning(1)
scope.camera.set_image_mode(ImageMode.EIGHT_BIT)
```

A camera setting changed while a run, a diagnostic or a home holds the scope is
refused (`exclusive_activity_running`). During a recording only what would
falsify its frames -- binning, frame size, image mode -- is refused.

## Frames and captures

**Frame validity.** After a change -- an LED lit, an exposure set, the stage
moved -- the camera keeps delivering frames that were exposed before the change
took effect. A capture waits until a frame counts for every recorded change,
by that change's rule: the **clock rule**, where the camera stamps when each
frame started exposing, counts a frame that started after the change was in
force; the **credit rule** counts a frame once enough frames have arrived since
the change. `scope.capabilities.frame_rules` answers each source's rule, and
`scope.imaging.frames_until_valid()` how many frames the capture would still
wait. A capture never infers validity from pixel content; when the scope keeps
changing longer than its budget, it raises `CaptureFailedError`
(`deadline_expired`).

**Which channel a still is.** A still is named for the lit LED -- the first in
address order when more than one is lit -- else for the shown channel. The
camera holds that channel's stored gain, exposure and auto-gain for the frame,
whatever came before: a run, a go-to, an LED lit through `scope.illumination`.
After a raw camera write (`scope.camera.set_gain_db` and the other setters) the
written values stand, and the frame's record says so: its `held_channel` is
None.

**A still.** `session.capture_still(channel)` takes one frame and returns a
`CapturedStill`: the read-only `image` and its `record`, a `FrameRecord` of
what the instrument was when the frame was taken.

```python
scope.motion.move_to_well('B2')             # over the simulator's specimen
session.show_channel('BF')
scope.illumination.led_on('BF', 5)
still = session.capture_still('BF', all_ones_check=True)
record = still.record
print(still.image.shape, still.image.dtype)
print(record.held_channel, record.exposure_ms, record.gain_db, record.illumination_ma)
assert record.dark is False                 # lit, and light was seen
assert record.saturated_fraction < 0.98     # measured, because all_ones_check asked
```

**Dark and saturated frames are returned, not refused.** With an LED lit, a
frame with no lit pixel is retried until `timeout_s` and then returned, its
`record.dark` True; with nothing lit a dark frame is expected and is not judged
(`dark` None). With `all_ones_check=True` a frame at least 98% saturated is
taken again once and then returned, its `saturated_fraction` measured. Read the
record to tell; never re-measure the pixels.

**A saved still.** `session.manual_capture.capture(...)` takes the still
LumaViewPro's Capture button takes, and saves it: the folder, the name, the
summing and the format come from the settings. It returns an `Operation` of a
`SavedStill` per file written, each its `file` and the record of it.

```python
saving = session.manual_capture.capture(false_color_on=False)
saved = saving.result(timeout=30)
print(saved[0].file, saved[0].channel, saved[0].statistics.maximum)
scope.illumination.leds_off()
```

A bare image carries no record. A file with the instrument's record in it is
saved by the API that took the frame -- a still, a run, a recording -- and
never by your program from the array ([Saved files](#saved-files)).

**Live frames.** `scope.imaging.add_frame_listener(callback)` hands each frame
the camera delivers to `callback(frame)`, a read-only `Frame` (`image`, `seq`,
`timestamp`, `arrived_ns`, `significant_bits`, `chunks`), on the camera's own
thread. A listener has 24 ms per frame: one that overruns its budget or raises
30 frames in a row is removed, and an outcome says so. Copy what you keep, and
return quickly.

```python
import threading

arrived = threading.Event()
frames = []

def on_frame(frame):
    frames.append(frame.seq)
    arrived.set()

scope.imaging.add_frame_listener(on_frame, name='guide')
assert arrived.wait(timeout=10)
scope.imaging.remove_frame_listener(on_frame)
```

## Saved files

**What a still holds.** A still is saved under the live folder in the format
the settings name (`OutputFormat`: TIFF, OME-TIFF or JPG; a run may instead
build an OME-TIFF hyperstack per well). A fluorescence
still is mono: its false colour is a choice made when it is shown, never in its
pixels. A frame deeper than 8 bits keeps the sensor's values right-aligned (a
12-bit frame is 0 to 4095 in a 16-bit container), and the file states the true
depth as `SignificantBits`: divide by `(1 << SignificantBits) - 1` to scale it,
never by 65535. A sum of frames states the bits it can reach -- 14 for four
12-bit frames -- and `FramesSummed` how many it holds.

**The image mode decides the depth and the encoding.** `ImageMode.EIGHT_BIT`
captures and saves 8-bit mono. The full-depth modes capture what the camera
has -- 12 bits where it delivers 12 -- and save it right-aligned
(`SCIENTIFIC`), shifted to fill the 16-bit container (`SCALED`), or as
three-channel RGB in the channel's colour (`FALSE_COLOR_RGB`); each states its
`SignificantBits`.

**The record in the file.** Each file carries what the instrument was when the
frame was taken: the objective, the position, each lit channel's current and
when it was read, the exposure, gain, black level and binning, the timestamps
and where they came from, whether the camera held the channel's settings, and
the frame-validity judgments. An OME-TIFF keeps the exposure and position in
its OME `Plane` and the rest in a `MapAnnotation` (namespace `etaluma.com/lvp`),
which Bio-Formats -- Fiji, QuPath, OMERO -- shows; a plain TIFF carries the same
keys in its `Plane` block. Every key, its unit and what a sum does to it is the
reference's [saved-file facts](API_REFERENCE.md#saved-file-facts). A fact the
frame did not know is absent, never a stand-in.

**Derived files.** A z-projection, a stitch, a composite or a video states a
fact only when every input states the same: a composite of several channels
states no current, and a sum of slices no `FramesSummed`. A run's hyperstack is
an OME-TIFF per well and scan, its channels coloured by OME `Channel.Color`; in
Fiji open it with `Plugins > Bio-Formats > Importer`, colour mode Composite.

**Recordings.** A recording is saved as numbered TIFFs, each frame with its
own record (position, whether the stage was moving, the channel and its
current), or as an MP4 played at the frames' own times; either way a
`recording_manifest.json` holds what was measured: the frames delivered, their
timestamps and the rate. Frames of one channel are also built into a
hyperstack.

## Protocols and runs

**A protocol** is a list of steps -- each a place on the protocol's plate, a
channel, and how to capture it -- with a schedule. Three members give you one:
`session.new_protocol(...)` builds one over every well of the selected plate,
`session.create_empty_protocol()` starts one with no steps, and
`session.load_protocol(path)` reads a file. Which channels a built protocol
captures is each channel's acquire setting (`set_channel_acquire`).

```python
session.set_channel_acquire('BF', 'image')       # BF is captured by every step
session.update_settings('BF.illumination_ma', 5)

protocol = session.create_empty_protocol()
scope.motion.move_to_well('B2')
print(session.add_step(protocol, channels=('BF',)))     # a step here, on the protocol's plate
print(protocol.step_records()[0])

path = session.save_protocol(protocol, session.live_folder_path('Protocols/guide.tsv'))
assert session.load_protocol(path).num_steps() == 1
```

```python
session.update_settings('zstack.range', 20.0)    # a stack's extent and step are settings
session.update_settings('zstack.step_size', 5.0)
plate = session.new_protocol(tiling='2x2', use_zstacking=True)
print(plate.num_steps(), plate.tiling())
```

A protocol is edited only through the session (`add_step`, `update_step`,
`rename_step`, `delete_step`, `apply_tiling`, `apply_zstacking`,
`set_protocol_schedule`, ...), and each edit is a guarded write, refused under
another's control session. Its records (`step_records()`) are read-only, and a
run works on its own copy.

**The schedule** is in seconds in the API: `period_s` between scans and
`duration_s` in all. The protocol file keeps them in minutes and hours.

```python
session.set_protocol_schedule(protocol, period_s=600.0, duration_s=3600.0)
assert protocol.num_scans() == 6
```

**The protocol file** is tab-separated. Its first row is
`Protocol.PROTOCOL_FILE_HEADER`, then the version, the period (minutes), the
duration (hours), the plate and the capture root, then a table of steps whose
columns are `Protocol.CURRENT_COLUMNS`:

```python
from modules.protocol import Protocol

print(Protocol.CURRENT_VERSION, Protocol.CURRENT_COLUMNS)
```

`X` and `Y` are plate millimetres from the plate's top-left, `Z` stage
micrometres, `Illumination` milliamps, `Gain` dB, `Exposure` milliseconds, and
`Sum` the frames summed. A file of an earlier version is converted when it is
read; a newer one is refused (`ProtocolFormatError`). Loading is strict about a
cell's type and lenient about its value: a value no run could use is reported
when the file is read, and refused when a run starts.

**Runs.** A run is started by the session's protocol runner and returns an
`Operation` at once:

| Member | Runs |
|---|---|
| `start_single_scan(protocol)` | every step once |
| `start_protocol(protocol)` | every scan of the schedule |
| `start_autofocus(channel)` | an autofocus where the stage is |
| `start_autofocus_all_steps(protocol)` | an autofocus at each step, written into the protocol |
| `start_zstack(channel)` | a stack of the stored range where the stage is |
| `start_composite(...)` | one capture per acquiring channel, merged |

```python
from modules.run_outcome import RunStatus

runner = session.create_protocol_runner()
started = runner.start_single_scan(protocol, sequence_name='guide')
outcome = started.result(timeout=120)
assert outcome.status is RunStatus.COMPLETED, outcome.message
print(outcome.captures)                     # asked, captured, failed, saturated, dark
print(started.files.result(timeout=60))     # the run's images: written, or how many were not
```

A run that cannot start is refused before it takes the scope:
`ProtocolRunRefusedError`, its `reason` naming why (`position_unknown`,
`zstack_not_configured`, `exclusive_activity_running`, ...). Once started, a run
always ends with a `RunOutcome` -- it does not raise -- whose `status` is one of
`completed`, `incomplete`, `aborted`, `failed` or `failed_at_start`, with a
`reason` and the words a person reads.

**An Operation** is the handle every `start_` member returns. `result()` waits
for the run to let go of the scope and answers its outcome; `done()` says
whether it has; `progress()` answers where it has got; `stop()` asks it to stop
and returns at once -- the run unwinds, puts the scope back, and ends
`aborted`. A run's images keep writing after it lets go: `files` is their own
operation. In-process each run also has a waiting form, `run_single_scan`,
`run_protocol`, ...; over REST a run is a job ([REST](#rest)).

```python
import threading
from modules.run_events import RunEvents

ended = threading.Event()
events = RunEvents(run_ended=lambda outcome, run_dir, protocol: ended.set())
running = runner.start_protocol(protocol, sequence_name='guide_stopped', events=events)
running.stop()
assert ended.wait(timeout=60)
assert running.result().status is RunStatus.ABORTED
```

**Where a run writes.** Under the live folder, `ProtocolData/<time>`: one
image per step and channel, the protocol it ran, and a record of each capture.
A run's video steps record as a manual recording does
([Saved files](#saved-files)), and its hyperstacks are built once the run ends.

**Post-processing** builds derived files from a run's folder:
`session.post_processing.build(folder, zproject=..., stitch=..., composite=...,
video=...)` builds the projections first, then the stitches, the composites
and the videos, each group of a run from every frame it recorded or not at all.
A build runs on its own lane, beside the next run, and is refused while the
run it would read is still writing.

## Events and listeners

Each change the scope makes is delivered to the listeners registered for it,
and over REST as an event on the event stream. The
[events table](API_REFERENCE.md#events) lists every event, the member that
delivers it, its record (the callback's arguments, in order) and how it
reaches a REST client. Each `add_*_listener` has its `remove_*_listener`.

```python
import threading

settled = threading.Event()

def on_position(axis, at):
    if axis == 'Z' and at.state == 'idle':
        settled.set()

scope.motion.add_position_listener(on_position)
scope.motion.move_z(4500.0)
assert settled.wait(timeout=10)
scope.motion.remove_position_listener(on_position)
```

**Threads.** A listener is called on the thread that made the change -- the
IO lane for a move or an LED, the camera's thread for a frame -- so it returns
quickly and does not call back into the API and wait. A GUI host that needs its
callbacks on its UI thread tells the process once, before its first session,
through `ScopeSession.set_ui_dispatcher`; a script never sets it.

**A run's events** are the `RunEvents` handlers given to its start:
`scan_started`, `step_started`, `frame_captured`, `scan_ended`, `run_ended`
and `files_written`, in that order. `run_ended` comes once the run has let go
of the scope, and `files_written` once its images are on disk: chain the next
work from `files_written`, not from `run_ended`.

**Levels and edges.** The run-state listener is a level: it is called once at
registration with the state now, then on every change, so a display that
registers late is not wrong. Every other listener hears changes from its
registration on; read the current value first.

## REST

### Serving it

```bash
python -m rest --simulate      # a simulated scope; leave out --simulate for the hardware
```

The server brings the scope up as LumaViewPro does -- the user's settings, the
startup home, the plugins -- and serves on `127.0.0.1` at the port in
`rest_api.port` (shipped 8000). It takes the same single-instance lock as
LumaViewPro, so it will not start while LumaViewPro or another server drives
the scope. A failed startup home is reported and the server still serves, so a
client can home again. Ctrl-C closes it: new calls are refused
(`server_closing`), the session shuts down with the event stream open, the
stream's last event is `closing`, and the process exits `0`, or `1` when the
session did not close cleanly.

LumaViewPro serves the same API beside its window when `rest_api.enabled` is
set in its settings (shipped `false`). To serve a session your own program has
brought up, build the application and serve it with uvicorn:

```python norun
import uvicorn
from rest.app import build_app

uvicorn.run(build_app(session), host='127.0.0.1', port=8000)    # until Ctrl-C
session.shutdown()
```

Serve it on `127.0.0.1` only: it has no credentials, so anyone who can reach
the port drives the scope. `GET /api` answers the versions served; the OpenAPI
3.1 description of every route is at `/api/v1/openapi.json`, with an
interactive reference at `/docs`. Each route's description lists the refusals
and faults its member declares, under the status each answers.

### Routes

The session is the root: its member `m` is `/api/v1/m`, and each sub-object is
a path segment, so `session.scope.motion.move_xy` is
`POST /api/v1/scope/motion/move_xy`.

- **A read is `GET`**: a property, a published field, or a method that changes
  nothing. A read method's arguments are its query string
  (`GET /api/v1/scope/motion/get_axis_limits?axis=Z`); a list is its key
  repeated, and a record one key per field, `<parameter>.<field>`.
- **A setting is `PUT` on its read's URL**, with the setter's arguments as a
  JSON object; the answer is the read after the write, the value the scope took
  (`PUT /api/v1/scope/camera/binning` with `{"size": 2}` answers `2`). A
  setting kept per channel takes the channel in the body and the query
  (`PUT /api/v1/channel_acquire` with `{"channel": "BF", "mode": "image"}`,
  then `GET /api/v1/channel_acquire?channel=BF`). The setter itself has no
  route.
- **Every other call is `POST`**, its arguments a JSON object by parameter
  name; `null` is None, and a parameter left out takes its default. A member
  that hands out a running call is a `POST` even when it changes nothing
  (`last_run`).
- **The answer** to a finished call is `200` with its value as the body, `null`
  for None.
- **Arguments are checked before the member runs**: a key it does not take, a
  value of the wrong type, a missing one, or a number that is not finite is
  refused `422` (`invalid_request`), its `errors` naming each argument. A
  number where a channel is named is refused here too, as `invalid_request`.
- **Paths are names in the live folder**, `/`-separated
  (`"ProtocolData/run1"`). A name that leaves it is refused
  (`outside_live_folder`); `"."` is the live folder itself. An answered path is
  `{"name": ..., "host_path": ...}`.
- **Units are the API's**: an axis position crosses as
  `{"position_um": ..., "state": ...}`, and a duration as a number of seconds.

In-process members -- those taking a callback, returning a live object, or
waiting on the caller's thread -- have no route; each has a wire form beside
it (a listener is an event, a waiting call is a job, `acting_in_control` is
the `Control-Session` header).

### Handles

A live object -- a protocol, a protocol runner -- crosses as a handle,
`{"handle": "1", "type": "Protocol"}`. Its members are at
`/api/v1/handles/<type>/<id>/<member>` (`GET /api/v1/handles/Protocol/1/period_s`),
and a parameter that takes it is sent its id (`{"protocol": "1"}`).
`GET /api/v1/handles` lists every handle held, oldest first.
`DELETE /api/v1/handles/<type>/<id>` forgets the id, never the object; the
session's protocol runner is every client's and is kept (`422`,
`handle_shared`). At most 1000 handles are held.

### Jobs

Every call runs on its own thread, and its route waits for it as long as the
client says: `Prefer: wait=<seconds>`, 2 s when it says nothing, at most 60 s.
`Prefer: respond-async` asks for no wait at all. A call still running then
answers `202 Accepted` with `Location: /api/v1/jobs/<id>` and the job:

```json
{"id": "3", "member": "handles/ProtocolRunner/2/start_single_scan", "status": "running",
 "requested": "2026-10-29T11:47:56.043-07:00", "ended": null, "progress": null}
```

A member that starts an operation -- a run, a move, a home, a still -- is
answered `202` with the operation's job as soon as it has started, whatever
the wait. `GET /api/v1/jobs/<id>`, itself waiting by `Prefer: wait`, reads a
job; once it has ended it carries `result` or `error` (the problem). A job's
`status` says whether the call answered a problem, not how the operation went:
a stopped run's job is `completed` with its outcome `aborted`, a stopped move's
`failed` with its stop notice. `DELETE /api/v1/jobs/<id>` stops a running
operation (`202`) or forgets a finished job (`204`). Finished jobs are kept up
to 1000 and 24 hours. Take a job's id from the answer that made it, never count
it: every call held as a job takes the next id.

A `POST` or `PUT` that may be sent twice carries `Idempotency-Key`: sent again
with the same key, it runs nothing and is answered what the first was, with
`Idempotent-Replayed: true`. A key is held for 24 hours.

### The control session

While a control session is held, every write carries its id in the
`Control-Session` header; any other write is refused `409`
(`control_session_held`), and a request naming an ended session `422`
(`control_session_ended`). `POST /api/v1/take_control` with `{"name": ...}`
answers the id; `POST /api/v1/end_control` with `{"session_id": ...}` ends it.

### Problems

Every answer that is not a result is an RFC 9457 problem,
`application/problem+json`:

```json
{"type": "urn:lumascope:problem:not_offered", "title": "Not One of the Values Offered",
 "detail": "axis must be one of X, Y, Z; 'W' is not one.", "status": 422,
 "instance": "urn:uuid:3faa1296-...", "kind": "refusal", "reason": "not_offered",
 "remedy": null, "argument": "axis", "offered": ["X", "Y", "Z"]}
```

`title` and `detail` are the words a person is shown; `reason` is the code to
branch on; `kind` is the outcome's; `remedy` is sent back as
`POST /api/v1/apply_remedy` with `{"remedy": ...}`. A refusal carries the
fields its class publishes -- `argument`, `offered`, `limits`, `holder` -- so a
client corrects its request without reading the words.

| Status | Means |
|---|---|
| `409` | A refusal the scope's state gave: the same request may succeed later. Also a move or home a stop ended (`kind` `notice`). |
| `422` | A refusal of the request as sent: change it. Also `invalid_request` and `handle_shared`. |
| `500` | A fault: the call failed. |
| `404` | No such route, handle, job or file. |
| `405` / `415` | The wrong method; a body that is not JSON. |
| `503` | With `Retry-After`, try again later (`overloaded`, `no_frame_yet`). Without it, nothing succeeds until the application restarts (`not_connected`, `scope_disconnected`, `session_closing`, `server_closing`). |

### The event stream

`GET /api/v1/events` is a server-sent event stream. Its first event is
`status`, the scope as `GET /api/v1/status` reads it; after that every change
is an event, named as in the [events table](API_REFERENCE.md#events), its data
the event's record. A run's events carry `run`, `{"job": "<id>"}`, the job of
the run that sent them, and every job's end is a `job_ended` event. An outcome
nobody asked for is an `outcome` event.

A client that reconnects with `Last-Event-ID` is sent what it missed; when the
server no longer holds it (more than 1024 events ago, or a restarted server),
it is sent `reset` and then `status`, where it starts again. A client that
cannot hold a stream open polls `GET /api/v1/events/held?after=<id>`. The last
event is `closing`.

### The live view and files

- `GET /api/v1/live` -- the camera's frames as MJPEG, for a browser's `<img>`.
- `GET /api/v1/live.jpg` -- one frame.
- `GET /api/v1/files/<name>` -- a file in the live folder, as an attachment; a
  `Range` request resumes a large download.
- `PUT /api/v1/files/<name>` -- a new file in the live folder; a file already
  at the name is refused (`file_exists`), so an upload never replaces a
  capture.
- `GET /api/v1/previews/<name>` -- a saved image downscaled to a JPEG for
  display (`max_width`).

The live view and previews are display: 8-bit grey, lossy, with no metadata.
The data is the file. Every answer carries `X-Request-ID`, which a problem
names as its `instance`, and each request is one line of
`lumaviewpro_rest_api.log`.

### A walkthrough

A session with `curl` against a server started with `--simulate`, reading the
answers with `jq`. Each line's comment is the status it answers; a handle's or
a job's id is read from the answer that made it. Every line here is run, in
this order, in LumaViewPro's test suite.

```bash
API=http://127.0.0.1:8000/api/v1

curl -s $API/status                                         # 200: the scope's work, axes, parts and camera
curl -s "$API/scope/motion/get_axis_limits?axis=W"          # 422 not_offered: offered X, Y, Z
curl -s -H 'Prefer: wait=60' --json '{"z_um": 4000.0}' $API/scope/motion/move_z    # 200
curl -s --json '{"channel": "BF", "illumination_ma": 5.0}' $API/scope/illumination/led_on   # 200: 5.0
curl -s --json '{}' $API/scope/illumination/leds_off        # 200
```

Build a one-step protocol and run it. A run is answered `202` at once with its
job; read the job with `Prefer: wait` until it has ended.

```bash
curl -s -X PUT --json '{"channel": "BF", "mode": "image"}' $API/channel_acquire   # 200: "image"
PROTOCOL=$(curl -s --json '{}' $API/create_empty_protocol | jq -r .handle)        # 200
curl -s -H 'Prefer: wait=60' --json '{"well": "B2"}' $API/scope/motion/move_to_well   # 200
curl -s --json "{\"protocol\": \"$PROTOCOL\", \"channels\": [\"BF\"]}" $API/add_step   # 200: ["custom0000_BF"]
RUNNER=$(curl -s --json '{}' $API/create_protocol_runner | jq -r .handle)         # 200
JOB=$(curl -s --json "{\"protocol\": \"$PROTOCOL\"}" $API/handles/ProtocolRunner/$RUNNER/start_single_scan | jq -r .id)   # 202
curl -s -H 'Prefer: wait=60' $API/jobs/$JOB                 # 200: "status": "completed", "result" the RunOutcome
curl -s "$API/live_folder_listing?name=ProtocolData"        # 200: the run's folder
```

Take the control session, and see another write refused while it is held.

```bash
CONTROL=$(curl -s --json '{"name": "walkthrough"}' $API/take_control | jq -r .)   # 200: the id, for this client alone
curl -s --json '{"z_um": 3000.0}' $API/scope/motion/move_z                          # 409 control_session_held
curl -s -H "Control-Session: $CONTROL" -H 'Prefer: wait=60' --json '{"z_um": 3000.0}' $API/scope/motion/move_z   # 200
curl -s --json "{\"session_id\": \"$CONTROL\"}" $API/end_control                    # 200
curl -s $API/events/held                                    # 200: where a poll of the events begins
```

A client that holds a connection open reads the events as they happen with
`curl -N $API/events`.

## The simulator

`simulate=True` builds a scope whose boards and camera are simulated, and every
member works on it as on the hardware: a change is proven there before it
reaches a bench. The camera images a specimen fixed to the stage, so moving the
stage moves the image.

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
guarded write, refused on a real scope (`needs_simulated_scope`) and on a board
that cannot show it (`not_injectable_on_board`). `injected()` answers what is
injected, and `clear()` removes it all.

Which board can show a fault depends on how the model is simulated. Each
member's own fault, and its arguments, are in the
[reference](API_REFERENCE.md#SimulationAPI):

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
| `refuse_camera_write`, `refuse_camera_read`, `stall_stream`, `delay_delivery` | camera | camera | refused |
| `refuse_stream_start`, `refuse_led_write`, `bind_camera_usb_driver` | refused | refused | FX2 |

A fault is armed for the next command, not the one in flight. A removed camera,
an unplugged or silenced FX2 and a reboot under way stay as they are after
`clear()`, as on a real scope.

```python
scope.simulation.drop_replies('motor', 1)   # the motor board misses its next reply
print(scope.simulation.injected())
```

```python raises=HardwareCommandRefusedError
scope.motion.move_z(4000.0)                 # position_unread: the move was not sent
```

```python
scope.simulation.clear()
scope.simulation.remove_camera()            # as if its cable were pulled
assert 'camera' in scope.unconnected_parts()
```

```python raises=HardwareCommandRefusedError
scope.camera.set_gain_db(2.0)               # not_connected, until a restart
```

```python model=LS720
scope.simulation.interlock(lid_open=True)
print(scope.motion.interlocks())
```

```python raises=HardwareCommandRefusedError
scope.motion.move_xy(x_um=1000.0)           # lid_open_before_motion
```

`create` also arms a fault before bring-up, so what bring-up does with it is
what a test sees: `sim_model` builds the scope as another model than the
settings declare, `sim_camera_stall` stops the camera's frames for a while, and
`sim_file_stall` holds the save drive. Over REST the faults are at
`/api/v1/scope/simulation/<member>`.

## Plugins

A plugin extends LumaViewPro: a post-processor, a live frame processor, a
panel in the window. It is a Python package loaded at startup, handed the
session, and it uses the API this guide describes. How to write one, its
lifecycle, and which plugin API level a host offers are in
[`PluginTutorial.md`](PluginTutorial.md).

## Changelog

Every change to the API after `4.0.0`, recorded in the same commit as the
change. Before `4.0.0` the API is fluid by design, and nothing is recorded here.

| Version | Date | Type | Member | Change |
|---|---|---|---|---|

- **Type** `additive` (a new optional argument, field or member): no version
  bump beyond the patch.
- **Type** `behavior-change`: a minor version bump.
- **Type** `rename` / `removal`: a deprecation cycle (the old name kept with a
  `FutureWarning`, removed in the next major version), or a major version bump.

(No entries yet: 4.0.0 has not shipped.)
