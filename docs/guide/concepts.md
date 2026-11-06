# Concepts

How the API fits together, and what every task uses: the architecture, bringing a session up, outcomes, who may act, state and settings, and units.

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
sub-APIs, one per part.

| Sub-API | What it does |
|---|---|
| `scope.motion` | moves the stage, the focus and the turret |
| `scope.illumination` | lights the LEDs |
| `scope.camera` | holds the camera's settings |
| `scope.imaging` | takes frames |
| `scope.capabilities` | says what this scope has |
| `scope.channel_identity` | names its channels |
| `scope.runtime_state` | answers derived state: the objective, the pixel size, plate coordinates |
| `scope.diagnostics` | reads the boards for support |
| `scope.simulation` | injects faults into a simulated scope |

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
when its class names it in `@api_fields`; the
[reference](../api-reference.md) lists exactly those. A member marked *in-process*
is for a Python caller only -- it takes a callback, returns a live object, or
blocks -- and has no route. Anything unmarked is not API, whatever its name: it
may change in any release.

**REST is the same API.** Each route is one `@api` member, at a path that
follows how you reach it from the session: `session.take_control` is
`POST /api/v1/take_control`, `scope.motion.move_z` is
`POST /api/v1/scope/motion/move_z`. A read is a `GET`, a setting is a `PUT` on
its read's URL (`session.select_labware` writes `PUT /api/v1/labware_id`), and
every other call is a `POST`. The same refusals, faults and
events reach a REST client as reach a Python caller ([REST](rest.md)).

## Bringing a session up

**Settings.** A LumaViewPro installation keeps the user's configuration in
`data/current.json`, the live file, and ships `data/settings.json`, the template
a new key's value comes from. `ScopeSession.load_user_settings` reads the user's
configuration as LumaViewPro does: the live file, checked against the template,
repaired, and completed with keys newer releases added. Settings are an argument
of `create`, never read behind your back.

```python
from lumascope import ScopeSession

root = '.'                                   # the LumaViewPro source folder, which holds data/
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
that is refused (`'position_unknown'`). `run_application_session()` is the
startup motion LumaViewPro itself runs at launch: every axis homed, then the
turret moved to slot 1. `start_application_session()` starts the same motion
and returns its [`Operation`](python.md#operations) without waiting. While the
home runs, it holds the scope like a run: other requests are refused naming the
home (`'home_in_flight'`).

**What `create` reads from the settings.** A dict you build yourself is
completed from the template, and each of its values is judged as a write of it
would be: a value outside its range is refused (`SettingRefusedError`, naming
the path), never clamped. When the motor board reports a model the catalogue
knows and it differs from `settings['microscope']`, bring-up writes the board's
model into the dict you passed: the hardware outranks the stored selection.
`settings['simulator_tier']` picks the simulator's fidelity
([The simulator](simulator.md#the-simulator)).

**Saving.** Given `settings_file`, the session saves every accepted write
itself, within a second and at shutdown. Without it nothing is saved, which is
what a test or a one-off script usually wants.

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
`create(..., simulate=False)` is refused `ScopeInUseError` (`'scope_in_use'`)
with nothing opened. A simulated scope takes no such lock.

**Shutting down.** `session.shutdown()` finishes what the session is doing --
it stops a live run and recording, then waits for their files and any
post-processing to be written -- and then turns the LEDs off, stops motion and
disconnects the scope. From its first step, new work is refused
(`'session_closing'`). A second call does nothing.

```python
session.shutdown()
assert session.live_work.closed
```

## Outcomes

Everything the API does ends in a return or an outcome, and every outcome is an
exception class under `LumascopeError`. An outcome
raised to you is yours to report; the API has not shown or logged it as a
failure on your behalf. An outcome nobody asked for -- a camera stream that
stalls, a run's file writer that stops -- reaches the outcome listeners
([below](python.md#hearing-outcomes)).

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
from lumascope import PositionOutOfRangeError, RefusalCause

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
[Outcomes in the reference](../api-reference.md#outcomes).

**Refusals every member shares.** Three refusals are made at every member's
door, so no member lists them:

- An argument of another type than the member declares is refused
  `ArgumentTypeRefusedError` (`'wrong_argument_type'`).
- A value outside a closed set -- an axis name, an image mode, an output
  format -- is refused `ArgumentRefusedError` (`'not_offered'`), its `offered`
  listing what is.
- A number that is not a number (NaN) is refused (`'not_a_number'`; a setting
  written by path, `'wrong_kind'`).

```python raises=ArgumentTypeRefusedError:wrong_argument_type
scope.motion.move_z('5000')                 # a str where a float is declared
```

```python raises=ArgumentRefusedError:not_offered
scope.motion.get_axis_limits('W')           # no axis W: offered X, Y, Z
```

**Remedies.** A refusal that one action answers carries it as `remedy`: the
remedy's name and the words of the offer. Take it through
`session.apply_remedy(refusal.remedy)`, the one door to every remedy and the
one place its name becomes an action; the remedy still decides whether it
applies now.

## Who may act

**The control session.** A client that must not be interfered with -- a plate
reader running a long experiment, an integration that owns the scope -- takes
the control session. While it is held, only work in it writes to the scope:
every other caller's command, setting, file, run, recording, home or diagnostic
is refused `ControlSessionRefusedError` (`'control_session_held'`), naming the
session. A stop is never refused, and the display preferences stay open to
everyone. The session is taken only while nothing of anyone else's is under
way. It ends when its holder goes silent: once 180 s pass with no call in it,
no request naming it and nothing it started -- a move, a run, a recording --
still running, the next caller it would refuse ends it instead, and the
holder's next call is refused `'control_session_ended'`. A holder that waits
longer between calls renews it (`session.renew_control(session_id)`).

```python
session_id = session.take_control('plate reader')     # the id goes to the taker alone
with session.acting_in_control(session_id):
    scope.motion.move_z(100)                          # work in the session
print(session.control)                                # who holds it, and since when
```

```python raises=ControlSessionRefusedError:control_session_held
scope.motion.move_z(200)                    # outside the session: refused while it is held
```

```python
ended = session.end_control(session_id)     # ends it, and stops a run or recording taken in it
assert session.control is None
```

Over REST, work in the session is a request carrying the `Control-Session`
header with the id. A request naming a session that has ended, or was never
issued, is refused `'control_session_ended'`. The person at the scope can take it
back without the id (`take_back_control`, in-process only).

**The holders.** A run, an autofocus, a recording, a diagnostic and a home each
hold the scope for their length. A command that would change what one is doing
is refused `HardwareCommandRefusedError` (`'exclusive_activity_running'`), its
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
| `'axis_absent'` | The scope has no such motion part: an axis, the turret, or no motors at all on a manual model. |
| `'not_on_this_scope'` | The scope has no such part: an LED, a channel, fan control. |
| `'not_connected'` | The part exists but is not connected: never came up, or its cable was pulled. |

```python raises=HardwareCommandRefusedError:axis_absent
scope.motion.move_turret(2)                 # this scope has no turret: axis_absent
```

**A lost part stays lost.** Nothing reconnects a part during a session. A part
that is not connected is named by `scope.unconnected_parts()` until LumaViewPro
restarts, so `'not_connected'` is a restart refusal (REST 503), and a run that
loses a part ends at once (`'hardware_disconnected'`).

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

```python raises=SettingRefusedError:out_of_range
session.update_settings('video.max_fps', 300)    # out_of_range: at most 200
```

A setting that a member owns is written only through that member, since the
member also applies it: `binning` is the camera's (`scope.camera.set_binning`),
`microscope` the session's `select_model`. Writing it by path is refused
`'has_member'`, its `member` naming the door. The reasons a path write is refused
-- `'has_member'`, `'host_only'`, `'installation_only'`, `'derived'`, `'not_a_setting'`,
`'block'`, `'wrong_kind'`, `'out_of_range'`, `'not_offered'` -- are
[`SettingRefusedError`'s](../api-reference.md#SettingRefusedError).

```python raises=SettingRefusedError:has_member
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
(`'gain_db_out_of_range'`).

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
from lumascope import PlateLocation

scope.motion.move_to_well('B2')
scope.motion.move_to_plate(PlateLocation(x_mm=23.38, y_mm=20.24, z_um=None))
print(session.get_current_plate_position())                 # X, Y in plate mm; Z in um
```

**A position to keep is read at rest.** A position you will act on later -- a
step, a bookmark, a sweep's centre -- is read while its axes are still, by
`scope.motion.positions_to_keep`. It is refused `'axis_moving'` while an axis it
names moves, and `'position_unknown'` while one is unhomed: a number is never
answered for a position the controller cannot vouch for.

```python
kept = scope.motion.positions_to_keep(('X', 'Y', 'Z'), then='saving a step')
print(kept['Z'])
```
