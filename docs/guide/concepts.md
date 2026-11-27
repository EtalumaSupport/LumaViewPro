# Concepts

What every task uses: the words the guide uses, the architecture's detail, outcomes, bringing a session up, who may act, state and settings, and units.

## Words this guide uses

Each word is used throughout the guide in the sense given here; the link is
where it is explained in full.

- **bring-up** -- what `ScopeSession.create` does: it connects the scope and
  configures it from the stored settings, with the camera streaming. It moves
  nothing ([Bringing a session up](#bringing-a-session-up)).
- **Classic** -- a Lumascope Classic, one of the FX2 models: the LS460, LS560,
  LS620 and LS720, whose one USB device serves both the camera and the LEDs
  ([Supported Lumascopes](../api-guide.md#supported-lumascopes)).
- **control session** -- a client's claim to be the scope's only writer, taken
  by `session.take_control`. While it is held, every other caller's write is
  refused `'control_session_held'`; a stop never is
  ([The control session](#the-control-session)).
- **door** -- a way into the API that every call of its kind passes, with
  refusals of its own. The doors are the check of a member's arguments, which
  every member is entered through
  ([Refusals every member shares](#refusals-every-member-shares)); a hardware
  lane; a guarded write (a setting or a file written); and a post-processing
  build's admission. The reference names a member's doors
  beside its marks (`lane_call`, `guarded_write`, ...) and lists their
  refusals with its own. `session.apply_remedy` is the one door to every
  remedy ([Remedies](#remedies)).
- **held** -- said of five things, each named with what holds it. Work that
  holds the scope -- a run (an autofocus among them), a recording, a home or a
  diagnostic -- is its **holder** for its length ([The holders](#the-holders)).
  A client holds the control session, never the scope. The camera holds a
  channel's settings when it runs on that channel's stored gain, exposure and
  auto-gain. A frame's record names that channel as its `held_channel`
  ([Frames and captures](captures-and-files.md#frames-and-captures)). A
  **held read** is a camera property that answers what the camera last took
  or reported, without asking it ([Camera](light-and-camera.md#camera)). Over
  REST, the server holds the handles and jobs it lists
  ([Limits](rest.md#limits)).
- **handle** -- over REST, how a live object such as a protocol or a protocol
  runner is named: `{"handle": "1", "type": "Protocol"}`, its members under
  `/api/v1/handles/<type>/<id>/<member>` ([Handles](rest.md#handles)). An operation is
  a job over REST, not a handle.
- **in-process** -- said of a member a Python caller in the same process
  calls and a REST client does not: it takes a callback, returns a live
  object, or waits. It has no route ([What is API](#what-is-api)).
- **installation** and **source folder** -- an installation is a folder
  LumaViewPro runs from, with its `data/` (the settings and the catalogs). A
  program run from source has the source folder, the one holding
  `lumaviewpro.py`, as its installation
  ([Where it runs](python.md#where-it-runs)); the guide calls it the source
  folder. A setting that is
  installation-only is edited only in that folder's `data/current.json`.
- **lane** -- a thread that runs one kind of command one at a time, in the
  order the commands arrived from every caller. The lanes are the IO lane (the
  LEDs and motion), the CAMERA lane, the session's FILE lane for saving images,
  and the post-processing lane for builds ([The lanes](#the-lanes)).
- **Layer Settings** -- the block of a protocol file holding each channel's
  settings. A Load, the window's Load button, puts the block into the
  session's channel settings ([The protocol file](runs.md#the-protocol-file)).
- **level** and **edge** -- a level listener is called once when it is added,
  with the state then, and on every change after; an edge listener hears only
  the changes after it is added
  ([Events and listeners](python.md#events-and-listeners)).
- **live folder** -- the folder captures are saved under, the `live_folder`
  setting (`./capture` in the source folder, as shipped). Every path over REST
  is a name in it ([Routes](rest.md#routes)).
- **operation**, **running call** and **job** -- an operation, an `Operation`,
  is what every `start_` member returns: work under way, which you wait on,
  stop or read the progress of ([Operations](python.md#operations)). The
  reference calls it a running call. Over REST an operation, and any call
  still running past its wait, is a job ([Jobs](rest.md#jobs)).
- **outcome** -- every end of a call but a returned value: an exception class
  under `LumascopeError`, of one kind (a refusal, a fault, a notice or a quiet
  outcome) and carrying a `reason` ([Outcomes](#outcomes)).
- **position to keep** -- a position read while its axes are still, by
  `scope.motion.positions_to_keep`, for work that acts on it later
  ([A position to keep is read at rest](motion.md#a-position-to-keep-is-read-at-rest)).
- **raw write** -- a camera or LED setter called directly (`scope.camera`'s
  setters, `scope.illumination`'s), not through a channel's stored settings.
  A raw camera write stands until a channel's settings are put back
  ([Camera](light-and-camera.md#camera)).
- **reported** -- said of an outcome delivered rather than raised: it reaches
  the outcome listeners, and over REST it is an `outcome` event
  ([Hearing outcomes](python.md#hearing-outcomes)).
- **shown channel** -- the one channel the session shows, as LumaViewPro's
  window does. `session.show_channel` puts its stored gain, exposure and
  auto-gain on the camera, `session.light_shown_channel` lights its LED, and a
  still is named for it; a run's steps and a go-to-step move it too
  ([Illumination](light-and-camera.md#illumination)).
- **source** of a change -- what a capture waits out after it: an LED, a gain,
  an exposure, a move, a binning and the rest, each with its own count of
  frames ([Frames and captures](captures-and-files.md#frames-and-captures)).
- **startup motion** -- the motion LumaViewPro runs at launch, and the REST
  server before a client's first call: every axis homed, then the turret moved
  to slot 1 (`session.run_application_session`)
  ([Bringing a session up](#bringing-a-session-up)).
- **support bundle** -- what Etaluma support asks for: the Tech Support
  Report, a ZIP of the boards' and camera's answers, the data and the logs
  ([Logging](python.md#logging)).
- **wire form** -- how a REST client reaches what an in-process member does:
  a listener is an event, and a call that waits is its `start_` form, read as a
  job ([Routes](rest.md#routes)). What crosses between a REST server and its
  client the guide calls "over REST".

## Architecture

The layers, the session and the scope's areas are the hub's
[architecture at a glance](../api-guide.md#the-architecture-at-a-glance);
this is their detail.

### The lanes

Each lane is one thread. The IO lane runs every LED and motion
command and the CAMERA lane every camera command, each one at a time, in the
order it arrived from every caller: your script, the window, a REST client. A
command waits until the hardware has it, so calling the API from several
threads is safe. The session adds a FILE lane for
saving images, a worker pool, and the thread that runs protocols.

### Where a call starts

Every caller starts from the session. What is the
session's -- the settings, the shown channel, runs, captures, who may act,
outcomes -- is its member (`session.show_channel`, `session.manual_capture`).
The hardware is `session.scope.<api>` (`session.scope.motion.move_z`,
`session.scope.camera.set_binning`). Each capability has one spelling.

### What is API

Everything published is imported from one module,
`lumascope` (`from lumascope import ScopeSession`), and the
[reference](../api-reference.md) lists every member and field of it. Anything
under `_lumascope` is internal and may change in any release. A member marked
*in-process* is for a Python caller only, and has no route: it takes a
callback, returns a live object, or blocks. In the source, a published member
carries the `@api` mark and a published field is named in its class's
`@api_fields`.

### REST is the same API

Each member is a route at the path you reach it by
from the session, and the same refusals, faults and events reach a REST client
as reach a Python caller. The server adds routes of its own for jobs, handles,
files, events and the live view. How a member becomes its route is
[Routes](rest.md#routes).

## Outcomes

Everything the API does ends in a return or an outcome, and every outcome is an
exception class under `LumascopeError`. An outcome
raised to you is yours to report; the API has not shown or logged it as a
failure on your behalf. An outcome nobody asked for -- a camera stream that
stalls, a run's file writer that stops -- reaches the outcome listeners
([Hearing outcomes](python.md#hearing-outcomes)).

### Kinds

Each outcome class is of one kind:

| Kind | Means |
|---|---|
| refusal | Nothing was done: the request was wrong, or the scope's state did not allow it. |
| fault | Something broke while it was being done. |
| notice | Nothing failed and nothing was refused, but you should know: a value set at the hardware's limit, a move someone stopped. |
| quiet | An expected answer that needs no report: a wait that ran out. |

### Reasons and causes

Every outcome carries a `reason`, the code your
program branches on. Every refusal and quiet outcome carries a `cause`:
whether the same request could succeed later. So does a notice that ends
your call, such as a stopped move; a notice that is only reported has none.

```python
from lumascope import PositionOutOfRangeError, RefusalCause

try:
    scope.motion.move_z(1_000_000)          # far beyond Z's travel
except PositionOutOfRangeError as refused:
    assert refused.cause is RefusalCause.REQUEST
    print(refused.reason, '--', refused)    # the reason, then the words a person reads
```

| Cause | Means | Retry? |
|---|---|---|
| `RefusalCause.STATE` | The scope's state refused it: a run is live, an axis is not homed. | Yes, once the state has changed. |
| `RefusalCause.REQUEST` | The request cannot succeed as sent: a bad argument, hardware this scope lacks. | No; change the request. |
| `RefusalCause.RESTART` | Nothing succeeds until the application restarts: a part is disconnected, the session is closing. | No. |

Over REST a problem carries the cause as `cause`, its value (`state`,
`request`, `restart`), and it decides the status, `409`, `422` or `503`; a
fault is `500` ([Problems](rest.md#problems)).

### `Raises:` is the contract

Each member's docstring lists what it may raise,
by class and reason; a member that raises nothing says `Nothing.`. The reference
shows each member's list, and for a route the REST status of each reason. Every
outcome class, with each reason's cause and status, is under
[Outcomes in the reference](../api-reference.md#outcomes).

### Refusals every member shares

Two refusals are made where every member is
entered, so no member lists them:

- An argument of another type than the member declares is refused
  `ArgumentTypeRefusedError` (`'wrong_argument_type'`).
- A value outside a closed set -- an axis name, an image mode, an output
  format -- is refused `ArgumentRefusedError` (`'not_offered'`), its `offered`
  listing what is.

A number that is not finite -- NaN or an infinity -- is refused by the member
that takes it, `ArgumentRefusedError` (`'not_a_number'`; a setting written by
path, `SettingRefusedError` with the same reason), and its `Raises:` lists it.
A number outside its fixed rule is `'out_of_range'` at either door, with
`limits` when the rule has both ends. Over REST it is refused
before the member runs (`'invalid_request'`).

```python raises=ArgumentTypeRefusedError:wrong_argument_type
scope.motion.move_z('5000')                 # a str where a float is declared
```

```python
from lumascope import ArgumentRefusedError

try:
    scope.motion.axis_limits('W')       # no axis W
except ArgumentRefusedError as refused:
    assert (refused.reason, refused.offered) == ('not_offered', ('X', 'Y', 'Z', 'T'))
```

```python raises=ArgumentRefusedError:not_a_number
scope.motion.move_z(float('nan'))           # refused by move_z itself
```

### Remedies

A refusal that LumaViewPro can answer itself, on your say-so,
carries that action as `remedy`: its name (`RemedyName`:
`recover_file_writer`, `repair_camera_driver`) and the words of the offer.
A refusal whose answer is your own call, such as a home before a move
(`'position_unknown'`), has `remedy` None, and its words say what to do. Take a remedy through
`session.apply_remedy(refusal.remedy)`, the one door to every remedy and the
one place its name becomes an action. The remedy still decides whether it
applies now.

## Bringing a session up

Every host brings a scope up the same way, through `ScopeSession.create`:
LumaViewPro at launch, the REST server as it starts, and a script as
[Bringing a session up in Python](python.md#bringing-a-session-up-in-python)
shows. `create` returns a session whose scope is connected and configured -- the
stored objective, plate, frame, binning and image mode applied -- and whose
camera is streaming. A new install does not yet know which objective is
mounted. Until its question is answered (below), a saved still and a run are
refused `ObjectiveUnknownError` (`'none_selected'`). `create` does **not** move
anything. The scope's position is unknown until it is homed. A move or a run
before that is refused (`'position_unknown'`), naming each axis with its state
as data in `axes_without_position`:
`{"X": "unknown", "Y": "unknown", "Z": "unknown"}` over REST. A step saved then is refused the same way,
`'position_unknown'`, and a run that loses a position ends
`'position_lost'` with the axes in its `RunOutcome`. `run_application_session()` is the
startup motion LumaViewPro itself runs at launch: every axis homed, then the
turret moved to slot 1. `start_application_session()` starts the same motion
and returns its [`Operation`](python.md#operations) without waiting. While the
home runs it holds the scope ([The holders](#the-holders);
[Homing is required](motion.md#homing-is-required)). A home that fails is not raised: it is reported as an
outcome ([Hearing outcomes](python.md#hearing-outcomes); over REST an `outcome`
event). The motion ends there, without the turret move, so read
`scope.motion.axes_without_position` before you trust a position.
LumaViewPro and the REST server (`python -m lumascope.rest`) run the startup
motion before a client's first call, and still serve after a failed home, so a
client can home again.

When the session's first startup motion ends, it reads the PC once: memory
available, free space on the save drive, and each other program's share of the
CPU since `create`. On Windows it also reads the PC's GPUs, each typed as
Windows types it: integrated or discrete on a PC with one of each, the discrete
one being the GPU Windows names High performance. A host whose window draws
hands `create` the GPU it draws on (`drawing_gpu`). The reading is
`host_reading`, and `status` carries it. `LowSystemResourcesNotice` is reported
then, once, when memory is under 4 GB, free space under 50 GB, another program
over 30% of the machine, or the window draws on the integrated GPU beside a
discrete one.

### The plate, the objective and the model

The plate is chosen by
`select_labware`, which answers whether it changed. The installation's plates
and objectives are `scope.all_labware` and `scope.all_objectives`, in the
order they are offered. Each record carries the `id` `select_labware` or
`select_objective` takes. `scope.labware` and `scope.objective` look one up by
it, and `scope.labware` also takes a name a plate has retired. A name the
catalog does not hold is refused `CatalogNameRefusedError`, whose `offered`
lists the names it does hold.
The objective is chosen by
`select_objective` ([The turret](motion.md#the-turret)). On a scope with a
turret, the objective is the assignment of the slot in the light path. Which
objective is in which slot is set by `assign_turret_objective`.
`session.objective_id` raises `ObjectiveUnknownError` while the objective in the
light path is unknown, its reason saying why (`'slot_unassigned'` while the slot
holds none).
Where nobody has said which objective is in the light path, the session asks
rather than guesses. `objective_question` answers the question owed, or None,
and `confirm_objective` answers it. On a turret scope, binding a slot with
`assign_turret_objective` is the same answer. The pixel size and field of view follow from
the objective and the camera, and refuse with the same cause while either is
unknown. A display reads both as a
state: `runtime_state.optics_status` answers the objective and the field of
view, each its value or the refusal that says why it is unknown, raised to
no one.

Some scopes report their model and some do not. For one that does not,
`select_model` stores the model it is brought up as next time. The running
model never changes: a model that is not the running one is reported as
`'scope_model_deferred'`, waiting for the next start. A scope whose board names
its model (`scope.channel_identity.model_named_by_board`) runs as the board's
model at every start. So `select_model` there is refused
(`SettingRefusedError`, `'model_named_by_board'`), and nothing is stored.
`session.selected_model` is the stored model, `scope.channel_identity.model` the
running one, and `session.model_at_next_start` the model the next start runs
as while it differs from the running one, else None.

```python model=LS850T
print([plate.id for plate in scope.all_labware])        # every plate offered
print(scope.labware('384 well microplate').columns)     # 24
print(session.select_labware('384 well microplate'))    # True: the plate changed
question = session.objective_question
if question is not None:                                # which objective is in slot 1?
    session.confirm_objective(question.proposed, slot=question.turret_slot)
print(session.objective_id, scope.runtime_state.pixel_size_um)
```

### One session per scope

A real scope is opened by one session at a time:
while LumaViewPro, a REST server or another script drives the microscope, or
another real scope is open in the same process, `create(..., simulate=False)`
is refused `ScopeInUseError` (`'scope_in_use'`) with nothing opened. A
simulated scope takes no such lock. LumaViewPro and the REST server also run
one at a time on a computer, simulated or not: a second one says another is
already running and does not start. The lock is a port on this computer,
`lvp_lock_port` in the installation's settings, its `data/current.json`
(shipped `43101`), so two installations on one computer share it unless one's
port is changed there.

## Who may act

### The control session

A client that must not be interfered with -- a plate
reader running a long experiment, an integration that owns the scope -- takes
the control session. While it is held, only work in it writes to the scope:
every other caller's command, setting, file, run, recording, home or diagnostic
is refused `ControlSessionRefusedError` (`'control_session_held'`), naming the
session. A stop is never refused, and the display preferences stay open to
everyone. The session is taken only while nothing of anyone else's is under
way. It ends when its holder goes silent. Once 180 s pass with no call in it,
no request naming it, and nothing it started still running (a move, a run, a
recording), the next caller it would refuse ends it instead. The holder's next
call is then refused `'control_session_ended'`. Any call in it renews it. Over
REST, any request naming it renews it, a read or a job poll included. A holder
that waits longer between calls renews it with no other call (`session.renew_control(session_id)`; over REST
`POST /api/v1/renew_control` with `{"session_id": ...}`).

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

Work naming a session that has ended, or was never issued, is refused
`'control_session_ended'`: in Python `acting_in_control` and `end_control` with
its id, over REST a write carrying it in its `Control-Session` header
([The control session](rest.md#the-control-session)). A stop needs no
session: in Python it is made outside the block, and over REST a stop or a
read carrying an ended session's id is answered. The person at the scope
can take it back without the id (`take_back_control`, in-process only).

`take_control` itself is refused while anything of anyone else's is under way:
another session (`'control_session_held'`), work holding the scope
(`'exclusive_activity_running'`), a setting or file being written
(`'write_in_flight'`), or work that has let the scope go still finishing, in
that work's own words: a run still ending or a recording closing its file
(`'exclusive_activity_running'`, naming it), a finished run's images or video
(`'files_writing'`, or `'files_writing_stalled'` with the writer's recovery),
or a build (`'post_processing_running'`). Turning the LEDs off is a write, not a stop: while a
session is held, `leds_off()` from outside it is refused.

### The holders

A run, an autofocus, a recording, a diagnostic and a home each
hold the scope for their length. A command that would change what one is doing
is refused `HardwareCommandRefusedError` (`'exclusive_activity_running'`), its
`holder` naming the work and, for a run, `holder_trigger` the host that started
it (`gui`, `rest` or `api`; None for any other holder). A recording refuses only
what would falsify its frames: the frame size, the binning, the image mode,
the turret, a light of an LED that is dark or at 0 mA, or the shown channel.
It leaves gain, exposure, focus and a lit LED's current open. What holds the scope now is `session.live_work` and
`session.held_by_other()`, and `session.status` answers it with the rest of the
scope's state (`GET /api/v1/status`, the read a client makes first).

```python tier=realistic
homing = scope.motion.start_home()          # a home, which takes seconds on this tier
```

```python raises=HardwareCommandRefusedError:exclusive_activity_running
scope.motion.move_z(100.0)                  # refused while the home holds the scope: holder home
```

```python
homing.result(timeout=120)
```

### Parts a scope does not have

A command for hardware the scope lacks is
refused before anything moves, lights or is written. Its `missing`
(a `MissingPart`) names the part, in Python and in the REST problem body:
its `part` (a `Part`), with the `axis`, `channel` or `address` that says
which where the part is one of several. A missing Ex589 LED is
`{"part": "channel_led", "channel": "Ex589", "axis": null, "address": null}`.

| Reason | Means |
|---|---|
| `'axis_absent'` | The scope has no such motion part: an axis, the turret, or no motors at all on a manual model. |
| `'not_on_this_scope'` | The scope has no such part: an LED, a channel, fan control. |
| `'not_connected'` | A board the model has (a `Board`: the motor controller, the LED controller, the camera) is not connected: it never came up, or its cable was pulled. A board is also a `Part`, by the same value. |

```python raises=HardwareCommandRefusedError:axis_absent
scope.motion.move_turret(2)                 # this scope has no turret: axis_absent
```

### A lost part stays lost

Nothing reconnects a part during a session. A part
that is not connected is named by `scope.unconnected_parts` until LumaViewPro
restarts, so `'not_connected'` is a restart refusal (REST 503), and a run that
loses a part ends at once: `'camera_disconnected'` when the camera alone is
gone, as a recording ends, and `'hardware_disconnected'` for any other board or
several. A run refused `'not_connected'` and a run that ends either way name
their boards as data, in `unconnected_parts` on the `ProtocolRunRefusedError`
and on the run's `RunOutcome`: `["led"]` over REST.

## State and settings

### One store per fact

Each piece of the scope's state has one owner, and
every reader asks it: the user's settings are the session's, which LED is lit
is the illumination API's, where an axis is the motion API's. Nothing keeps a
copy to fall out of step.

### Settings by path

A setting is named by its dotted path into the settings
-- `video.max_fps`, `zstack.range` -- and read and written through the
session. A write is judged before it is stored: a value outside its range is
refused, and the refusal names the path and the range. A value past a limit of
the attached hardware is set at that limit and reported instead
([Intent and what was delivered](#intent-and-what-was-delivered)).

```python
session.update_settings('video.max_fps', 30)
assert session.setting('video.max_fps') == 30
print(session.setting('video'))         # a whole block, as a dict
```

### A channel's settings

A channel's settings are one record,
[`ChannelSettings`](../api-reference.md#ChannelSettings).
`session.channel_settings(channel)` reads them all.
`session.set_channel_settings(channel, ...)` writes any of them, judged whole:
every field named is stored, or none. It answers the record as stored. A
field left out is left as it is; None is a value (`acquire=None` stops the
channel acquiring).

A write to the shown channel's gain, exposure or current is on the camera and
its LED when it returns, whichever member makes it: this one, a Load of a
protocol's Layer Settings, or a step visited. A write to a channel not shown is
only stored, and is used the next time that channel is shown or captured. While
a run or a diagnostic holds the scope, a write is only stored, and the shown
channel's settings go to the camera and its LED when the run or diagnostic
ends. While a home holds the scope, a write that would change the shown
channel's camera or LED is refused. Through this member or a Load, the camera
and LED take the shown channel's write before it is stored. If either refuses
it, the value is not stored: the write raises (`CameraSettingRejected` for the
camera), and the channel keeps the value it had.

```python
bf = session.set_channel_settings('BF', illumination_ma=5, gain_db=2.0)
assert session.channel_settings('BF') == bf
```

```python raises=SettingRefusedError:out_of_range
session.update_settings('video.max_fps', 300)    # out_of_range: at most 200
```

A setting that a member owns is written only through that member, since the
member also applies it: `binning` is the camera's (`scope.camera.set_binning`),
`microscope` the session's `select_model`. Writing or reading it by path is
refused `'has_member'`, its `member` naming the member to call. The reasons a path write is refused
-- `'has_member'`, `'host_only'`, `'installation_only'`, `'not_a_setting'`,
`'block'`, `'not_a_number'`, `'wrong_kind'`, `'out_of_range'`, `'not_offered'` -- are
[`SettingRefusedError`'s](../api-reference.md#SettingRefusedError).

```python raises=SettingRefusedError:has_member
session.update_settings('binning', 2)       # has_member: scope.camera.set_binning
```

A listener hears every accepted write, with the paths it changed:

```python
changed = []
session.add_settings_listener(changed.append)
session.update_settings('video.max_fps', 60)
assert ('video.max_fps',) in changed
session.remove_settings_listener(changed.append)
```

### Intent and what was delivered

A setting keeps the value asked for, even
when the hardware delivers less; the member that drives the hardware answers
what it delivered. An LED board drives in steps -- whole milliamps on the
EL-0940, about 3.29 mA on the FX2 models -- so `led_on` answers the current it
commanded, the nearest step. A camera snaps or quantizes a gain or an exposure, so
`set_gain_db` and `set_exposure_ms` answer the value now in effect. A stored
gain or exposure outside the camera's range stays stored. What the camera would
be given for it is `scope.camera.applied_gain_db_for(gain)`,
`applied_exposure_ms_for(exposure)` and `applied_auto_gain_for(...)`.
Bring-up reports a stored value it capped (`'channel_setting_capped'`). A gain
asked of the camera directly outside its range is set at the range's end, and
the call reports it (`'value_clamped'`).

A channel's gain, exposure or current written past the camera's range or the
LED board's maximum, or a sum above 100, is stored at the limit. It is reported
the same way, once for the write. A run's step past one runs at the limit, and
its images and run folder record the value used. The protocol keeps what it
says, and the run reports the steps once when it starts.

```python
assert scope.illumination.led_on('BF', 10.37) == 10.0      # the board's step
scope.illumination.leds_off()
applied = scope.camera.applied_gain_db_for(999.0)
assert applied.capped and applied.applied_db == scope.camera.setting_limits.gain_max_db
```

### What the settings hold, and what they do not

`settings_snapshot`
answers a copy of every setting, for work that must not see a write made while
it runs (in-process; a REST client reads a setting by its path). Positions, the lit LED and the camera's live values are not settings:
each is read from its owner.

## Units and coordinate frames

A quantity's unit is the last part of its name: `z_um`, `exposure_ms`,
`gain_db`, `illumination_ma`, `period_s`. Times the API takes and answers are
seconds; the protocol file and the settings keep a run's period in minutes and
its duration in hours, as LumaViewPro's window shows them.

### Two frames

The stage moves in **stage micrometers**, each axis measured
from where it homes; X and Y count from the stage's bottom-right, so a larger
stage X is further left on the plate. A plate is addressed in **plate
millimeters**, X and Y from the plate's top-left corner. The two are related by
the plate's geometry and the **stage offset**, where the plate sits on the
stage.

```python
print(session.labware_id)                                   # the plate selected
print(scope.runtime_state.stage_offset)               # where it sits on the stage
plate = scope.runtime_state.stage_to_plate(60000.0, 40000.0)   # stage um -> plate mm
stage = scope.runtime_state.plate_to_stage(plate.x_mm, plate.y_mm)
assert round(stage.x_um) == 60000 and round(stage.y_um) == 40000
```

Both conversions use the selected plate and the live offset unless you name
another (`labware_id=`, `stage_offset=`). A conversion is not a range check: a
position off the stage's travel converts, and the move to it is refused. A protocol carries its own plate, and
its steps' X and Y are in plate millimeters on that plate; a run keeps the
offset it started with.

Moving X and Y in plate terms is motion.md's
[Moving in plate terms](motion.md#moving-in-plate-terms).
