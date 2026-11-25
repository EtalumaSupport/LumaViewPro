# Python

What a Python caller in the same process uses and a REST client does not:
bringing a session up, logging, hearing bring-up and outcomes, live frames,
events and listeners, and operations. A still as an image,
`session.capture_still`, is [A still](captures-and-files.md#a-still).

## Bringing a session up in Python

### Where it runs

The API is the `lumascope` package in LumaViewPro's
source; the installed app keeps it inside its program file, where no other
Python reaches it. Get the source and run its install script, as LumaViewPro's
[README](https://github.com/EtalumaSupport/LumaViewPro#getting-started) says
(Python 3.12 or 3.13): the script makes `lumascope` importable by the Python
it installed into, and that Python runs your program. A Python it did not
install into fails at the import, `ModuleNotFoundError: No module named
'lumascope'`. A Python whose installed packages are not the ones
`requirements.txt` pins is refused at the import, `PackagesNotAsPinnedError`.
The refusal names each package, the version installed, the version pinned and
the install script. When that Python can install into itself, it also gives the
pip line that fixes it. Re-run the install script after a pull. The source folder, the one
holding `lumaviewpro.py` and `data/`, is your program's installation (`root` below),
and the shipped live folder `./capture` is under it. The logs are under the
source folder `lumascope` is imported from ([Logging](#logging)).

### Settings

A LumaViewPro installation keeps the user's configuration in
`data/current.json`, the live file, and ships `data/settings.json`, the template
a new key's value comes from. `ScopeSession.load_user_settings` reads the user's
configuration as LumaViewPro does: the live file, checked against the template,
repaired, and completed with keys newer releases added. Where LumaViewPro would
fall back to the template and ask, it refuses: a `current.json` it cannot use is
`SettingsFileError`, and a folder that is no installation `'not_an_installation'`.
A source folder that LumaViewPro has never run in has no `current.json`: the
template is read instead, and the first saved write creates the file.
Each call reads the files afresh, so the dict it answers is yours. A plugin
inside a running LumaViewPro reads the session's settings with
`session.setting(path)`. Settings are an argument of `create`, never read
behind your back.

```python bringup
from lumascope import ScopeSession

root = '.'                                   # the LumaViewPro source folder, which holds data/; here, the working folder
settings = ScopeSession.load_user_settings(root)
session = ScopeSession.create(
    settings,
    source_path=root,                        # the installation the scope is built from
    simulate=True,                           # False (the default) opens the real hardware
    settings_file=ScopeSession.user_settings_file(root),   # where accepted writes are saved
)
session.run_application_session()           # the startup motion: home, then the turret
scope = session.scope
question = session.objective_question      # a new install: which objective is mounted?
if question is not None:                     # until answered, a saved still and a run are refused
    session.confirm_objective(question.proposed, slot=question.turret_slot)
```

`create` warns once (`FutureWarning`) that the API is pre-release until
`4.0.0` ([the note](../api-guide.md)). Its other keywords are
[its reference's](../api-reference.md#ScopeSession.create).
`run_application_session(disable_homing=True)` runs no startup motion at all.

### What `create` reads from the settings

The session holds its own copy of
the dict you pass: `create` changes nothing in it, and a write into it afterwards
changes no setting; `session.update_settings(path, value)` does. The copy is
completed from the template, and each of its values is judged as a write of it
would be: a value outside its range is refused (`SettingRefusedError`, naming
the path), never clamped. When the motor board names its model, the scope
runs as the board's model at every start. `settings['microscope']` is left as
it was saved (`session.selected_model`), for a scope that does not report its
own ([Bringing a session up](concepts.md#bringing-a-session-up)).
`settings['simulator_tier']` picks the simulator's fidelity
([The simulator](simulator.md#the-simulator)).

### Saving

Given `settings_file`, the session saves every accepted write
itself, within a second and at shutdown, once the scope is up. While the
settings are provisional (`session.bring_up_record.settings_set_aside` is
not None), or when the scope found no hardware at bring-up, writes are kept
unsaved and the session says so once (`SettingsNotSavedNotice`). Without `settings_file` nothing is
saved, which is what a test or a one-off script usually wants.

### Logging

Importing the API configures no logging: your `basicConfig()`,
the root logger and your own records stay yours. The session's first call
(`load_user_settings` or `create`) starts LumaViewPro's log folder,
`logs/LVP_Log` under the source folder `lumascope` is imported from, whatever
`root` names, so a script leaves the support bundle without a line of setup.
LumaViewPro logs everything under one logger, `LVP`, and
the files hang off it. Its main-log records also reach your logging as any
library's do, so a handler on `LVP` captures them. Its level is the setting
`logging.default.level` (shipped `VERBOSE`). `create` applies it and the end
of the startup motion applies it again, so a `setLevel` on `LVP` made before
then is replaced. To quiet it, set `logging.default.level` (a level's name,
such as `WARNING`) in the settings handed to `create`. Seven trace areas under it write
their own files and do not reach your handlers: `LVP.serial`, `LVP.camera`,
`LVP.api`, `LVP.protocol`, `LVP.autofocus`, `LVP.metrics` and
`LVP.gui_interactions`. Call
`lumascope.configure_logging(log_files=False)` before the first session call to
open no file at all, or `lumascope.configure_logging()` to start the folder before
it. Logging is configured once per process: a call asking for another logging
than the process already has is refused (`LoggingConfigurationRefusedError`,
`'logging_configured_otherwise'`), and changes nothing.

### Hearing bring-up

Bring-up has happened by the time `create` returns. What
it found -- a camera not found, a stored value replaced, a part missing -- is
reported as outcomes. To hear them, pass your listener to `create`
(`outcome_listener=`). A client that comes later reads the record instead
(each part's `up`, and `missing`, the parts this scope should have and lacks)
and `scope.motion_expected` (False for a model with no motion). The record's
`substitutions` and `settings_set_aside` name what bring-up replaced:

```python
record = session.bring_up_record
for part in record.parts:
    print(part.part, 'up' if part.up else 'not up')
assert record.missing == ()                # the simulator has every part
```

### A record as a dict

Every record the API answers -- the bring-up record,
`session.status`, a still's record -- has `to_dict()`: its fields as plain Python, a
nested record a dict, an enum its string, a sequence a list. It is for a caller
that takes a dict and not a record, such as MATLAB's `struct()`:

```python
facts = record.to_dict()
assert facts['missing'] == []
print([part['part'] for part in facts['parts']])
```

### Hearing it come up

Bring-up takes a few seconds on hardware. To say what
it is doing while it runs, such as a progress bar or a status line, pass
`on_progress=`, the same callback the support report takes. It is called
before each step, on the thread that called `create`. It is given the
percentage of the steps already done (0, 25, 50, 75) and the step's words (`Connecting the Motor Controller`,
`Connecting the LED Controller`, `Connecting the Camera`, `Configuring the scope`).
LumaViewPro calls `create` on a thread of its own and shows these steps while
its window stays live.

### Shutting down

`session.shutdown()` first finishes what the session is doing:
it stops a live run and recording, then waits for their files and any
post-processing to be written. Then it turns the LEDs off, stops motion and
disconnects the scope. From its first step, new work that would take the scope
-- a run, a recording, a home, a diagnostic -- is refused `SessionClosingError`
(`'session_closing'`). Once it has closed, every command is refused
`'scope_disconnected'`. It waits for the work's files with no time limit.
`discard_pending_video()` drops the video frames still queued, so a close stuck
behind a slow disk can end. A second call during the close waits for it.

```python
session.shutdown()
assert session.live_work.closed
```

## Hearing outcomes

Every outcome the API reports rather than raises is delivered to every
outcome listener in the process, as a
[`Notification`](../api-reference.md#Notification). That includes an outcome
nobody asked for, and a notice that answers a call you just made (`solicited`
True), such as the model saved for the next start. `shown` is whether it is for
display now. An outcome muted when it happened -- during an unattended run, at
shutdown, or the same title shown less than 10 s before -- arrives with `shown`
False. It arrives again, with the same `outcome_id` and `shown` True, if it is
later shown. A program that displays opens only the shown ones; one that records keeps
them all. Each carries the outcome's `reason` and `cause`, so a listener
branches on them as a caller does on a raised refusal. A report made with no
outcome behind it has kind `unclassified` and no `cause`. A quiet outcome is
never delivered. `fatal` marks one that reaches a person even while an
unattended run mutes the rest.

```python
heard = []

def on_outcome(note):
    heard.append(note)
    if note.shown:
        print(note.kind, note.title, '--', note.message)

session.add_outcome_listener(on_outcome)
```

Over REST every notice, and every outcome nobody asked for, is an `outcome`
event on the event stream, sent once, at its first delivery: a muted one shown
later is not sent again. A refusal or fault that answered a call is not sent,
its asker having had it as the call's answer ([REST](rest.md)).

## Live frames

`scope.imaging.add_frame_listener(listener)` hands each frame
the camera delivers to `listener(frame)`, a read-only
[`Frame`](../api-reference.md#Frame) (`image`, `seq`, `stored`, `arrived_ns`,
`wall_ns`, `significant_bits`, `chunks`), on the camera's own
thread. A listener has 24 ms per frame: one that overruns its budget or raises
30 frames in a row is removed, and an outcome says so. Keeping a frame is safe:
it is read-only and no later frame changes it, so draw on a copy. Return
quickly.

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

## Events and listeners

Each change the scope makes is delivered to the listeners registered for it,
and over REST as an event on the event stream -- but the frame events, which
the live view carries. The
[events table](../api-reference.md#events) lists every event, the member that
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

**Threads.** A listener is called on the thread that made the change: the
IO lane for a move or an LED, the camera's thread for a frame. So a listener
returns quickly, and does not call back into the API and wait. A listener's
raise is reported once and the other listeners are still called, so a listener
need not guard itself. A host with a window tells the process once, before its
first session, how to reach its UI thread (`ScopeSession.set_ui_dispatcher`).
Run events given `on_ui_thread=True` and the lanes' completions arrive there, while a listener
still schedules its own drawing. A script never sets it.

**A run's events** are the `RunEvents` handlers given to its start:
`scan_started`, `step_started`, `frame_captured`, `scan_ended`, `run_ended`
and `files_written`, in that order, and `video_progress` as a video step
records and writes. `run_ended` comes once the run has let go of the scope, and
`files_written` once its images are on disk: chain the next work from
`files_written`, not from `run_ended`. A composite's `run_ended` comes once its
merge settles, which may be after its `files_written`. A handler's raise is
reported under its event's name. A `frame_captured` handler takes four
arguments, `(image, frames_summed, frame_significant_bits, channel)`, the last
the channel the frame was taken for. A handler written for the three before it
raises on every frame. A still (`manual_capture.start_save_still`) takes
`events` as well and sends `frame_captured` alone, once its file is written.

**Levels and edges.** The run-state listener is a level: it is called once at
registration with the state now, then on every change, so a display that
registers late is not wrong. Every other listener hears changes from its
registration on; read the current value first.

## Operations

**An Operation** is what every `start_` member returns. It is typed by what its
`result()` answers: `Operation[RunOutcome]` for a run, `Operation[None]` for a
move or a home, `Operation[tuple[SavedStill, ...]]` for a still and
`Operation[RecordingFinished]` for a recording (its `end_reason`, and how long
saving ran on). `result()` waits
for the run to let go of the scope and answers its outcome. `done()` says
whether it has, and `progress()` answers where it has got. `stop()` asks it to
stop and returns at once: the run unwinds, puts the scope back, and ends
`aborted`. A run's images keep writing after it lets go: `files` is their own
operation ([A run's files](runs.md#a-runs-files)). `result()` and `exception()` take a timeout. On the UI thread of
a host with a window, they are refused (`'wait_on_ui_thread'`) until the
operation is done, as every waiting member is. A still's `stop()` does nothing, and a move's or a
home's stop can be refused or fail as `stop_motion()` can. In-process each run
also has a waiting form (`run_single_scan`, `run_protocol`, ...). Over REST a run
is a job ([Jobs](rest.md#jobs)).

```python
import threading
from lumascope import RunEvents, RunStatus

session.set_channel_settings('BF', acquire='image')
protocol = session.create_empty_protocol()          # one step, at B2
scope.motion.move_to_well('B2')
session.add_step(protocol, channels=('BF',))
session.set_protocol_schedule(protocol, period_s=600.0, duration_s=3600.0)
runner = session.protocol_runner()

ended = threading.Event()
events = RunEvents(run_ended=lambda outcome, run_dir, protocol: ended.set())
running = runner.start_protocol(protocol, sequence_name='guide_stopped', events=events)
running.stop()
assert ended.wait(timeout=60)
assert running.result().status is RunStatus.ABORTED
```
